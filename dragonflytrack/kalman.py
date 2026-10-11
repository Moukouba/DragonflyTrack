"""Kalman box filter and IMM (Interacting Multiple Model) filter for DragonflyTrack-X."""

import numpy as np

CHI2_4D_95 = 9.4877


def tlbr_to_xyah(b):
    x1, y1, x2, y2 = b
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, w / h, h])


def xyah_to_tlbr(s):
    cx, cy, a, h = s
    h = max(h, 1e-3)
    a = max(a, 1e-3)
    w = a * h
    return np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5])


def iou_batch(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    aa = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    bb = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def diou_batch(a, b):
    """Distance-IoU matrix between boxes a (M,4) and b (N,4). Returns shape (M,N) in range [-1, 1]."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iou = iou_batch(a, b)

    cx_a = (a[:, 0] + a[:, 2]) * 0.5
    cy_a = (a[:, 1] + a[:, 3]) * 0.5
    cx_b = (b[:, 0] + b[:, 2]) * 0.5
    cy_b = (b[:, 1] + b[:, 3]) * 0.5

    # Center distances squared
    rho2 = (cx_a[:, None] - cx_b[None, :]) ** 2 + (cy_a[:, None] - cy_b[None, :]) ** 2

    # Enclosing box diagonal squared
    c_x1 = np.minimum(a[:, None, 0], b[None, :, 0])
    c_y1 = np.minimum(a[:, None, 1], b[None, :, 1])
    c_x2 = np.maximum(a[:, None, 2], b[None, :, 2])
    c_y2 = np.maximum(a[:, None, 3], b[None, :, 3])

    c2 = np.maximum(c_x2 - c_x1, 1e-6) ** 2 + np.maximum(c_y2 - c_y1, 1e-6) ** 2
    return iou - (rho2 / c2)


def giou_batch(a, b):
    """Generalized-IoU matrix between boxes a (M,4) and b (N,4). Returns shape (M,N) in range [-1, 1]."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iou = iou_batch(a, b)

    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter

    c_x1 = np.minimum(a[:, None, 0], b[None, :, 0])
    c_y1 = np.minimum(a[:, None, 1], b[None, :, 1])
    c_x2 = np.maximum(a[:, None, 2], b[None, :, 2])
    c_y2 = np.maximum(a[:, None, 3], b[None, :, 3])

    c_area = np.maximum(c_x2 - c_x1, 0) * np.maximum(c_y2 - c_y1, 0)
    c_area = np.maximum(c_area, 1e-9)

    return iou - ((c_area - union) / c_area)


class KalmanBoxFilter:
    def __init__(self, bbox, h_ref=30.0):
        self.x = np.zeros(8)
        self.x[:4] = tlbr_to_xyah(bbox)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        k = max(self.x[3], 1.0) / h_ref
        r = np.array([1.0, 1.0, 0.1, 1.0]) * k
        p = np.array([10.0, 10.0, 0.1, 10.0]) * k
        self.R = np.diag(r**2)
        self.P = np.diag(np.concatenate([p, p]) ** 2)
        self._q_scale = 1.0

    def _Q(self):
        h = max(self.x[3], 1.0)
        s = 0.05
        q = np.array([
            (s * h) ** 2,
            (s * h) ** 2,
            (s * 0.1) ** 2,
            (s * h) ** 2,
            (s * h) ** 2 / 16,
            (s * h) ** 2 / 16,
            (s * 1e-2) ** 2,
            (s * h) ** 2 / 16,
        ])
        return np.diag(q) * self._q_scale

    def _constrain_box_state(self):
        for value_idx, velocity_idx, minimum in ((2, 6, 1e-3), (3, 7, 1.0)):
            if self.x[value_idx] < minimum:
                self.x[value_idx] = minimum
                if self.x[velocity_idx] < 0.0:
                    self.x[velocity_idx] = 0.0

    def predict(self):
        self.x = self.F @ self.x
        self._constrain_box_state()
        self.P = self.F @ self.P @ self.F.T + self._Q()

    def update(self, bbox, score=None, position_only=False):
        zf = tlbr_to_xyah(bbox)
        H, z, R = self.H, zf, self.R
        if position_only:
            H, z, R = self.H[:2], zf[:2], self.R[:2, :2]
        if score is not None:
            R = R * (1.0 / max(float(np.clip(score, 0.05, 1.0)), 0.05))
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        self._constrain_box_state()
        I = np.eye(8) - K @ H
        self.P = I @ self.P @ I.T + K @ R @ K.T

    def update_velocity(self, vx, vy, alpha=0.7):
        """Observation-Centric Momentum (OC-SORT): smooth velocity using virtual velocity."""
        self.x[4] = alpha * self.x[4] + (1 - alpha) * vx
        self.x[5] = alpha * self.x[5] + (1 - alpha) * vy

    def gating_distance(self, z_xyah):
        y = z_xyah - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.solve(S, y))

    def apply_affine(self, A):
        A = np.asarray(A, float)
        L, t = A[:2, :2], A[:2, 2]
        sc = float(np.sqrt(abs(np.linalg.det(L))))
        M = np.eye(8)
        M[0:2, 0:2] = L
        M[4:6, 4:6] = L
        M[3, 3] = sc
        M[7, 7] = sc
        self.x = M @ self.x
        self.x[0:2] += t
        self.P = M @ self.P @ M.T

    def gating_distances(self, Z):
        Y = Z - self.x[:4]
        S = self.H @ self.P @ self.H.T + self.R
        return np.einsum("md,md->m", Y, np.linalg.solve(S, Y.T).T)


class IMMKalmanBoxFilter:
    def __init__(self, bbox, q_maneuver=8.0, stay=0.95, h_ref=30.0):
        self.models = [
            KalmanBoxFilter(bbox, h_ref=h_ref),
            KalmanBoxFilter(bbox, h_ref=h_ref),
        ]
        self.models[1]._q_scale = q_maneuver
        self.mu = np.array([0.9, 0.1])
        p = float(np.clip(stay, 0.5, 0.999))
        self.Pi = np.array([[p, 1 - p], [1 - p, p]])
        self._mp = self.mu.copy()
        self._xc = self.models[0].x.copy()
        self._Pc = self.models[0].P.copy()

    @property
    def x(self):
        return self._xc

    @property
    def P(self):
        return self._Pc

    @property
    def maneuver_prob(self):
        return float(self.mu[1])

    def _combine(self, w):
        xs = np.stack([m.x for m in self.models])
        x = (w[:, None] * xs).sum(0)
        P = np.zeros_like(self.models[0].P)
        for j, m in enumerate(self.models):
            d = m.x - x
            P += w[j] * (m.P + np.outer(d, d))
        self._xc, self._Pc = x, P

    def predict(self):
        c = np.clip(self.Pi.T @ self.mu, 1e-12, None)
        w = (self.Pi * self.mu[:, None]) / c[None, :]
        xs = [m.x.copy() for m in self.models]
        Ps = [m.P.copy() for m in self.models]
        for j, m in enumerate(self.models):
            x0 = sum(w[i, j] * xs[i] for i in range(2))
            P0 = sum(
                w[i, j] * (Ps[i] + np.outer(xs[i] - x0, xs[i] - x0))
                for i in range(2)
            )
            m.x, m.P = x0, P0
            m.predict()
        self._mp = c
        self._combine(c)

    def update(self, bbox, score=None, position_only=False):
        zf = tlbr_to_xyah(bbox)
        z = zf[:2] if position_only else zf
        ll = np.zeros(2)
        for j, m in enumerate(self.models):
            H = m.H[:2] if position_only else m.H
            R = m.R[:2, :2] if position_only else m.R
            if score is not None:
                R = R * (1.0 / max(float(np.clip(score, 0.05, 1.0)), 0.05))
            y = z - H @ m.x
            S = H @ m.P @ H.T + R
            d2 = float(y.T @ np.linalg.solve(S, y))
            _, ldet = np.linalg.slogdet(S)
            ll[j] = -0.5 * d2 - 0.5 * ldet
            m.update(bbox, score=score, position_only=position_only)
        l = ll + np.log(self._mp + 1e-300)
        l -= l.max()
        w = np.exp(l)
        s = w.sum()
        self.mu = w / s if s > 1e-300 else self._mp
        self._combine(self.mu)

    def update_velocity(self, vx, vy, alpha=0.7):
        for m in self.models:
            m.update_velocity(vx, vy, alpha=alpha)
        self._combine(self.mu)

    def gating_distance(self, z_xyah):
        H = self.models[0].H
        S = H @ self._Pc @ H.T + self.models[0].R
        y = z_xyah - H @ self._xc
        return float(y.T @ np.linalg.solve(S, y))

    def apply_affine(self, A):
        for m in self.models:
            m.apply_affine(A)
        self._combine(self.mu)

    def gating_distances(self, Z):
        H = self.models[0].H
        S = H @ self._Pc @ H.T + self.models[0].R
        Y = Z - H @ self._xc
        return np.einsum("md,md->m", Y, np.linalg.solve(S, Y.T).T)
