"""Motion saliency and camera-motion compensation (GMC) for DragonflyTrack-X."""

import numpy as np
from scipy.ndimage import gaussian_filter


def _shift(img, dy, dx):
    H, W = img.shape
    out = np.empty_like(img)
    ys = slice(max(0, -dy), H - max(0, dy))
    yd = slice(max(0, dy), H + min(0, dy))
    xs = slice(max(0, -dx), W - max(0, dx))
    xd = slice(max(0, dx), W + min(0, dx))
    out[ys, xs] = img[yd, xd]
    if dy > 0:
        out[H - dy :, :] = out[H - dy - 1 : H - dy, :]
    if dy < 0:
        out[:-dy, :] = out[-dy : -dy + 1, :]
    if dx > 0:
        out[:, W - dx :] = out[:, W - dx - 1 : W - dx]
    if dx < 0:
        out[:, :-dx] = out[:, -dx : -dx + 1]
    return out


class MotionSaliency:
    """Reichardt small-target motion saliency correlator."""

    def __init__(self, s_small=1.5, s_wide=5.0, kappa=1.0):
        self.s_small, self.s_wide, self.kappa = s_small, s_wide, kappa
        self.prev = None
        self.dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __call__(self, frame_rgb, A=None):
        f = np.asarray(frame_rgb)
        gray = (
            f.mean(axis=2, dtype=np.float32)
            if f.ndim == 3
            else f.astype(np.float32)
        )
        Ls = gaussian_filter(gray, self.s_small)
        Lw = gaussian_filter(gray, self.s_wide)
        if self.prev is None:
            self.prev = (Ls, Lw)
            return np.zeros_like(gray)
        ps, pw = self.prev
        if A is not None and not np.allclose(A, np.eye(2, 3), atol=1e-6):
            import cv2

            Hh, Ww = gray.shape
            ps = cv2.warpAffine(
                ps,
                np.asarray(A, np.float64),
                (Ww, Hh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            pw = cv2.warpAffine(
                pw,
                np.asarray(A, np.float64),
                (Ww, Hh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        S = Wm = None
        for d in self.dirs:
            s_ = np.maximum(ps * _shift(Ls, *d) - _shift(ps, *d) * Ls, 0)
            w_ = np.maximum(pw * _shift(Lw, *d) - _shift(pw, *d) * Lw, 0)
            S = s_ if S is None else np.maximum(S, s_)
            Wm = w_ if Wm is None else np.maximum(Wm, w_)
        self.prev = (Ls, Lw)
        m = np.clip(S - self.kappa * Wm, 0, None)
        pk = m.max()
        return m / pk if pk > 1e-8 else m


class CameraMotion:
    """Global camera-motion estimation via Lucas-Kanade optical flow and RANSAC."""

    def __init__(
        self,
        scale=0.5,
        max_corners=300,
        min_inliers=10,
        max_scale_change=0.2,
        deadband_px=0.25,
        deadband_scale=1e-3,
    ):
        import cv2

        self.cv2 = cv2
        self.scale, self.mc, self.min_in = scale, max_corners, min_inliers
        self.max_ds = max_scale_change
        self.db_px, self.db_s = deadband_px, deadband_scale
        self.prev = None
        self.last_inliers = 0

    def reset(self):
        self.prev = None

    def __call__(self, frame_rgb, dets=None):
        cv2 = self.cv2
        f = np.asarray(frame_rgb)
        if f.dtype != np.uint8:
            f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
        g = cv2.resize(
            g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA
        )
        A = np.eye(2, 3)
        self.last_inliers = 0
        if self.prev is not None and self.prev.shape == g.shape:
            mask = np.full(g.shape, 255, np.uint8)
            if dets is not None and len(dets):
                for b in np.asarray(dets)[:, :4]:
                    x1, y1, x2, y2 = (np.asarray(b) * self.scale).astype(int)
                    mask[
                        max(0, y1 - 2) : max(0, y2 + 2),
                        max(0, x1 - 2) : max(0, x2 + 2),
                    ] = 0
            p0 = cv2.goodFeaturesToTrack(self.prev, self.mc, 0.01, 8, mask=mask)
            if p0 is not None and len(p0) >= self.min_in:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None)
                ok = st.ravel() == 1
                if ok.sum() >= self.min_in:
                    M, inl = cv2.estimateAffinePartial2D(
                        p0[ok],
                        p1[ok],
                        method=cv2.RANSAC,
                        ransacReprojThreshold=2.0,
                    )
                    if (
                        M is not None
                        and inl is not None
                        and int(inl.sum()) >= self.min_in
                    ):
                        sc = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
                        if abs(sc - 1.0) <= self.max_ds:
                            A = M.astype(np.float64)
                            A[:, 2] /= self.scale
                            self.last_inliers = int(inl.sum())
                            if (
                                np.hypot(*A[:, 2]) < self.db_px
                                and abs(sc - 1.0) < self.db_s
                                and abs(A[1, 0]) < self.db_s
                            ):
                                A = np.eye(2, 3)
        self.prev = g
        return A
