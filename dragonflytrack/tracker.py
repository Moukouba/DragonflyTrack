"""Core DragonflyTrack-X tracker implementation."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import (
    KalmanBoxFilter,
    IMMKalmanBoxFilter,
    tlbr_to_xyah,
    xyah_to_tlbr,
    iou_batch,
    diou_batch,
    CHI2_4D_95,
)
from .memory import IdentityMemory
from .motion import MotionSaliency, CameraMotion

TENTATIVE, TRACKED, LOST, REMOVED = 0, 1, 2, 3


@dataclass
class Track:
    tid: int
    kf: object
    score: float
    state: int = TENTATIVE
    hits: int = 1
    time_since_update: int = 0
    last_reid_frame: int = -(10**9)
    last_obs_c: object = None      # center of the last ASSOCIATED detection
    last_obs_frame: int = 0

    def bbox(self):
        return xyah_to_tlbr(self.kf.x[:4])


class DragonflyTrackX:
    """DragonflyTrack-X multi-object tracker with Distance-IoU matching and occlusion memory recovery."""

    def __init__(
        self,
        det_thresh=0.4,
        low_thresh=0.1,
        new_thresh=0.45,
        max_age=60,
        confirm_hits=2,
        coast_frames=5,
        w_iou=1.0,
        w_mot=1.0,
        w_app=1.5,
        iou_gate=0.3,
        maha_gate=CHI2_4D_95,
        lost_maha_gate=60.0,
        reid_veto=0.70,           # was 0.60: stricter appearance veto prevents swaps
        reid_min=0.60,
        beta_intercept=0.35,
        use_imm=True,
        fovea_k=2,
        use_stmd_recovery=False,
        reid_max_age=15,
        saliency_thresh=0.45,
        dup_iou=0.6,
        repel_strength=0.03,
        conflict_dist=60.0,
        conflict_sim=0.55,
        min_quality=0.12,
        reid_dim=64,
        spawn_gate_age=None,
        h_ref=30.0,
        gmc=False,
        use_diou=True,
        use_ocsort=True,
    ):
        self.det_thresh = det_thresh
        self.low_thresh = low_thresh
        self.new_thresh = new_thresh
        self.max_age = max_age
        self.confirm_hits = confirm_hits
        self.coast = coast_frames
        self.w_iou, self.w_mot, self.w_app = w_iou, w_mot, w_app
        self.iou_gate = iou_gate
        self.maha_gate = maha_gate
        self.lost_maha = lost_maha_gate
        self.reid_veto = reid_veto
        self.reid_min = reid_min
        self.beta = beta_intercept
        self.use_imm = use_imm
        self.fovea_k = fovea_k
        self.use_stmd = use_stmd_recovery
        self.reid_max_age = min(max_age, reid_max_age)
        self.sal_thr = saliency_thresh
        self.dup_iou = dup_iou
        self.repel_strength = repel_strength
        self.conflict_dist = conflict_dist
        self.conflict_sim = conflict_sim
        self.spawn_gate_age = (
            self.reid_max_age if spawn_gate_age is None else spawn_gate_age
        )
        self.h_ref = h_ref
        self.use_diou = use_diou
        # use_ocsort: after re-associating a track that was unmatched for >1 frame, blend its
        # velocity with the virtual velocity (new detection - LAST OBSERVED center) / gap.
        # No measured benefit on the synthetic benchmark (see tests); kept as an option.
        self.use_ocsort = use_ocsort
        self.gmc = CameraMotion() if gmc is True else (gmc or None)
        self.memory = IdentityMemory(dim=reid_dim, min_quality=min_quality)
        self.saliency = MotionSaliency()
        self.tracks = []
        self._next_id = 1
        self.frame_idx = 0
        self.cam_A = None
        # set `tracker.trace = []` to record every association decision as tuples
        # (frame, stage 'A'|'B'|'C'|'NEW', track_id, det_score, iou, mahalanobis_d2, appearance_sim, frames_since_update)
        self.trace = None
        self.stats = {
            "foveal_ops": 0,
            "lazy_ops": 0,
            "new": 0,
            "deleted": 0,
            "dup_blocked": 0,
            "reanchors": 0,
            "repels": 0,
            "ocsort_updates": 0,
        }

    @staticmethod
    def _det_centers(boxes):
        b = np.asarray(boxes, float)
        return np.stack(
            [(b[:, 0] + b[:, 2]) * 0.5, (b[:, 1] + b[:, 3]) * 0.5], axis=1
        )

    def _align(self, tracks, boxes):
        v = np.array([t.kf.x[4:6] for t in tracks])
        vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        d = (
            self._det_centers(boxes)[None, :, :]
            - np.array([t.kf.x[:2] for t in tracks])[:, None, :]
        )
        dn = d / np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-9)
        return np.clip(np.einsum("nmd,nd->nm", dn, vn), -1, 1)

    def _interception(self, tracks, boxes):
        if not len(tracks):
            return np.ones((0, len(boxes)))
        v = np.array([t.kf.x[4:6] for t in tracks])
        n = np.linalg.norm(v, axis=1)
        if (n < 0.5).all():
            return np.ones((len(tracks), len(boxes)))
        a = self._align(tracks, boxes)
        return 1.0 - self.beta * np.clip(a, 0, None) ** 2

    def _observe(self, t, box):
        b = np.asarray(box, float)
        t.last_obs_c = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5])
        t.last_obs_frame = self.frame_idx

    def _quality(self, box, score):
        w = max(0.0, box[2] - box[0])
        h = max(0.0, box[3] - box[1])
        return float(
            np.clip(np.sqrt(w * h) / (100.0 * self.h_ref / 30.0), 0, 1)
            * np.clip(score, 0, 1)
        )

    def _find_conflicts(self):
        out = []
        for i in range(len(self.tracks)):
            ta = self.tracks[i]
            if not self.memory.has(ta.tid):
                continue
            ca = ta.bbox()
            pa = np.array([(ca[0] + ca[2]) / 2, (ca[1] + ca[3]) / 2])
            for j in range(i + 1, len(self.tracks)):
                tb = self.tracks[j]
                if not self.memory.has(tb.tid):
                    continue
                cb = tb.bbox()
                pb = np.array([(cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2])
                if np.linalg.norm(pa - pb) > self.conflict_dist:
                    continue
                s = float(self.memory.mu[ta.tid] @ self.memory.mu[tb.tid])
                if s >= self.conflict_sim:
                    out.append((ta.tid, tb.tid))
        return out

    def update(self, dets, frame=None, embedder=None, cam=None):
        d_emb = getattr(embedder, "dim", None)
        if d_emb is not None and int(d_emb) != self.memory.W.shape[1]:
            raise ValueError(
                f"embedder.dim={d_emb} but DragonflyTrackX(reid_dim={self.memory.W.shape[1]}); "
                f"construct the tracker with reid_dim={d_emb} (track_video() does this automatically)"
            )
        self.frame_idx += 1
        dets = np.asarray(dets, float).reshape(-1, 5)
        ok = (
            np.isfinite(dets).all(axis=1)
            & (dets[:, 2] > dets[:, 0])
            & (dets[:, 3] > dets[:, 1])
        )
        dets = dets[ok]
        dets[:, 4] = np.clip(dets[:, 4], 0.0, 1.0)
        for t in self.tracks:
            t.kf.predict()
            t.time_since_update += 1

        A = cam
        if A is None and self.gmc is not None and frame is not None:
            A = self.gmc(frame, dets)
        if A is not None and not np.allclose(A, np.eye(2, 3), atol=1e-9):
            for t in self.tracks:
                t.kf.apply_affine(A)
        self.cam_A = A

        active = [t for t in self.tracks if t.state in (TRACKED, TENTATIVE)]
        lost = [t for t in self.tracks if t.state == LOST]

        sal = (
            self.saliency(frame, A)
            if (self.use_stmd and frame is not None)
            else None
        )

        if sal is not None and lost:
            H, W = sal.shape
            for t in lost:
                if t.time_since_update < 2:
                    continue
                tb = t.bbox()
                if (
                    len(dets)
                    and iou_batch(tb[None, :], dets[:, :4]).max() >= 0.2
                ):
                    continue
                cx, cy = t.kf.x[0], t.kf.x[1]
                x1 = int(np.clip(cx - 40, 0, W - 1))
                x2 = int(np.clip(cx + 40, 1, W))
                y1 = int(np.clip(cy - 40, 0, H - 1))
                y2 = int(np.clip(cy + 40, 1, H))
                if x2 <= x1 or y2 <= y1:
                    continue
                win = sal[y1:y2, x1:x2]
                idx = int(np.argmax(win))
                resp = float(win.ravel()[idx])
                if resp < self.sal_thr:
                    continue
                py, px = np.unravel_index(idx, win.shape)
                gx, gy = x1 + px, y1 + py
                if np.hypot(gx - cx, gy - cy) > 20.0:
                    continue
                w2 = win.copy()
                w2[py, px] = 0.0
                if w2.max() > 0.0 and resp / (w2.max() + 1e-9) < 1.3:
                    continue
                nb = np.array([
                    gx - (tb[2] - tb[0]) / 2,
                    gy - (tb[3] - tb[1]) / 2,
                    gx + (tb[2] - tb[0]) / 2,
                    gy + (tb[3] - tb[1]) / 2,
                ])
                t.kf.update(nb, score=0.3 * resp, position_only=True)
                self.stats["reanchors"] += 1

        high_mask = dets[:, 4] >= self.det_thresh
        low_mask = ~high_mask & (dets[:, 4] >= self.low_thresh)
        hi_rows = np.where(high_mask)[0]
        lo_rows = np.where(low_mask)[0]
        high, low = dets[hi_rows], dets[lo_rows]

        matched, used = {}, set()
        det_feats = {}
        if active and len(high):
            P = np.array([t.bbox() for t in active])
            iou = iou_batch(P, high[:, :4])
            ov_dist = diou_batch(P, high[:, :4]) if self.use_diou else iou
            Zh = np.array([tlbr_to_xyah(b) for b in high[:, :4]])
            Z = np.stack([t.kf.gating_distances(Zh) for t in active])
            if embedder is not None:
                appearance_candidates = np.where(
                    ((Z <= self.maha_gate) & (iou >= self.iou_gate)).any(axis=0)
                )[0]
                if len(appearance_candidates):
                    emb = embedder.extract(
                        frame, high[appearance_candidates][:, :4]
                    )
                    for k, di in enumerate(appearance_candidates):
                        det_feats[int(di)] = emb[k]
                    self.stats["lazy_ops"] += len(appearance_candidates)
            inter = self._interception(active, high)
            cost = (
                self.w_iou * (1 - ov_dist)
                + self.w_mot * (1 - np.exp(-0.5 * Z / self.maha_gate))
            ) * inter
            vetoed = np.zeros_like(cost, bool)
            zexp = {j: self.memory._expand(f) for j, f in det_feats.items()}
            for i, t in enumerate(active):
                if not self.memory.has(t.tid):
                    continue
                for j, f in det_feats.items():
                    s = self.memory.similarity(t.tid, f, z=zexp[j])
                    if s < self.reid_veto:
                        vetoed[i, j] = True
                    else:
                        cost[i, j] += self.w_app * (1 - s)
            feas = (iou >= self.iou_gate) & (Z <= self.maha_gate) & ~vetoed
            r, c = linear_sum_assignment(np.where(feas, cost, 1e6))
            for i, j in zip(r, c):
                if feas[i, j]:
                    matched[i] = j
                    used.add(j)
            for i, j in matched.items():
                t = active[i]
                if self.trace is not None:
                    _s = (self.memory.similarity(t.tid, det_feats[j], z=zexp[j])
                          if (j in det_feats and self.memory.has(t.tid)) else None)
                    self.trace.append((self.frame_idx, "A", t.tid, float(high[j, 4]), float(iou[i, j]),
                                       float(Z[i, j]), _s, t.time_since_update))
                t.kf.update(high[j, :4], score=float(high[j, 4]))
                self._observe(t, high[j, :4])
                t.score = float(high[j, 4])
                t.hits += 1
                t.time_since_update = 0
                f = det_feats.get(j)
                if f is not None and self.memory.has(t.tid):
                    self.memory.update(
                        t.tid, f, self._quality(high[j, :4], t.score)
                    )
                    t.last_reid_frame = self.frame_idx

        alive = [active[i] for i in range(len(active)) if i not in matched]
        rem_high = [j for j in range(len(high)) if j not in used]

        if alive and len(low):
            P = np.array([t.bbox() for t in alive])
            iou = iou_batch(P, low[:, :4])
            ov_dist = diou_batch(P, low[:, :4]) if self.use_diou else iou
            feas = iou >= self.iou_gate
            cost = 1 - ov_dist
            low_sim = None
            low_feats = None
            if embedder is not None:
                low_feats = embedder.extract(frame, low[:, :4])
                low_exp = self.memory.expand_many(low_feats)
                low_sim = np.zeros_like(iou)
                for i, t in enumerate(alive):
                    if not self.memory.has(t.tid):
                        feas[i] = False
                        continue
                    for j, feature in enumerate(low_feats):
                        low_sim[i, j] = self.memory.similarity(
                            t.tid, feature, z=low_exp[j]
                        )
                feas &= low_sim >= self.reid_min
                cost += self.w_app * (1 - low_sim)
                self.stats["lazy_ops"] += len(low)
            r, c = linear_sum_assignment(np.where(feas, cost, 1e6))
            used_lo = set()
            for i, j in zip(r, c):
                if feas[i, j]:
                    t = alive[i]
                    if self.trace is not None:
                        self.trace.append((self.frame_idx, "B", t.tid, float(low[j, 4]), float(iou[i, j]),
                                           None, float(low_sim[i, j]) if low_sim is not None else None,
                                           t.time_since_update))
                    t.kf.update(low[j, :4], score=float(low[j, 4]))
                    self._observe(t, low[j, :4])
                    t.score = float(low[j, 4])
                    t.hits += 1
                    t.time_since_update = 0
                    if low_feats is not None and self.memory.has(t.tid):
                        self.memory.update(
                            t.tid,
                            low_feats[j],
                            self._quality(low[j, :4], t.score),
                        )
                        t.last_reid_frame = self.frame_idx
                    alive[i] = None
                    used_lo.add(j)
            alive = [t for t in alive if t is not None]
            rem_low = [j for j in range(len(low)) if j not in used_lo]
        else:
            rem_low = list(range(len(low)))

        sel = (
            np.r_[
                hi_rows[np.array(rem_high, int)], lo_rows[np.array(rem_low, int)]
            ]
            if (len(rem_high) or len(rem_low))
            else np.empty(0, int)
        )
        leftover = dets[sel] if len(sel) else np.empty((0, 5))
        e_sel = (
            embedder.extract(frame, leftover[:, :4])
            if (embedder is not None and len(leftover))
            else None
        )
        if e_sel is not None:
            self.stats["lazy_ops"] += len(leftover)

        reclaim = [
            t for t in lost if t.time_since_update <= self.reid_max_age
        ] + alive
        if reclaim and len(leftover):
            iouC = iou_batch(
                np.array([t.bbox() for t in reclaim]), leftover[:, :4]
            )
            ovC = (
                diou_batch(np.array([t.bbox() for t in reclaim]), leftover[:, :4])
                if self.use_diou
                else iouC
            )
            Zl = np.array([tlbr_to_xyah(b) for b in leftover[:, :4]])
            ZC = np.stack([t.kf.gating_distances(Zl) for t in reclaim])
            inter = self._interception(reclaim, leftover)
            if e_sel is not None:
                ze = self.memory.expand_many(e_sel)
                cos = np.array([
                    [
                        self.memory.similarity(t.tid, e_sel[j], z=ze[j])
                        for j in range(len(leftover))
                    ]
                    for t in reclaim
                ])
                # Primary appearance-based feasibility
                app_feas = (cos >= self.reid_min) & (ZC <= self.lost_maha)
                # Spatial fallback: a briefly-LOST track can reclaim a strongly
                # overlapping detection even if appearance is momentarily unreliable.
                spatial_feas = np.zeros_like(app_feas)
                for i, t in enumerate(reclaim):
                    if t.state == LOST and t.time_since_update <= 5:
                        spatial_feas[i] = (
                            (iouC[i] >= 0.5) & (ZC[i] <= self.lost_maha)
                        )
                feas = app_feas | spatial_feas
                # Prefer appearance cost where appearance is trusted; otherwise
                # fall back to IoU cost for the spatial-only matches.
                cost = np.where(
                    app_feas, (1 - cos) * inter, (1 - ovC) * inter
                )
            else:
                cost = (1 - ovC) * inter
                feas = (iouC >= 0.2) & (ZC <= self.lost_maha)
            for i, t in enumerate(reclaim):
                if e_sel is not None and not self.memory.has(t.tid):
                    feas[i] = False
            r, c = linear_sum_assignment(np.where(feas, cost, 1e6))
            took_j = set()
            for i, j in zip(r, c):
                if feas[i, j]:
                    t = reclaim[i]
                    time_gap = t.time_since_update
                    if self.trace is not None:
                        self.trace.append((self.frame_idx, "C", t.tid, float(leftover[j, 4]), float(iouC[i, j]),
                                           float(ZC[i, j]), float(cos[i, j]) if e_sel is not None else None, time_gap))
                    if self.use_ocsort and time_gap > 1 and t.last_obs_c is not None:
                        new_c = (leftover[j, :2] + leftover[j, 2:4]) * 0.5
                        dt = max(self.frame_idx - t.last_obs_frame, 1)
                        vx_virt = (new_c[0] - t.last_obs_c[0]) / dt
                        vy_virt = (new_c[1] - t.last_obs_c[1]) / dt
                        t.kf.update_velocity(vx_virt, vy_virt, alpha=0.5)
                        self.stats["ocsort_updates"] += 1

                    t.kf.update(leftover[j, :4], score=float(leftover[j, 4]))
                    self._observe(t, leftover[j, :4])
                    t.score = float(leftover[j, 4])
                    t.hits += 1
                    t.time_since_update = 0
                    t.state = TRACKED if t.state == LOST else t.state
                    if e_sel is not None:
                        self.memory.update(
                            t.tid,
                            e_sel[j],
                            self._quality(leftover[j, :4], t.score),
                        )
                        t.last_reid_frame = self.frame_idx
                    took_j.add(j)
            keep = np.array(
                [j for j in range(len(leftover)) if j not in took_j], int
            )
            leftover = leftover[keep]
            e_sel = e_sel[keep] if e_sel is not None else None

        for a, b in self._find_conflicts():
            self.memory.repel(a, b, self.repel_strength)
            self.stats["repels"] += 1

        blockers = [
            t for t in self.tracks if t.time_since_update <= self.spawn_gate_age
        ]
        spawnable = []
        for j in range(len(leftover)):
            box, sc = leftover[j, :4], leftover[j, 4]
            if sc < self.new_thresh:
                continue
            compatible_blockers = []
            for t in blockers:
                # A recently LOST track always blocks a new spawn in its vicinity,
                # regardless of current appearance similarity. This prevents a
                # duplicate ID from being created while an occluded track is
                # still within its re-association window.
                if t.state == LOST:
                    compatible_blockers.append(t)
                    continue
                if (
                    e_sel is not None
                    and self.memory.has(t.tid)
                    and self.memory.similarity(t.tid, e_sel[j]) < self.reid_veto
                ):
                    continue
                compatible_blockers.append(t)
            if compatible_blockers:
                blocker_boxes = np.array([t.bbox() for t in compatible_blockers])
                if iou_batch(box[None, :], blocker_boxes).max() > self.dup_iou:
                    self.stats["dup_blocked"] += 1
                    continue
                if embedder is not None:
                    z = tlbr_to_xyah(box)
                    if any(
                        t.kf.gating_distance(z) <= self.lost_maha
                        for t in compatible_blockers
                    ):
                        self.stats["dup_blocked"] += 1
                        continue
            spawnable.append(j)

        for t in self.tracks:
            if t.state == TENTATIVE:
                if t.hits >= self.confirm_hits:
                    t.state = TRACKED
                elif t.time_since_update >= 1:
                    t.state = REMOVED
                    self.memory.forget(t.tid)
            elif t.state == TRACKED and t.time_since_update >= 1:
                t.state = LOST
        for t in self.tracks:
            if t.state == LOST and t.time_since_update > self.max_age:
                t.state = REMOVED
                self.memory.forget(t.tid)
                self.stats["deleted"] += 1

        for j in spawnable:
            box = leftover[j, :4]
            kf = (
                IMMKalmanBoxFilter(box, h_ref=self.h_ref)
                if self.use_imm
                else KalmanBoxFilter(box, h_ref=self.h_ref)
            )
            e = e_sel[j] if e_sel is not None else None
            t = Track(tid=self._next_id, kf=kf, score=float(leftover[j, 4]))
            self._next_id += 1
            self._observe(t, box)
            if self.trace is not None:
                self.trace.append((self.frame_idx, "NEW", t.tid, float(leftover[j, 4]), None, None, None, 0))
            self.tracks.append(t)
            if e is not None:
                self.memory.update(t.tid, e, self._quality(box, t.score))
                t.last_reid_frame = self.frame_idx
            self.stats["new"] += 1

        self.tracks = [t for t in self.tracks if t.state != REMOVED]
        out = []
        for t in self.tracks:
            if t.state == TRACKED or (
                t.state == LOST and t.time_since_update <= self.coast
            ):
                out.append((t.tid, t.bbox(), t.score))
        return out