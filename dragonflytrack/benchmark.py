"""Synthetic benchmark generator, unit tests, and evaluation metrics for DragonflyTrack-X."""

import os
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanBoxFilter, IMMKalmanBoxFilter, tlbr_to_xyah, iou_batch
from .memory import IdentityMemory
from .motion import MotionSaliency, _shift
from .embedder import SimulatedReIDEmbedder
from .tracker import DragonflyTrackX, Track, TRACKED, LOST
from .adapters import MOTWriter


def make_scenario(T=160, W=960, H=540, seed=0, amp_scale=1.0):
    rng = np.random.default_rng(seed)
    n = 6
    objs = []
    for i in range(n):
        e = rng.normal(size=64)
        e /= np.linalg.norm(e)
        objs.append(
            dict(
                p=rng.uniform([80, 80], [W - 80, H - 80]),
                v=rng.uniform(-3, 3, 2),
                A=rng.uniform(1.5, 4.0, 2) * amp_scale,
                w=rng.uniform(0.03, 0.07, 2),
                ph=rng.uniform(0, 2 * np.pi, 2),
                s=rng.uniform(24, 38),
                e=e,
            )
        )
    u = rng.normal(size=64)
    u -= (u @ objs[0]["e"]) * objs[0]["e"]
    u /= np.linalg.norm(u)
    objs[1]["e"] = 0.55 * objs[0]["e"] + np.sqrt(1 - 0.55**2) * u
    objs[1]["e"] /= np.linalg.norm(objs[1]["e"])
    twin_off = np.array([12.0, 3.0])
    seg = (30, 75)
    t_cross = 75
    forced = {0: {50, 60}, 1: {40, 65}}
    v_pre = None
    frames, gt_frames = [], {}
    for t in range(T):
        if t == seg[0]:
            v_pre = [objs[0]["v"].copy(), objs[1]["v"].copy()]
        img = np.full((H, W, 3), 0.05, np.float32)
        img += (0.02 * rng.normal(size=(H, W, 3))).astype(np.float32)
        boxes, ids = [], []
        for i, o in enumerate(objs):
            if i == 1 and seg[0] <= t < seg[1]:
                o["p"] = objs[0]["p"] + twin_off
                o["v"] = objs[0]["v"].copy()
            else:
                o["p"] = (
                    o["p"] + o["v"] + o["A"] * np.sin(o["w"] * t + o["ph"])
                )
                for k in range(2):
                    lim = W if k == 0 else H
                    if o["p"][k] < 60 or o["p"][k] > lim - 60:
                        o["v"][k] *= -1
                        o["p"][k] = np.clip(o["p"][k], 60, lim - 60)
            s = o["s"]
            box = np.r_[o["p"] - s / 2, o["p"] + s / 2]
            ids.append(i)
            boxes.append(box)
            cx, cy = o["p"]
            r = s / 2
            x1 = int(max(0, cx - r))
            x2 = int(min(W, cx + r + 1))
            y1 = int(max(0, cy - r))
            y2 = int(min(H, cy + r + 1))
            yy, xx = np.mgrid[y1:y2, x1:x2]
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            img[y1:y2, x1:x2][m] = 0.24 + 0.12 * i
        if t == t_cross and v_pre is not None:
            shared = objs[1]["v"].copy()
            objs[0]["v"] = v_pre[1]
            objs[1]["v"] = shared
        frames.append(np.clip(img, 0, 1))
        gt_frames[t + 1] = [(i + 1, b) for i, b in zip(ids, boxes)]
    return frames, gt_frames, objs, forced, seg


def make_dets(gt_frames, miss, forced, seg, seed=1, W=960, H=540):
    rng = np.random.default_rng(seed)
    out = {}
    for f, gt in gt_frames.items():
        t = f - 1
        d = []
        for gid, box in gt:
            i = gid - 1
            m = rng.random() < miss
            if seg[0] <= t < seg[1] and t in forced.get(i, set()):
                m = True
            if not m:
                d.append(
                    np.r_[
                        box + rng.normal(0, 1.5, 4), rng.uniform(0.55, 0.95)
                    ]
                )
                if rng.random() < 0.12:
                    d.append(
                        np.r_[
                            box + rng.normal(0, 4, 4), rng.uniform(0.15, 0.45)
                        ]
                    )
        for _ in range(2):
            x, y = rng.uniform(0, W - 60), rng.uniform(0, H - 60)
            d.append(
                np.array([
                    x,
                    y,
                    x + rng.uniform(20, 50),
                    y + rng.uniform(20, 50),
                    rng.uniform(0.1, 0.4),
                ])
            )
        out[f] = np.array(d).reshape(-1, 5)
    return out


def evaluate_full(ts_frames, gt_frames, iou_thr=0.5):
    last_map, pair_counts = {}, {}
    idsw = tp = fp = fn = total_gt = total_res = 0
    switches = []
    for f in sorted(gt_frames):
        gt = gt_frames[f]
        res = ts_frames.get(f, [])
        gB = np.array([b for _, b in gt])
        gids = [i for i, _ in gt]
        rB = np.array([b for _, b, _ in res]) if res else np.empty((0, 4))
        rids = [i for i, _, _ in res]
        total_gt += len(gids)
        total_res += len(rids)
        iou = iou_batch(gB, rB)
        pairs = sorted(
            (
                (iou[gi, rj], gi, rj)
                for gi in range(len(gB))
                for rj in range(len(rB))
                if iou[gi, rj] >= iou_thr
            ),
            key=lambda x: -x[0],
        )
        sg, sr = set(), set()
        for v, gi, rj in pairs:
            if gi in sg or rj in sr:
                continue
            sg.add(gi)
            sr.add(rj)
            if gids[gi] in last_map and last_map[gids[gi]] != rids[rj]:
                idsw += 1
                switches.append((f, gids[gi]))
            last_map[gids[gi]] = rids[rj]
            pair_counts[(gids[gi], rids[rj])] = (
                pair_counts.get((gids[gi], rids[rj]), 0) + 1
            )
        tp += len(sg)
        fn += len(gB) - len(sg)
        fp += len(rB) - len(sr)
    mota = 1.0 - (fn + fp + idsw) / max(1, total_gt)
    if pair_counts:
        gu = sorted({g for g, _ in pair_counts})
        tu = sorted({t for _, t in pair_counts})
        G = {g: i for i, g in enumerate(gu)}
        T = {t: i for i, t in enumerate(tu)}
        C = np.zeros((len(gu), len(tu)))
        for (g, t), cnt in pair_counts.items():
            C[G[g], T[t]] = cnt
        ri, ci = linear_sum_assignment(-C)
        idtp = float(C[ri, ci].sum())
    else:
        idtp = 0.0
    idf1 = 2 * idtp / max(1.0, 2 * idtp + (total_res - idtp) + (total_gt - idtp))
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "IDSW": idsw,
        "MOTA": round(mota, 4),
        "IDF1": round(idf1, 4),
        "switches": switches,
    }


def unit_tests():
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    kf = KalmanBoxFilter([100, 100, 130, 130])
    p = np.array([115.0, 115.0])
    for _ in range(15):
        kf.predict()
        p = p + 3.0
        kf.update([p[0] - 15, p[1] - 15, p[0] + 15, p[1] + 15], score=0.9)
    assert np.linalg.norm(kf.x[:2] - p) < 2.0
    assert (
        kf.gating_distance(tlbr_to_xyah([p[0] - 15, p[1] - 15, p[0] + 15, p[1] + 15]))
        < 9.49
    )
    assert (
        kf.gating_distance(tlbr_to_xyah([p[0] + 20, p[1] + 20, p[0] + 50, p[1] + 50]))
        > 9.49
    )

    imm = IMMKalmanBoxFilter([100, 100, 130, 130])
    p = np.array([115.0, 115.0])
    for _ in range(10):
        imm.predict()
        p = p + 3.0
        imm.update([p[0] - 15, p[1] - 15, p[0] + 15, p[1] + 15], score=0.9)
    m_smooth = imm.maneuver_prob
    for _ in range(6):
        imm.predict()
        p = p + 12.0
        imm.update([p[0] - 15, p[1] - 15, p[0] + 15, p[1] + 15], score=0.9)
    assert imm.maneuver_prob > max(m_smooth, 0.2), (m_smooth, imm.maneuver_prob)

    mem = IdentityMemory(dim=64, seed=1)
    f = rng.normal(size=64)
    f /= np.linalg.norm(f)
    mem.update(1, f, 0.05)
    assert not mem.has(1)
    for _ in range(30):
        g = f + 0.05 * rng.normal(size=64)
        g /= np.linalg.norm(g)
        mem.update(1, g, 0.9)
    assert len(mem.bank[1]) <= 8
    mem.update(2, rng.normal(size=64), 0.9)
    mem.update(3, rng.normal(size=64), 0.9)
    s0 = float(mem.mu[2] @ mem.mu[3])
    mem.repel(2, 3)
    assert float(mem.mu[2] @ mem.mu[3]) < s0

    sal = MotionSaliency()
    W, H = 120, 120
    yy, xx = np.mgrid[0:H, 0:W]
    blob = lambda cx: np.exp(-((xx - cx) ** 2 + (yy - 60.0) ** 2) / 40.0)
    fr1 = np.zeros((H, W, 3), np.float32)
    fr1[..., 0] = blob(40)
    sal(fr1)
    fr2 = np.zeros((H, W, 3), np.float32)
    fr2[..., 0] = blob(43)
    s1 = sal(fr2)
    assert s1.max() > 0.5
    assert s1[55:66, 36:52].max() > 0.3
    assert s1[0:20, 0:20].max() < 0.05

    trk = DragonflyTrackX()
    stub = SimpleNamespace(
        kf=SimpleNamespace(
            x=np.array([100.0, 100.0, 1.0, 30.0, 4.0, 0.0, 0.0, 0.0])
        )
    )
    a = trk._align([stub], np.array([[130.0, 85.0, 160.0, 115.0, 0.9]]))
    assert abs(a[0, 0] - 1.0) < 1e-9, a

    trk = DragonflyTrackX()
    assert trk.update(np.empty((0, 5))) == []
    trk.update(np.array([[10, 10, 40, 40, 0.9]]))
    assert trk.update(np.empty((0, 5))) == []

    for box_filter in (
        KalmanBoxFilter([100, 100, 130, 140]),
        IMMKalmanBoxFilter([100, 100, 130, 140]),
    ):
        models = getattr(box_filter, "models", [box_filter])
        for model in models:
            model.x[2:4] = [0.01, 2.0]
            model.x[6:8] = [-0.1, -3.0]
        box_filter.predict()
        assert box_filter.x[2] >= 1e-3 and box_filter.x[3] >= 1.0
        assert box_filter.x[6] >= 0.0 and box_filter.x[7] >= 0.0

    spawn_guard = DragonflyTrackX(max_age=60, reid_dim=64, reid_min=1.0)
    assert spawn_guard.reid_max_age == 15
    assert spawn_guard.spawn_gate_age == spawn_guard.reid_max_age
    stale_track = Track(
        tid=1,
        kf=KalmanBoxFilter([10, 10, 40, 70]),
        score=0.9,
        state=LOST,
        time_since_update=20,
    )
    spawn_guard.tracks.append(stale_track)
    spawn_guard._next_id = 2
    spawn_guard.memory.update(1, np.eye(64)[0], 1.0)
    new_feature = np.eye(64)[0]
    different_embedder = SimpleNamespace(
        dim=64,
        extract=lambda frame, boxes: np.repeat(
            new_feature[None, :], len(boxes), axis=0
        ),
    )
    spawn_guard.update(
        np.array([[45, 10, 75, 70, 0.9]]),
        frame=np.zeros((100, 100, 3), np.uint8),
        embedder=different_embedder,
    )
    assert spawn_guard._next_id == 3
    assert [t.tid for t in spawn_guard.tracks] == [1, 2]
    assert spawn_guard.tracks[0].state == LOST
    assert spawn_guard.stats["new"] == 1

    other_feature = np.eye(64)[1]
    other_embedder = SimpleNamespace(
        dim=64,
        extract=lambda frame, boxes: np.repeat(
            other_feature[None, :], len(boxes), axis=0
        ),
    )
    for confidence in (0.9, 0.3):
        appearance_guard = DragonflyTrackX(reid_dim=64)
        old_track = Track(
            tid=1,
            kf=IMMKalmanBoxFilter([10, 10, 40, 70]),
            score=0.9,
            state=TRACKED,
        )
        appearance_guard.tracks.append(old_track)
        appearance_guard._next_id = 2
        appearance_guard.memory.update(1, np.eye(64)[0], 1.0)
        appearance_guard.trace = []
        appearance_guard.update(
            np.array([[10, 10, 40, 70, confidence]]),
            frame=np.zeros((100, 100, 3), np.uint8),
            embedder=other_embedder,
        )
        assert not any(
            event[1] in ("A", "B", "C") and event[2] == 1
            for event in appearance_guard.trace
        )
        if confidence >= appearance_guard.new_thresh:
            assert appearance_guard._next_id == 3
            assert [t.tid for t in appearance_guard.tracks] == [1, 2]
        else:
            assert appearance_guard._next_id == 2
            assert old_track.state == LOST
        assert (
            appearance_guard.memory.similarity(1, other_feature)
            < appearance_guard.reid_veto
        )

    for cls in (KalmanBoxFilter, IMMKalmanBoxFilter):
        f = cls([100, 100, 130, 135])
        for _ in range(4):
            f.predict()
            f.update([102, 101, 131, 136], score=0.8)
        Zs = np.array([
            tlbr_to_xyah([90 + 3 * k, 95, 125 + 3 * k, 134]) for k in range(8)
        ])
        assert np.allclose(
            [f.gating_distance(z) for z in Zs], f.gating_distances(Zs)
        )

    print("unit_tests: all passed")


def run_config(
    name,
    frames,
    gt_frames,
    dets,
    objs,
    use_reid,
    use_imm,
    beta,
    recovery,
    out_path=None,
):
    lat = {i: objs[i]["e"].copy() for i in range(len(objs))}
    rng = np.random.default_rng(99)
    emb = SimulatedReIDEmbedder(dim=64, noise=0.02, seed=5) if use_reid else None
    trk = DragonflyTrackX(
        use_imm=use_imm, beta_intercept=beta, use_stmd_recovery=recovery
    )
    ts_frames = {}
    writer = MOTWriter(out_path) if out_path else None
    t0 = time.perf_counter()
    for f in sorted(gt_frames):
        for i in lat:
            lat[i] = lat[i] + rng.normal(0, 0.02, 64)
            lat[i] /= np.linalg.norm(lat[i])
        gt = gt_frames[f]
        if emb is not None:
            emb.register(np.array([b for _, b in gt]), [i - 1 for i, _ in gt], lat)
        out = trk.update(dets[f], frame=frames[f - 1], embedder=emb)
        ts_frames[f] = out
        if writer:
            writer.write(f, out)
    ms = (time.perf_counter() - t0) / max(1, len(gt_frames)) * 1e3
    if writer:
        writer.close()
    m = evaluate_full(ts_frames, gt_frames)
    sw = m.pop("switches")
    near = [s for s in sw if 70 <= s[0] <= 90]
    st = trk.stats
    print(
        f"{name:42s} MOTA={m['MOTA']:.4f} IDF1={m['IDF1']:.4f} "
        f"IDSW={m['IDSW']:3d} (70-90: {len(near):2d}) TP={m['TP']} "
        f"FP={m['FP']} FN={m['FN']} IDs={trk._next_id-1} new={st['new']} "
        f"del={st['deleted']} dupblk={st['dup_blocked']} "
        f"foveal={st['foveal_ops']} lazy={st['lazy_ops']} "
        f"reanchor={st['reanchors']} repel={st['repels']} | {ms:.1f} ms/frame"
    )
    return m


def run_demo():
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results_x"
    )
    os.makedirs(out_dir, exist_ok=True)
    print("== component unit tests ==")
    unit_tests()
    frames, gt_frames, objs, forced, seg = make_scenario()
    dets = make_dets(gt_frames, 0.10, forced, seg)
    print("== running baseline demo ==")
    run_config(
        "FULL (IMM+ReID+interception)",
        frames,
        gt_frames,
        dets,
        objs,
        True,
        True,
        0.35,
        False,
        os.path.join(out_dir, "dragonflytrack_x.txt"),
    )
