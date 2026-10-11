"""Input video/image sources, detector adapters, and driver utilities."""

import os
import numpy as np
from .tracker import DragonflyTrackX, TRACKED, LOST


class VideoFrameSource:
    """Iterates (frame_id, RGB uint8) from a video file or camera index."""

    def __init__(self, path, every=1, start=1):
        import cv2

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        self.every = max(1, int(every))
        self._fid = start

    def __iter__(self):
        return self

    def __next__(self):
        bgr = None
        for _ in range(self.every):
            ok, bgr = self.cap.read()
            self._fid += 1
            if not ok:
                self.cap.release()
                raise StopIteration
        import cv2

        return self._fid - 1, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class ImageFolderSource:
    """Iterates (frame_id, RGB) frames from a folder (sorted by filename)."""

    def __init__(self, folder, pattern="*"):
        import glob
        import cv2

        self.files = sorted(glob.glob(os.path.join(folder, pattern)))
        if not self.files:
            raise IOError(f"no frames match {os.path.join(folder, pattern)}")

    def __iter__(self):
        import cv2

        for fid, fp in enumerate(self.files, start=1):
            bgr = cv2.imread(fp)
            if bgr is None:
                continue
            yield fid, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class YOLODetectorAdapter:
    """Ultralytics YOLO detector adapter returning (N, 5) [x1, y1, x2, y2, conf]."""

    def __init__(
        self,
        model,
        conf=0.10,
        iou=0.60,
        imgsz=None,
        classes=(0,),
        device=None,
        verbose=False,
    ):
        if isinstance(model, str):
            from ultralytics import YOLO

            model = YOLO(model)
        self.model = model
        self.conf, self.iou, self.classes = conf, iou, classes
        self.imgsz, self.device, self.verbose = imgsz, device, verbose

    def __call__(self, frame_rgb, fid=None):
        from PIL import Image

        frame_rgb = np.asarray(frame_rgb)
        if frame_rgb.dtype != np.uint8:
            frame_rgb = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)
        kw = dict(
            conf=self.conf,
            iou=self.iou,
            classes=list(self.classes),
            verbose=self.verbose,
        )
        if self.imgsz is not None:
            kw["imgsz"] = self.imgsz
        if self.device is not None:
            kw["device"] = self.device
        res = self.model.predict(
            source=Image.fromarray(np.ascontiguousarray(frame_rgb)), **kw
        )[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 5), np.float32)
        xyxy = b.xyxy.cpu().numpy().astype(np.float32)
        conf = b.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
        return np.hstack([xyxy, conf]).astype(np.float32)


class PrecomputedDetections:
    """Replay precomputed MOT detections text file."""

    def __init__(self, path, default_conf=0.6):
        self.dets = {}
        with open(path) as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                try:
                    fid = int(float(parts[0]))
                    x, y, w, h = (float(v) for v in parts[2:6])
                    c = float(parts[6])
                except ValueError:
                    continue
                if c <= 0:
                    c = default_conf
                self.dets.setdefault(fid, []).append(
                    [x, y, x + w, y + h, float(np.clip(c, 0.0, 1.0))]
                )

    def __call__(self, frame_rgb=None, fid=None):
        return np.asarray(
            self.dets.get(fid if fid is not None else 0, []), np.float32
        ).reshape(-1, 5)


class MOTWriter:
    """Writes tracking results in MOT Challenge text format."""

    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, frame_id, tracks):
        for tid, tlbr, score in tracks:
            x1, y1, x2, y2 = tlbr
            self.f.write(
                f"{frame_id},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{score:.3f},-1,-1,-1\n"
            )

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def draw_tracks(frame, tracks):
    """Draw bounding boxes and track IDs on RGB frame."""
    import cv2

    img = np.asarray(frame).copy()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    for tid, b, s in tracks:
        x1, y1, x2, y2 = (int(v) for v in b)
        # local RandomState: same per-ID colors as before, but no reseeding of the
        # process-wide numpy RNG on every drawn box
        rs = np.random.RandomState(int(tid) % (2**32))
        color = tuple(int(c) for c in rs.randint(50, 230, size=3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, max(th, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


_draw = draw_tracks


def track_video(
    source,
    detector,
    embedder=None,
    tracker=None,
    out_path=None,
    max_frames=None,
    show=False,
    progress=0,
):
    """Driver function for end-to-end video tracking."""
    if tracker is None:
        tracker = DragonflyTrackX(
            reid_dim=(
                embedder.dim
                if embedder is not None and hasattr(embedder, "dim")
                else 64
            )
        )
    results, n = {}, 0
    writer = MOTWriter(out_path) if out_path else None
    try:
        for fid, frame in source:
            frame = np.asarray(frame)
            dets = detector(frame, fid)
            out = tracker.update(dets, frame=frame, embedder=embedder)
            results[fid] = out
            if writer:
                writer.write(fid, out)
            n += 1
            if progress and n % progress == 0:
                live = sum(
                    1 for t in tracker.tracks if t.state in (TRACKED, LOST)
                )
                print(f"[frame {fid}] live={live} dets={len(dets)}")
            if show:
                import cv2

                cv2.imshow(
                    "DragonflyTrack-X",
                    cv2.cvtColor(draw_tracks(frame, out), cv2.COLOR_RGB2BGR),
                )
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
            if max_frames is not None and n >= max_frames:
                break
    finally:
        if writer:
            writer.close()
        closer = getattr(source, "close", None)
        if callable(closer):
            closer()
    return tracker, results
