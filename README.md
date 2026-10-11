# DragonflyTrack-X: Multi-Object Tracking Module

`DragonflyTrack-X` is a modular, high-performance Multi-Object Tracker (MOT) designed for complex tracking scenarios involving sharp maneuvers, dense crowds, moving cameras, and target occlusions.

It merges bio-inspired motion saliency and quality-gated identity memory with engineering-grade BYTE-style detection cascading, Interacting Multiple Model (IMM) Kalman filtering, Distance-IoU (DIoU) association metrics, OC-SORT observation-centric momentum, and global camera-motion compensation (GMC).

---

## Features

- **Distance-IoU (DIoU) & GIoU Matching**: Replaces standard IoU with DIoU metric penalizing spatial center distances, enabling far superior matching for non-overlapping or fast-moving targets.
- **OC-SORT Observation-Centric Momentum**: Calculates virtual velocity vectors upon re-association of lost tracks to smooth out trajectory jumps post-occlusion.
- **Interacting Multiple Model (IMM) Kalman Filtering**: Dynamically blends cruise and maneuver motion models to track unpredictable target acceleration without manual noise tuning.
- **Sparse Identity Memory**: Employs random-projection feature expansion, quality-gated von Mises-Fisher (vMF) prototypes, exemplar bank eviction, and prototype repulsion.
- **Camera-Motion Compensation (GMC)**: Optical-flow corner tracking with Lucas-Kanade + RANSAC similarity transform (`gmc=True`) warps predicted track state and covariance matrices between frames.
- **Modular Inputs and Detectors**: Supports Ultralytics YOLO models, PyTorch ReID embedders, video streams, image sequences, and precomputed MOT detections.

---

## Directory Structure

```
Dragonfly/
âââ dragonflytrack/            # Core Python Package
â   âââ __init__.py            # Package exports (DragonflyTrackX, YOLODetectorAdapter, etc.)
â   âââ kalman.py              # KalmanBoxFilter, IMMKalmanBoxFilter, DIoU & GIoU metrics
â   âââ memory.py              # IdentityMemory (vMF prototype, random projections)
â   âââ motion.py              # MotionSaliency & CameraMotion (GMC optical flow)
â   âââ embedder.py            # ReID embedder interfaces (TorchReIDEmbedder, etc.)
â   âââ tracker.py             # Main DragonflyTrackX tracking logic & Track state dataclass
â   âââ adapters.py            # VideoFrameSource, YOLODetectorAdapter, MOTWriter, draw_tracks
â   âââ benchmark.py           # Synthetic scenario generator, MOTA/IDF1 metrics, unit_tests
âââ footage/                   # Sample video inputs
â   âââ Shopping People Commerce Mall Many Crowd Walking Free Stock vide.mp4
âââ results_x/                 # Tracking outputs (MOT text format & annotated video)
â   âââ footage_tracks.txt
â   âââ footage_tracked.mp4
âââ dragonflytrack_paper.tex  # Mathematical formulation source
âââ dragonflytrack_mathematics.pdf
âââ test_footage.py            # Executable test script for running tracking on video footage
âââ pyproject.toml             # Modern Python build configuration
âââ setup.py                   # Setup script for editable package installation
âââ README.md                  # Comprehensive documentation
```

---

## Installation & Environment

```bash
# Activate conda environment
conda activate tensorflow

# Install the package and Ultralytics YOLO dependencies
pip install -e ".[yolo]"
```

---

## Quickstart & Usage

```python
import cv2
from dragonflytrack import (
    DragonflyTrackX,
    VideoFrameSource,
    YOLODetectorAdapter,
    HSVColorHistEmbedder,
    draw_tracks,
)

# 1. Initialize the bundled video, YOLO26m detector, and appearance embedder
source = VideoFrameSource(
    "footage/Shopping People Commerce Mall Many Crowd Walking Free Stock vide.mp4"
)
detector = YOLODetectorAdapter("yolo26m.pt", conf=0.15, classes=(0,))  # 0: person
embedder = HSVColorHistEmbedder()

# 2. Initialize tracker with DIoU matching, OC-SORT momentum, and GMC
tracker = DragonflyTrackX(
    det_thresh=0.4,
    low_thresh=0.1,
    reid_max_age=15,
    reid_dim=embedder.dim,
    gmc=True,
    use_diou=True,
    use_ocsort=True,
)

# 3. Process video frame by frame
for fid, frame in source:
    dets = detector(frame, fid)                 # (N, 5) [x1, y1, x2, y2, conf]
    tracks = tracker.update(dets, frame=frame, embedder=embedder)

    annotated = draw_tracks(frame, tracks)
    cv2.imshow("DragonflyTrack-X", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == 27:
        break

source.close()
cv2.destroyAllWindows()
```

Lost tracks are eligible for identity recovery for 15 frames by default (`reid_max_age`). After that window, a detection starts a new track with a fresh, monotonically increasing ID. Increase `reid_max_age` when preserving identity across longer occlusions is more important than avoiding stale-ID reassignment.

---

## Verification & Unit Testing

Run the built-in component tests and the footage tracker test:

```bash
python -c "from dragonflytrack.benchmark import unit_tests; unit_tests()"
python test_footage.py yolo26m.pt
```

Output:
```
unit_tests: all passed
```

The footage test writes MOT-format tracks and an annotated video to `results_x/`.
