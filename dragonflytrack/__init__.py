"""DragonflyTrack-X: Multi-Object Tracker.

Packed into a usable, modular package.
"""

from .kalman import (
    KalmanBoxFilter,
    IMMKalmanBoxFilter,
    tlbr_to_xyah,
    xyah_to_tlbr,
    iou_batch,
    diou_batch,
    giou_batch,
    CHI2_4D_95,
)
from .memory import IdentityMemory
from .motion import MotionSaliency, CameraMotion
from .embedder import (
    ReIDEmbedder,
    SimulatedReIDEmbedder,
    HSVColorHistEmbedder,
    CallableReIDEmbedder,
    TorchReIDEmbedder,
    LightweightTorchReIDEmbedder,
)
from .tracker import (
    Track,
    DragonflyTrackX,
    TENTATIVE,
    TRACKED,
    LOST,
    REMOVED,
)
from .adapters import (
    VideoFrameSource,
    ImageFolderSource,
    YOLODetectorAdapter,
    PrecomputedDetections,
    MOTWriter,
    draw_tracks,
    track_video,
)

__version__ = "1.4.2"
__all__ = [
    "KalmanBoxFilter",
    "IMMKalmanBoxFilter",
    "tlbr_to_xyah",
    "xyah_to_tlbr",
    "iou_batch",
    "diou_batch",
    "giou_batch",
    "CHI2_4D_95",
    "IdentityMemory",
    "MotionSaliency",
    "CameraMotion",
    "ReIDEmbedder",
    "SimulatedReIDEmbedder",
    "HSVColorHistEmbedder",
    "CallableReIDEmbedder",
    "TorchReIDEmbedder",
    "LightweightTorchReIDEmbedder",
    "Track",
    "DragonflyTrackX",
    "TENTATIVE",
    "TRACKED",
    "LOST",
    "REMOVED",
    "VideoFrameSource",
    "ImageFolderSource",
    "YOLODetectorAdapter",
    "PrecomputedDetections",
    "MOTWriter",
    "draw_tracks",
    "track_video",
]
