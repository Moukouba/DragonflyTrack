"""ReID embedder adapters and wrappers for DragonflyTrack-X."""

import numpy as np
from .kalman import iou_batch


class ReIDEmbedder:
    """Base ReID embedder interface."""

    def extract(self, frame, tlbrs):
        raise NotImplementedError


class SimulatedReIDEmbedder(ReIDEmbedder):
    """Simulated ReID embedder for synthetic benchmark testing."""

    def __init__(self, dim=64, noise=0.02, seed=0):
        self.dim, self.noise = dim, noise
        self.rng = np.random.default_rng(seed)
        self._boxes = np.empty((0, 4))
        self._ids = np.empty(0, int)
        self._lat = {}

    def register(self, boxes, ids, latents):
        self._boxes = np.asarray(boxes, float).reshape(-1, 4)
        self._ids = np.asarray(ids, int)
        self._lat = latents

    def extract(self, frame, tlbrs):
        tlbrs = np.atleast_2d(np.asarray(tlbrs, float))
        out = self.rng.normal(0, 1.0, (len(tlbrs), self.dim))
        if len(self._boxes):
            iou = iou_batch(tlbrs, self._boxes)
            best, ok = iou.argmax(1), iou.max(1) >= 0.5
            for i, (j, good) in enumerate(zip(best, ok)):
                if good:
                    out[i] = self._lat[self._ids[j]] + self.rng.normal(
                        0, self.noise, self.dim
                    )
        return out / np.maximum(
            np.linalg.norm(out, axis=1, keepdims=True), 1e-9
        )


class HSVColorHistEmbedder(ReIDEmbedder):
    """Fast spatial HSV color histogram appearance embedder."""

    def __init__(self, h_bins=8, s_bins=4, v_bins=4):
        import cv2

        self.cv2 = cv2
        self.h_bins, self.s_bins, self.v_bins = h_bins, s_bins, v_bins
        self.dim = (h_bins * s_bins * v_bins) * 2  # 128 dims

    def extract(self, frame, tlbrs):
        if frame is None or len(tlbrs) == 0:
            return np.zeros((len(tlbrs), self.dim), np.float32)
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:   # float [0,1] frames: bin ranges assume 0..255
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        H, W = frame.shape[:2]
        hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2HSV)

        out = []
        for x1, y1, x2, y2 in tlbrs:
            ax1, ay1 = int(max(0, x1)), int(max(0, y1))
            ax2, ay2 = int(min(W, x2)), int(min(H, y2))
            if ax2 <= ax1 or ay2 <= ay1:
                out.append(np.zeros(self.dim, np.float32))
                continue
            crop = hsv[ay1:ay2, ax1:ax2]
            ch = crop.shape[0]
            top, bot = crop[: ch // 2], crop[ch // 2 :]

            h_top = self.cv2.calcHist(
                [top], [0, 1, 2], None, [self.h_bins, self.s_bins, self.v_bins], [0, 180, 0, 256, 0, 256]
            ).ravel()
            h_bot = self.cv2.calcHist(
                [bot], [0, 1, 2], None, [self.h_bins, self.s_bins, self.v_bins], [0, 180, 0, 256, 0, 256]
            ).ravel()

            vec = np.concatenate([h_top, h_bot])
            n = np.linalg.norm(vec)
            out.append(vec / n if n > 1e-9 else vec)

        return np.array(out, np.float32)


class CallableReIDEmbedder(ReIDEmbedder):
    """Wraps any crop->embedding callable into the ReIDEmbedder interface."""

    def __init__(
        self, crop_fn, dim, crop_size=(128, 256), margin=0.10, batch=128
    ):
        import cv2

        self._cv2 = cv2
        self.crop_fn, self.dim = crop_fn, dim
        self.crop_size, self.margin, self.batch = crop_size, margin, batch

    def _crops(self, frame, tlbrs):
        H, W = frame.shape[:2]
        ch, cw = self.crop_size[1], self.crop_size[0]
        out = np.zeros((len(tlbrs), ch, cw, 3), np.uint8)
        for i, (x1, y1, x2, y2) in enumerate(tlbrs):
            w, h = x2 - x1, y2 - y1
            mx, my = w * self.margin, h * self.margin
            ax1 = int(max(0, x1 - mx))
            ay1 = int(max(0, y1 - my))
            ax2 = int(min(W, x2 + mx))
            ay2 = int(min(H, y2 + my))
            if ax2 <= ax1 or ay2 <= ay1:
                continue
            patch = frame[ay1:ay2, ax1:ax2]
            out[i] = self._cv2.resize(
                patch, self.crop_size, interpolation=self._cv2.INTER_LINEAR
            )
        return out

    def extract(self, frame, tlbrs):
        if frame is None:
            raise ValueError("CallableReIDEmbedder needs frame pixels")
        tlbrs = np.atleast_2d(np.asarray(tlbrs, float))
        if len(tlbrs) == 0:
            return np.zeros((0, self.dim), np.float32)
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        crops = self._crops(frame, tlbrs)
        feats = [
            np.asarray(self.crop_fn(crops[s : s + self.batch]), float)
            for s in range(0, len(crops), self.batch)
        ]
        z = np.vstack(feats) if feats else np.zeros((0, self.dim))
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)


class TorchReIDEmbedder(CallableReIDEmbedder):
    """Wraps a PyTorch ReID model into the ReIDEmbedder interface."""

    def __init__(
        self,
        model,
        dim,
        crop_size=(128, 256),
        device="cpu",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        batch=128,
        margin=0.10,
    ):
        import torch

        model = model.to(device).eval()
        self._torch = torch
        self._mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor(std).view(1, 3, 1, 1).to(device)

        def crop_fn(crops):
            with torch.no_grad():
                x = torch.from_numpy(crops).to(device).permute(0, 3, 1, 2)
                x = x.float().div_(255.0)
                x = (x - self._mean) / self._std
                return model(x).cpu().numpy()

        super().__init__(
            crop_fn,
            dim=dim,
            crop_size=crop_size,
            batch=batch,
            margin=margin,
        )


class LightweightTorchReIDEmbedder(TorchReIDEmbedder):
    """Pretrained MobileNetV3 ReID feature extractor."""

    def __init__(self, crop_size=(128, 256), device=None, batch=128):
        import torch
        import torchvision.models as models

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        model.classifier = torch.nn.Identity()
        dim = 576

        super().__init__(
            model, dim=dim, crop_size=crop_size, device=device, batch=batch
        )
