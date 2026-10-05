"""Small CNN: observation-camera image → reduced coordinates q.

Optional PyTorch dependency (``pip install -e '.[ml]'``). The rest of the
vision pipeline (marker baselines, ROM force) does not need torch.

Architecture keeps a coarse spatial layout (no global average pool) so the
network can see *where* the finger bends; outputs are trained in units of
``q / q_rms`` and rescaled on predict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:  # pragma: no cover
        raise ImportError("ImageCNN requires PyTorch; install with: pip install -e '.[ml]'") from e
    return torch, nn


def _feature_hw(h: int, w: int, n_stride2: int = 4) -> tuple[int, int]:
    for _ in range(n_stride2):
        h = (h + 1) // 2
        w = (w + 1) // 2
    return h, w


class _QNet:
    @staticmethod
    def build(in_ch: int, r: int, image_hw: tuple[int, int], hidden: int = 128, dropout: float = 0.2):
        torch, nn = _torch()
        fh, fw = _feature_hw(*image_hw)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.Conv2d(128, hidden, 3, stride=2, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(hidden * fh * fw, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(256, r),
                )

            def forward(self, x):
                return self.head(self.features(x))

        return Net()


@dataclass
class ImageCNNModel:
    """Trained image → q regressor."""

    r: int
    in_ch: int
    image_hw: tuple[int, int]
    hidden: int = 128
    dropout: float = 0.2
    q_scale: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    _net: Any = field(default=None, repr=False)
    _device: str = "cpu"

    def __post_init__(self) -> None:
        torch, _ = _torch()
        self.image_hw = (int(self.image_hw[0]), int(self.image_hw[1]))
        self._net = _QNet.build(self.in_ch, self.r, self.image_hw, self.hidden, self.dropout)
        self._device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._net.to(self._device)

    def _to_tensor(self, images: np.ndarray):
        torch, _ = _torch()
        x = np.asarray(images, np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        if x.ndim == 3:
            x = x[None]
        if x.shape[-1] in (1, 3):
            x = np.transpose(x, (0, 3, 1, 2))
        return torch.from_numpy(np.ascontiguousarray(x)).to(self._device)

    def predict(self, images: np.ndarray) -> np.ndarray:
        torch, _ = _torch()
        self._net.eval()
        with torch.no_grad():
            y = self._net(self._to_tensor(images)).cpu().numpy()
        if self.q_scale is not None:
            y = y * np.asarray(self.q_scale, float)
        return y

    def predict_dataset(self, vds, indices: np.ndarray | None = None, batch_size: int = 64) -> np.ndarray:
        idx = np.arange(len(vds)) if indices is None else np.asarray(indices, int)
        out = np.zeros((len(idx), self.r))
        for i0 in range(0, len(idx), batch_size):
            sl = idx[i0 : i0 + batch_size]
            out[i0 : i0 + len(sl)] = self.predict(vds.images[sl])
        return out

    def save(self, path: str | Path) -> Path:
        torch, _ = _torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "r": self.r, "in_ch": self.in_ch, "image_hw": self.image_hw,
                "hidden": self.hidden, "dropout": self.dropout,
                "q_scale": None if self.q_scale is None else np.asarray(self.q_scale),
                "meta": self.meta,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ImageCNNModel":
        torch, _ = _torch()
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        hw = tuple(blob.get("image_hw") or blob["meta"].get("image_hw") or (144, 192))
        m = cls(
            r=int(blob["r"]), in_ch=int(blob["in_ch"]), image_hw=hw,
            hidden=int(blob.get("hidden", 128)), dropout=float(blob.get("dropout", 0.2)),
        )
        m.q_scale = None if blob["q_scale"] is None else np.asarray(blob["q_scale"], float)
        m.meta = dict(blob.get("meta") or {})
        m._net.load_state_dict(blob["state_dict"])
        m._net.to(m._device)
        return m


def train_image_cnn(
    vds,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    hidden: int = 128,
    dropout: float = 0.2,
    patience: int = 15,
    seed: int = 0,
) -> tuple[ImageCNNModel, dict]:
    """Train ImageCNNModel; returns model and history dict."""
    torch, nn = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    in_ch = int(vds.image_shape[-1])
    r = int(vds.q.shape[1])
    hw = (int(vds.image_shape[0]), int(vds.image_shape[1]))
    model = ImageCNNModel(r=r, in_ch=in_ch, image_hw=hw, hidden=hidden, dropout=dropout)
    q_train = vds.q[train_mask]
    q_scale = np.maximum(np.sqrt(np.mean(q_train ** 2, axis=0)), 1e-12)
    model.q_scale = q_scale

    opt = torch.optim.AdamW(model._net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    train_idx = np.nonzero(train_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    history = {"train_rmse": [], "val_rmse": [], "val_q_rel": [], "best_epoch": 0}
    best_state, best_val, wait = None, np.inf, 0

    def batch_tensors(sl):
        x = model._to_tensor(vds.images[sl])
        y = torch.from_numpy((vds.q[sl] / q_scale).astype(np.float32)).to(model._device)
        return x, y

    for epoch in range(epochs):
        model._net.train()
        order = rng.permutation(train_idx)
        for i0 in range(0, len(order), batch_size):
            sl = order[i0 : i0 + batch_size]
            x, y = batch_tensors(sl)
            pred = model._net(x)
            loss = nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        pred_tr = model.predict_dataset(vds, train_idx[: min(512, len(train_idx))])
        pred_va = model.predict_dataset(vds, val_idx)
        tr = float(np.sqrt(np.mean(((pred_tr - vds.q[train_idx[: len(pred_tr)]]) / q_scale) ** 2)))
        va = float(np.sqrt(np.mean(((pred_va - vds.q[val_idx]) / q_scale) ** 2)))
        q_rel = float(np.mean(np.linalg.norm(pred_va - vds.q[val_idx], axis=1) / np.maximum(np.linalg.norm(vds.q[val_idx], axis=1), 1e-30)))
        history["train_rmse"].append(tr)
        history["val_rmse"].append(va)
        history["val_q_rel"].append(q_rel)
        if va < best_val - 1e-5:
            best_val, best_state, wait = va, {k: v.detach().cpu().clone() for k, v in model._net.state_dict().items()}, 0
            history["best_epoch"] = epoch
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model._net.load_state_dict(best_state)
        model._net.to(model._device)
    model.meta.update({
        "epochs_ran": len(history["val_rmse"]), "best_val_rmse": float(best_val),
        "best_val_q_rel": float(history["val_q_rel"][history["best_epoch"]]),
        "device": model._device, "image_hw": hw,
    })
    return model, history
