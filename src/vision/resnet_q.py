"""ResNet-18 image → reduced coordinates q (marker-free visual state estimation).

The network only predicts the *state* ``q`` (same POD basis / convention as
``q_sim = Φ_rᵀ u``); force and contact location are obtained afterwards by
the existing reduced mechanics (``ReducedModel`` / ``UnknownContactLocalizer``).
The network is never trained on forces.

* Pure PyTorch implementation of ResNet-18 (no torchvision dependency); the
  stem is the standard 7×7/2 conv + max-pool, followed by global average
  pooling and a linear head with ``r`` outputs.
* Targets are normalised per mode, ``(q − μ) / σ``, with the statistics stored
  in :class:`QNormalizer` and saved next to the weights (checkpoint + JSON).
* Input images are ``uint8`` RGB (or grayscale); per-channel mean/std are
  computed on the training set and also stored.

Optional dependency (``pip install -e '.[ml]'``): torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:  # pragma: no cover
        raise ImportError("ResNetQModel requires PyTorch; install with: pip install -e '.[ml]'") from e
    return torch, nn


# ---------------------------------------------------------------- normalisation
@dataclass
class QNormalizer:
    """Per-mode affine normalisation of the reduced coordinates."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, q: np.ndarray, std_floor: float = 1e-12) -> "QNormalizer":
        q = np.asarray(q, float)
        return cls(mean=q.mean(axis=0), std=np.maximum(q.std(axis=0), std_floor))

    def normalize(self, q: np.ndarray) -> np.ndarray:
        return (np.asarray(q, float) - self.mean) / self.std

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, float) * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QNormalizer":
        return cls(mean=np.asarray(d["mean"], float), std=np.asarray(d["std"], float))

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


@dataclass
class ImageNormalizer:
    """Per-channel mean/std on [0, 1] intensities."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, images_uint8: np.ndarray, max_samples: int = 512) -> "ImageNormalizer":
        x = np.asarray(images_uint8[:max_samples], np.float32) / 255.0
        return cls(mean=x.mean(axis=(0, 1, 2)), std=np.maximum(x.std(axis=(0, 1, 2)), 1e-3))

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImageNormalizer":
        return cls(mean=np.asarray(d["mean"], np.float32), std=np.asarray(d["std"], np.float32))


# ---------------------------------------------------------------- network
def build_resnet18(in_ch: int, out_dim: int, width: int = 64):
    """ResNet-18 (BasicBlock ×[2,2,2,2]) with a linear regression head."""
    torch, nn = _torch()

    class BasicBlock(nn.Module):
        def __init__(self, cin: int, cout: int, stride: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.relu = nn.ReLU(inplace=True)
            self.down = None
            if stride != 1 or cin != cout:
                self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

        def forward(self, x):
            idt = x if self.down is None else self.down(x)
            y = self.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            return self.relu(y + idt)

    class ResNet18Q(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            w = int(width)
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, w, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
            chans = [w, 2 * w, 4 * w, 8 * w]
            layers, cin = [], w
            for i, c in enumerate(chans):
                stride = 1 if i == 0 else 2
                layers += [BasicBlock(cin, c, stride), BasicBlock(c, c, 1)]
                cin = c
            self.layers = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(cin, out_dim)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(self.head.bias)
            nn.init.normal_(self.head.weight, std=1e-2)

        def forward(self, x):
            return self.head(torch.flatten(self.pool(self.layers(self.stem(x))), 1))

    return ResNet18Q()


def _pick_device() -> str:
    torch, _ = _torch()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------- model wrapper
@dataclass
class ResNetQModel:
    """Image → q regressor with stored normalisation statistics."""

    r: int
    in_ch: int
    image_hw: tuple[int, int]
    width: int = 64
    q_norm: QNormalizer | None = None
    img_norm: ImageNormalizer | None = None
    meta: dict = field(default_factory=dict)
    _net: Any = field(default=None, repr=False)
    _device: str = "cpu"

    def __post_init__(self) -> None:
        self.image_hw = (int(self.image_hw[0]), int(self.image_hw[1]))
        self._net = build_resnet18(self.in_ch, self.r, self.width)
        self._device = _pick_device()
        self._net.to(self._device)

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self._net.parameters()))

    # -------------------------------------------------------- tensors
    def _to_tensor(self, images: np.ndarray):
        torch, _ = _torch()
        x = np.asarray(images, np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        if x.ndim == 3:
            x = x[None]
        if self.img_norm is not None:
            x = (x - self.img_norm.mean) / self.img_norm.std
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        return torch.from_numpy(np.ascontiguousarray(x)).to(self._device)

    def predict(self, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """``images`` (S, H, W, C) uint8 or float → q (S, r) in physical units."""
        torch, _ = _torch()
        self._net.eval()
        images = np.asarray(images)
        if images.ndim == 3:
            images = images[None]
        out = np.zeros((len(images), self.r))
        with torch.no_grad():
            for i0 in range(0, len(images), batch_size):
                z = self._net(self._to_tensor(images[i0 : i0 + batch_size])).cpu().numpy()
                out[i0 : i0 + len(z)] = z
        return self.q_norm.denormalize(out) if self.q_norm is not None else out

    def predict_indices(self, images: np.ndarray, indices: np.ndarray, batch_size: int = 64) -> np.ndarray:
        idx = np.asarray(indices, int)
        return self.predict(images[idx], batch_size=batch_size)

    # -------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        torch, _ = _torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": "resnet18",
                "state_dict": {k: v.detach().cpu() for k, v in self._net.state_dict().items()},
                "r": self.r, "in_ch": self.in_ch, "image_hw": list(self.image_hw), "width": self.width,
                "q_norm": None if self.q_norm is None else self.q_norm.to_dict(),
                "img_norm": None if self.img_norm is None else self.img_norm.to_dict(),
                "meta": self.meta,
            },
            path,
        )
        # human-readable copy of the normalisation statistics next to the weights
        stats = {"q_norm": None if self.q_norm is None else self.q_norm.to_dict(),
                 "img_norm": None if self.img_norm is None else self.img_norm.to_dict(),
                 "r": self.r, "image_hw": list(self.image_hw), "meta": self.meta}
        path.with_suffix(".norm.json").write_text(json.dumps(stats, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ResNetQModel":
        torch, _ = _torch()
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        m = cls(r=int(blob["r"]), in_ch=int(blob["in_ch"]), image_hw=tuple(blob["image_hw"]), width=int(blob.get("width", 64)))
        m.q_norm = None if blob.get("q_norm") is None else QNormalizer.from_dict(blob["q_norm"])
        m.img_norm = None if blob.get("img_norm") is None else ImageNormalizer.from_dict(blob["img_norm"])
        m.meta = dict(blob.get("meta") or {})
        m._net.load_state_dict(blob["state_dict"])
        m._net.to(m._device)
        return m


# ---------------------------------------------------------------- augmentation
def _augment_batch(x, rng: np.random.Generator, max_shift: int = 4):
    """Photometric jitter + small integer translation, per sample (train only, normalised NCHW)."""
    torch, _ = _torch()
    n = x.shape[0]
    gain = torch.from_numpy(rng.uniform(0.85, 1.15, (n, 1, 1, 1)).astype(np.float32)).to(x.device)
    bias = torch.from_numpy(rng.uniform(-0.15, 0.15, (n, 1, 1, 1)).astype(np.float32)).to(x.device)
    x = x * gain + bias
    if max_shift > 0:
        pad = torch.nn.functional.pad(x, (max_shift,) * 4, mode="replicate")
        H, W = x.shape[-2:]
        out = torch.empty_like(x)
        dx = rng.integers(0, 2 * max_shift + 1, n)
        dy = rng.integers(0, 2 * max_shift + 1, n)
        for i in range(n):
            out[i] = pad[i, :, dy[i] : dy[i] + H, dx[i] : dx[i] + W]
        x = out
    return x


# ---------------------------------------------------------------- teacher reliability
def teacher_snr_weights(q_teacher: np.ndarray, q_reference: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Per-mode weight = fraction of the teacher's variance that is signal.

    ``w_k = clip(var(q_ref_k) / var(q_teacher_k), floor, 1)``: a mode whose
    teacher label is dominated by estimation noise (variance ≫ the true spread)
    gets a small weight; a clean mode keeps weight 1.
    """
    v_t = np.var(np.asarray(q_teacher, float), axis=0)
    v_r = np.var(np.asarray(q_reference, float), axis=0)
    return np.clip(v_r / np.maximum(v_t, 1e-30), float(floor), 1.0)


# ---------------------------------------------------------------- training
def train_resnet_q(
    images: np.ndarray,
    q_target: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    q_reference: np.ndarray | None = None,
    mode_weights: np.ndarray | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 12,
    augment: bool = True,
    width: int = 64,
    seed: int = 0,
    log: Callable[[str], None] | None = print,
) -> tuple[ResNetQModel, dict]:
    """Train ``images → q_target`` with a normalised MSE loss.

    ``q_target`` is the supervision (``q_sim`` or ``q_marker``). ``q_reference``
    (optional, e.g. ``q_sim`` when training on the marker teacher) is only used
    for reporting the validation error against the exact simulation state.
    ``mode_weights`` (optional, shape ``(r,)``, in [0, 1]) scale the per-mode
    normalised squared error, e.g. an SNR estimate of a noisy teacher so that
    noise-dominated modes do not dominate the loss (see
    :func:`teacher_snr_weights`). Early stopping / model selection uses the
    same weighted normalised validation RMSE against ``q_target``.
    """
    torch, nn = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    images = np.asarray(images)
    q_target = np.asarray(q_target, float)
    train_idx = np.nonzero(np.asarray(train_mask, bool))[0]
    val_idx = np.nonzero(np.asarray(val_mask, bool))[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("empty train or validation split")

    H, W, C = images.shape[1:]
    model = ResNetQModel(r=int(q_target.shape[1]), in_ch=int(C), image_hw=(int(H), int(W)), width=int(width))
    model.q_norm = QNormalizer.fit(q_target[train_idx])
    model.img_norm = ImageNormalizer.fit(images[train_idx])
    z_target = model.q_norm.normalize(q_target).astype(np.float32)
    w = np.ones(model.r, np.float32) if mode_weights is None else np.asarray(mode_weights, np.float32).reshape(-1)
    if w.shape != (model.r,):
        raise ValueError("mode_weights must have shape (r,)")
    w_t = torch.from_numpy(w).to(model._device)

    opt = torch.optim.AdamW(model._net.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = int(np.ceil(len(train_idx) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(epochs * steps_per_epoch, 1), pct_start=0.15)
    history: dict[str, Any] = {"train_loss": [], "val_rmse_norm": [], "val_q_rel_target": [], "val_q_rel_reference": [], "best_epoch": 0}
    best_state, best_val, wait = None, np.inf, 0

    def q_rel(a: np.ndarray, b: np.ndarray) -> float:
        # median per-sample relative error over samples with non-trivial |q| (zero-load frames excluded)
        nb = np.linalg.norm(b, axis=1)
        keep = nb > 1e-3 * max(float(nb.max()), 1e-30)
        if not np.any(keep):
            return float("nan")
        return float(np.median(np.linalg.norm(a[keep] - b[keep], axis=1) / nb[keep]))

    for epoch in range(epochs):
        model._net.train()
        order = rng.permutation(train_idx)
        losses = []
        for i0 in range(0, len(order), batch_size):
            sl = order[i0 : i0 + batch_size]
            if len(sl) < 2:  # BatchNorm needs > 1 sample
                continue
            x = model._to_tensor(images[sl])
            if augment:
                x = _augment_batch(x, rng)
            y = torch.from_numpy(z_target[sl]).to(model._device)
            loss = torch.mean(w_t * (model._net(x) - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.item()))

        pred_va = model.predict(images[val_idx])
        va_norm = float(np.sqrt(np.mean(w * (model.q_norm.normalize(pred_va) - z_target[val_idx]) ** 2)))
        rel_t = q_rel(pred_va, q_target[val_idx])
        rel_r = q_rel(pred_va, np.asarray(q_reference, float)[val_idx]) if q_reference is not None else float("nan")
        history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["val_rmse_norm"].append(va_norm)
        history["val_q_rel_target"].append(rel_t)
        history["val_q_rel_reference"].append(rel_r)
        if log is not None:
            extra = f"  q_rel(ref) {rel_r:.3f}" if q_reference is not None else ""
            log(f"  epoch {epoch + 1:3d}/{epochs}  loss {history['train_loss'][-1]:.4f}  val nRMSE {va_norm:.4f}  q_rel(target) {rel_t:.3f}{extra}")
        if va_norm < best_val - 1e-5:
            best_val, wait = va_norm, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model._net.state_dict().items()}
            history["best_epoch"] = epoch
        else:
            wait += 1
            if wait >= patience:
                if log is not None:
                    log(f"  early stop at epoch {epoch + 1} (best {history['best_epoch'] + 1})")
                break

    if best_state is not None:
        model._net.load_state_dict(best_state)
        model._net.to(model._device)
    b = history["best_epoch"]
    model.meta.update({
        "epochs_ran": len(history["val_rmse_norm"]), "best_epoch": int(b),
        "best_val_rmse_norm": float(best_val),
        "best_val_q_rel_target": float(history["val_q_rel_target"][b]),
        "best_val_q_rel_reference": float(history["val_q_rel_reference"][b]),
        "device": model._device, "image_hw": [int(H), int(W)], "n_params": model.n_params,
        "mode_weights": w.tolist(),
        "train": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                  "patience": patience, "augment": bool(augment), "width": int(width), "seed": seed,
                  "n_train": int(len(train_idx)), "n_val": int(len(val_idx))},
    })
    return model, history
