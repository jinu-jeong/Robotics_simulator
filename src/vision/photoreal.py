"""Photoreal post-process for synthetic camera frames (offline + online).

Default is **on**. Flat Taichi RGB is lifted toward a camera-like look before
it becomes training data or the Stage-E network input.

Backends
  preview   – always available (PIL grade / grain / vignette); fast enough for
              closed-loop sim ticks on a laptop
  diffusers – optional SD + ControlNet (Canny) when ``diffusers`` is installed
  auto      – diffusers if importable, else preview
  off       – identity (also ``enabled: false``)

Deploy on a real robot does not run this module: the webcam already is
photoreal. This path exists for **sim** renders used in dataset gen and in the
``mode=nn`` closed-loop demo so train/serve distributions match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

_WARNED_FALLBACK = False


@dataclass
class PhotorealConfig:
    """Shared offline / online photoreal settings (defaults = enabled)."""

    enabled: bool = True
    backend: str = "auto"  # auto | preview | diffusers | off
    strength: float = 0.40  # how hard preview / SD pushes away from flat RGB
    seed: int = 0
    prompt: str = (
        "photorealistic soft robotic silicone finger, natural workshop lighting, "
        "subtle subsurface scattering, realistic plastic and metal, depth of field"
    )
    negative_prompt: str = "cartoon, flat shading, plastic toy, lowres, blurry, watermark"
    # diffusers (optional)
    model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_id: str = "lllyasviel/sd-controlnet-canny"
    num_inference_steps: int = 12
    guidance_scale: float = 4.5
    controlnet_scale: float = 0.85
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "PhotorealConfig":
        d = dict(cfg or {})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kw = {k: d[k] for k in d if k in known and k != "extra"}
        extra = {k: v for k, v in d.items() if k not in known}
        if "extra" in d and isinstance(d["extra"], dict):
            extra = {**extra, **d["extra"]}
        return cls(**kw, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", {}) or {}
        d.update(extra)
        return d


def _to_uint8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype == np.uint8:
        return np.ascontiguousarray(x)
    x = np.asarray(x, float)
    if x.max() <= 1.0 + 1e-6:
        x = x * 255.0
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _preview_enhance(img: np.ndarray, *, strength: float, seed: int) -> np.ndarray:
    """Fast camera-like grade (no GPU). Keeps layout; softens flat GGUI look."""
    from PIL import Image, ImageEnhance, ImageFilter

    s = float(np.clip(strength, 0.0, 1.0))
    u8 = _to_uint8(img)
    if u8.ndim == 2:
        u8 = np.stack([u8] * 3, axis=-1)
    elif u8.shape[-1] == 4:
        u8 = u8[..., :3]
    h, w = u8.shape[:2]
    im = Image.fromarray(u8, mode="RGB")

    # Mild sharpen + contrast / color (looks less clay-like).
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(40 + 80 * s), threshold=2))
    im = ImageEnhance.Contrast(im).enhance(1.0 + 0.35 * s)
    im = ImageEnhance.Color(im).enhance(1.0 + 0.45 * s)
    im = ImageEnhance.Brightness(im).enhance(1.0 + 0.08 * s)

    arr = np.asarray(im, dtype=np.float32)
    # Soft vignette
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2)
    vignette = 1.0 - (0.22 * s) * np.clip(r, 0.0, 1.0) ** 2
    arr *= vignette[..., None]
    # Warm key + cool fill (split-tone-ish)
    warm = np.array([1.04, 1.01, 0.96], np.float32)
    arr = arr * (1.0 + (warm - 1.0) * s)
    # Film grain
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    grain = rng.normal(0.0, 3.5 + 6.0 * s, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + grain, 0, 255)
    base = u8.astype(np.float32)
    t = float(np.clip(0.55 + 0.40 * s, 0.0, 1.0))
    out = (1.0 - t) * base + t * arr
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


class PhotorealEnhancer:
    """Stateful enhancer (diffusers pipe cached on first use)."""

    def __init__(self, cfg: PhotorealConfig | Mapping[str, Any] | None = None) -> None:
        self.cfg = cfg if isinstance(cfg, PhotorealConfig) else PhotorealConfig.from_mapping(cfg)
        self._pipe = None
        self._resolved: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled) and str(self.cfg.backend).lower() != "off"

    def _resolve_backend(self) -> str:
        if self._resolved is not None:
            return self._resolved
        b = str(self.cfg.backend).lower().strip()
        if not self.cfg.enabled or b == "off":
            self._resolved = "off"
            return self._resolved
        if b == "preview":
            self._resolved = "preview"
            return self._resolved
        if b in ("auto", "diffusers"):
            try:
                import diffusers  # noqa: F401
                import torch  # noqa: F401

                self._resolved = "diffusers"
                return self._resolved
            except Exception:
                global _WARNED_FALLBACK
                if b == "diffusers":
                    raise
                if not _WARNED_FALLBACK:
                    print(
                        "[photoreal] diffusers not installed — using fast 'preview' backend "
                        "(pip install diffusers transformers accelerate for SD+ControlNet)"
                    )
                    _WARNED_FALLBACK = True
                self._resolved = "preview"
                return self._resolved
        self._resolved = "preview"
        return self._resolved

    def _diffusers_enhance(self, img: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        u8 = _to_uint8(img)
        if u8.ndim == 2:
            u8 = np.stack([u8] * 3, axis=-1)
        pil = Image.fromarray(u8, mode="RGB")
        # Keep network / dataset resolution; SD works better near 512 but we stay native.
        w, h = pil.size
        # round to multiple of 8
        W = max(8, (w // 8) * 8)
        H = max(8, (h // 8) * 8)
        if (W, H) != (w, h):
            pil = pil.resize((W, H), Image.Resampling.LANCZOS)

        if self._pipe is None:
            from diffusers import (  # type: ignore
                ControlNetModel,
                StableDiffusionControlNetImg2ImgPipeline,
                UniPCMultistepScheduler,
            )

            dtype = torch.float16 if torch.backends.mps.is_available() or torch.cuda.is_available() else torch.float32
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            controlnet = ControlNetModel.from_pretrained(self.cfg.controlnet_id, torch_dtype=dtype)
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.cfg.model_id, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)
            self._pipe = pipe
            self._device = device

        # Canny condition from the sim frame (locks silhouette).
        import cv2

        gray = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        cond = Image.fromarray(edges).convert("RGB")

        g = torch.Generator(device=self._device if self._device != "mps" else "cpu")
        g = g.manual_seed(int(self.cfg.seed))
        strength = float(np.clip(0.25 + 0.55 * self.cfg.strength, 0.2, 0.85))
        out = self._pipe(
            prompt=self.cfg.prompt,
            negative_prompt=self.cfg.negative_prompt,
            image=pil,
            control_image=cond,
            strength=strength,
            num_inference_steps=int(self.cfg.num_inference_steps),
            guidance_scale=float(self.cfg.guidance_scale),
            controlnet_conditioning_scale=float(self.cfg.controlnet_scale),
            generator=g,
        ).images[0]
        if out.size != (w, h):
            out = out.resize((w, h), Image.Resampling.LANCZOS)
        return np.asarray(out, dtype=np.uint8)

    def __call__(self, img: np.ndarray, *, seed: int | None = None) -> np.ndarray:
        if not self.enabled:
            return _to_uint8(img)
        backend = self._resolve_backend()
        if backend == "off":
            return _to_uint8(img)
        sd = int(self.cfg.seed if seed is None else seed)
        if backend == "diffusers":
            try:
                return self._diffusers_enhance(img)
            except Exception as e:  # noqa: BLE001
                global _WARNED_FALLBACK
                if not _WARNED_FALLBACK:
                    print(f"[photoreal] diffusers failed ({e}); falling back to preview")
                    _WARNED_FALLBACK = True
                self._resolved = "preview"
        return _preview_enhance(img, strength=float(self.cfg.strength), seed=sd)


def enhance_image(img: np.ndarray, cfg: PhotorealConfig | Mapping[str, Any] | None = None, *, seed: int | None = None) -> np.ndarray:
    """One-shot helper (builds a short-lived enhancer). Prefer :class:`PhotorealEnhancer` in loops."""
    return PhotorealEnhancer(cfg)(img, seed=seed)
