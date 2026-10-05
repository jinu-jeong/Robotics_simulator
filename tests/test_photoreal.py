"""Photoreal post-process defaults to on; preview backend is dependency-free."""

from __future__ import annotations

import numpy as np

from src.vision.photoreal import PhotorealConfig, PhotorealEnhancer, enhance_image


def test_photoreal_default_enabled():
    cfg = PhotorealConfig()
    assert cfg.enabled is True
    assert cfg.backend == "auto"


def test_preview_changes_image_keeps_shape():
    rng = np.random.default_rng(0)
    img = rng.integers(40, 200, size=(64, 80, 3), dtype=np.uint8)
    out = enhance_image(img, {"enabled": True, "backend": "preview", "strength": 0.5, "seed": 1})
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_photoreal_off_is_identity():
    img = np.full((32, 40, 3), 120, dtype=np.uint8)
    out = PhotorealEnhancer({"enabled": False})(img)
    assert np.array_equal(out, img)
