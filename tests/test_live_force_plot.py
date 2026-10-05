"""Smoke test for force-strip rendering (no interactive window)."""

from __future__ import annotations

import numpy as np
import pytest

from src.grasp.live_force_plot import render_force_strip, stack_camera_and_force


def test_render_force_strip_shape():
    log = {
        "t": [0.0, 0.1, 0.2, 0.3],
        "lam_true": [0.0, 0.8, 1.5, 2.0],
        "lam_meas": [0.0, 0.7, 1.4, 1.9],
        "force_target": [2.2, 2.2, 2.2, 2.2],
        "hold_force": [1.3, 1.3, 1.3, 1.3],
    }
    img = render_force_strip(log, width=320, height=160)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] == 160 and img.shape[1] == 320
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0 + 1e-6


def test_display_force_estimate_only_smooths_the_estimate():
    from src.grasp.live_force_plot import display_force_estimate

    ym = np.array([0.0, 0.0, 3.5, 3.1, 3.5, 3.1])  # gated, then jittery and biased high
    out = display_force_estimate(ym, ema=0.5)
    assert out[0] == 0.0 and out[1] == 0.0
    assert out[2] == 3.5
    assert np.ptp(out[3:]) < np.ptp(ym[3:])  # less jitter
    assert out[-1] == pytest.approx(3.3, abs=0.1)  # bias is kept: nothing pulls it toward GT


def test_stack_camera_and_force():
    cam = np.zeros((40, 80, 3), np.float32)
    frc = np.ones((20, 100, 3), np.float32) * 0.5
    out = stack_camera_and_force(cam, frc)
    assert out.shape == (60, 100, 3)
