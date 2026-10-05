"""Kalman / EMA / batch filters on a short q stream."""

from __future__ import annotations

import numpy as np

from src.vision.temporal import KalmanQ, filter_sequence


def test_kalman_static_matches_mean():
    rng = np.random.default_rng(0)
    q = np.array([0.04, -0.01])
    zs = q + rng.normal(0, 0.01, size=(20, 2))
    kf = KalmanQ.create(2, process=0.0, measure=1e-4)
    for z in zs:
        est = kf.update(z)
    assert np.linalg.norm(est - zs.mean(axis=0)) < 1e-6
    assert np.linalg.norm(est - q) < 0.6 * np.median(np.linalg.norm(zs - q, axis=1))


def test_filter_sequence_kinds():
    zs = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
    raw = filter_sequence(zs, "raw")
    batch = filter_sequence(zs, "batch")
    ema = filter_sequence(zs, "ema", ema_alpha=1.0)
    assert np.allclose(raw[-1], [5, 4])
    assert np.allclose(batch[-1], zs.mean(axis=0))
    assert np.allclose(ema[-1], [5, 4])
    kal = filter_sequence(zs, "kalman", process=0.0, measure=1.0)
    assert kal.shape == zs.shape
