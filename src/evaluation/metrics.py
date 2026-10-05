"""Shared numeric summaries for end-to-end evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def relative_errors(hat: np.ndarray, true: np.ndarray, axis: int | None = None) -> np.ndarray:
    hat = np.asarray(hat, float)
    true = np.asarray(true, float)
    if axis is None and hat.ndim > 1:
        num = np.linalg.norm(hat - true, axis=-1)
        den = np.linalg.norm(true, axis=-1)
    else:
        num = np.abs(hat - true) if axis is None else np.linalg.norm(hat - true, axis=axis)
        den = np.abs(true) if axis is None else np.linalg.norm(true, axis=axis)
    return num / np.maximum(den, 1e-30)


def summarize(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    v = np.asarray(values, float).reshape(-1)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {f"{prefix}mean": float("nan"), f"{prefix}median": float("nan"), f"{prefix}p90": float("nan")}
    return {
        f"{prefix}mean": float(np.mean(v)),
        f"{prefix}median": float(np.median(v)),
        f"{prefix}p90": float(np.percentile(v, 90)),
        f"{prefix}max": float(np.max(v)),
    }


def force_metrics(mag_hat: np.ndarray, mag_true: np.ndarray, F_hat: np.ndarray | None = None, F_true: np.ndarray | None = None) -> dict[str, Any]:
    mag_hat = np.asarray(mag_hat, float).reshape(-1)
    mag_true = np.asarray(mag_true, float).reshape(-1)
    rel = relative_errors(mag_hat, mag_true)
    out: dict[str, Any] = {
        **summarize(rel, "force_rel_"),
        "force_mae_N": float(np.mean(np.abs(mag_hat - mag_true))),
        "force_rmse_N": float(np.sqrt(np.mean((mag_hat - mag_true) ** 2))),
        "force_rel_by_sample": rel,
        "mag_hat": mag_hat,
        "mag_true": mag_true,
    }
    if F_hat is not None and F_true is not None:
        Fa, Ft = np.asarray(F_hat, float), np.asarray(F_true, float)
        cos = np.sum(Fa * Ft, axis=1) / np.maximum(np.linalg.norm(Fa, axis=1) * np.linalg.norm(Ft, axis=1), 1e-30)
        out["direction_cos_mean"] = float(np.mean(cos))
    return out


def timing_stats(times_s: np.ndarray) -> dict[str, float]:
    t = np.asarray(times_s, float).reshape(-1)
    return {"time_mean_ms": float(1e3 * np.mean(t)), "time_median_ms": float(1e3 * np.median(t)), "time_p90_ms": float(1e3 * np.percentile(t, 90))}
