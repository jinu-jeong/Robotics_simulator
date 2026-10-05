"""Re-project existing FEM markers under controlled cameras — no image regen.

Used to answer: at what (n_views, pixel noise, resolution) does unknown-contact
search close. Each view is an exact calibrated pinhole; only marker pixel
noise is varied. ``q`` is geometric Gauss–Newton per view, then averaged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..rendering.synthetic_camera import camera_ring
from ..vision.marker_model import GeometricMarkerModel
from .metrics import relative_errors, summarize


@dataclass(frozen=True)
class SNRCondition:
    n_views: int
    noise_px: float
    res_scale: int

    @property
    def key(self) -> str:
        return f"v{self.n_views}_n{self.noise_px:g}_s{self.res_scale}"


def default_conditions(
    n_views: Iterable[int] = (1, 2, 4, 8),
    noise_px: Iterable[float] = (0.2, 0.5, 1.0),
    res_scales: Iterable[int] = (1, 2, 4),
    base_noise: float = 0.5,
) -> list[SNRCondition]:
    """Full (views × noise) at scale 1, plus extra scales at ``base_noise``."""
    views = [int(v) for v in n_views]
    noises = [float(n) for n in noise_px]
    scales = [int(s) for s in res_scales]
    out: list[SNRCondition] = []
    seen: set[tuple[int, float, int]] = set()
    for v in views:
        for n in noises:
            t = (v, n, 1)
            if t not in seen:
                seen.add(t)
                out.append(SNRCondition(*t))
        for s in scales:
            if s == 1:
                continue
            t = (v, float(base_noise), s)
            if t not in seen:
                seen.add(t)
                out.append(SNRCondition(*t))
    return out


def estimate_q_multiview(
    geo: GeometricMarkerModel,
    cameras,
    mesh,
    markers,
    u: np.ndarray,
    noise_px: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Return ``(mean q, visible-marker fraction)`` over the given cameras."""
    qs = []
    vis_frac = []
    p = markers.positions(mesh, u)
    for cam in cameras:
        vis = markers.visibility(cam, mesh, u)
        vis_frac.append(float(np.mean(vis)))
        uv, _ = cam.project(p)
        if noise_px > 0:
            uv = uv + rng.normal(0.0, float(noise_px), uv.shape)
        qs.append(geo.predict_one(cam, uv, vis))
    return np.mean(np.stack(qs, axis=0), axis=0), float(np.mean(vis_frac))


def evaluate_condition(
    *,
    geo: GeometricMarkerModel,
    localizer,
    cameras,
    mesh,
    markers,
    samples: list[dict[str, Any]],
    noise_px: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Unknown-contact metrics for one (cameras, noise) setting."""
    n = len(samples)
    q_hat = np.zeros((n, geo.r))
    pos = np.zeros(n)
    mag = np.zeros(n)
    vis = np.zeros(n)
    q_true = np.stack([s["q"] for s in samples])
    F_true = np.array([s["force"] for s in samples], float)
    for i, s in enumerate(samples):
        q_hat[i], vis[i] = estimate_q_multiview(
            geo, cameras, mesh, markers, s["u"], noise_px, rng,
        )
        loc = localizer.localize(q_hat[i])
        pos[i] = np.linalg.norm(loc.contact.position - s["contact"])
        mag[i] = loc.magnitude
    q_rel = relative_errors(q_hat, q_true)
    f_rel = np.abs(mag - F_true) / np.maximum(F_true, 1e-30)
    high = F_true > 1.0
    return {
        "n": int(n),
        "visible_frac": float(np.mean(vis)),
        **summarize(q_rel, "q_rel_"),
        **summarize(1e3 * pos, "contact_err_mm_"),
        **summarize(f_rel, "force_rel_"),
        "contact_err_mm_median_Fgt1": float(np.median(1e3 * pos[high])) if high.any() else float("nan"),
        "force_rel_median_Fgt1": float(np.median(f_rel[high])) if high.any() else float("nan"),
        "q_rel": q_rel,
        "contact_err_mm": 1e3 * pos,
        "force_rel": f_rel,
        "mag_true": F_true,
    }


def pick_heldout_fem(vds, test_idx: np.ndarray, max_samples: int | None) -> np.ndarray:
    """Unique FEM indices in the vision test split, stride-subsampled."""
    fids = np.unique(np.asarray(vds.fem_index)[np.asarray(test_idx, int)])
    fids = np.sort(fids)
    if max_samples is not None and len(fids) > int(max_samples):
        pick = np.linspace(0, len(fids) - 1, int(max_samples), dtype=int)
        fids = fids[pick]
    return fids
