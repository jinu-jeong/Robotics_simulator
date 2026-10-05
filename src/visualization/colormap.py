"""Small dependency-free colormaps returning float32 RGB in [0, 1]."""

from __future__ import annotations

import numpy as np

_VIRIDIS = np.array(
    [
        [0.267004, 0.004874, 0.329415],
        [0.282656, 0.140926, 0.457517],
        [0.253935, 0.265254, 0.529983],
        [0.206756, 0.371758, 0.553117],
        [0.163625, 0.471133, 0.558148],
        [0.127568, 0.566949, 0.550556],
        [0.134692, 0.658636, 0.517649],
        [0.266941, 0.748751, 0.440573],
        [0.477504, 0.821444, 0.318195],
        [0.741388, 0.873449, 0.149561],
        [0.993248, 0.906157, 0.143936],
    ]
)

_JET = np.array(
    [
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 1.0, 0.5],
        [1.0, 1.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ]
)

_COOLWARM = np.array(
    [
        [0.230, 0.299, 0.754],
        [0.552, 0.690, 0.996],
        [0.865, 0.865, 0.865],
        [0.958, 0.603, 0.482],
        [0.706, 0.016, 0.150],
    ]
)

_TABLES = {"viridis": _VIRIDIS, "jet": _JET, "coolwarm": _COOLWARM}


def available_colormaps() -> list[str]:
    return sorted(_TABLES) + ["grey"]


def colormap(values: np.ndarray, name: str = "viridis", vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Map scalars to RGB.

    ``vmin``/``vmax`` default to the data range; a degenerate range maps
    everything to the lowest color (so an undeformed mesh is uniformly dark).
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    lo = float(np.min(v)) if vmin is None else float(vmin)
    hi = float(np.max(v)) if vmax is None else float(vmax)
    if hi - lo <= 1e-300:
        t = np.zeros_like(v)
    else:
        t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    if name == "grey":
        g = 0.25 + 0.7 * t
        return np.stack([g, g, g], axis=1).astype(np.float32)
    table = _TABLES.get(name)
    if table is None:
        raise KeyError(f"unknown colormap {name!r}; available: {available_colormaps()}")
    x = np.linspace(0.0, 1.0, len(table))
    rgb = np.stack([np.interp(t, x, table[:, c]) for c in range(3)], axis=1)
    return rgb.astype(np.float32)
