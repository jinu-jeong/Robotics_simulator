"""Marker observation features and geometric image Jacobians.

For a known camera and a POD basis Φ, marker image displacements are (to first
order) linear in the reduced coordinates:

    Δuv ≈ J(c) q,   J_{:,k} = ∂π(p(ε e_k)) / ∂ε |_{ε=0}

where π is the pinhole projection and p(u) are the marker world positions on
the deformed surface. Solving the (masked, ridge-regularised) least-squares
problem recovers q without a learned network – a mechanics-informed baseline.
"""

from __future__ import annotations

import numpy as np

from ..rendering.markers import SurfaceMarkers
from ..rom.pod import PODBasis
from ..visualization.camera import ObservationCamera
from .dataset import VisionDataset


def undeformed_marker_pixels(vds: VisionDataset, markers: SurfaceMarkers, mesh) -> np.ndarray:
    """(S, M, 2) projections of undeformed markers with each sample's camera."""
    p0 = markers.positions(mesh, None)
    out = np.zeros_like(vds.marker_px_clean)
    for s in range(len(vds)):
        out[s] = vds.camera(s).project(p0)[0]
    return out


def marker_delta_features(
    marker_px: np.ndarray,
    uv0: np.ndarray,
    visible: np.ndarray,
) -> np.ndarray:
    """Flattened Δuv with invisible markers zeroed -> (S, 2M) or (2M,) for one sample."""
    single = marker_px.ndim == 2
    if single:
        marker_px = marker_px[None]
        uv0 = uv0[None]
        visible = visible[None]
    d = (np.asarray(marker_px, float) - np.asarray(uv0, float)).reshape(len(marker_px), -1)
    mask = np.repeat(np.asarray(visible, bool), 2, axis=1)
    d = np.where(mask, d, 0.0)
    return d[0] if single else d


def mode_marker_displacements(markers: SurfaceMarkers, mesh, basis: PODBasis) -> np.ndarray:
    """(M, 3, r) world displacement of each marker under unit POD mode k."""
    p0 = markers.positions(mesh, None)
    return np.stack([markers.positions(mesh, basis.mode(k)) - p0 for k in range(basis.r)], axis=-1)


def image_jacobian(
    camera: ObservationCamera,
    p0: np.ndarray,
    dp_modes: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """(2M, r) Jacobian ∂(uv)/∂q at the undeformed configuration for one camera."""
    uv0, _ = camera.project(p0)
    M, _, r = dp_modes.shape
    J = np.zeros((2 * M, r))
    for k in range(r):
        uv_k, _ = camera.project(p0 + eps * dp_modes[:, :, k])
        J[:, k] = ((uv_k - uv0) / eps).reshape(-1)
    return J


def solve_q_from_jacobian(
    J: np.ndarray,
    delta_uv: np.ndarray,
    visible: np.ndarray,
    ridge: float = 1e-4,
    huber: float | None = None,
) -> np.ndarray:
    """Masked ridge LS: argmin_q ‖W (J q − Δuv)‖² + ridge ‖q‖².

    ``huber`` (pixels) applies one IRLS reweight after an unweighted solve.
    """
    y = np.asarray(delta_uv, float).reshape(-1)
    mask = np.repeat(np.asarray(visible, bool).reshape(-1), 2)
    Jm, ym = J[mask], y[mask]
    if Jm.size == 0:
        return np.zeros(J.shape[1])
    A = Jm.T @ Jm + float(ridge) * np.eye(J.shape[1])
    q = np.linalg.solve(A, Jm.T @ ym)
    if huber is None or huber <= 0:
        return q
    resid = Jm @ q - ym
    w = np.minimum(1.0, float(huber) / np.maximum(np.abs(resid), 1e-9))
    A = (Jm * w[:, None]).T @ Jm + float(ridge) * np.eye(J.shape[1])
    return np.linalg.solve(A, Jm.T @ (w * ym))
