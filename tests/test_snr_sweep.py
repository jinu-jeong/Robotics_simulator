"""SNR sweep helpers: camera ring and clean multi-view q recovery."""

from __future__ import annotations

import numpy as np

from src.evaluation.snr_sweep import default_conditions, estimate_q_multiview
from src.rendering.markers import make_marker_grid
from src.rendering.synthetic_camera import camera_ring
from src.fem.finger_model import FingerFEMModel
from src.rom.pod import compute_pod
from src.vision.marker_model import GeometricMarkerModel


def _small_finger():
    model = FingerFEMModel.from_config({
        "geometry": {"length": 0.1, "width": 0.012, "height": 0.01},
        "mesh": {"nx": 12, "ny": 3, "nz": 2},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    })
    model.restrict_contact_surface("top")
    markers = make_marker_grid(model.mesh, model.geometry, "top", nx=5, ny=2, margin_rel=0.1)
    return model, markers


def test_camera_ring_count_and_visibility():
    model, markers = _small_finger()
    cfg = {"target_rel": [0.55, 0.5, 0.5], "position_offset_rel": [0.05, -1.1, 0.55],
           "up": [0.0, 0.0, 1.0], "fov_y_deg": 40.0}
    cams = camera_ring(model.geometry, cfg, 4, 192, 144)
    assert len(cams) == 4
    pos = np.stack([c.position for c in cams])
    assert np.unique(np.round(pos, 6), axis=0).shape[0] == 4
    vis = np.mean([markers.visibility(c, model.mesh, None).mean() for c in cams])
    assert vis > 0.8


def test_default_conditions_cover_axes():
    conds = default_conditions()
    keys = {(c.n_views, c.noise_px, c.res_scale) for c in conds}
    assert (1, 0.5, 1) in keys and (8, 0.2, 1) in keys
    assert (4, 0.5, 4) in keys
    assert (4, 0.2, 4) not in keys  # extra scales only at base noise


def test_multiview_clean_recovers_q():
    model, markers = _small_finger()
    res, _ = model.solve_normal_contact(model.geometry.point_from_relative([0.7, 0.5, 1.0]), 2.0)
    res2, _ = model.solve_normal_contact(model.geometry.point_from_relative([0.5, 0.5, 1.0]), 1.0)
    basis = compute_pod(np.stack([res.u.reshape(-1), res2.u.reshape(-1)], axis=1)).truncate(2)
    geo = GeometricMarkerModel(basis, markers, model.mesh, ridge=1e-6, use_noisy=False, n_iter=2)
    cfg = {"target_rel": [0.55, 0.5, 0.5], "position_offset_rel": [0.05, -1.1, 0.55],
           "up": [0.0, 0.0, 1.0], "fov_y_deg": 40.0}
    cams = camera_ring(model.geometry, cfg, 2, 192, 144)
    rng = np.random.default_rng(0)
    q_hat, vis = estimate_q_multiview(geo, cams, model.mesh, markers, res.u, 0.0, rng)
    q = basis.project(res.u)
    rel = np.linalg.norm(q_hat - q) / max(np.linalg.norm(q), 1e-30)
    assert vis > 0.8
    assert rel < 0.08
