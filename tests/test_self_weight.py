"""Finger self-weight (ρ g body force) and its compensation in the grasp estimator."""

from __future__ import annotations

import numpy as np
import pytest

from src.fem.finger_model import FingerFEMModel
from src.grasp.estimator import GraspEstimator
from src.grasp.mechanics import GraspMechanics

G = 9.80665
L, W, H = 0.1, 0.02, 0.01
RHO = 1200.0


def _model(gravity: bool, nx: int = 16, ny: int = 4, nz: int = 4) -> FingerFEMModel:
    return FingerFEMModel.from_config({
        "geometry": {"length": L, "width": W, "height": H},
        "mesh": {"nx": nx, "ny": ny, "nz": nz},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4, "density": RHO},
        "boundary_conditions": {"fixed_surface": "root"},
        "gravity": {"enabled": gravity, "vector": [0.0, 0.0, -G]},
    })


def test_body_force_vector_sums_to_weight():
    m = _model(True)
    f = m.fem.body_force_vector([0.0, 0.0, -G]).reshape(-1, 3)
    weight = RHO * L * W * H * G
    assert f.sum(axis=0) == pytest.approx([0.0, 0.0, -weight], abs=1e-9 * weight)
    # zero density → no load
    m0 = FingerFEMModel.from_config({
        "geometry": {"length": L, "width": W, "height": H}, "mesh": {"nx": 8, "ny": 2, "nz": 2},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    })
    assert not m0.has_gravity
    assert not np.any(m0.fem.body_force_vector([0.0, 0.0, -G]))


def test_self_weight_sag_matches_beam_theory():
    """Uniformly loaded cantilever: δ_tip = w L⁴ / (8 E I).

    Euler–Bernoulli neglects shear (~1 % here) and the 3D root constraint, so compare
    the FEM/theory ratio of the self-weight sag with that of a 1 N tip load on the same
    mesh; with quadratic elements both are within a few % of theory and agree with each other.
    """
    m = _model(True, nx=24, ny=4, nz=4)
    tipnodes = m.mesh.nodes_on_plane(0, L)
    I = W * H**3 / 12.0
    w = RHO * G * W * H
    delta_g = w * L**4 / (8.0 * m.material.E * I)
    tip_g = -m.gravity_displacement()[tipnodes, 2].mean()
    res_p, _ = m.solve_point_force(m.geometry.point_from_relative([1.0, 0.5, 1.0]), np.array([0.0, 0.0, -1.0]))
    tip_p = -res_p.u[tipnodes, 2].mean()
    delta_p = L**3 / (3.0 * m.material.E * I)
    assert tip_g > 0.0
    assert 0.9 < tip_g / delta_g < 1.0
    assert abs(tip_g / delta_g - tip_p / delta_p) < 0.02
    # reactions balance the weight
    res = m.solve_gravity()
    assert res.equilibrium_residual() < 1e-6
    assert res.reaction_resultant()[2] == pytest.approx(RHO * L * W * H * G, rel=1e-6)


def test_gravity_rotates_with_finger_frame_and_is_cached():
    m = _model(True)
    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # local x → world −z (hanging down)
    g_loc = m.gravity_local(R)
    assert g_loc == pytest.approx([G, 0.0, 0.0])  # gravity along local +x: pure tension, ~no bending
    u_up = m.gravity_displacement(R)
    u_flat = m.gravity_displacement()
    assert np.linalg.norm(u_up, axis=1).max() < 0.05 * np.linalg.norm(u_flat, axis=1).max()
    assert m.gravity_displacement(R) is u_up  # cached per orientation


def test_mechanics_superposes_gravity_and_compensates_force():
    m = _model(True)
    mech = GraspMechanics.build(m, [0.85, 1.0, 0.5], [0.028, 0.024, 0.01], mass=0.12, mu=0.45)
    assert mech.has_gravity
    u_g = mech.gravity_displacement()
    assert np.linalg.norm(u_g, axis=1).max() > 1e-5  # 0.1 mm-scale sag
    u2 = mech.displacement(2.0)
    u0 = mech.displacement(0.0)
    assert np.allclose(u0, u_g)
    res, _ = m.solve_normal_contact(mech.contact_local, 2.0)
    assert np.allclose(u2 - u_g, res.u)
    # opening ↔ force stay inverse of each other with the gravity indentation term
    g = mech.opening_from_force(1.3)
    assert mech.force_from_opening(g) == pytest.approx(1.3, rel=1e-9)

    # inverse estimator: compensation removes the sag exactly; without it the force is biased
    est = GraspEstimator.build(mech, {"mode": "inverse", "unknown_contact": False})
    err_c = abs(est.measure(u2, 2.0) - 2.0)
    assert err_c < 1e-9
    est.gravity_compensation = False
    err_nc = abs(est.measure(u2, 2.0) - 2.0)
    # The full-field LS projects the −z sag onto the −y contact response; on the hex-dominant
    # mesh the two are almost orthogonal (no Kuhn y–z coupling), so the bias is small but nonzero.
    assert err_nc > 1e-7 and err_nc > 100 * err_c


def test_marker_estimator_compensates_self_weight_in_pixel_space():
    m = _model(True)
    mech = GraspMechanics.build(m, [0.85, 1.0, 0.5], [0.028, 0.024, 0.01], mass=0.12, mu=0.45)
    cfg = {
        "mode": "markers", "pixel_noise_std": 0.0, "unknown_contact": False,
        "marker_nx": 6, "marker_ny": 2,
        "camera": {"target_rel": [0.55, 0.5, 0.5], "position_offset_rel": [0.05, -1.1, 0.55],
                   "fov_y_deg": 40.0, "image_width": 160, "image_height": 120},
    }
    u = mech.displacement(2.0)
    est = GraspEstimator.build(mech, dict(cfg), seed=0)
    hat = est.measure(u, 2.0)
    assert abs(hat - 2.0) / 2.0 < 0.12
    est_nc = GraspEstimator.build(mech, {**cfg, "gravity_compensation": False}, seed=0)
    hat_nc = est_nc.measure(u, 2.0)
    assert abs(hat_nc - 2.0) > abs(hat - 2.0)  # sag outside span(Φ) biases the LS fit
