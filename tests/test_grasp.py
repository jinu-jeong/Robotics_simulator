"""Stages A / B grasp: compliance, Coulomb lift, GT vs marker estimator."""

from __future__ import annotations

import numpy as np
import pytest

from src.fem.finger_model import FingerFEMModel
from src.grasp.estimator import GraspEstimator
from src.grasp.mechanics import GraspMechanics
from src.grasp.sim import GraspSim
from src.grasp.world import ParallelJawWorld


def _tiny_model() -> FingerFEMModel:
    return FingerFEMModel.from_config({
        "geometry": {"length": 0.1, "width": 0.02, "height": 0.01},
        "mesh": {"nx": 12, "ny": 3, "nz": 3},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    })


@pytest.fixture(scope="module")
def mech() -> GraspMechanics:
    return GraspMechanics.build(_tiny_model(), [0.85, 1.0, 0.5], [0.028, 0.024, 0.01], mass=0.12, mu=0.45)


def test_world_inner_faces_face_each_other(mech):
    w = ParallelJawWorld(mech.model.geometry)
    p = mech.contact_local
    g = 0.03
    left = w.left_contact_world(p, g, 0.0)
    right = w.right_contact_world(p, g, 0.0)
    assert np.isclose(left[1], -0.5 * g)
    assert np.isclose(right[1], 0.5 * g)
    assert np.isclose(left[0], right[0])


def test_stiffness_and_force_from_opening(mech):
    assert mech.stiffness > 0
    assert mech.force_from_opening(mech.object_width + 0.01) == 0.0
    lam = mech.force_from_opening(mech.object_width - 0.002)
    assert lam == pytest.approx(mech.stiffness * 0.001, rel=1e-6)
    # closing more increases force
    assert mech.force_from_opening(mech.object_width - 0.004) > lam
    assert mech.hold_force == pytest.approx(0.12 * 9.80665 / (2 * 0.45), rel=1e-6)
    assert mech.can_hold(mech.hold_force)
    assert not mech.can_hold(0.5 * mech.hold_force)


def test_soft_object_is_a_series_spring(mech):
    import dataclasses
    k_obj = 800.0
    soft = dataclasses.replace(mech, object_stiffness=k_obj)
    assert mech.is_rigid and not soft.is_rigid
    g = mech.object_width - 0.004
    k_series = 1.0 / (2.0 / mech.stiffness + 1.0 / k_obj)
    lam = soft.force_from_opening(g)
    assert lam == pytest.approx(k_series * 0.004, rel=1e-9)
    assert lam < mech.force_from_opening(g)
    assert soft.opening_from_force(lam) == pytest.approx(g, abs=1e-12)
    assert soft.object_size_at(lam)[1] == pytest.approx(mech.object_width - lam / k_obj)
    # the finger sees only λ: same displacement field as the rigid case at equal force
    assert np.allclose(soft.displacement(lam), mech.displacement(lam))


def test_object_compliance_estimate(mech):
    import dataclasses
    for k_obj in (np.inf, 2000.0, 600.0):
        m = dataclasses.replace(mech, object_stiffness=k_obj)
        g = m.opening_from_force(2.0)
        assert m.estimate_object_compliance(g, 2.0) == pytest.approx(1.0 / k_obj, abs=1e-9)
    # a force over-estimate reads as a stiffer object
    m = dataclasses.replace(mech, object_stiffness=600.0)
    g = m.opening_from_force(2.0)
    assert m.estimate_object_compliance(g, 2.2) < 1.0 / 600.0


def test_soft_object_lifts(mech):
    import dataclasses
    soft = dataclasses.replace(mech, object_stiffness=1000.0)
    est = GraspEstimator.build(soft, {"mode": "gt"})
    sim = GraspSim(
        soft, est, opening_start=0.032, opening_min=0.012, close_speed=0.02,
        lift_height=0.03, lift_speed=0.04, force_target=soft.hold_force + 0.6,
        force_tol=0.20, kp=0.006, ki=0.001, settle_s=0.20, dt=0.01,
    )
    sim.run(t_max=8.0)
    assert sim.success()
    assert sim.state.opening < mech.opening_from_force(sim.state.lam_true)


def test_stage_a_lifts_when_target_above_threshold(mech):
    est = GraspEstimator.build(mech, {"mode": "gt"})
    sim = GraspSim(
        mech, est, opening_start=0.032, opening_min=0.016, close_speed=0.02,
        lift_height=0.03, lift_speed=0.04, force_target=mech.hold_force + 0.6,
        force_tol=0.20, kp=0.006, ki=0.001, settle_s=0.20, dt=0.01,
    )
    sim.run(t_max=6.0)
    assert sim.success()
    assert sim.state.obj_z > 0.5 * sim.lift_height


def test_stage_a_drops_when_target_below_threshold(mech):
    est = GraspEstimator.build(mech, {"mode": "gt"})
    sim = GraspSim(
        mech, est, opening_start=0.032, opening_min=0.016, close_speed=0.02,
        lift_height=0.03, lift_speed=0.04, force_target=0.5 * mech.hold_force,
        force_tol=0.1, kp=0.006, ki=0.001, settle_s=0.15, dt=0.01,
    )
    sim.run(t_max=6.0)
    assert not sim.success()
    assert sim.state.obj_z == pytest.approx(0.5 * mech.object_size[2], abs=1e-6)


def test_marker_estimator_recovers_force(mech):
    est = GraspEstimator.build(mech, {
        "mode": "markers", "pixel_noise_std": 0.0,
        "unknown_contact": False,
        "marker_nx": 6, "marker_ny": 2,
        "camera": {"target_rel": [0.55, 0.5, 0.5], "position_offset_rel": [0.05, -1.1, 0.55],
                   "fov_y_deg": 40.0, "image_width": 160, "image_height": 120},
    }, seed=0)
    u = mech.displacement(2.0)
    hat = est.measure(u, 2.0)
    assert abs(hat - 2.0) / 2.0 < 0.12
    assert est.kf is not None


def test_kalman_is_default_and_resets(mech):
    est = GraspEstimator.build(mech, {
        "mode": "markers", "pixel_noise_std": 0.5,
        "unknown_contact": False,
        "marker_nx": 6, "marker_ny": 2,
        "camera": {"target_rel": [0.55, 0.5, 0.5], "position_offset_rel": [0.05, -1.1, 0.55],
                   "fov_y_deg": 40.0, "image_width": 160, "image_height": 120},
    }, seed=1)
    assert est.q_filter == "kalman" and est.kf is not None
    u = mech.displacement(2.0)
    first = est.measure(u, 2.0)
    for _ in range(24):
        est.measure(u, 2.0)
    held = est.measure(u, 2.0)
    est.reset_filter()
    after = est.measure(u, 2.0)
    assert abs(held - 2.0) <= abs(first - 2.0) + 1e-9
    assert est.kf.n_updates == 1
    assert after != held or abs(after - 2.0) < 0.5
