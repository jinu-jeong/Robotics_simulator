"""Stage C: 7-DoF FK / IK and arm-mounted grasp lift."""

from __future__ import annotations

import numpy as np
import pytest

from src.fem.finger_model import FingerFEMModel
from src.grasp.arm import SerialArm7, make_T, pose_error
from src.grasp.arm_sim import build_arm_grasp_sim, grasp_T
from src.grasp.estimator import GraspEstimator
from src.grasp.mechanics import GraspMechanics
from src.grasp.sim import GraspSim
from src.grasp.world import ParallelJawWorld


def test_fk_zero_pose():
    arm = SerialArm7()
    T = arm.fk(np.zeros(7))
    expect = np.array([arm.L_upper + arm.L_fore + arm.L_flange, 0.0, arm.d_base])
    assert np.allclose(T[:3, 3], expect, atol=1e-12)
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-12)


def test_ik_reaches_grasp_pose():
    arm = SerialArm7()
    T_des = make_T(np.eye(3), [0.22, 0.02, 0.03])
    q0 = np.array([0.0, 0.55, 0.0, 1.70, 0.0, 0.70, 0.0])
    q, err = arm.solve(T_des, q0, n_iter=120, tol=1.5e-3)
    e = pose_error(arm.fk(q), T_des)
    assert err < 2.5e-3
    assert np.linalg.norm(e[:3]) < 2.5e-3
    assert np.linalg.norm(e[3:]) < 0.04


def test_world_T_ee_matches_scalar_lift():
    g = FingerFEMModel.from_config({
        "geometry": {"length": 0.1, "width": 0.02, "height": 0.01},
        "mesh": {"nx": 4, "ny": 2, "nz": 2},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    }).geometry
    w = ParallelJawWorld(g)
    nodes = np.array([[0.08, 0.02, 0.005], [0.0, 0.0, 0.0]])
    lift = 0.03
    T = make_T(np.eye(3), [0.0, 0.0, lift])
    a = w.left_nodes(nodes, 0.03, lift)
    b = w.left_nodes(nodes, 0.03, 0.0, T_ee=T)
    assert np.allclose(a, b)


def _tiny_grasp() -> GraspSim:
    model = FingerFEMModel.from_config({
        "geometry": {"length": 0.1, "width": 0.02, "height": 0.01},
        "mesh": {"nx": 12, "ny": 3, "nz": 3},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    })
    mech = GraspMechanics.build(model, [0.85, 1.0, 0.5], [0.028, 0.024, 0.01], mass=0.12, mu=0.45)
    est = GraspEstimator.build(mech, {"mode": "gt"})
    return GraspSim(
        mech, est, opening_start=0.032, opening_min=0.016, close_speed=0.02,
        lift_height=0.03, lift_speed=0.04, force_target=mech.hold_force + 0.6,
        force_tol=0.20, kp=0.006, ki=0.001, settle_s=0.20, dt=0.01,
    )


def test_grasp_T_places_contact_on_object():
    grasp = _tiny_grasp()
    p_obj = np.array([0.32, 0.0, 0.005])
    T = grasp_T(grasp.mech.contact_local, p_obj)
    w = grasp.mech.world
    left = w.left_contact_world(grasp.mech.contact_local, grasp.mech.object_width, T_ee=T)
    assert np.allclose(left[0], p_obj[0], atol=1e-9)
    assert np.allclose(left[2], p_obj[2], atol=1e-9)
    assert left[1] == pytest.approx(-0.5 * grasp.mech.object_width)


def test_arm_stage_a_lifts():
    sim = build_arm_grasp_sim(_tiny_grasp(), {
        "object_xy": [0.32, 0.0],
        "ik_max_dq": 0.12,
        "ik_tol": 3.0e-3,
    })
    sim.run(t_max=14.0)
    assert sim.success()
    assert sim.obj_z > 0.5 * sim.grasp.lift_height


def test_world_estimator_recovers_force_under_T_ee():
    model = FingerFEMModel.from_config({
        "geometry": {"length": 0.1, "width": 0.02, "height": 0.01},
        "mesh": {"nx": 12, "ny": 3, "nz": 3},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    })
    mech = GraspMechanics.build(model, [0.85, 1.0, 0.5], [0.028, 0.024, 0.01], mass=0.12, mu=0.45)
    T = make_T(np.eye(3), [0.235, 0.0, 0.0])
    est = GraspEstimator.build(mech, {
        "mode": "world", "pixel_noise_std": 0.0,
        "unknown_contact": False,
        "marker_nx": 6, "marker_ny": 2,
        "world_camera": {
            "position": [0.26, -0.18, 0.12],
            "target": [0.28, -0.02, 0.015],
            "fov_y_deg": 42.0, "image_width": 160, "image_height": 120,
        },
    }, seed=0)
    lam = 2.0
    u = mech.displacement(lam)
    hat = est.measure(u, lam, T_ee=T, opening=mech.opening_from_force(lam))
    assert est.last_visible is not None and int(est.last_visible.sum()) >= 4
    assert abs(hat - lam) / lam < 0.15


def test_world_unknown_contact_localizes_and_recovers_force():
    model = FingerFEMModel.from_config({
        "geometry": {"length": 0.1, "width": 0.02, "height": 0.01},
        "mesh": {"nx": 12, "ny": 3, "nz": 3},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    })
    mech = GraspMechanics.build(model, [0.85, 1.0, 0.5], [0.028, 0.024, 0.01], mass=0.12, mu=0.45)
    T = make_T(np.eye(3), [0.235, 0.0, 0.0])
    est = GraspEstimator.build(mech, {
        "mode": "world", "pixel_noise_std": 0.0,
        "unknown_contact": True,
        "pod_n_x": 3, "pod_n_z": 2, "q_modes": 6,
        "lock_votes": 1,
        "q_min_norm": 1.0e-3,
        "q_lock_norm": 1.0e-3,
        "max_residual_rel": 0.5,
        "lock_residual_rel": 0.5,
        "lock_min_force": 0.2,
        "vote_clear_bad": 99,
        "marker_nx": 6, "marker_ny": 2,
        "unknown_contact_cfg": {"face_stride": 3, "r_loc": 4, "refine": True, "refine_n": 5},
        "world_camera": {
            "position": [0.26, -0.18, 0.12],
            "target": [0.28, -0.02, 0.015],
            "fov_y_deg": 42.0, "image_width": 160, "image_height": 120,
        },
    }, seed=0)
    assert est.unknown_contact and est.localizer is not None
    lam = 2.0
    u = mech.displacement(lam)
    hat = est.measure(u, lam, T_ee=T, opening=mech.opening_from_force(lam))
    assert est.last_contact_local is not None
    assert est.last_contact_err_m is not None and est.last_contact_err_m < 0.008
    assert abs(hat - lam) / lam < 0.25


def test_contact_lock_waits_for_open_votes():
    model = FingerFEMModel.from_config({
        "geometry": {"length": 0.1, "width": 0.02, "height": 0.01},
        "mesh": {"nx": 12, "ny": 3, "nz": 3},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    })
    mech = GraspMechanics.build(model, [0.85, 1.0, 0.5], [0.028, 0.024, 0.01], mass=0.12, mu=0.45)
    T = make_T(np.eye(3), [0.235, 0.0, 0.0])
    est = GraspEstimator.build(mech, {
        "mode": "world", "pixel_noise_std": 0.0, "unknown_contact": True,
        "pod_n_x": 3, "pod_n_z": 2, "q_modes": 6, "lock_votes": 1,
        "q_min_norm": 1.0e-3, "q_lock_norm": 1.0e-3, "max_residual_rel": 0.5,
        "lock_residual_rel": 0.5, "lock_min_force": 0.2, "vote_clear_bad": 99,
        "marker_nx": 6, "marker_ny": 2,
        "unknown_contact_cfg": {"face_stride": 3, "r_loc": 4, "refine": True, "refine_n": 5},
        "world_camera": {"position": [0.26, -0.18, 0.12], "target": [0.28, -0.02, 0.015],
                         "fov_y_deg": 42.0, "image_width": 160, "image_height": 120},
    }, seed=0)
    u = mech.displacement(2.0)
    g = mech.opening_from_force(2.0)
    est.lock_votes_open = False
    est.measure(u, 2.0, T_ee=T, opening=g)
    assert est._locked_contact is None and not est._contact_votes
    est.lock_votes_open = True
    est.measure(u, 2.0, T_ee=T, opening=g)
    assert est._locked_contact is not None


def test_arm_stage_d_lifts():
    model = FingerFEMModel.from_config({
        "geometry": {"length": 0.1, "width": 0.02, "height": 0.01},
        "mesh": {"nx": 12, "ny": 3, "nz": 3},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    })
    mech = GraspMechanics.build(model, [0.85, 1.0, 0.5], [0.028, 0.024, 0.01], mass=0.12, mu=0.45)
    est = GraspEstimator.build(mech, {
        "mode": "world", "pixel_noise_std": 0.0,
        "unknown_contact": True,
        "pod_n_x": 3, "pod_n_z": 2, "q_modes": 6,
        "lock_votes": 1,
        "q_min_norm": 1.0e-3,
        "q_lock_norm": 1.0e-3,
        "max_residual_rel": 0.5,
        "lock_residual_rel": 0.5,
        "lock_min_force": 0.2,
        "vote_clear_bad": 99,
        "marker_nx": 6, "marker_ny": 2,
        "unknown_contact_cfg": {"face_stride": 3, "r_loc": 4, "refine": True, "refine_n": 5},
        "world_camera": {
            "position": [0.26, -0.18, 0.12],
            "target": [0.28, -0.02, 0.015],
            "fov_y_deg": 42.0, "image_width": 160, "image_height": 120,
        },
    }, seed=0)
    grasp = GraspSim(
        mech, est, opening_start=0.032, opening_min=0.016, close_speed=0.02,
        lift_height=0.03, lift_speed=0.04, force_target=mech.hold_force + 0.6,
        force_tol=0.25, kp=0.006, ki=0.001, settle_s=0.20, dt=0.01,
    )
    sim = build_arm_grasp_sim(grasp, {
        "object_xy": [0.32, 0.0],
        "ik_max_dq": 0.12,
        "ik_tol": 3.0e-3,
    })
    sim.run(t_max=14.0)
    assert sim.success()
    assert est.last_contact_err_m is not None
    assert est.last_contact_err_m < 0.012


def test_arm_scene_builds():
    from src.grasp.scene import build_arm_grasp_state

    sim = build_arm_grasp_sim(_tiny_grasp(), {"object_xy": [0.32, 0.0]})
    st = build_arm_grasp_state(sim)
    assert st.n_nodes > 0
    assert any(o.name.startswith("link") for o in st.objects)
    assert any(o.name == "object" for o in st.objects)
