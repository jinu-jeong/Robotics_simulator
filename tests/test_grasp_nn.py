"""Stage E: marker-free world-camera scene rendering + ResNet q in the arm grasp loop."""

from __future__ import annotations

import numpy as np
import pytest

from src.fem.finger_model import FingerFEMModel
from src.grasp.arm_sim import build_arm_grasp_sim
from src.grasp.estimator import GraspEstimator
from src.grasp.mechanics import GraspMechanics
from src.grasp.nn_vision import (
    GraspSceneAppearance,
    RemoteGraspRenderer,
    basis_fingerprint,
    compose_grasp_scene,
    nn_camera,
    two_finger_faces,
)
from src.grasp.sim import GraspSim


@pytest.fixture(scope="module")
def tiny_world_sim():
    model = FingerFEMModel.from_config({
        "geometry": {"length": 0.1, "width": 0.02, "height": 0.01},
        "mesh": {"nx": 12, "ny": 3, "nz": 3},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    })
    mech = GraspMechanics.build(model, [0.85, 1.0, 0.5], [0.028, 0.024, 0.01], mass=0.12, mu=0.45)
    est = GraspEstimator.build(mech, {
        "mode": "world", "q_modes": 4, "pod_n_x": 3, "pod_n_z": 2, "pixel_noise_std": 0.0,
        "world_camera": {"image_width": 96, "image_height": 72},
        "nn": {"image_width": 64, "image_height": 48, "every": 2},
    })
    grasp = GraspSim(
        mech, est, opening_start=0.032, opening_min=0.016, close_speed=0.02,
        lift_height=0.03, lift_speed=0.04, force_target=mech.hold_force + 0.6,
        force_tol=0.20, kp=0.006, ki=0.001, settle_s=0.20, dt=0.01,
    )
    return build_arm_grasp_sim(grasp, {"object_xy": [0.32, 0.0]})


def test_compose_scene_and_camera(tiny_world_sim):
    sim = tiny_world_sim
    mech, est = sim.mech, sim.estimator
    mesh = mech.model.mesh
    u = mech.displacement(1.5)
    ap = GraspSceneAppearance()
    nodes, objs = compose_grasp_scene(
        mech, u, mech.opening_from_force(1.5), sim.T_grasp, sim._ee_object_center(sim.T_grasp), None,
        table_center=np.array([0.28, 0.0, -0.01]), table_size=np.array([0.42, 0.24, 0.02]), appearance=ap,
    )
    assert nodes.shape == (2 * mesh.n_nodes, 3) and len(objs) == 2
    assert two_finger_faces(mesh).shape == (2 * len(mesh.surface_faces), 3)
    # left finger really is displaced by the (rotated) FEM field
    nodes0, _ = compose_grasp_scene(
        mech, np.zeros_like(u), mech.opening_from_force(1.5), sim.T_grasp, sim._ee_object_center(sim.T_grasp), None,
        table_center=np.array([0.28, 0.0, -0.01]), table_size=np.array([0.42, 0.24, 0.02]), appearance=ap,
    )
    d = np.linalg.norm(nodes[: mesh.n_nodes] - nodes0[: mesh.n_nodes], axis=1)
    assert np.isclose(d.max(), np.linalg.norm(u, axis=1).max())
    cam = nn_camera(est.world_camera, 64, 48)
    assert (cam.image_width, cam.image_height) == (64, 48) and np.allclose(cam.position, est.world_camera.position)
    fp = basis_fingerprint(est.geo.basis.Phi)
    assert fp == est.basis_fingerprint and len(fp) == 12
    assert GraspSceneAppearance.from_dict(ap.to_dict()) == ap


def test_remote_renderer_and_nn_mode_in_arm_loop(tiny_world_sim, tmp_path):
    pytest.importorskip("torch")
    from src.vision.resnet_q import QNormalizer, ResNetQModel

    sim = tiny_world_sim
    mech, est = sim.mech, sim.estimator
    mesh = mech.model.mesh
    try:
        rend = RemoteGraspRenderer(64, 48, 2 * mesh.n_nodes, two_finger_faces(mesh), GraspSceneAppearance())
    except Exception as e:  # pragma: no cover - machine dependent
        pytest.skip(f"GGUI renderer not available in a child process: {e}")
    try:
        u = mech.displacement(1.0)
        nodes, objs = compose_grasp_scene(
            mech, u, mech.opening_from_force(1.0), sim.T_grasp, sim._ee_object_center(sim.T_grasp), None,
            table_center=np.array([0.28, 0.0, -0.01]), table_size=np.array([0.42, 0.24, 0.02]), appearance=GraspSceneAppearance(),
        )
        cam = nn_camera(est.world_camera, 64, 48)
        img = rend.render(nodes, cam, objs, seed=0)
        assert img.shape == (48, 64, 3) and img.dtype == np.uint8 and img.std() > 5
    finally:
        rend.close()

    # untrained (tiny) network with the estimator's q convention: the whole nn path must run end to end
    r = est.geo.basis.r
    net = ResNetQModel(r=r, in_ch=3, image_hw=(48, 64), width=8)
    net.q_norm = QNormalizer(mean=np.zeros(r), std=np.full(r, 1e-3))
    net.meta.update({"basis_fingerprint": est.basis_fingerprint, "target": "sim", "input": "raw"})
    ckpt = net.save(tmp_path / "tiny_nn.pt")
    est.nn_cfg = {"checkpoint": str(ckpt), "image_width": 64, "image_height": 48, "remote": True}
    sim.reset()
    sim.set_mode("nn")
    try:
        for _ in range(40):
            sim.step()
        assert est.nn is not None and est.last_image is not None and est.last_image.shape == (48, 64, 3)
        assert est._nn_step > 0 and est.nn.n_frames == -(-est._nn_step // est.nn_every)  # one frame per camera tick
        assert np.isfinite(sim.grasp.state.lam_meas)
        # a wrong basis is refused
        net.meta["basis_fingerprint"] = "deadbeef0000"
        bad = net.save(tmp_path / "bad_nn.pt")
        est.close()
        est.nn_cfg["checkpoint"] = str(bad)
        with pytest.raises(ValueError):
            est.ensure_nn()
    finally:
        est.close()
        sim.set_mode("gt")
