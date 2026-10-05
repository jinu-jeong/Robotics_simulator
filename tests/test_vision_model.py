"""Milestone 6 tests: marker→q geometry, ridge fit, splits, vision→force pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.fem.finger_model import FingerFEMModel
from src.rendering.markers import make_marker_grid
from src.rom.pod import compute_pod
from src.vision.dataset import VisionDataset
from src.vision.features import image_jacobian, marker_delta_features, mode_marker_displacements, solve_q_from_jacobian
from src.vision.marker_model import GeometricMarkerModel, RidgeMarkerModel, fuse_q_by_fem_index
from src.vision.pipeline import VisionMechanicsPipeline, build_rom
from src.vision.splits import contact_ids_from_positions, force_ids_from_magnitudes, make_vision_split
from src.visualization.camera import ObservationCamera


@pytest.fixture(scope="module")
def finger_pack():
    model = FingerFEMModel.from_config({
        "geometry": {"length": 0.1, "width": 0.012, "height": 0.01},
        "mesh": {"nx": 12, "ny": 3, "nz": 2},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    })
    model.restrict_contact_surface("top")
    markers = make_marker_grid(model.mesh, model.geometry, "top", nx=5, ny=2, margin_rel=0.1)
    contacts = [model.geometry.point_from_relative([x, 0.5, 1.0]) for x in (0.4, 0.55, 0.7, 0.85)]
    cols, meta = [], []
    for cpos in contacts:
        for F in (0.5, 1.5, 2.5):
            res, contact = model.solve_normal_contact(cpos, F)
            cols.append(res.u.reshape(-1))
            meta.append({"position": contact.position.copy(), "normal": contact.normal.copy(),
                         "F": float(F), "vector": (-F * contact.normal).copy(), "u": res.u.copy()})
    U = np.stack(cols, axis=1)
    basis = compute_pod(U)
    basis = basis.truncate(min(4, basis.r))
    rom = build_rom(model, basis, "top")
    target = model.geometry.point_from_relative([0.55, 0.5, 0.5])
    cam = ObservationCamera(
        position=target + model.geometry.length * np.array([0.05, -1.1, 0.55]),
        target=target, fov_y_deg=40.0, image_width=160, image_height=120,
    )
    return model, markers, basis, rom, meta, cam


def test_image_jacobian_recovers_q_from_clean_markers(finger_pack):
    model, markers, basis, rom, meta, cam = finger_pack
    p0 = markers.positions(model.mesh, None)
    dp = mode_marker_displacements(markers, model.mesh, basis)
    J = image_jacobian(cam, p0, dp, eps=1e-4)
    assert J.shape == (2 * len(markers), basis.r)
    sample = meta[-2]
    q = basis.project(sample["u"])
    uv, _ = cam.project(markers.positions(model.mesh, sample["u"]))
    uv0, _ = cam.project(p0)
    q_hat = solve_q_from_jacobian(J, uv - uv0, np.ones(len(markers), bool), ridge=1e-8)
    rel = np.linalg.norm(q_hat - q) / np.linalg.norm(q)
    assert rel < 0.05, f"geometric q recovery too poor: {rel}"


def test_geometric_model_and_force_pipeline(finger_pack):
    model, markers, basis, rom, meta, cam = finger_pack
    geo = GeometricMarkerModel(basis, markers, model.mesh, ridge=1e-6, use_noisy=False)
    sample = meta[-1]
    q_hat = geo.predict_one(cam, cam.project(markers.positions(model.mesh, sample["u"]))[0], np.ones(len(markers), bool))
    c = rom.mapping.locate(sample["position"])
    est = rom.estimate_force(q_hat, c, method="displacement", mode="normal")
    assert abs(est.magnitude - sample["F"]) / sample["F"] < 0.08


def test_iterative_and_huber_match_clean_geometric(finger_pack):
    model, markers, basis, rom, meta, cam = finger_pack
    sample = meta[-1]
    uv, _ = cam.project(markers.positions(model.mesh, sample["u"]))
    vis = np.ones(len(markers), bool)
    q0 = GeometricMarkerModel(basis, markers, model.mesh, ridge=1e-6, use_noisy=False).predict_one(cam, uv, vis)
    q1 = GeometricMarkerModel(
        basis, markers, model.mesh, ridge=1e-6, use_noisy=False, n_iter=2, huber=1.5,
    ).predict_one(cam, uv, vis)
    rel = np.linalg.norm(q1 - q0) / max(np.linalg.norm(q0), 1e-30)
    assert rel < 0.05
    c = rom.mapping.locate(sample["position"])
    est = rom.estimate_force(q1, c, method="displacement", mode="normal")
    assert abs(est.magnitude - sample["F"]) / sample["F"] < 0.08


def test_fuse_q_by_fem_index():
    q = np.array([[1.0, 0.0], [3.0, 2.0], [0.0, 1.0]])
    fused = fuse_q_by_fem_index(q, np.array([0, 0, 1]))
    assert np.allclose(fused[0], [2.0, 1.0]) and np.allclose(fused[1], [2.0, 1.0])
    assert np.allclose(fused[2], [0.0, 1.0])


def test_ridge_marker_roundtrip(tmp_path, finger_pack):
    model, markers, basis, rom, meta, cam = finger_pack
    S, M, r = len(meta), len(markers), basis.r
    marker_px = np.zeros((S, M, 2), np.float32)
    q = np.zeros((S, r))
    cameras = []
    for s, sample in enumerate(meta):
        uv, _ = cam.project(markers.positions(model.mesh, sample["u"]))
        marker_px[s] = uv + np.random.default_rng(s).normal(0, 0.05, uv.shape)
        q[s] = basis.project(sample["u"])
        cameras.append(cam.to_dict())
    vds = VisionDataset(
        images=np.zeros((S, 8, 8, 3), np.uint8), marker_px=marker_px, marker_px_clean=marker_px.copy(),
        marker_visible=np.ones((S, M), bool), q=q,
        force_magnitude=np.array([m["F"] for m in meta]), force_vector=np.stack([m["vector"] for m in meta]),
        contact_position=np.stack([m["position"] for m in meta]), contact_normal=np.stack([m["normal"] for m in meta]),
        camera_K=np.stack([cam.intrinsics()] * S), camera_T_cw=np.stack([cam.extrinsics()] * S),
        camera_position=np.stack([cam.position] * S), camera_fov_y_deg=np.full(S, cam.fov_y_deg),
        fem_index=np.arange(S), appearance=[{}] * S, meta={"cameras": cameras},
    )
    train = np.ones(S, bool)
    train[-3:] = False
    ridge = RidgeMarkerModel.fit(vds, train, markers, model.mesh, ridge_lambda=1e-3, use_noisy=True)
    path = ridge.save(tmp_path / "ridge.npz")
    ridge2 = RidgeMarkerModel.load(path, markers, model.mesh)
    ridge2.uv0_cache = ridge.uv0_cache
    q_hat = ridge2.predict(vds, np.nonzero(~train)[0])
    rel = np.linalg.norm(q_hat - q[~train], axis=1) / np.linalg.norm(q[~train], axis=1)
    assert rel.mean() < 0.15

    pipe = VisionMechanicsPipeline(basis, rom, lambda idx: ridge.predict(vds, idx), name="ridge")
    metrics = pipe.evaluate(vds, np.nonzero(~train)[0])
    assert metrics["force_rel_median"] < 0.2


def test_marker_delta_features_masks_invisible():
    px = np.array([[10.0, 20.0], [30.0, 40.0]])
    uv0 = np.zeros((2, 2))
    vis = np.array([True, False])
    f = marker_delta_features(px, uv0, vis)
    assert f.shape == (4,)
    assert np.allclose(f[:2], [10, 20]) and np.allclose(f[2:], 0)


def test_splits_contact_holdout():
    fem_pos = np.array([[0.03 + 0.02 * i, 0.0, 0.01] for i in range(4) for _ in range(3)])
    assert contact_ids_from_positions(fem_pos).max() + 1 == 4

    class DS:
        fem_index = np.repeat(np.arange(12), 2)
        force_magnitude = np.repeat(np.tile([0.5, 1.5, 2.5], 4), 2)

        def __len__(self):
            return len(self.fem_index)

    split = make_vision_split(DS(), fem_pos, hold_every=2)
    assert split.meta["n_contacts"] == 4
    assert split.meta["n_held_contacts"] == 2
    assert split.train.sum() + split.test.sum() == 24
    assert not np.any(split.train & split.test)
    assert force_ids_from_magnitudes(np.array([0.5, 1.5, 0.5])).tolist() == [0, 1, 0]


def test_image_cnn_optional_overfit(tmp_path):
    pytest.importorskip("torch")
    from src.vision.image_model import ImageCNNModel, train_image_cnn

    S, H, W, r = 48, 32, 48, 3
    rng = np.random.default_rng(0)
    q = rng.normal(size=(S, r)) * np.array([0.1, 0.01, 0.001])
    images = np.zeros((S, H, W, 3), np.uint8)
    for s in range(S):
        images[s] = np.clip(128 + 400 * q[s, 0], 0, 255)

    class Fake:
        pass

    fake = Fake()
    fake.image_shape = (H, W, 3)
    fake.images = images
    fake.q = q

    train = np.zeros(S, bool)
    train[:36] = True
    model, _hist = train_image_cnn(fake, train, ~train, epochs=40, batch_size=16, lr=1e-3, patience=20, hidden=32, seed=0)
    pred = model.predict(images[~train])
    corr = np.corrcoef(pred[:, 0], q[~train, 0])[0, 1]
    assert corr > 0.8
    out = model.save(tmp_path / "cnn_test.pt")
    model2 = ImageCNNModel.load(out)
    assert np.allclose(model2.predict(images[:4]), model.predict(images[:4]), atol=1e-5)
