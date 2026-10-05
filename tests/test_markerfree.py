"""Marker-free experiment tests: paired dataset, paired rendering, ResNet-18 image → q."""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry.finger import FingerGeometry, make_rectangular_finger_mesh
from src.rendering.markers import make_marker_grid
from src.rendering.synthetic_camera import RenderAppearance, nominal_camera
from src.vision.paired_dataset import PairedVisionDataset
from src.visualization.camera import ObservationCamera


def _fake_paired(S=6, H=6, W=8, M=4, r=5, seed=0) -> PairedVisionDataset:
    rng = np.random.default_rng(seed)
    cams = [ObservationCamera(position=[0.05, -0.1 - 0.01 * s, 0.07], target=[0.05, 0.005, 0.005], image_width=W, image_height=H) for s in range(S)]
    q_sim = rng.normal(size=(S, r))
    return PairedVisionDataset(
        images_marker=rng.integers(0, 255, (S, H, W, 3), dtype=np.uint8), images_raw=rng.integers(0, 255, (S, H, W, 3), dtype=np.uint8),
        marker_px=rng.normal(size=(S, M, 2)).astype(np.float32), marker_px_clean=np.zeros((S, M, 2), np.float32),
        marker_visible=np.ones((S, M), bool), q_sim=q_sim, q_marker=q_sim + 0.1 * rng.normal(size=(S, r)),
        force_magnitude=np.linspace(0.1, 3, S), force_vector=np.zeros((S, 3)), contact_position=np.zeros((S, 3)),
        contact_normal=np.tile([0, 0, 1.0], (S, 1)), camera_K=np.stack([c.intrinsics() for c in cams]),
        camera_T_cw=np.stack([c.extrinsics() for c in cams]), camera_position=np.stack([c.position for c in cams]),
        camera_fov_y_deg=np.array([c.fov_y_deg for c in cams]), fem_index=np.arange(S),
        appearance=[{"background_color": [0, 0, 0]}] * S, meta={"cameras": [c.to_dict() for c in cams], "note": "t"},
    )


def test_paired_dataset_roundtrip_and_views(tmp_path):
    ds = _fake_paired()
    path = ds.save(tmp_path / "p.npz")
    ds2 = PairedVisionDataset.load(path)
    assert len(ds2) == 6 and ds2.image_shape == (6, 8, 3) and ds2.r == 5
    assert np.array_equal(ds2.images_raw, ds.images_raw) and np.array_equal(ds2.images_marker, ds.images_marker)
    assert np.allclose(ds2.q_sim, ds.q_sim) and np.allclose(ds2.q_marker, ds.q_marker)
    assert not np.allclose(ds2.q_sim, ds2.q_marker)  # labels are kept distinct
    # VisionDataset views select image modality and q label explicitly
    v_raw = ds2.as_vision_dataset(images="raw", q="sim")
    v_mk = ds2.as_vision_dataset(images="marker", q="marker")
    assert v_raw.images is ds2.images_raw and np.allclose(v_raw.q, ds2.q_sim)
    assert v_mk.images is ds2.images_marker and np.allclose(v_mk.q, ds2.q_marker)
    assert v_raw.meta["paired_view"] == {"images": "raw", "q": "sim"}
    assert np.allclose(v_raw.camera(2).position, ds.camera_position[2])
    with pytest.raises(ValueError):
        ds2.q("oracle")
    sub = ds2.subset([1, 3])
    assert len(sub) == 2 and np.allclose(sub.q_sim, ds2.q_sim[[1, 3]]) and len(sub.appearance) == 2
    assert "q_marker_vs_sim_rel_median" in ds2.summary()


def test_paired_dataset_rejects_mismatched_pairs():
    ds = _fake_paired()
    with pytest.raises(ValueError):
        PairedVisionDataset(**{**ds.__dict__, "images_raw": ds.images_raw[:-1]})


def test_paired_render_identical_outside_markers():
    """Marker-on and marker-off renders of the same state differ only at the marker discs."""
    try:
        from src.rendering.synthetic_camera import SyntheticCameraRenderer

        r = SyntheticCameraRenderer(160, 120)
    except Exception as e:  # pragma: no cover - machine dependent
        pytest.skip(f"GGUI renderer not available: {e}")
    try:
        g = FingerGeometry(length=0.1, width=0.012, height=0.01)
        mesh = make_rectangular_finger_mesh(g, nx=10, ny=3, nz=2)
        cam = nominal_camera(g, {"target_rel": [0.55, 0.5, 0.5], "position_offset_rel": [0.05, -1.1, 0.55], "fov_y_deg": 40.0}, 160, 120)
        r.set_mesh(mesh.n_nodes, mesh.surface_faces)
        mk = make_marker_grid(mesh, g, "top", 5, 2, margin_rel=0.1)
        u = np.zeros((mesh.n_nodes, 3))
        u[:, 2] = -0.02 * (mesh.nodes[:, 0] / g.length) ** 2
        p, n = mk.positions(mesh, u), mk.normals(mesh, u)
        ap = RenderAppearance(marker_radius=0.12 * g.height, marker_color=(1.0, 0.1, 0.05), image_noise_std=0.02)
        im_on = r.render(mesh.nodes + u, cam, ap, markers_xyz=p, marker_normals=n, rng=np.random.default_rng(7))
        im_off = r.render(mesh.nodes + u, cam, ap, markers_xyz=None, rng=np.random.default_rng(7))
        diff = np.abs(im_on - im_off).max(axis=-1)
        changed = diff > 0.05
        assert 0.0 < changed.mean() < 0.05  # only small disc regions differ
        uv, _ = cam.project(p)
        ys, xs = np.nonzero(changed)
        d = np.linalg.norm(np.stack([xs + 0.5, ys + 0.5], 1)[:, None] - uv[None], axis=2).min(axis=1)
        assert d.max() < 6.0  # every changed pixel lies near a projected marker
        # marker-free image contains no marker-coloured pixels
        red = (im_off[..., 0] > 0.55) & (im_off[..., 1] < 0.45)
        assert not red.any()
    finally:
        r.destroy()


def test_resnet_q_normaliser_and_overfit(tmp_path):
    pytest.importorskip("torch")
    from src.vision.resnet_q import QNormalizer, ResNetQModel, train_resnet_q

    qn = QNormalizer.fit(np.array([[1.0, 10.0], [3.0, 30.0]]))
    assert np.allclose(qn.mean, [2, 20]) and np.allclose(qn.denormalize(qn.normalize([[3.0, 30.0]])), [[3, 30]])
    assert np.allclose(QNormalizer.from_dict(qn.to_dict()).std, qn.std)

    S, H, W, r = 48, 40, 56, 3
    rng = np.random.default_rng(0)
    q = rng.normal(size=(S, r)) * np.array([0.1, 0.01, 0.001])
    images = np.full((S, H, W, 3), 40, np.uint8)
    for s in range(S):
        x = int(np.clip(28 + 150 * q[s, 0], 4, W - 5))
        images[s, 8:32, x - 3 : x + 3] = 220
    train = np.zeros(S, bool)
    train[:36] = True
    model, hist = train_resnet_q(images, q, train, ~train, epochs=25, batch_size=12, lr=1e-3, patience=25, width=8, augment=False, log=None)
    pred = model.predict(images[~train])
    assert np.corrcoef(pred[:, 0], q[~train, 0])[0, 1] > 0.8
    assert model.q_norm is not None and model.img_norm is not None and hist["best_epoch"] >= 0
    path = model.save(tmp_path / "rq.pt")
    assert (tmp_path / "rq.norm.json").exists()
    model2 = ResNetQModel.load(path)
    assert np.allclose(model2.predict(images[:4]), model.predict(images[:4]), atol=1e-5)
    assert np.allclose(model2.q_norm.mean, model.q_norm.mean)
