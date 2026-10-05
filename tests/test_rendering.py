"""Milestone 5 tests: surface markers, pinhole camera model, synthetic renderer, vision dataset."""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry.finger import FingerGeometry, make_rectangular_finger_mesh
from src.rendering.markers import SurfaceMarkers, finger_face_ids, make_marker_grid
from src.rendering.synthetic_camera import RenderAppearance, disc_triangles, jitter_camera, nominal_camera
from src.visualization.camera import ObservationCamera


@pytest.fixture(scope="module")
def geo_mesh():
    g = FingerGeometry(length=0.1, width=0.012, height=0.01)
    return g, make_rectangular_finger_mesh(g, nx=10, ny=3, nz=2)


@pytest.fixture(scope="module")
def cam(geo_mesh):
    g, _ = geo_mesh
    return nominal_camera(g, {"target_rel": [0.55, 0.5, 0.5], "position_offset_rel": [0.05, -1.1, 0.55], "fov_y_deg": 40.0}, 320, 240)


# ------------------------------------------------------------------ markers
def test_marker_grid_lies_on_face_and_follows_displacement(geo_mesh):
    g, mesh = geo_mesh
    mk = make_marker_grid(mesh, g, "top", nx=6, ny=2, margin_rel=0.1)
    assert len(mk) == 12
    p = mk.positions(mesh)
    assert np.allclose(p[:, 2], g.height)
    assert p[:, 0].min() >= 0.1 * g.length - 1e-12 and p[:, 0].max() <= 0.9 * g.length + 1e-12
    assert np.allclose(mk.barycentric.sum(axis=1), 1.0) and (mk.barycentric >= -1e-12).all()
    # rigid translation of the mesh translates the markers exactly
    u = np.tile([0.001, -0.002, 0.003], (mesh.n_nodes, 1))
    assert np.allclose(mk.positions(mesh, u) - p, u[0])
    assert np.allclose(mk.normals(mesh), [0, 0, 1])


def test_marker_grid_other_faces(geo_mesh):
    g, mesh = geo_mesh
    for face, axis, val in [("bottom", 2, 0.0), ("side_pos_y", 1, g.width), ("side_neg_y", 1, 0.0), ("tip", 0, g.length)]:
        mk = make_marker_grid(mesh, g, face, nx=3, ny=2)
        assert np.allclose(mk.positions(mesh)[:, axis], val)
        assert len(finger_face_ids(mesh, g, face)) > 0
    with pytest.raises(KeyError):
        finger_face_ids(mesh, g, "nope")


def test_marker_roundtrip_dict(geo_mesh):
    g, mesh = geo_mesh
    mk = make_marker_grid(mesh, g, "top", 4, 2)
    mk2 = SurfaceMarkers.from_dict(mk.to_dict())
    assert np.array_equal(mk.face_ids, mk2.face_ids) and np.allclose(mk.barycentric, mk2.barycentric)


def test_visibility_facing_and_frustum(geo_mesh, cam):
    g, mesh = geo_mesh
    top = make_marker_grid(mesh, g, "top", 5, 2)
    bottom = make_marker_grid(mesh, g, "bottom", 5, 2)
    assert top.visibility(cam, mesh).all()  # camera is above the finger
    assert not bottom.visibility(cam, mesh).any()  # bottom face points away
    # a camera far off to the side sees nothing inside the image
    far = ObservationCamera(position=cam.position + np.array([5.0, 0, 0]), target=cam.target + np.array([5.0, 0, 0]),
                            fov_y_deg=cam.fov_y_deg, image_width=320, image_height=240)
    assert not top.visibility(far, mesh).any()


# ------------------------------------------------------------------ pinhole model
def test_projection_center_and_axes(cam):
    uv, z = cam.project(cam.target[None, :])
    assert np.allclose(uv[0], [cam.image_width / 2, cam.image_height / 2])
    assert z[0] > 0
    R = cam.rotation_world_from_camera
    d = np.linalg.norm(cam.target - cam.position)
    right = cam.target + 0.01 * R[:, 0]
    down = cam.target + 0.01 * R[:, 1]
    uv_r, _ = cam.project(right[None, :])
    uv_d, _ = cam.project(down[None, :])
    assert uv_r[0, 0] > cam.image_width / 2 and abs(uv_r[0, 1] - cam.image_height / 2) < 1e-6
    assert uv_d[0, 1] > cam.image_height / 2 and abs(uv_d[0, 0] - cam.image_width / 2) < 1e-6
    # focal length from the vertical FOV
    K = cam.intrinsics()
    fy = 0.5 * cam.image_height / np.tan(np.deg2rad(cam.fov_y_deg) / 2)
    assert np.isclose(K[1, 1], fy) and np.isclose(uv_d[0, 1] - cam.image_height / 2, fy * 0.01 / d, rtol=1e-6)


def test_extrinsics_roundtrip(cam):
    T = cam.extrinsics()
    R_wc, t = T[:3, :3].T, T[:3, 3]
    assert np.allclose(-R_wc @ t, cam.position)
    assert np.allclose(R_wc.T @ R_wc, np.eye(3))
    assert np.allclose(R_wc[:, 2], cam.forward)


def test_jitter_camera_is_bounded_and_seeded(geo_mesh, cam):
    g, _ = geo_mesh
    jit = {"position_rel": 0.05, "target_rel": 0.02, "fov_deg": 2.0}
    c1 = jitter_camera(cam, g, jit, np.random.default_rng(3))
    c2 = jitter_camera(cam, g, jit, np.random.default_rng(3))
    assert np.allclose(c1.position, c2.position) and c1.fov_y_deg == c2.fov_y_deg
    assert np.abs(c1.position - cam.position).max() <= 0.05 * g.length + 1e-12
    assert abs(c1.fov_y_deg - cam.fov_y_deg) <= 2.0
    assert (c1.image_width, c1.image_height) == (cam.image_width, cam.image_height)


def test_disc_triangles_geometry():
    c = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]])
    n = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    tri = disc_triangles(c, n, radius=0.1, n_seg=8, lift_rel=0.0)
    assert tri.shape == (2 * 8 * 3, 3)
    t0 = tri[: 8 * 3]
    assert np.allclose(t0[:, 2], 1.0)  # first disc lies in its plane
    assert np.allclose(np.linalg.norm(t0[1::3, :2], axis=1), 0.1)  # rim points at radius r
    assert np.allclose(t0.reshape(8, 3, 3).mean(axis=(0, 1)), c[0], atol=0.05)  # centroid ≈ centre


# ------------------------------------------------------------------ renderer (needs a GGUI-capable Taichi backend)
def _renderer_or_skip(w, h):
    try:
        from src.rendering.synthetic_camera import SyntheticCameraRenderer

        return SyntheticCameraRenderer(w, h)
    except Exception as e:  # pragma: no cover - machine dependent
        pytest.skip(f"GGUI renderer not available: {e}")


def test_renderer_matches_pinhole_projection(geo_mesh, cam):
    """Rendered marker discs must appear where ObservationCamera.project puts them."""
    from scipy import ndimage

    g, mesh = geo_mesh
    r = _renderer_or_skip(cam.image_width, cam.image_height)
    try:
        r.set_mesh(mesh.n_nodes, mesh.surface_faces)
        mk = make_marker_grid(mesh, g, "top", 5, 2, margin_rel=0.1)
        u = np.zeros((mesh.n_nodes, 3))
        u[:, 2] = -0.02 * (mesh.nodes[:, 0] / g.length) ** 2  # bend the finger down
        p, n = mk.positions(mesh, u), mk.normals(mesh, u)
        ap = RenderAppearance(marker_radius=0.12 * g.height, marker_color=(1.0, 0.1, 0.05), finger_color=(0.7, 0.7, 0.7),
                              background_color=(0.1, 0.1, 0.1))
        img = r.render(mesh.nodes + u, cam, ap, markers_xyz=p, marker_normals=n)
        assert img.shape == (cam.image_height, cam.image_width, 3) and 0.0 <= img.min() and img.max() <= 1.0
        mask = (img[..., 0] > 0.55) & (img[..., 1] < 0.45)
        lab, nb = ndimage.label(mask)
        assert nb == len(mk)
        cm = np.array(ndimage.center_of_mass(mask, lab, range(1, nb + 1)))
        blobs = np.stack([cm[:, 1] + 0.5, cm[:, 0] + 0.5], axis=1)
        uv, _ = cam.project(p)
        d = np.linalg.norm(uv[:, None] - blobs[None], axis=2).min(axis=1)
        assert d.max() < 1.0, f"renderer / pinhole mismatch: max {d.max():.2f} px"

        # appearance knobs: background colour is reproduced, noise & grayscale change the output shape/statistics
        corner = img[0, 0]
        assert np.allclose(corner, ap.background_color, atol=0.02)
        img_g = r.render(mesh.nodes + u, cam, RenderAppearance(grayscale=True), rng=np.random.default_rng(0))
        assert img_g.shape == (cam.image_height, cam.image_width, 1)
        img_n = r.render(mesh.nodes + u, cam, RenderAppearance(image_noise_std=0.05), rng=np.random.default_rng(0))
        img_c = r.render(mesh.nodes + u, cam, RenderAppearance(image_noise_std=0.0))
        assert 0.02 < np.std(img_n - img_c) < 0.08
        with pytest.raises(ValueError):
            r.render(mesh.nodes + u, ObservationCamera(image_width=64, image_height=48))
    finally:
        r.destroy()


def test_vision_dataset_roundtrip(tmp_path):
    from src.vision.dataset import VisionDataset

    S, M, H, W = 3, 4, 6, 8
    rng = np.random.default_rng(0)
    cams = [ObservationCamera(position=[0.05, -0.1 - 0.01 * s, 0.07], target=[0.05, 0.005, 0.005], image_width=W, image_height=H) for s in range(S)]
    ds = VisionDataset(
        images=rng.integers(0, 255, (S, H, W, 3), dtype=np.uint8), marker_px=rng.normal(size=(S, M, 2)).astype(np.float32),
        marker_px_clean=np.zeros((S, M, 2), np.float32), marker_visible=np.ones((S, M), bool), q=rng.normal(size=(S, 5)),
        force_magnitude=np.linspace(0.1, 3, S), force_vector=np.zeros((S, 3)), contact_position=np.zeros((S, 3)),
        contact_normal=np.tile([0, 0, 1.0], (S, 1)), camera_K=np.stack([c.intrinsics() for c in cams]),
        camera_T_cw=np.stack([c.extrinsics() for c in cams]), camera_position=np.stack([c.position for c in cams]),
        camera_fov_y_deg=np.array([c.fov_y_deg for c in cams]), fem_index=np.arange(S),
        appearance=[{"background_color": [0, 0, 0]}] * S, meta={"cameras": [c.to_dict() for c in cams], "note": "test"},
    )
    path = ds.save(tmp_path / "v.npz")
    ds2 = VisionDataset.load(path)
    assert len(ds2) == S and ds2.image_shape == (H, W, 3) and ds2.n_markers == M
    assert np.array_equal(ds2.images, ds.images) and np.allclose(ds2.q, ds.q)
    assert ds2.meta["note"] == "test" and ds2.appearance[0]["background_color"] == [0, 0, 0]
    c = ds2.camera(1)
    assert np.allclose(c.position, cams[1].position) and np.allclose(c.intrinsics(), ds2.camera_K[1])
    # extrinsics-only reconstruction (no camera dicts in meta) reproduces the projection
    ds2.meta.pop("cameras")
    c_fallback = ds2.camera(1)
    pts = rng.normal(size=(5, 3)) * 0.01 + [0.05, 0.005, 0.005]
    assert np.allclose(c_fallback.project(pts)[0], cams[1].project(pts)[0], atol=1e-6)
    assert ds2.summary()["n_samples"] == S
