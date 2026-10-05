"""Head-less smoke tests of the Taichi viewer (skipped if no GGUI backend)."""

import numpy as np
import pytest

from src.geometry.finger import FingerGeometry, make_rectangular_finger_mesh, root_fixed_nodes
from src.visualization.camera import ObservationCamera
from src.visualization.state import ContactPoint, ForceVector, ObjectGeometry, VisualizationState

try:  # Taichi GGUI may be unavailable on CI machines without Vulkan/Metal.
    from src.visualization.taichi_viewer import TaichiViewer

    _viewer = TaichiViewer(show_window=False)
    _GGUI_OK = True
except Exception as e:  # pragma: no cover
    _viewer = None
    _GGUI_OK = False
    _GGUI_ERR = repr(e)

pytestmark = pytest.mark.skipif(not _GGUI_OK, reason="Taichi GGUI not available")


def _demo_state(nx=8, ny=2, nz=1, with_u=True):
    geom = FingerGeometry(0.1, 0.02, 0.01)
    mesh = make_rectangular_finger_mesh(geom, nx, ny, nz)
    u = None
    if with_u:
        u = np.zeros_like(mesh.nodes)
        u[:, 2] = -2e-3 * (mesh.nodes[:, 0] / geom.length) ** 2
    fixed = root_fixed_nodes(mesh)
    tip = mesh.nodes[np.argmax(mesh.nodes[:, 0] + mesh.nodes[:, 2])]
    return VisualizationState(
        nodes_original=mesh.nodes,
        surface_faces=mesh.surface_faces,
        element_edges=mesh.element_edges,
        surface_edges=mesh.surface_edges,
        displacement=u,
        fixed_nodes=fixed,
        contact_points=[ContactPoint(tip)],
        ground_truth_force=ForceVector(tip, [0, 0, -1.0]),
        estimated_force=ForceVector(tip, [0.05, 0, -0.9]),
        objects=[ObjectGeometry.sphere(tip + [0, 0, 0.004], 0.004, attach_to=tip)],
        observation_camera=ObservationCamera(position=[0.05, -0.1, 0.05], target=[0.05, 0.01, 0.005]),
    )


def test_render_all_modes_produce_distinct_images():
    v = _viewer
    v.set_state(_demo_state())
    imgs = {}
    for mode in ("original", "deformed", "overlay"):
        v.options.mode = mode
        img = v.render_offscreen()
        assert img.shape == (v.res[1], v.res[0], 3)
        assert np.isfinite(img).all()
        imgs[mode] = img
    bg = np.array(v.cfg["background_color"], dtype=np.float32)
    for mode, img in imgs.items():
        non_bg = np.mean(np.any(np.abs(img - bg) > 0.02, axis=-1))
        assert non_bg > 0.01, f"{mode}: nothing rendered"
    assert np.abs(imgs["original"] - imgs["deformed"]).mean() > 1e-3
    assert np.abs(imgs["overlay"] - imgs["deformed"]).mean() > 1e-4


def test_amplification_changes_image_but_not_physics():
    v = _viewer
    st = _demo_state()
    u_before = st.displacement.copy()
    v.set_state(st)
    v.options.mode = "deformed"
    v.options.amplification = 1.0
    a = v.render_offscreen()
    v.options.amplification = 50.0
    b = v.render_offscreen()
    assert np.abs(a - b).mean() > 1e-3
    assert np.array_equal(st.displacement, u_before)
    assert np.allclose(v.renderer.mesh.positions["display"].to_numpy()[: st.n_nodes], (st.nodes_original + 50.0 * u_before).astype(np.float32), atol=1e-6)


def test_fixed_nodes_do_not_move_in_display_buffer():
    v = _viewer
    st = _demo_state()
    v.set_state(st)
    v.options.mode = "deformed"
    v.options.amplification = 100.0
    v.render_offscreen()
    disp = v.renderer.mesh.positions["display"].to_numpy()[: st.n_nodes]
    assert np.allclose(disp[st.fixed_nodes], st.nodes_original[st.fixed_nodes].astype(np.float32), atol=1e-6)


def test_toggles_and_state_without_displacement():
    v = _viewer
    v.set_state(_demo_state(with_u=False))
    o = v.options
    for attr in ("show_wireframe", "show_tet_edges", "show_nodes", "show_fixed_nodes", "show_contact",
                 "show_gt_force", "show_estimated_force", "show_objects", "show_observation_camera"):
        setattr(o, attr, True)
    for mode in ("original", "deformed", "overlay"):
        o.mode = mode
        img = v.render_offscreen()
        assert np.isfinite(img).all()
    o.color_mode = "solid"
    v.render_offscreen()
    lines = v.renderer.summary_lines(o)
    assert any("nodes:" in ln for ln in lines)


def test_topology_change_reallocates_buffers():
    v = _viewer
    v.set_state(_demo_state(nx=4))
    n1 = v.renderer.mesh.n_nodes
    v.set_state(_demo_state(nx=12))
    assert v.renderer.mesh.n_nodes != n1
    v.render_offscreen()


def test_amplification_steps_and_key_callbacks():
    v = _viewer
    v.set_state(_demo_state())
    v.options.amplification = 10.0
    v.step_amplification(+1)
    assert v.options.amplification == 20.0
    v.step_amplification(-1)
    v.step_amplification(-1)
    assert v.options.amplification == 5.0
    called = []
    v.register_key("Left", lambda vw: called.append(True), "test")
    v._key_callbacks["Left"][0](v)
    assert called == [True]
    assert "Left" in v.help_text()


def test_screenshot(tmp_path):
    v = _viewer
    v.set_state(_demo_state())
    p = v.save_screenshot(tmp_path / "shot.png")
    assert p.exists() and p.stat().st_size > 1000


def test_keypoints_and_overlay_inset():
    """Markers are drawn (toggle changes the image); the PIP inset shows up in the corner and the 3D scene stays visible."""
    v = _viewer
    st = _demo_state()
    kp = st.nodes_original[st.surface_faces[:20, 0]]
    st.keypoints = kp
    st.keypoints_visible = np.arange(len(kp)) % 2 == 0
    v.set_state(st)
    v.set_overlay_image(None)
    v.options.show_keypoints = True
    a = v.render_offscreen()
    v.options.show_keypoints = False
    b = v.render_offscreen()
    assert np.abs(a - b).max() > 0.1

    inset = np.zeros((30, 40, 3), np.float32)
    inset[:] = (0.0, 1.0, 0.0)
    v.set_overlay_image(inset, corner="top-right", max_width_frac=0.2)
    c = v.render_offscreen()
    H, W = c.shape[:2]
    x0, y_top, w, h = v._overlay_rect
    assert x0 + w <= W and y_top + h <= H
    assert np.allclose(c[y_top + h // 2, x0 + w // 2], (0.0, 1.0, 0.0), atol=0.02)  # inset visible, y flipped correctly
    assert np.abs(c[:, : x0 - 20] - b[:, : x0 - 20]).max() < 2.5 / 255  # rest of the frame (3D scene) unchanged (1 LSB rounding)
    v.set_overlay_image(None)
    assert np.abs(v.render_offscreen() - b).max() < 1e-3
