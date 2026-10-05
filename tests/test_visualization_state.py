"""Tests for VisualizationState, cameras and colormaps (no Taichi runtime needed)."""

import numpy as np
import pytest

from src.geometry.finger import FingerGeometry, make_rectangular_finger_mesh
from src.visualization.camera import ObservationCamera, OrbitCamera, look_at_rotation
from src.visualization.colormap import available_colormaps, colormap
from src.visualization.state import ContactPoint, ForceVector, ObjectGeometry, VisualizationState


@pytest.fixture(scope="module")
def mesh():
    return make_rectangular_finger_mesh(FingerGeometry(0.1, 0.02, 0.01), 4, 2, 1)


def test_state_displaced_amplification_is_visual_only(mesh):
    u = np.zeros_like(mesh.nodes)
    u[:, 2] = -1e-3 * mesh.nodes[:, 0] / 0.1
    st = VisualizationState(mesh.nodes, mesh.surface_faces, element_edges=mesh.element_edges, surface_edges=mesh.surface_edges, displacement=u)
    assert np.allclose(st.nodes_deformed, mesh.nodes + u)
    assert np.allclose(st.displaced(10.0), mesh.nodes + 10.0 * u)
    # physics arrays untouched
    assert np.allclose(st.displacement, u)
    assert st.max_displacement() == pytest.approx(1e-3)
    assert st.topology_key() == (mesh.n_nodes, len(mesh.surface_faces), len(mesh.element_edges), len(mesh.surface_edges))
    # legacy input: element connectivity -> corner edges
    st2 = VisualizationState(mesh.nodes, mesh.surface_faces, tetrahedra=mesh.hexes)
    assert st2.element_edges.shape[1] == 2 and st2.topology_key()[3] == -1


def test_state_without_displacement(mesh):
    st = VisualizationState(mesh.nodes, mesh.surface_faces)
    assert np.allclose(st.displaced(50.0), mesh.nodes)
    assert st.displacement_magnitude().shape == (mesh.n_nodes,)
    assert st.max_displacement() == 0.0


def test_state_validation_errors(mesh):
    with pytest.raises(ValueError):
        VisualizationState(mesh.nodes, mesh.surface_faces, displacement=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        VisualizationState(mesh.nodes, mesh.surface_faces, fixed_nodes=[mesh.n_nodes + 5])
    with pytest.raises(ValueError):
        VisualizationState(mesh.nodes, np.array([[0, 1, mesh.n_nodes]]))


def test_forces_and_contacts(mesh):
    p = mesh.nodes[5]
    gt = ForceVector(p, [0, 0, -2.0], label="gt")
    est = ForceVector(p, [0, 0, -1.8], label="est")
    st = VisualizationState(
        mesh.nodes, mesh.surface_faces,
        contact_points=[ContactPoint(p, node_index=5)],
        ground_truth_force=gt, estimated_force=est,
        extra_forces=[ForceVector(p, [1.0, 0, 0])],
    )
    roles = [r for r, _ in st.all_forces()]
    assert roles == ["gt", "est", "extra"]
    assert gt.magnitude == pytest.approx(2.0)
    assert np.allclose(gt.direction, [0, 0, -1])
    assert ForceVector(p, [0, 0, 0]).direction.tolist() == [0, 0, 0]


def test_object_geometry_and_scene_bounds(mesh):
    sph = ObjectGeometry.sphere([0.2, 0.0, 0.0], 0.01, attach_to=[0.1, 0.01, 0.01])
    st = VisualizationState(mesh.nodes, mesh.surface_faces, objects=[sph])
    lo, hi = st.scene_bounds(include_objects=True)
    assert hi[0] >= 0.21 - 1e-9
    lo_f, hi_f = st.finger_bounds()
    assert hi_f[0] == pytest.approx(0.1)
    assert sph.attach_to.shape == (3,)


def test_colormaps():
    v = np.linspace(0, 1, 11)
    for name in available_colormaps():
        rgb = colormap(v, name)
        assert rgb.shape == (11, 3)
        assert rgb.dtype == np.float32
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    flat = colormap(np.zeros(5), "viridis")
    assert np.allclose(flat, flat[0])  # degenerate range -> uniform color
    with pytest.raises(KeyError):
        colormap(v, "nope")


def test_look_at_rotation_is_orthonormal_and_forward():
    R = look_at_rotation([0, -1, 0.5], [0, 0, 0], [0, 0, 1])
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    fwd = np.array([0, 1, -0.5]) / np.linalg.norm([0, 1, -0.5])
    assert np.allclose(R[:, 2], fwd)
    assert R[:, 1] @ np.array([0, 0, 1]) < 0  # camera +y points down


def test_observation_camera_projection_and_frustum():
    cam = ObservationCamera(position=[0, -1, 0], target=[0, 0, 0], up=[0, 0, 1], fov_y_deg=90, image_width=200, image_height=100)
    uv, z = cam.project(np.array([[0, 0, 0]]))
    assert np.allclose(uv[0], [100, 50])  # target projects to the image center
    assert z[0] == pytest.approx(1.0)
    # a point above the target must appear higher in the image (smaller v)
    uv_up, _ = cam.project(np.array([[0, 0, 0.2]]))
    assert uv_up[0, 1] < 50
    # a point to the camera's right (+x in world here) appears at larger u
    uv_right, _ = cam.project(np.array([[0.2, 0, 0]]))
    assert uv_right[0, 0] > 100
    corners = cam.frustum_corners(1.0)
    assert corners.shape == (4, 3)
    assert np.allclose(corners[:, 1], 0.0)  # image plane through the target
    v, e = cam.frustum_lines()
    assert e.max() < len(v)
    K = cam.intrinsics()
    assert K[1, 1] == pytest.approx(50.0)  # fy = (H/2)/tan(45 deg)
    T = cam.extrinsics()
    assert np.allclose(T[:3, :3] @ cam.position + T[:3, 3], 0.0)


def test_orbit_camera_controls():
    cam = OrbitCamera(target=[0, 0, 0], distance=1.0, yaw_deg=0.0, pitch_deg=0.0)
    assert np.allclose(cam.position, [1, 0, 0])
    cam.orbit(90.0, 0.0)
    assert np.allclose(cam.position, [0, 1, 0], atol=1e-12)
    cam.orbit(0.0, 45.0)
    assert cam.position[2] == pytest.approx(np.sin(np.deg2rad(45)))
    cam.zoom(0.5)
    assert cam.distance == pytest.approx(0.5)
    cam.pan(0.1, 0.0)
    assert not np.allclose(cam.target, 0.0)
    cam.fit([0, 0, 0], [1, 1, 1], distance_rel=2.0)
    assert np.allclose(cam.target, 0.5)
    assert cam.distance == pytest.approx(2.0 * np.sqrt(3))
    cam.zoom(3.0)
    cam.reset()
    assert cam.distance == pytest.approx(2.0 * np.sqrt(3))
    cam.pitch_deg = 0.0
    cam.orbit(0.0, 500.0)
    assert cam.pitch_deg == 89.0  # clamped
