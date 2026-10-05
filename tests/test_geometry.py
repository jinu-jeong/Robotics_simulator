"""Tests for the finger mesh generator, mesh utilities and primitives."""

import numpy as np
import pytest

from src.geometry.finger import FingerGeometry, TetMesh, make_rectangular_finger_mesh, root_fixed_nodes
from src.geometry.mesh_utils import (
    euler_characteristic,
    extract_surface_triangles,
    fix_tet_orientation,
    is_closed_surface,
    nearest_node,
    tet_signed_volumes,
    triangle_areas,
    triangle_normals,
    unique_edges,
    vertex_normals,
)
from src.geometry.primitives import arrow_mesh, box_mesh, cone_mesh, cylinder_mesh, sphere_mesh, to_triangle_soup

GEOM = FingerGeometry(length=0.1, width=0.02, height=0.01)


@pytest.fixture(scope="module")
def mesh() -> TetMesh:
    return make_rectangular_finger_mesh(GEOM, nx=10, ny=2, nz=2)


def test_node_and_element_counts(mesh):
    nx, ny, nz = 10, 2, 2
    n_corner = (nx + 1) * (ny + 1) * (nz + 1)
    n_tet_cells = round(0.2 * nx) * ny * nz  # tip 20 % of the length -> Kuhn T10
    assert mesh.n_hexes == nx * ny * nz - n_tet_cells
    assert mesh.n_tets == 6 * n_tet_cells
    assert mesh.tets.shape[1] == 10 and mesh.hexes.shape[1] == 20
    # one mid node per unique corner edge
    assert mesh.n_nodes == n_corner + len(mesh.element_edges)
    assert mesh.n_dofs == 3 * mesh.n_nodes
    assert 0.75 < mesh.hex_volume_fraction() < 0.85
    # mid-edge nodes sit at edge midpoints
    e = mesh.element_edges
    assert np.allclose(mesh.nodes[n_corner:], 0.5 * (mesh.nodes[e[:, 0]] + mesh.nodes[e[:, 1]]))


def test_hex_tet_interface_ties():
    m = make_rectangular_finger_mesh(GEOM, 10, 2, 2)
    # one hanging node per interface quad (ny*nz quads), slaved to the 8 quad nodes
    assert m.n_ties == 2 * 2
    assert m.tie_masters.shape == (m.n_ties, 8)
    assert np.allclose(m.tie_weights.sum(axis=1), 1.0)
    # the slave is the centre of its master quad and is never a surface node
    centre = np.einsum("sk,skj->sj", m.tie_weights, m.nodes[m.tie_masters])
    assert np.allclose(centre, m.nodes[m.tie_slaves])
    assert not np.isin(m.tie_slaves, m.surface_nodes).any()
    assert make_rectangular_finger_mesh(GEOM, 6, 2, 2, tet_fraction=0.0).n_tets == 0
    all_tet = make_rectangular_finger_mesh(GEOM, 6, 2, 2, tet_fraction=1.0)
    assert all_tet.n_hexes == 0 and all_tet.n_ties == 0
    assert is_closed_surface(all_tet.surface_faces)


def test_all_tets_positive_and_volume_matches_box(mesh):
    vols = mesh.volumes()
    assert np.all(vols > 0)
    assert mesh.total_volume() == pytest.approx(GEOM.length * GEOM.width * GEOM.height, rel=1e-12)


def test_surface_is_closed_box_shell(mesh):
    faces = mesh.surface_faces
    assert is_closed_surface(faces)
    assert euler_characteristic(faces) == 2
    # total boundary area equals the analytic box surface area
    L, W, H = GEOM.length, GEOM.width, GEOM.height
    assert triangle_areas(mesh.nodes, faces).sum() == pytest.approx(2 * (L * W + L * H + W * H), rel=1e-12)


def test_surface_normals_point_outward(mesh):
    n = triangle_normals(mesh.nodes, mesh.surface_faces)
    centroid = mesh.nodes[mesh.surface_faces].mean(axis=1) - mesh.nodes.mean(axis=0)
    assert np.all(np.einsum("ij,ij->i", n, centroid) > 0)
    # each normal is axis aligned for a box
    assert np.allclose(np.abs(n).max(axis=1), 1.0)


def test_vertex_normals_on_top_surface(mesh):
    vn = vertex_normals(mesh.nodes, mesh.surface_faces)
    top_interior = np.nonzero(
        (np.abs(mesh.nodes[:, 2] - GEOM.height) < 1e-12)
        & (mesh.nodes[:, 0] > 1e-9) & (mesh.nodes[:, 0] < GEOM.length - 1e-9)
        & (mesh.nodes[:, 1] > 1e-9) & (mesh.nodes[:, 1] < GEOM.width - 1e-9)
    )[0]
    assert len(top_interior) > 0
    assert np.allclose(vn[top_interior], [0.0, 0.0, 1.0])


def test_fix_tet_orientation_flips_negative_elements(mesh):
    corners = mesh.tets[:, :4]
    assert np.all(tet_signed_volumes(mesh.nodes, corners) > 0)
    flipped = corners[:, [0, 1, 3, 2]]
    assert np.all(tet_signed_volumes(mesh.nodes, flipped) < 0)
    fixed = fix_tet_orientation(mesh.nodes, flipped)
    assert np.all(tet_signed_volumes(mesh.nodes, fixed) > 0)


def test_surface_extraction_of_all_tet_mesh_is_outward():
    m = make_rectangular_finger_mesh(GEOM, 6, 2, 2, tet_fraction=1.0)
    flipped = m.tets[:, [0, 1, 3, 2]]  # negatively oriented corner tets
    faces = extract_surface_triangles(m.nodes, flipped)
    n = triangle_normals(m.nodes, faces)
    centroid = m.nodes[faces].mean(axis=1) - m.nodes.mean(axis=0)
    assert len(faces) == len(m.surface_tris)  # corner triangulation = parent tri6 faces
    assert np.all(np.einsum("ij,ij->i", n, centroid) > 0)


def test_unique_edges_counts(mesh):
    # Corner edges of all elements = element_edges; boundary corner edges are a subset.
    te = np.unique(np.concatenate([unique_edges(mesh.tets), unique_edges(mesh.hexes)]), axis=0)
    assert np.array_equal(te, mesh.element_edges)
    se = mesh.surface_edges
    assert np.all(se[:, 0] < se[:, 1])
    assert len(se) < len(te)
    assert set(map(tuple, se)) <= set(map(tuple, te))
    assert unique_edges(mesh.surface_faces).shape[1] == 2  # display triangles


def test_root_fixed_nodes(mesh):
    fixed = root_fixed_nodes(mesh)
    ny, nz = 2, 2
    assert len(fixed) == (ny + 1) * (nz + 1) + ny * (nz + 1) + nz * (ny + 1)  # corners + mid-edge nodes
    assert np.allclose(mesh.nodes[fixed, 0], 0.0)


def test_nearest_node_restricted_to_candidates(mesh):
    target = GEOM.point_from_relative([0.9, 0.5, 1.0])
    idx = nearest_node(mesh.nodes, target, candidates=mesh.surface_nodes)
    assert idx in mesh.surface_nodes
    assert np.allclose(mesh.nodes[idx], target, atol=GEOM.length / 10 / 2 + 1e-12)


def test_invalid_resolution_raises():
    with pytest.raises(ValueError):
        make_rectangular_finger_mesh(GEOM, 0, 1, 1)


@pytest.mark.parametrize(
    "name,vf",
    [
        ("box", box_mesh([0.1, 0.2, 0.3], [1.0, 2.0, 3.0])),
        ("sphere", sphere_mesh([0, 0, 0], 0.5, 8, 12)),
        ("cylinder", cylinder_mesh([0, 0, 0], [1, 1, 1], 0.2, 12)),
        ("cone", cone_mesh([0, 0, 0], [0, 1, 0], 0.3, 10)),
    ],
)
def test_primitives_closed_and_outward(name, vf):
    v, f = vf
    assert is_closed_surface(f), name
    assert euler_characteristic(f) == 2, name
    n = triangle_normals(v, f)
    c = v[f].mean(axis=1) - v.mean(axis=0)
    assert np.all(np.einsum("ij,ij->i", n, c) > 0), name


def test_sphere_radius_and_box_extent():
    v, _ = sphere_mesh([1, 2, 3], 0.25)
    assert np.allclose(np.linalg.norm(v - [1, 2, 3], axis=1), 0.25)
    v, _ = box_mesh([0, 0, 0], [2, 4, 6])
    assert np.allclose(v.max(axis=0) - v.min(axis=0), [2, 4, 6])


def test_arrow_geometry_length_and_direction():
    origin = np.array([0.0, 0.0, 0.0])
    d = np.array([0.0, 0.0, -1.0])
    v, f = arrow_mesh(origin, d, length=0.05, shaft_radius=0.001, head_radius=0.002, head_length=0.01)
    proj = v @ d
    assert proj.min() == pytest.approx(0.0, abs=1e-12)   # tail at origin
    assert proj.max() == pytest.approx(0.05, abs=1e-12)  # tip at length
    lateral = np.linalg.norm(v - np.outer(proj, d), axis=1)
    assert lateral.max() == pytest.approx(0.002, abs=1e-12)
    soup = to_triangle_soup(v, f)
    assert soup.shape == (3 * len(f), 3)
