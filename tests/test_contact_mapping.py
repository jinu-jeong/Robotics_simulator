"""Tests for the point-contact load operator B(c)."""

import numpy as np
import pytest

from src.contact.contact_mapping import PointContactMapping
from src.geometry.finger import FingerGeometry, make_rectangular_finger_mesh

GEOM = FingerGeometry(0.1, 0.02, 0.01)


@pytest.fixture(scope="module")
def mapping():
    return PointContactMapping(make_rectangular_finger_mesh(GEOM, 10, 2, 2))


def test_locate_projects_onto_top_surface(mapping):
    c = mapping.locate([0.043, 0.007, 0.05])  # above the top face
    assert np.allclose(c.position, [0.043, 0.007, GEOM.height])
    assert np.allclose(c.normal, [0, 0, 1])
    assert c.distance == pytest.approx(0.04)
    assert np.all(c.barycentric >= -1e-12) and c.barycentric.sum() == pytest.approx(1.0)
    # barycentric interpolation reproduces the point
    verts = mapping.mesh.nodes[c.node_indices]
    assert np.allclose(c.barycentric @ verts, c.position)


def test_locate_at_node_gives_single_weight(mapping):
    node = mapping.mesh.surface_nodes[7]
    c = mapping.locate(mapping.mesh.nodes[node])
    assert c.distance == pytest.approx(0.0, abs=1e-15)
    assert c.nearest_node == node
    assert np.isclose(c.barycentric.max(), 1.0)


def test_operator_unit_resultant_and_sparsity(mapping):
    c = mapping.locate([0.061, 0.0123, 0.02])
    d = np.array([0.0, 0.0, -1.0])
    B = mapping.operator(c, d)
    assert B.shape == (mapping.mesh.n_dofs, 1)
    f = B.toarray().reshape(-1, 3)
    assert np.allclose(f.sum(axis=0), d)
    assert np.count_nonzero(np.linalg.norm(f, axis=1)) <= 3
    assert set(np.nonzero(np.linalg.norm(f, axis=1))[0]).issubset(set(c.node_indices.tolist()))


def test_normal_load_vector_points_into_body(mapping):
    f, c = mapping.normal_load_vector([0.09, 0.01, 0.01], 2.5)
    assert np.allclose(f.reshape(-1, 3).sum(axis=0), [0, 0, -2.5])
    f2, _ = mapping.load_vector([0.09, 0.01, 0.01], [0, 0, -2.5])
    assert np.allclose(f, f2)
    f0, _ = mapping.load_vector([0.09, 0.01, 0.01], [0, 0, 0])
    assert np.allclose(f0, 0.0)


def test_side_face_normal(mapping):
    c = mapping.locate([0.05, -0.01, 0.005])  # left of the y = 0 face
    assert np.allclose(c.normal, [0, -1, 0])
    assert np.allclose(c.position, [0.05, 0.0, 0.005])
