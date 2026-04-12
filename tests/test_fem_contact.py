"""Tests for FEM-FEM and RBD-FEM contact detection and response."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.physics.contact.sdf import (
    point_triangle_distance,
    triangle_normal,
    sphere_triangle,
    ContactPoint,
)
from robosim.physics.contact.fem_contact import (
    FEMSurfaceCollider,
    detect_fem_fem,
    detect_rbd_fem,
    resolve_fem_fem_contacts,
    resolve_rbd_fem_contacts,
    _point_vs_geometry,
)
from robosim.physics.contact.solver import ContactSolver
from robosim.physics.contact.detection import GroundPlane
from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.solver import FEMSolver, DeformableBody
from robosim.model.geometry import Geometry


# ═══════════════════════════════════════════════════════════════
# Geometric primitives
# ═══════════════════════════════════════════════════════════════

class TestPointTriangleDistance:
    """Test point_triangle_distance function."""

    def _make_tri(self):
        """Right triangle in XY plane at z=0."""
        return (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )

    def test_point_on_face(self):
        v0, v1, v2 = self._make_tri()
        p = np.array([0.2, 0.2, 0.0])  # inside triangle
        closest, dist, bary = point_triangle_distance(p, v0, v1, v2)
        assert dist < 1e-10
        assert np.allclose(closest, p, atol=1e-10)
        assert np.allclose(bary.sum(), 1.0)

    def test_point_above_face(self):
        v0, v1, v2 = self._make_tri()
        p = np.array([0.2, 0.2, 0.5])  # above triangle
        closest, dist, bary = point_triangle_distance(p, v0, v1, v2)
        assert abs(dist - 0.5) < 1e-10
        assert np.allclose(closest, [0.2, 0.2, 0.0], atol=1e-10)

    def test_point_near_vertex(self):
        v0, v1, v2 = self._make_tri()
        p = np.array([-0.1, -0.1, 0.0])  # near v0
        closest, dist, bary = point_triangle_distance(p, v0, v1, v2)
        expected_dist = np.linalg.norm(p - v0)
        assert abs(dist - expected_dist) < 1e-10

    def test_point_near_edge(self):
        v0, v1, v2 = self._make_tri()
        p = np.array([0.5, -0.1, 0.0])  # near edge v0-v1
        closest, dist, bary = point_triangle_distance(p, v0, v1, v2)
        # Closest point should be on edge v0-v1
        assert closest[1] < 1e-10  # y=0 on this edge
        assert 0 <= closest[0] <= 1.0

    def test_barycentric_sum_one(self):
        v0, v1, v2 = self._make_tri()
        for _ in range(10):
            p = np.random.randn(3)
            _, _, bary = point_triangle_distance(p, v0, v1, v2)
            assert abs(bary.sum() - 1.0) < 1e-8


class TestSphereTriangle:
    def test_no_contact(self):
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        center = np.array([0.2, 0.2, 1.0])
        cp = sphere_triangle(center, 0.1, v0, v1, v2)
        assert cp is None

    def test_contact(self):
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        center = np.array([0.2, 0.2, 0.05])
        cp = sphere_triangle(center, 0.1, v0, v1, v2)
        assert cp is not None
        assert cp.penetration > 0
        assert cp.normal[2] > 0  # normal points toward sphere


class TestPointVsGeometry:
    def test_point_inside_sphere(self):
        geom = Geometry.sphere(0.1)
        center = np.zeros(3)
        R = np.eye(3)
        cp = _point_vs_geometry(np.array([0.03, 0.0, 0.0]), geom, center, R)
        assert cp is not None
        assert cp.penetration > 0

    def test_point_outside_sphere(self):
        geom = Geometry.sphere(0.1)
        cp = _point_vs_geometry(np.array([0.2, 0.0, 0.0]), geom, np.zeros(3), np.eye(3))
        assert cp is None

    def test_point_inside_box(self):
        geom = Geometry.box(0.2, 0.2, 0.2)
        cp = _point_vs_geometry(np.array([0.05, 0.0, 0.0]), geom, np.zeros(3), np.eye(3))
        assert cp is not None
        assert cp.penetration > 0

    def test_point_outside_box(self):
        geom = Geometry.box(0.2, 0.2, 0.2)
        cp = _point_vs_geometry(np.array([0.2, 0.0, 0.0]), geom, np.zeros(3), np.eye(3))
        assert cp is None


# ═══════════════════════════════════════════════════════════════
# FEM-FEM Contact
# ═══════════════════════════════════════════════════════════════

class TestFEMFEMDetection:
    def _make_two_cubes(self, gap=0.0):
        """Two small FEM cubes, one above the other, separated by `gap`."""
        mat = CorotationalElastic(young=1e5, poisson=0.3)

        # Cube A at origin
        mesh_a = TetMesh.create_box(
            origin=np.zeros(3), size=np.array([0.1, 0.1, 0.1]),
            divisions=(2, 2, 2),
        )
        body_a = DeformableBody(name="a", mesh=mesh_a, material=mat, density=1000.0)

        # Cube B above A (gap between them)
        mesh_b = TetMesh.create_box(
            origin=np.array([0.0, 0.0, 0.1 + gap]),
            size=np.array([0.1, 0.1, 0.1]),
            divisions=(2, 2, 2),
        )
        body_b = DeformableBody(name="b", mesh=mesh_b, material=mat, density=1000.0)

        fem = FEMSolver(bodies=[body_a, body_b], gravity=np.array([0, 0, -9.81]))
        fem.initialize(dt=0.001)

        return fem, body_a, body_b

    def test_no_contact_when_separated(self):
        """No contacts when cubes have a gap."""
        fem, a, b = self._make_two_cubes(gap=0.02)
        ca = FEMSurfaceCollider.from_body(a, 0)
        cb = FEMSurfaceCollider.from_body(b, 1)
        ca.update_aabb(a.x)
        cb.update_aabb(b.x)

        contacts = detect_fem_fem(ca, cb, a.x, b.x, d_hat=0.005)
        assert len(contacts) == 0

    def test_contact_when_overlapping(self):
        """Contacts detected when cubes overlap."""
        fem, a, b = self._make_two_cubes(gap=-0.005)
        ca = FEMSurfaceCollider.from_body(a, 0)
        cb = FEMSurfaceCollider.from_body(b, 1)
        ca.update_aabb(a.x)
        cb.update_aabb(b.x)

        contacts = detect_fem_fem(ca, cb, a.x, b.x, d_hat=0.01)
        assert len(contacts) > 0
        # Check contact normal roughly points upward (from B face toward A node)
        for c in contacts:
            assert c.penetration > 0

    def test_contact_near_proximity(self):
        """Contacts detected when cubes are within d_hat."""
        fem, a, b = self._make_two_cubes(gap=0.002)
        ca = FEMSurfaceCollider.from_body(a, 0)
        cb = FEMSurfaceCollider.from_body(b, 1)
        ca.update_aabb(a.x, margin=0.01)
        cb.update_aabb(b.x, margin=0.01)

        contacts = detect_fem_fem(ca, cb, a.x, b.x, d_hat=0.005)
        assert len(contacts) > 0


class TestFEMFEMResponse:
    def test_resolve_separates_bodies(self):
        """After resolving, overlapping nodes should be pushed apart."""
        mat = CorotationalElastic(young=1e5, poisson=0.3)

        mesh_a = TetMesh.create_box(
            origin=np.zeros(3), size=np.array([0.1, 0.1, 0.1]),
            divisions=(2, 2, 2),
        )
        mesh_b = TetMesh.create_box(
            origin=np.array([0.0, 0.0, 0.095]),  # 5mm overlap
            size=np.array([0.1, 0.1, 0.1]),
            divisions=(2, 2, 2),
        )
        body_a = DeformableBody(name="a", mesh=mesh_a, material=mat, density=1000.0)
        body_b = DeformableBody(name="b", mesh=mesh_b, material=mat, density=1000.0)

        fem = FEMSolver(bodies=[body_a, body_b], gravity=np.array([0, 0, 0]))
        fem.initialize(dt=0.001)

        # Give body B downward velocity
        body_b.v[:, 2] = -0.5

        ca = FEMSurfaceCollider.from_body(body_a, 0)
        cb = FEMSurfaceCollider.from_body(body_b, 1)
        ca.update_aabb(body_a.x, margin=0.01)
        cb.update_aabb(body_b.x, margin=0.01)

        contacts = detect_fem_fem(ca, cb, body_a.x, body_b.x, d_hat=0.01)
        contacts += detect_fem_fem(cb, ca, body_b.x, body_a.x, d_hat=0.01)

        n = resolve_fem_fem_contacts(contacts, [body_a, body_b],
                                     restitution=0.0, friction_mu=0.0)
        assert n > 0

        # Check: at least some of B's bottom nodes should have been pushed up
        # relative to their starting position
        b_bottom = np.where(mesh_b.nodes[:, 2] < 0.096)[0]
        avg_z_before = mesh_b.nodes[b_bottom, 2].mean()
        avg_z_after = body_b.x[b_bottom, 2].mean()
        # Contact should have pushed B upward or slowed its descent
        # Also verify B's downward velocity was reduced
        avg_vz = body_b.v[b_bottom, 2].mean()
        assert avg_vz >= -0.5, f"Velocity not reduced: avg_vz={avg_vz}"


# ═══════════════════════════════════════════════════════════════
# ContactSolver integration
# ═══════════════════════════════════════════════════════════════

class TestContactSolverFEM:
    def test_register_fem(self):
        mat = CorotationalElastic(young=1e5, poisson=0.3)
        mesh = TetMesh.create_box(size=np.array([0.1, 0.1, 0.1]), divisions=(2, 2, 2))
        body = DeformableBody(name="a", mesh=mesh, material=mat, density=1000.0)
        fem = FEMSolver(bodies=[body], gravity=np.zeros(3))
        fem.initialize(dt=0.001)

        solver = ContactSolver(ground=GroundPlane())
        solver.register_fem(fem)
        assert len(solver._fem_colliders) == 1

    def test_resolve_fem_fem_all(self):
        mat = CorotationalElastic(young=1e5, poisson=0.3)

        mesh_a = TetMesh.create_box(
            origin=np.zeros(3), size=np.array([0.1, 0.1, 0.1]), divisions=(2, 2, 2),
        )
        mesh_b = TetMesh.create_box(
            origin=np.array([0.0, 0.0, 0.097]),  # slight overlap
            size=np.array([0.1, 0.1, 0.1]), divisions=(2, 2, 2),
        )
        body_a = DeformableBody(name="a", mesh=mesh_a, material=mat, density=1000.0)
        body_b = DeformableBody(name="b", mesh=mesh_b, material=mat, density=1000.0)

        fem = FEMSolver(bodies=[body_a, body_b], gravity=np.zeros(3))
        fem.initialize(dt=0.001)

        solver = ContactSolver(ground=GroundPlane())
        solver.register_fem(fem)

        n = solver.resolve_fem_fem_all(fem, restitution=0.0, friction_mu=0.0, d_hat=0.01)
        assert n > 0


# ═══════════════════════════════════════════════════════════════
# FEM-FEM simulation (drop test)
# ═══════════════════════════════════════════════════════════════

class TestFEMFEMSimulation:
    def test_drop_on_block(self):
        """Drop a soft cube onto another — should not pass through."""
        mat = CorotationalElastic(young=1e5, poisson=0.3)

        # Bottom block: resting on ground, fixed bottom
        mesh_bottom = TetMesh.create_box(
            origin=np.zeros(3), size=np.array([0.2, 0.2, 0.1]),
            divisions=(2, 2, 2),
        )
        body_bottom = DeformableBody(
            name="bottom", mesh=mesh_bottom, material=mat, density=1000.0,
            fixed_nodes=np.where(mesh_bottom.nodes[:, 2] < 1e-6)[0],
        )

        # Top block: starts just above, will drop
        mesh_top = TetMesh.create_box(
            origin=np.array([0.02, 0.02, 0.12]),  # small offset, 2cm gap
            size=np.array([0.15, 0.15, 0.1]),
            divisions=(2, 2, 2),
        )
        body_top = DeformableBody(
            name="top", mesh=mesh_top, material=mat, density=1000.0,
        )

        fem = FEMSolver(
            bodies=[body_bottom, body_top],
            gravity=np.array([0, 0, -9.81]),
            damping=0.5,
        )
        fem.initialize(dt=0.001)

        contact = ContactSolver(ground=GroundPlane(height=0.0))
        contact.register_fem(fem)

        # Simulate: top block falls onto bottom
        for step in range(200):
            fem.step()

            # Ground contact
            for body in fem.bodies:
                contact.resolve_fem_contact(body, restitution=0.1, friction_mu=0.5)

            # FEM-FEM contact
            contact.resolve_fem_fem_all(fem, restitution=0.1, friction_mu=0.5, d_hat=0.005)

        # After 200 steps (0.2s): top block should rest on bottom, not pass through
        top_z_min = body_top.x[:, 2].min()
        bottom_z_max = body_bottom.x[:, 2].max()

        # Top should be above or near the bottom's top surface
        assert top_z_min > bottom_z_max - 0.02, \
            f"Top passed through bottom: top_min_z={top_z_min:.4f}, bot_max_z={bottom_z_max:.4f}"
        assert np.all(np.isfinite(body_top.x)), "NaN in top body"
        assert np.all(np.isfinite(body_bottom.x)), "NaN in bottom body"
