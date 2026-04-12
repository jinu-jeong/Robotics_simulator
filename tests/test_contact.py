"""Tests for contact detection and response."""

import numpy as np
import pytest

from robosim.math.transforms import Transform
from robosim.model.geometry import Geometry
from robosim.physics.contact.sdf import (
    ContactPoint,
    sphere_ground,
    box_ground,
    cylinder_ground,
    sphere_sphere,
    points_ground,
)
from robosim.physics.contact.detection import AABB, compute_aabb, ContactDetector, GroundPlane
from robosim.physics.contact.response import ContactParams, compute_contact_force


class TestSDF:
    def test_sphere_above_ground(self):
        """Sphere above ground: no contact."""
        cp = sphere_ground(center=np.array([0, 0, 1.0]), radius=0.5)
        assert cp is None

    def test_sphere_touching_ground(self):
        """Sphere exactly at ground: penetration = 0, no contact."""
        cp = sphere_ground(center=np.array([0, 0, 0.5]), radius=0.5)
        assert cp is None  # penetration = 0 is not penetrating

    def test_sphere_penetrating_ground(self):
        """Sphere below ground: correct penetration and normal."""
        cp = sphere_ground(center=np.array([0, 0, 0.3]), radius=0.5)
        assert cp is not None
        assert abs(cp.penetration - 0.2) < 1e-10
        np.testing.assert_array_almost_equal(cp.normal, [0, 0, 1])

    def test_box_flat_on_ground(self):
        """Axis-aligned box partially below ground."""
        # Box centered at z=0.4, half-extent z=0.5 → bottom at z=-0.1
        R = np.eye(3)
        contacts = box_ground(
            center=np.array([0, 0, 0.4]),
            rotation=R,
            half_extents=np.array([0.5, 0.5, 0.5]),
        )
        # 4 bottom vertices are below ground
        assert len(contacts) == 4
        for cp in contacts:
            assert cp.penetration > 0
            np.testing.assert_array_almost_equal(cp.normal, [0, 0, 1])

    def test_box_above_ground(self):
        """Box fully above ground: no contacts."""
        contacts = box_ground(
            center=np.array([0, 0, 2.0]),
            rotation=np.eye(3),
            half_extents=np.array([0.5, 0.5, 0.5]),
        )
        assert len(contacts) == 0

    def test_sphere_sphere_contact(self):
        """Two overlapping spheres."""
        cp = sphere_sphere(
            np.array([0, 0, 0]), 1.0,
            np.array([1.5, 0, 0]), 1.0,
        )
        assert cp is not None
        assert abs(cp.penetration - 0.5) < 1e-10
        np.testing.assert_array_almost_equal(cp.normal, [-1, 0, 0])

    def test_sphere_sphere_no_contact(self):
        """Two far-apart spheres."""
        cp = sphere_sphere(
            np.array([0, 0, 0]), 0.5,
            np.array([3, 0, 0]), 0.5,
        )
        assert cp is None

    def test_points_ground(self):
        """Multiple points vs ground plane."""
        pts = np.array([
            [0, 0, 0.5],   # above
            [0, 0, -0.1],  # below
            [0, 0, -0.3],  # below
            [0, 0, 0.0],   # exactly on plane
        ])
        contacts = points_ground(pts, ground_height=0.0)
        assert len(contacts) == 2  # only the two below
        indices = [c[0] for c in contacts]
        assert 1 in indices
        assert 2 in indices


class TestAABB:
    def test_aabb_sphere(self):
        aabb = compute_aabb(Geometry.sphere(1.0), Transform.identity())
        np.testing.assert_array_almost_equal(aabb.min_pt, [-1, -1, -1])
        np.testing.assert_array_almost_equal(aabb.max_pt, [1, 1, 1])

    def test_aabb_box_identity(self):
        aabb = compute_aabb(Geometry.box(2, 2, 2), Transform.identity())
        np.testing.assert_array_almost_equal(aabb.min_pt, [-1, -1, -1])
        np.testing.assert_array_almost_equal(aabb.max_pt, [1, 1, 1])

    def test_aabb_overlap(self):
        a = AABB(np.array([0, 0, 0]), np.array([2, 2, 2]))
        b = AABB(np.array([1, 1, 1]), np.array([3, 3, 3]))
        assert a.overlaps(b)

    def test_aabb_no_overlap(self):
        a = AABB(np.array([0, 0, 0]), np.array([1, 1, 1]))
        b = AABB(np.array([2, 2, 2]), np.array([3, 3, 3]))
        assert not a.overlaps(b)


class TestContactResponse:
    def test_penalty_force_magnitude(self):
        """Penetrating sphere: force ≈ k * d."""
        cp = ContactPoint(
            point_a=np.array([0, 0, -0.1]),
            point_b=np.array([0, 0, 0.0]),
            normal=np.array([0, 0, 1.0]),
            penetration=0.1,
        )
        params = ContactParams(stiffness=1e4, damping=0, friction_mu=0,
                               max_penetration=1.0)  # no cap for this test
        cf = compute_contact_force(cp, np.zeros(3), np.zeros(3), params)
        assert cf is not None
        assert abs(cf.normal_force - 1000.0) < 1e-6  # k * d = 1e4 * 0.1
        np.testing.assert_array_almost_equal(cf.force, [0, 0, 1000.0])

    def test_no_force_when_separated(self):
        """No penetration → no force."""
        cp = ContactPoint(
            point_a=np.array([0, 0, 0.5]),
            point_b=np.array([0, 0, 0.0]),
            normal=np.array([0, 0, 1.0]),
            penetration=0.0,
        )
        cf = compute_contact_force(cp, np.zeros(3), np.zeros(3), ContactParams())
        assert cf is None

    def test_damping_reduces_bounce(self):
        """Damping adds force when approaching, reduces when separating."""
        cp = ContactPoint(
            point_a=np.array([0, 0, -0.1]),
            point_b=np.array([0, 0, 0]),
            normal=np.array([0, 0, 1.0]),
            penetration=0.1,
        )
        params = ContactParams(stiffness=1e4, damping=100, friction_mu=0)

        # No velocity: baseline
        cf_static = compute_contact_force(cp, np.zeros(3), np.zeros(3), params)

        # Approaching: v_n < 0 → damping adds force
        v_approach = np.array([0, 0, -1.0])
        cf_approach = compute_contact_force(cp, v_approach, np.zeros(3), params)

        # Separating slowly: v_n > 0 → damping reduces force
        v_separate = np.array([0, 0, 0.5])
        cf_separate = compute_contact_force(cp, v_separate, np.zeros(3), params)

        assert cf_static is not None
        assert cf_approach is not None
        assert cf_separate is not None
        assert cf_approach.normal_force > cf_static.normal_force
        assert cf_separate.normal_force < cf_static.normal_force

    def test_friction_opposes_sliding(self):
        """Friction force opposes tangential velocity."""
        cp = ContactPoint(
            point_a=np.array([0, 0, -0.1]),
            point_b=np.array([0, 0, 0]),
            normal=np.array([0, 0, 1.0]),
            penetration=0.1,
        )
        params = ContactParams(stiffness=1e4, damping=0, friction_mu=0.5)
        v_slide = np.array([1.0, 0, 0])  # sliding in +X
        cf = compute_contact_force(cp, v_slide, np.zeros(3), params)
        assert cf is not None
        # Friction should push in -X
        assert cf.force[0] < 0

    def test_friction_magnitude_bounded(self):
        """Friction force ≤ mu * normal_force."""
        cp = ContactPoint(
            point_a=np.array([0, 0, -0.1]),
            point_b=np.array([0, 0, 0]),
            normal=np.array([0, 0, 1.0]),
            penetration=0.1,
        )
        params = ContactParams(stiffness=1e4, damping=0, friction_mu=0.5)
        v_slide = np.array([10.0, 0, 0])
        cf = compute_contact_force(cp, v_slide, np.zeros(3), params)
        assert cf is not None
        f_tangential = np.linalg.norm(cf.force[:2])
        assert f_tangential <= params.friction_mu * cf.normal_force + 1e-6


class TestContactDetector:
    def test_sphere_ground_detection(self):
        """ContactDetector finds sphere-ground contact."""
        det = ContactDetector(ground=GroundPlane())
        det.add_body(
            Geometry.sphere(0.5),
            Transform.from_translation(np.array([0, 0, 0.3])),
        )
        contacts = det.detect_all()
        assert len(contacts) == 1
        cp, bid_a, bid_b = contacts[0]
        assert bid_b == -1  # ground
        assert cp.penetration > 0

    def test_sphere_above_ground_no_contact(self):
        det = ContactDetector(ground=GroundPlane())
        det.add_body(
            Geometry.sphere(0.5),
            Transform.from_translation(np.array([0, 0, 2.0])),
        )
        contacts = det.detect_all()
        assert len(contacts) == 0

    def test_two_spheres_collision(self):
        det = ContactDetector()
        det.add_body(Geometry.sphere(1.0), Transform.from_translation(np.array([0, 0, 0])))
        det.add_body(Geometry.sphere(1.0), Transform.from_translation(np.array([1.5, 0, 0])))
        contacts = det.detect_all()
        assert len(contacts) == 1
