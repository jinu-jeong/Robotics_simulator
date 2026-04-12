"""Tests for box-box and box-sphere collision detection."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robosim.physics.contact.sdf import (
    ContactPoint, box_box, box_sphere, mesh_ground,
)
from robosim.physics.contact.detection import (
    ContactDetector, GroundPlane, compute_aabb,
)
from robosim.model.geometry import Geometry
from robosim.math.transforms import Transform


I3 = np.eye(3)


class TestBoxBox:
    """SAT-based OBB-OBB collision tests."""

    def test_separated_boxes(self):
        """Two axis-aligned boxes far apart — no contact."""
        contacts = box_box(
            np.array([0.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
            np.array([3.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
        )
        assert len(contacts) == 0

    def test_overlapping_boxes(self):
        """Two axis-aligned boxes overlapping — should produce contacts."""
        contacts = box_box(
            np.array([0.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
            np.array([0.8, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
        )
        assert len(contacts) > 0
        for cp in contacts:
            assert cp.penetration > 0
            assert abs(np.linalg.norm(cp.normal) - 1.0) < 1e-6

    def test_face_contact_penetration(self):
        """Two boxes touching on a face — known penetration depth."""
        # Box A: [-0.5, 0.5] on each axis, centered at origin
        # Box B: centered at (0.9, 0, 0), same size
        # Overlap on X axis = 0.5 + 0.5 - 0.9 = 0.1
        contacts = box_box(
            np.array([0.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
            np.array([0.9, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
        )
        assert len(contacts) > 0
        # All contacts should have penetration around 0.1
        for cp in contacts:
            assert 0.0 < cp.penetration <= 0.15

    def test_rotated_box(self):
        """One box rotated 45 degrees — edge contact."""
        R45 = Rotation.from_euler('z', 45, degrees=True).as_matrix()
        contacts = box_box(
            np.array([0.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
            np.array([1.1, 0.0, 0.0]), R45, np.array([0.5, 0.5, 0.5]),
        )
        # Rotated box corner extends further, so should be overlapping
        # sqrt(2)*0.5 ≈ 0.707, center at 1.1, so extent is 1.1 - 0.5 - 0.707 < 0
        # Actually A extends to 0.5 on X, B center at 1.1, B extends 0.707 in X
        # So B extends from 1.1 - 0.707 = 0.393 to 1.1 + 0.707 = 1.807
        # A extends from -0.5 to 0.5
        # Overlap: 0.5 - 0.393 = 0.107
        assert len(contacts) > 0

    def test_just_touching(self):
        """Boxes barely touching — penetration near zero."""
        contacts = box_box(
            np.array([0.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
            np.array([1.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
        )
        # Exactly touching: pen = 0, should return no contacts or very small pen
        # SAT with pen <= 0 returns empty
        assert len(contacts) == 0

    def test_normal_direction(self):
        """Contact normal points from B to A."""
        contacts = box_box(
            np.array([0.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
            np.array([0.8, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
        )
        assert len(contacts) > 0
        # Normal should point in -X direction (from B toward A)
        for cp in contacts:
            assert cp.normal[0] < 0  # pointing from B to A (A is at origin, B at 0.8)

    def test_stacked_vertically(self):
        """Box on top of another box — Z-axis contact."""
        contacts = box_box(
            np.array([0.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
            np.array([0.0, 0.0, 0.9]), I3, np.array([0.5, 0.5, 0.5]),
        )
        assert len(contacts) > 0
        for cp in contacts:
            assert cp.penetration > 0
            # Normal should be roughly Z-axis
            assert abs(cp.normal[2]) > 0.9


class TestBoxSphere:
    """Box vs sphere collision tests."""

    def test_sphere_outside_box(self):
        """Sphere clearly outside box."""
        cp = box_sphere(
            np.array([0.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
            np.array([2.0, 0.0, 0.0]), 0.1,
        )
        assert cp is None

    def test_sphere_touching_face(self):
        """Sphere touching a face of the box."""
        cp = box_sphere(
            np.array([0.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
            np.array([0.7, 0.0, 0.0]), 0.3,
        )
        assert cp is not None
        assert cp.penetration > 0
        # Normal should point roughly in +X (from box face to sphere center)
        assert cp.normal[0] > 0.9

    def test_sphere_at_corner(self):
        """Sphere near a corner of the box."""
        cp = box_sphere(
            np.array([0.0, 0.0, 0.0]), I3, np.array([0.5, 0.5, 0.5]),
            np.array([0.6, 0.6, 0.6]), 0.3,
        )
        assert cp is not None
        assert cp.penetration > 0

    def test_rotated_box_sphere(self):
        """Sphere vs rotated box."""
        R45 = Rotation.from_euler('z', 45, degrees=True).as_matrix()
        cp = box_sphere(
            np.array([0.0, 0.0, 0.0]), R45, np.array([0.5, 0.5, 0.5]),
            np.array([0.8, 0.0, 0.0]), 0.2,
        )
        # After 45-deg rotation, box extends to sqrt(2)*0.5 ≈ 0.707 on X
        # Sphere at 0.8 with r=0.2 → gap = 0.8 - 0.707 - 0.2 < 0 → contact
        assert cp is not None


class TestMeshGround:
    """Mesh vs ground contact test."""

    def test_mesh_above_ground(self):
        verts = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float64)
        contacts = mesh_ground(verts, I3, np.zeros(3), ground_height=0.0)
        assert len(contacts) == 0

    def test_mesh_penetrating_ground(self):
        verts = np.array([[0, 0, -0.1], [1, 0, 0.5], [0, 1, 0.5]], dtype=np.float64)
        contacts = mesh_ground(verts, I3, np.zeros(3), ground_height=0.0)
        assert len(contacts) == 1
        assert contacts[0].penetration == pytest.approx(0.1, abs=1e-6)


class TestDetectorIntegration:
    """Test box collision through the ContactDetector."""

    def test_box_box_detector(self):
        detector = ContactDetector(ground=GroundPlane(height=0.0))
        T_a = Transform.from_translation(np.array([0, 0, 1.0]))
        T_b = Transform.from_translation(np.array([0.8, 0, 1.0]))
        detector.add_body(Geometry.box(1.0, 1.0, 1.0), T_a)
        detector.add_body(Geometry.box(1.0, 1.0, 1.0), T_b)
        results = detector.detect_all()
        # Should have body-body contacts (0, 1) and possibly ground contacts
        body_contacts = [(cp, a, b) for cp, a, b in results if b >= 0]
        assert len(body_contacts) > 0

    def test_box_sphere_detector(self):
        detector = ContactDetector(ground=None)
        T_a = Transform.from_translation(np.array([0, 0, 0]))
        T_b = Transform.from_translation(np.array([0.7, 0, 0]))
        detector.add_body(Geometry.box(1.0, 1.0, 1.0), T_a)
        detector.add_body(Geometry.sphere(0.3), T_b)
        results = detector.detect_all()
        assert len(results) > 0

    def test_existing_sphere_sphere_still_works(self):
        """Regression: existing sphere-sphere detection still works."""
        detector = ContactDetector(ground=None)
        T_a = Transform.from_translation(np.array([0, 0, 0]))
        T_b = Transform.from_translation(np.array([0.3, 0, 0]))
        detector.add_body(Geometry.sphere(0.2), T_a)
        detector.add_body(Geometry.sphere(0.2), T_b)
        results = detector.detect_all()
        assert len(results) > 0
