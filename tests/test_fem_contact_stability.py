"""Test FEM-RBD contact stability with high stiffness materials.

Verifies that the rate-limited contact correction prevents energy explosion
when a stiff FEM body touches an RBD primitive.
"""

import numpy as np
import pytest

from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.solver import DeformableBody, FEMSolver
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.contact.solver import ContactSolver
from robosim.physics.contact.detection import GroundPlane
from robosim.physics.contact.response import ContactParams
from robosim.physics.contact.fem_contact import (
    resolve_rbd_fem_contacts,
    RBDFEMContactPoint,
)
from robosim.model.factory import create_free_box
from robosim.physics.rbd.solver import RBDSolver


def _make_small_fem(center, size=0.05, E=1e6):
    """Create a small FEM deformable body.

    Returns (fem_solver, body).
    """
    mesh = TetMesh.create_box(
        origin=np.array(center, dtype=np.float64),
        size=np.array([size, size, size]),
        divisions=(2, 2, 2),
    )
    body = DeformableBody(
        name="test_fem",
        mesh=mesh,
        material=CorotationalElastic(young=E, poisson=0.3),
        density=1000.0,
    )
    solver = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]))
    return solver, body


def _make_box_robot(center, size=0.1):
    """Create a free box robot at the given position."""
    robot = create_free_box(
        name="box_robot",
        size=(size, size, size),
        mass=1.0,
        position=np.array(center),
    )
    solver = RBDSolver(robot=robot)
    return robot, solver


class TestFEMContactStability:
    """Verify rate-limited FEM-RBD contact doesn't explode."""

    def test_high_stiffness_no_explosion(self):
        """FEM body near RBD box with E=1e6 should stay calm."""
        box_center = [0.0, 0.0, 0.0]
        fem_center = [0.0, 0.0, 0.05]

        robot, rbd_solver = _make_box_robot(box_center, size=0.2)
        fem_solver, body = _make_small_fem(fem_center, size=0.05, E=1e6)

        dt = 0.001
        rbd_solver.initialize(dt=dt)
        fem_solver.initialize(dt=dt)

        contact = ContactSolver(
            ground=GroundPlane(height=-1.0),
            params=ContactParams(stiffness=5e4, damping=500, friction_mu=0.5),
        )
        contact.register_rbd(rbd_solver)
        contact.register_fem(fem_solver)

        max_speed = 0.0
        for step in range(100):
            fem_solver.step()
            contact.resolve_rbd_fem_all(
                rbd_solver, fem_solver,
                restitution=0.1, friction_mu=0.5,
                dt=dt,
            )
            speed = np.max(np.linalg.norm(body.v, axis=1))
            max_speed = max(max_speed, speed)

        final_speed = np.max(np.linalg.norm(body.v, axis=1))
        assert final_speed < 10.0, (
            f"FEM velocity exploded: final speed = {final_speed:.2f} m/s"
        )
        assert max_speed < 20.0, (
            f"FEM velocity spiked during sim: {max_speed:.2f} m/s"
        )

    def test_very_high_stiffness_no_explosion(self):
        """Even with E=1e8, contact should remain stable."""
        box_center = [0.0, 0.0, 0.0]
        fem_center = [0.0, 0.0, 0.06]

        robot, rbd_solver = _make_box_robot(box_center, size=0.2)
        fem_solver, body = _make_small_fem(fem_center, size=0.05, E=1e8)

        dt = 0.001
        rbd_solver.initialize(dt=dt)
        fem_solver.initialize(dt=dt)

        contact = ContactSolver(
            ground=GroundPlane(height=-1.0),
            params=ContactParams(stiffness=5e4, damping=500, friction_mu=0.5),
        )
        contact.register_rbd(rbd_solver)
        contact.register_fem(fem_solver)

        max_speed = 0.0
        for step in range(50):
            fem_solver.step()
            contact.resolve_rbd_fem_all(
                rbd_solver, fem_solver,
                restitution=0.1, friction_mu=0.5,
                dt=dt,
            )
            speed = np.max(np.linalg.norm(body.v, axis=1))
            max_speed = max(max_speed, speed)

        assert max_speed < 50.0, (
            f"Very high stiffness FEM velocity exploded: {max_speed:.2f} m/s"
        )

    def test_rate_limited_correction_clamps(self):
        """Position correction should be clamped to max_correction_speed * dt."""
        fem_solver, body = _make_small_fem(
            [0.0, 0.0, 0.0], size=0.05, E=1e6
        )
        fem_solver.initialize(dt=0.001)

        robot, _ = _make_box_robot([0.0, 0.0, -0.05], size=0.2)

        # Deep penetration: 5cm
        contacts = [RBDFEMContactPoint(
            fem_node_idx=0,
            link_idx=0,
            normal=np.array([0.0, 0.0, 1.0]),
            penetration=0.05,
            rbd_contact_point=np.array([0.0, 0.0, 0.0]),
            fem_body_idx=0,
        )]

        pos_before = body.x[0].copy()

        n = resolve_rbd_fem_contacts(
            contacts, robot, fem_solver.bodies,
            restitution=0.1, friction_mu=0.5,
            dt=0.001,
            max_correction_speed=2.0,
        )

        pos_after = body.x[0]
        correction_mag = np.linalg.norm(pos_after - pos_before)

        # max_correction = 2.0 * 0.001 = 0.002m
        assert correction_mag <= 0.003, (
            f"Correction not clamped: {correction_mag:.4f}m "
            f"(expected <= 0.002m)"
        )
        assert correction_mag > 0.0, "No correction applied"
        assert n == 1

    def test_deep_penetration_no_bounce(self):
        """Deep penetration should suppress bounce (restitution -> 0)."""
        fem_solver, body = _make_small_fem(
            [0.0, 0.0, 0.0], size=0.05, E=1e6
        )
        fem_solver.initialize(dt=0.001)

        robot, _ = _make_box_robot([0.0, 0.0, -0.05], size=0.2)

        # Node approaching the surface with velocity
        body.v[0] = np.array([0.0, 0.0, -5.0])

        contacts = [RBDFEMContactPoint(
            fem_node_idx=0,
            link_idx=0,
            normal=np.array([0.0, 0.0, 1.0]),
            penetration=0.05,  # deep
            rbd_contact_point=np.array([0.0, 0.0, 0.0]),
            fem_body_idx=0,
        )]

        resolve_rbd_fem_contacts(
            contacts, robot, fem_solver.bodies,
            restitution=0.5,
            friction_mu=0.5,
            dt=0.001,
            max_correction_speed=2.0,
        )

        # Deep penetration -> no bounce, velocity should be ~0 in normal
        v_z = body.v[0, 2]
        assert v_z >= -0.1, (
            f"Deep penetration should not bounce: v_z = {v_z:.3f}"
        )
        assert v_z < 3.0, (
            f"Deep penetration should not launch: v_z = {v_z:.3f}"
        )
