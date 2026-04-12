"""Tests for free-floating rigid objects via factory functions."""

import numpy as np
import pytest

from robosim.model.factory import create_free_box, create_free_sphere
from robosim.physics.rbd.solver import RBDSolver
from robosim.physics.rbd.algorithms import aba
from robosim.physics.contact.detection import GroundPlane
from robosim.physics.contact.solver import ContactSolver
from robosim.physics.contact.response import ContactParams


class TestFreeBox:
    """Test free-floating box creation and physics."""

    def test_creation(self):
        box = create_free_box("box", size=(0.1, 0.1, 0.1), mass=1.0)
        assert box.n_dof == 6
        assert len(box.links) == 7  # base + 5 virtual + body
        assert len(box.joints) == 6

    def test_initial_position(self):
        pos = np.array([1.0, 2.0, 3.0])
        box = create_free_box(position=pos)
        np.testing.assert_allclose(box.q[:3], pos)

    def test_free_fall_acceleration(self):
        """Free box should accelerate at -g in Z."""
        box = create_free_box(mass=1.0, position=np.array([0, 0, 1.0]))
        qdd = aba(box, box.q, box.qd, np.zeros(6))
        assert abs(qdd[2] - (-9.81)) < 0.01
        # No lateral acceleration
        assert abs(qdd[0]) < 1e-6
        assert abs(qdd[1]) < 1e-6

    def test_drop_and_settle(self):
        """Box drops from height and settles on ground."""
        box = create_free_box(
            size=(0.06, 0.06, 0.06), mass=0.5,
            position=np.array([0.0, 0.0, 0.3]),
        )
        solver = RBDSolver(robot=box)
        solver.initialize(dt=0.001)

        params = ContactParams(stiffness=5e3, damping=100, friction_mu=0.5)
        contact = ContactSolver(ground=GroundPlane(height=0.0), params=params)
        contact.register_rbd(solver)

        for _ in range(2000):
            solver.clear_external_forces()
            wrenches = contact.compute_rbd_contact_forces(solver)
            for link_idx, wrench in wrenches.items():
                solver.set_external_force(link_idx, wrench)
            solver.step()

        z = box.q[2]
        assert 0.02 < z < 0.06, f"Box should rest near ground, got z={z}"
        assert abs(box.qd[2]) < 0.5, "Box should be near rest"

    def test_no_lateral_drift(self):
        """Box dropped straight down should not drift horizontally."""
        box = create_free_box(
            size=(0.06, 0.06, 0.06), mass=0.5,
            position=np.array([0.0, 0.0, 0.2]),
        )
        solver = RBDSolver(robot=box)
        solver.initialize(dt=0.001)

        params = ContactParams(stiffness=5e3, damping=100, friction_mu=0.5)
        contact = ContactSolver(ground=GroundPlane(height=0.0), params=params)
        contact.register_rbd(solver)

        for _ in range(1500):
            solver.clear_external_forces()
            wrenches = contact.compute_rbd_contact_forces(solver)
            for link_idx, wrench in wrenches.items():
                solver.set_external_force(link_idx, wrench)
            solver.step()

        assert abs(box.q[0]) < 0.01, "No X drift"
        assert abs(box.q[1]) < 0.01, "No Y drift"


class TestFreeSphere:
    """Test free-floating sphere creation."""

    def test_creation(self):
        sphere = create_free_sphere("ball", radius=0.05, mass=0.3)
        assert sphere.n_dof == 6
        assert len(sphere.links) == 7

    def test_free_fall(self):
        sphere = create_free_sphere(mass=1.0, position=np.array([0, 0, 1.0]))
        qdd = aba(sphere, sphere.q, sphere.qd, np.zeros(6))
        assert abs(qdd[2] - (-9.81)) < 0.01
