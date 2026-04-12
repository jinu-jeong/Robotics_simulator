"""Tests for Rigid Body Dynamics solver (Featherstone algorithms)."""

import math

import numpy as np
import pytest

from robosim.math.spatial import SpatialInertia
from robosim.math.transforms import Transform
from robosim.model.joint import Joint, JointType
from robosim.model.link import Link
from robosim.model.robot import Robot
from robosim.physics.rbd.algorithms import aba, rnea, crba, gravity_torques
from robosim.physics.rbd.solver import RBDSolver


def make_single_pendulum(length: float = 1.0, mass: float = 1.0) -> Robot:
    """Create a simple pendulum: fixed base + single revolute joint + point mass."""
    base = Link(
        name="base",
        inertial=SpatialInertia(mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3))),
    )
    # Pendulum bob: point mass at the end of the link
    # Inertia of a thin rod about one end: I = (1/3) * m * L^2
    rod_inertia = (1.0 / 3.0) * mass * length**2
    bob = Link(
        name="bob",
        inertial=SpatialInertia(
            mass=mass,
            com=np.array([0.0, 0.0, -length / 2]),  # CoM at midpoint
            inertia=np.diag([rod_inertia, rod_inertia, 0.001]),
        ),
    )

    joint = Joint(
        name="pivot",
        joint_type=JointType.REVOLUTE,
        parent_link="base",
        child_link="bob",
        axis=np.array([0.0, 1.0, 0.0]),  # rotate about Y
        origin=Transform.identity(),
    )

    robot = Robot(name="pendulum", links=[base, bob], joints=[joint])
    robot.gravity = np.array([0.0, 0.0, -9.81])
    robot.build()
    return robot


def make_double_pendulum(
    l1: float = 0.5, l2: float = 0.5, m1: float = 1.0, m2: float = 1.0
) -> Robot:
    """Create a double pendulum."""
    base = Link(
        name="base",
        inertial=SpatialInertia(mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3))),
    )
    I1 = (1.0 / 3.0) * m1 * l1**2
    link1 = Link(
        name="link1",
        inertial=SpatialInertia(
            mass=m1,
            com=np.array([0.0, 0.0, -l1 / 2]),
            inertia=np.diag([I1, I1, 0.001]),
        ),
    )
    I2 = (1.0 / 3.0) * m2 * l2**2
    link2 = Link(
        name="link2",
        inertial=SpatialInertia(
            mass=m2,
            com=np.array([0.0, 0.0, -l2 / 2]),
            inertia=np.diag([I2, I2, 0.001]),
        ),
    )

    j1 = Joint(
        name="j1",
        joint_type=JointType.REVOLUTE,
        parent_link="base",
        child_link="link1",
        axis=np.array([0.0, 1.0, 0.0]),
        origin=Transform.identity(),
    )
    j2 = Joint(
        name="j2",
        joint_type=JointType.REVOLUTE,
        parent_link="link1",
        child_link="link2",
        axis=np.array([0.0, 1.0, 0.0]),
        origin=Transform.from_translation(np.array([0.0, 0.0, -l1])),
    )

    robot = Robot(name="double_pendulum", links=[base, link1, link2], joints=[j1, j2])
    robot.gravity = np.array([0.0, 0.0, -9.81])
    robot.build()
    return robot


class TestRNEA:
    def test_gravity_torque_hanging(self):
        """Pendulum hanging straight down (q=0) should have zero gravity torque."""
        robot = make_single_pendulum()
        # At q=0, link hangs along -Z (aligned with gravity) -> zero torque
        tau_g = gravity_torques(robot, np.array([0.0]))
        # Small due to numerical precision, but should be near zero
        assert abs(tau_g[0]) < 1e-10

    def test_gravity_torque_horizontal(self):
        """Pendulum at 90 degrees should have max gravity torque."""
        robot = make_single_pendulum(length=1.0, mass=1.0)
        # q = pi/2 means the pendulum is horizontal
        tau_g = gravity_torques(robot, np.array([math.pi / 2]))
        # Expected: m * g * L/2 (torque about pivot = weight * distance to CoM)
        expected = 1.0 * 9.81 * 0.5
        assert abs(abs(tau_g[0]) - expected) < 0.01

    def test_rnea_inverse_dynamics_consistency(self):
        """RNEA with zero velocity and acceleration should give gravity torques."""
        robot = make_single_pendulum()
        q = np.array([0.5])
        tau = rnea(robot, q, np.zeros(1), np.zeros(1))
        tau_g = gravity_torques(robot, q)
        np.testing.assert_array_almost_equal(tau, tau_g)


class TestABA:
    def test_free_fall_acceleration(self):
        """With zero torque, a horizontal pendulum should accelerate under gravity."""
        robot = make_single_pendulum(length=1.0, mass=1.0)
        q = np.array([math.pi / 2])  # horizontal
        qd = np.array([0.0])
        tau = np.array([0.0])

        qdd = aba(robot, q, qd, tau)
        # Should accelerate (negative angular acceleration, falling)
        assert abs(qdd[0]) > 1.0  # should be significantly nonzero

    def test_gravity_compensation(self):
        """ABA with gravity-compensating torque should give zero acceleration."""
        robot = make_single_pendulum()
        q = np.array([math.pi / 4])
        qd = np.array([0.0])
        tau_g = gravity_torques(robot, q)

        qdd = aba(robot, q, qd, tau_g)
        np.testing.assert_array_almost_equal(qdd, [0.0], decimal=8)

    def test_rnea_aba_consistency(self):
        """RNEA(q, qd, qdd) should give the torques that ABA inverts."""
        robot = make_double_pendulum()
        q = np.array([0.3, -0.5])
        qd = np.array([1.0, -0.5])
        qdd = np.array([2.0, -1.0])

        tau = rnea(robot, q, qd, qdd)
        qdd_recovered = aba(robot, q, qd, tau)
        np.testing.assert_array_almost_equal(qdd_recovered, qdd, decimal=6)


class TestCRBA:
    def test_mass_matrix_symmetric(self):
        """Mass matrix should be symmetric."""
        robot = make_double_pendulum()
        q = np.array([0.3, -0.5])
        M = crba(robot, q)
        np.testing.assert_array_almost_equal(M, M.T, decimal=10)

    def test_mass_matrix_positive_definite(self):
        """Mass matrix should be positive definite."""
        robot = make_double_pendulum()
        q = np.array([0.3, -0.5])
        M = crba(robot, q)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0)

    def test_mass_matrix_rnea_consistency(self):
        """M @ qdd should match RNEA torques (without gravity/Coriolis)."""
        robot = make_double_pendulum()
        robot.gravity = np.array([0.0, 0.0, 0.0])  # disable gravity
        q = np.array([0.0, 0.0])
        qd = np.array([0.0, 0.0])
        qdd = np.array([1.0, 0.5])

        M = crba(robot, q)
        tau_M = M @ qdd
        tau_rnea = rnea(robot, q, qd, qdd, gravity=np.zeros(3))
        np.testing.assert_array_almost_equal(tau_M, tau_rnea, decimal=6)


class TestRBDSolver:
    def test_energy_conservation_pendulum(self):
        """Energy should be approximately conserved for undamped pendulum."""
        robot = make_single_pendulum(length=1.0, mass=1.0)
        robot.q = np.array([math.pi / 3])  # start at 60 degrees
        robot.qd = np.array([0.0])

        solver = RBDSolver(robot=robot)
        solver.initialize(dt=0.0005)

        E0 = solver.total_energy()
        for _ in range(2000):  # 1 second of simulation
            solver.step()

        E_final = solver.total_energy()
        # Symplectic Euler conserves energy well; allow ~1% drift
        relative_error = abs(E_final - E0) / abs(E0)
        assert relative_error < 0.02, f"Energy drift: {relative_error*100:.2f}%"

    def test_pendulum_oscillation(self):
        """A pendulum released from rest should oscillate."""
        robot = make_single_pendulum()
        robot.q = np.array([math.pi / 6])
        robot.qd = np.array([0.0])

        solver = RBDSolver(robot=robot)
        solver.initialize(dt=0.001)

        # Track max displacement
        q_values = []
        for _ in range(2000):
            solver.step()
            q_values.append(solver.get_positions()[0])

        # Should cross zero (oscillate)
        signs = np.sign(q_values)
        sign_changes = np.sum(np.diff(signs) != 0)
        assert sign_changes >= 2, "Pendulum should oscillate through zero"

    def test_double_pendulum_no_explosion(self):
        """Double pendulum should not explode (energy bounded)."""
        robot = make_double_pendulum()
        robot.q = np.array([math.pi / 4, math.pi / 3])
        robot.qd = np.array([0.0, 0.0])

        solver = RBDSolver(robot=robot)
        solver.initialize(dt=0.001)

        E0 = solver.total_energy()
        for _ in range(5000):
            solver.step()

        E_final = solver.total_energy()
        # Energy should not grow unbounded
        assert abs(E_final) < abs(E0) * 5, "Energy should not explode"
