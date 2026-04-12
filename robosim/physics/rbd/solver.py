"""RBD Solver: manages rigid body dynamics simulation state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosim.model.robot import Robot
from robosim.physics.rbd.algorithms import aba, gravity_torques, rnea
from robosim.physics.rbd.integrator import semi_implicit_euler


@dataclass
class RBDSolver:
    """Rigid Body Dynamics solver implementing the PhysicsSolver interface.

    Wraps a Robot model and provides step-by-step simulation with
    Featherstone's ABA and semi-implicit Euler integration.
    """

    robot: Robot
    dt: float = 0.001

    # External torques applied by controllers
    _tau: np.ndarray | None = field(default=None, repr=False)
    # External spatial forces per link {link_idx: (6,) wrench}
    _f_ext: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    _time: float = field(default=0.0, repr=False)

    def initialize(self, dt: float) -> None:
        self.dt = dt
        self._tau = np.zeros(self.robot.n_dof)
        self._f_ext = {}
        self._time = 0.0

    @property
    def tau(self) -> np.ndarray:
        if self._tau is None:
            self._tau = np.zeros(self.robot.n_dof)
        return self._tau

    @tau.setter
    def tau(self, value: np.ndarray):
        self._tau = np.asarray(value, dtype=np.float64)

    @property
    def time(self) -> float:
        return self._time

    def set_external_force(self, link_idx: int, wrench: np.ndarray):
        """Set an external spatial force on a link.

        Parameters
        ----------
        link_idx : index of the link
        wrench : (6,) spatial force [torque(3); force(3)] in link frame
        """
        self._f_ext[link_idx] = np.asarray(wrench, dtype=np.float64)

    def clear_external_forces(self):
        self._f_ext.clear()

    def step(self, dt: float | None = None, external_forces: np.ndarray | None = None) -> None:
        """Advance the simulation by one time step.

        Parameters
        ----------
        dt : time step (uses self.dt if None)
        external_forces : (n_dof,) additional torques (added to self.tau)
        """
        if dt is None:
            dt = self.dt

        tau = self.tau.copy()
        if external_forces is not None:
            tau += external_forces

        f_ext = self._f_ext if self._f_ext else None

        q_new, qd_new = semi_implicit_euler(
            self.robot, self.robot.q, self.robot.qd, tau, dt, f_ext=f_ext
        )

        self.robot.q = q_new
        self.robot.qd = qd_new
        self._time += dt

    def get_positions(self) -> np.ndarray:
        return self.robot.q.copy()

    def get_velocities(self) -> np.ndarray:
        return self.robot.qd.copy()

    def compute_forward_dynamics(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Compute joint accelerations without advancing state."""
        return aba(self.robot, q, qd, tau)

    def compute_inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """Compute required torques for desired accelerations."""
        return rnea(self.robot, q, qd, qdd)

    def compute_gravity_torques(self, q: np.ndarray | None = None) -> np.ndarray:
        """Compute gravity compensation torques."""
        if q is None:
            q = self.robot.q
        return gravity_torques(self.robot, q)

    def kinetic_energy(self) -> float:
        """Compute kinetic energy: 0.5 * qd^T @ M(q) @ qd."""
        from robosim.physics.rbd.algorithms import crba
        M = crba(self.robot, self.robot.q)
        qd = self.robot.qd
        return 0.5 * float(qd @ M @ qd)

    def potential_energy(self) -> float:
        """Compute gravitational potential energy."""
        g = self.robot.gravity
        fk = self.robot.forward_kinematics()
        E = 0.0
        for i, link in enumerate(self.robot.links):
            if link.mass > 0:
                # World position of CoM
                com_local = link.inertial.com
                com_world = fk[i].apply_point(com_local)
                E -= link.mass * float(g @ com_world)
        return E

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()
