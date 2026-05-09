"""JointPD — simple joint-space proportional-derivative controller."""

from __future__ import annotations

import numpy as np

from robosim.control.base import Controller


class JointPD(Controller):
    """Joint-space PD controller with optional gravity compensation.

    Parameters
    ----------
    kp : proportional gain — scalar or per-dof (n_dof,) array
    kd : derivative gain   — scalar or per-dof (n_dof,) array
    max_torque : symmetric torque clamp — scalar (applied to all joints)
        or per-dof (n_dof,) array. A per-finger cap turns the position
        PD into an *impedance / force-limited* controller for that
        joint: the PD pushes up to the cap then saturates, so a finger
        commanded past a contact surface stops at the contact instead
        of driving deeper. Recommended for friction-grasping (constraint
        contact mode) where unbounded PD overrides contact reaction.

    Usage::

        ctrl = JointPD(kp=[280, 480, 200, 75, 5e4, 5e4],
                       kd=[ 35,  50,  20,  2, 100, 100])
        ctrl.target_q = HOME_Q
        tau = ctrl.compute(t, robot.q, robot.dq)
    """

    def __init__(
        self,
        kp: float | list | np.ndarray = 200.0,
        kd: float | list | np.ndarray = 30.0,
        max_torque: float | list | np.ndarray = 500.0,
        gravity_comp: bool = True,
    ):
        self._kp_raw = kp
        self._kd_raw = kd
        self._max_torque_raw = max_torque
        self.max_torque = max_torque       # back-compat: scalar attribute remains valid
        self.gravity_comp = gravity_comp

        self.target_q:  np.ndarray | None = None
        self.target_qd: np.ndarray | None = None

        # Broadcast to per-dof arrays lazily on first compute() call
        self._kp: np.ndarray | None = None
        self._kd: np.ndarray | None = None
        self._robot = None  # set by runner for gravity compensation

    # ── internal ──────────────────────────────────────────────────────────────

    def _init_gains(self, n_dof: int) -> None:
        kp = np.atleast_1d(np.asarray(self._kp_raw, dtype=float))
        kd = np.atleast_1d(np.asarray(self._kd_raw, dtype=float))
        mt = np.atleast_1d(np.asarray(self._max_torque_raw, dtype=float))
        self._kp = np.broadcast_to(kp, (n_dof,)).copy()
        self._kd = np.broadcast_to(kd, (n_dof,)).copy()
        self._max_torque_arr = np.broadcast_to(mt, (n_dof,)).copy()

    # ── Controller interface ───────────────────────────────────────────────────

    def compute(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        n = len(q)
        if self._kp is None:
            self._init_gains(n)
        if self.target_q is None:
            return np.zeros(n)

        q_des  = self.target_q
        qd_des = self.target_qd if self.target_qd is not None else np.zeros(n)

        tau = self._kp * (q_des - q) + self._kd * (qd_des - qd)

        if self.gravity_comp and self._robot is not None:
            from robosim.physics.rbd.algorithms import gravity_torques
            tau = tau + gravity_torques(self._robot, q)

        return np.clip(tau, -self._max_torque_arr, self._max_torque_arr)

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"JointPD(kp={self._kp_raw}, kd={self._kd_raw})"
