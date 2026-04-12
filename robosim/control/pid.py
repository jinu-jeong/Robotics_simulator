"""PID joint-space controller."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosim.control.base import Controller


@dataclass
class PIDController(Controller):
    """Independent PID controller for each joint.

    Parameters
    ----------
    n_dof : number of joints
    kp : (n_dof,) proportional gains
    kd : (n_dof,) derivative gains
    ki : (n_dof,) integral gains
    max_torque : per-joint torque clamp (scalar or array)
    """

    n_dof: int = 1
    kp: np.ndarray = field(default_factory=lambda: np.array([100.0]))
    kd: np.ndarray = field(default_factory=lambda: np.array([10.0]))
    ki: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    max_torque: float = 100.0

    # Target state
    _q_des: np.ndarray = field(default=None, repr=False)
    _qd_des: np.ndarray = field(default=None, repr=False)

    # Integral accumulator
    _integral: np.ndarray = field(default=None, repr=False)
    _prev_t: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self.kp = np.broadcast_to(np.asarray(self.kp, dtype=np.float64), (self.n_dof,)).copy()
        self.kd = np.broadcast_to(np.asarray(self.kd, dtype=np.float64), (self.n_dof,)).copy()
        self.ki = np.broadcast_to(np.asarray(self.ki, dtype=np.float64), (self.n_dof,)).copy()
        self._q_des = np.zeros(self.n_dof)
        self._qd_des = np.zeros(self.n_dof)
        self._integral = np.zeros(self.n_dof)
        self._prev_t = 0.0

    def set_target(self, q_des: np.ndarray, qd_des: np.ndarray | None = None):
        """Set desired joint positions (and optionally velocities)."""
        self._q_des = np.asarray(q_des, dtype=np.float64)
        self._qd_des = np.zeros(self.n_dof) if qd_des is None else np.asarray(qd_des, dtype=np.float64)

    def compute(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        dt = t - self._prev_t
        self._prev_t = t

        e = self._q_des - q
        ed = self._qd_des - qd

        if dt > 0 and dt < 0.1:
            self._integral += e * dt
            # Anti-windup: clamp integral
            i_max = self.max_torque / (np.abs(self.ki) + 1e-12)
            self._integral = np.clip(self._integral, -i_max, i_max)

        tau = self.kp * e + self.kd * ed + self.ki * self._integral
        return np.clip(tau, -self.max_torque, self.max_torque)

    def reset(self):
        self._integral[:] = 0.0
        self._prev_t = 0.0
