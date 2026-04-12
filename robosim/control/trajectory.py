"""Trajectory generation for joint-space motion."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosim.control.base import Controller


@dataclass
class Waypoint:
    """A timed joint-space waypoint."""
    t: float
    q: np.ndarray
    qd: np.ndarray | None = None


class TrajectoryController(Controller):
    """Follows a sequence of waypoints with cubic interpolation + PID tracking.

    Generates smooth minimum-jerk trajectories between waypoints and
    uses PD control to track them.
    """

    def __init__(
        self,
        n_dof: int,
        waypoints: list[Waypoint],
        kp: np.ndarray | float = 200.0,
        kd: np.ndarray | float = 30.0,
        max_torque: float = 100.0,
    ):
        self.n_dof = n_dof
        self.waypoints = sorted(waypoints, key=lambda w: w.t)
        self.kp = np.broadcast_to(np.atleast_1d(np.asarray(kp, dtype=np.float64)), (n_dof,)).copy()
        self.kd = np.broadcast_to(np.atleast_1d(np.asarray(kd, dtype=np.float64)), (n_dof,)).copy()
        self.max_torque = max_torque

        # Ensure waypoints have velocity info
        for wp in self.waypoints:
            wp.q = np.asarray(wp.q, dtype=np.float64)
            if wp.qd is None:
                wp.qd = np.zeros(n_dof)
            else:
                wp.qd = np.asarray(wp.qd, dtype=np.float64)

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Get desired (q, qd) at time t via cubic Hermite interpolation."""
        wps = self.waypoints

        if t <= wps[0].t:
            return wps[0].q.copy(), wps[0].qd.copy()
        if t >= wps[-1].t:
            return wps[-1].q.copy(), wps[-1].qd.copy()

        # Find segment
        for i in range(len(wps) - 1):
            if wps[i].t <= t < wps[i + 1].t:
                break

        w0, w1 = wps[i], wps[i + 1]
        T = w1.t - w0.t
        s = (t - w0.t) / T  # normalized time [0, 1]

        # Cubic Hermite basis
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2

        q_des = h00 * w0.q + h10 * T * w0.qd + h01 * w1.q + h11 * T * w1.qd

        # Derivative of Hermite
        ds = 1.0 / T
        dh00 = (6 * s**2 - 6 * s) * ds
        dh10 = (3 * s**2 - 4 * s + 1) * ds
        dh01 = (-6 * s**2 + 6 * s) * ds
        dh11 = (3 * s**2 - 2 * s) * ds

        qd_des = dh00 * w0.q + dh10 * T * w0.qd + dh01 * w1.q + dh11 * T * w1.qd

        return q_des, qd_des

    def compute(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        q_des, qd_des = self.evaluate(t)
        e = q_des - q
        ed = qd_des - qd
        tau = self.kp * e + self.kd * ed
        return np.clip(tau, -self.max_torque, self.max_torque)

    def reset(self):
        pass

    @property
    def duration(self) -> float:
        return self.waypoints[-1].t if self.waypoints else 0.0
