"""7-DoF serial arm: FK, geometric Jacobian (finite-diff), damped-LS IK.

Joint sequence (desk-scale anthropomorphic arm):

    0 yaw (z) at the base
    1 shoulder pitch (y)
    2 upper-arm roll (x)
    3 elbow pitch (y)
    4 forearm roll (x)
    5 wrist pitch (y)
    6 flange yaw (z)

The flange frame is the gripper EE: ``+x`` along the fingers, ``+y`` the
opening direction, ``+z`` finger thickness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def _roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return T


def _rotx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    return T


def _trans(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def rotvec_from_R(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (small-angle friendly)."""
    return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def pose_error(T: np.ndarray, T_des: np.ndarray) -> np.ndarray:
    """6-vector ``[p_des − p; rotvec(Rᵀ R_des)]`` (body-frame rotation)."""
    dp = T_des[:3, 3] - T[:3, 3]
    dw = rotvec_from_R(T[:3, :3].T @ T_des[:3, :3])
    return np.concatenate([dp, dw])


def make_T(R=None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, float).reshape(3, 3)
    if p is not None:
        T[:3, 3] = np.asarray(p, float).reshape(3)
    return T


@dataclass
class SerialArm7:
    """Product-of-transforms 7-DoF arm."""

    base: np.ndarray = field(default_factory=lambda: np.zeros(3))
    d_base: float = 0.14
    L_upper: float = 0.24
    L_fore: float = 0.20
    L_flange: float = 0.07
    q_min: np.ndarray = field(default_factory=lambda: np.array([-2.6, -0.8, -2.4, 0.05, -2.6, -2.6, -2.6]))
    q_max: np.ndarray = field(default_factory=lambda: np.array([2.6, 2.5, 2.4, 2.8, 2.6, 2.4, 2.6]))
    n: int = 7

    def fk(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float).reshape(7)
        T = _trans(*self.base)
        T = T @ _rotz(q[0]) @ _trans(z=self.d_base)
        T = T @ _roty(q[1]) @ _trans(x=self.L_upper)
        T = T @ _rotx(q[2])
        T = T @ _roty(q[3]) @ _trans(x=self.L_fore)
        T = T @ _rotx(q[4])
        T = T @ _roty(q[5])
        T = T @ _rotz(q[6]) @ _trans(x=self.L_flange)
        return T

    def frames(self, q: np.ndarray) -> list[np.ndarray]:
        """Joint origins (8 frames: base + 7 joints / flange)."""
        q = np.asarray(q, float).reshape(7)
        Ts = [_trans(*self.base)]
        T = Ts[0] @ _rotz(q[0]) @ _trans(z=self.d_base)
        Ts.append(T.copy())
        T = T @ _roty(q[1]) @ _trans(x=self.L_upper)
        Ts.append(T.copy())
        T = T @ _rotx(q[2])
        Ts.append(T.copy())
        T = T @ _roty(q[3]) @ _trans(x=self.L_fore)
        Ts.append(T.copy())
        T = T @ _rotx(q[4])
        Ts.append(T.copy())
        T = T @ _roty(q[5])
        Ts.append(T.copy())
        T = T @ _rotz(q[6]) @ _trans(x=self.L_flange)
        Ts.append(T.copy())
        return Ts

    def jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """6×7 geometric Jacobian of ``[p; rotvec]`` at ``q``."""
        T0 = self.fk(q)
        J = np.zeros((6, 7))
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = eps
            J[:, i] = pose_error(T0, self.fk(q + dq)) / eps
        return J

    def ik_step(
        self,
        q: np.ndarray,
        T_des: np.ndarray,
        damping: float = 2e-3,
        max_dq: float = 0.05,
        q_pref: np.ndarray | None = None,
        null_gain: float = 0.04,
        rot_weight: float = 0.35,
    ) -> tuple[np.ndarray, float]:
        """One damped-LS IK step. Returns ``(q_new, ‖e‖)`` with unweighted pose error."""
        q = np.asarray(q, float).reshape(7)
        e = pose_error(self.fk(q), T_des)
        w = np.array([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight])
        J = self.jacobian(q) * w[:, None]
        ew = w * e
        A = J @ J.T + float(damping) * np.eye(6)
        dq = J.T @ np.linalg.solve(A, ew)
        if q_pref is not None:
            N = np.eye(7) - J.T @ np.linalg.solve(A, J)
            dq = dq + null_gain * (N @ (np.asarray(q_pref, float) - q))
        n = np.linalg.norm(dq)
        if n > max_dq:
            dq *= max_dq / n
        q = np.clip(q + dq, self.q_min, self.q_max)
        err = float(np.linalg.norm(pose_error(self.fk(q), T_des)))
        return q, err

    def planar_seeds(self, T_des: np.ndarray) -> list[np.ndarray]:
        """Elbow-up / elbow-down analytic seeds (yaw + three pitches)."""
        R = T_des[:3, :3]
        wrist = T_des[:3, 3] - self.L_flange * R[:, 0]
        S = np.asarray(self.base, float) + np.array([0.0, 0.0, self.d_base])
        v = wrist - S
        q0 = float(np.arctan2(v[1], v[0]))
        c, s = np.cos(q0), np.sin(q0)
        tx = c * v[0] + s * v[1]
        ty = -v[2]
        L1, L2 = self.L_upper, self.L_fore
        r2 = tx * tx + ty * ty
        den = 2.0 * L1 * L2
        if den < 1e-12:
            return []
        cos_e = (r2 - L1 * L1 - L2 * L2) / den
        if abs(cos_e) > 1.02:
            return []
        cos_e = float(np.clip(cos_e, -1.0, 1.0))
        xd = c * R[0, 0] + s * R[1, 0]
        zd = R[2, 0]
        sigma = float(np.arctan2(-zd, xd))
        seeds = []
        for sign in (+1.0, -1.0):
            q3 = sign * float(np.arccos(cos_e))
            q1 = float(np.arctan2(ty, tx) - np.arctan2(L2 * np.sin(q3), L1 + L2 * np.cos(q3)))
            q5 = sigma - q1 - q3
            Ry = _roty(sigma)
            Rz = _rotz(q0)
            R6 = Ry[:3, :3].T @ Rz[:3, :3].T @ R
            q6 = float(np.arctan2(R6[1, 0], R6[0, 0]))
            q_raw = np.array([q0, q1, 0.0, q3, 0.0, q5, q6])
            if np.any(q_raw < self.q_min - 0.02) or np.any(q_raw > self.q_max + 0.02):
                continue
            seeds.append(np.clip(q_raw, self.q_min, self.q_max))
        return seeds

    def solve(
        self,
        T_des: np.ndarray,
        q0: np.ndarray,
        n_iter: int = 80,
        tol: float = 1.5e-3,
        **kw,
    ) -> tuple[np.ndarray, float]:
        starts = [np.asarray(q0, float).copy()]
        starts.extend(self.planar_seeds(T_des))
        best_q, best_err = starts[0], np.inf
        for q_s in starts:
            q = q_s.copy()
            err = float(np.linalg.norm(pose_error(self.fk(q), T_des)))
            for _ in range(n_iter):
                if err < tol:
                    break
                q, err = self.ik_step(q, T_des, **kw)
            if err < best_err:
                best_q, best_err = q, err
            if best_err < tol:
                break
        return best_q, float(best_err)
