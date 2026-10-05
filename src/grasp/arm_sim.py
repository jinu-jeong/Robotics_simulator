"""Stage C: 7-DoF arm carries the A/B parallel-jaw gripper.

Phases: home → reach (pregrasp) → approach_ee → squeeze → lift → hold/drop.

Jaw force control is the existing :class:`GraspSim` (``external_ee=True``).
The arm owns ``T_ee`` via damped-LS IK; when the object is held it rides
the flange, otherwise it sits on / falls toward the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .arm import SerialArm7, make_T, pose_error
from .sim import GraspSim


def grasp_T(contact_local: np.ndarray, object_center: np.ndarray, R=None) -> np.ndarray:
    """EE pose that places the inner-face contact on the object ±y faces."""
    c = np.asarray(contact_local, float).reshape(3)
    p_obj = np.asarray(object_center, float).reshape(3)
    R = np.eye(3) if R is None else np.asarray(R, float).reshape(3, 3)
    return make_T(R, p_obj - R @ np.array([c[0], 0.0, c[2]]))


@dataclass
class ArmGraspSim:
    grasp: GraspSim
    arm: SerialArm7
    q_home: np.ndarray
    T_pre: np.ndarray
    T_grasp: np.ndarray
    T_lift: np.ndarray
    object_xy: np.ndarray
    table_z: float = 0.0
    ik_damping: float = 2e-3
    ik_max_dq: float = 0.05
    ik_tol: float = 2.5e-3
    q: np.ndarray = field(default_factory=lambda: np.zeros(7))
    phase: str = "home"
    t: float = 0.0
    ik_err: float = 0.0
    obj_z: float = 0.0
    obj_vz: float = 0.0
    obj_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    object_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    log: dict[str, list] = field(default_factory=dict)
    _lift_z: float | None = None

    def __post_init__(self) -> None:
        self.grasp.external_ee = True
        self.reset()

    @property
    def mech(self):
        return self.grasp.mech

    @property
    def estimator(self):
        return self.grasp.estimator

    @property
    def T_ee(self) -> np.ndarray:
        return self.arm.fk(self.q)

    @property
    def dt(self) -> float:
        return self.grasp.dt

    def set_mode(self, mode: str) -> None:
        self.grasp.set_mode(mode)

    def reset(self) -> None:
        self.grasp.reset()
        self.grasp.external_ee = True
        self.q = np.asarray(self.q_home, float).copy()
        self.phase = "home"
        self.t = 0.0
        self.ik_err = 0.0
        half = 0.5 * self.mech.object_size[2]
        self.obj_z = self.table_z + half
        self.obj_vz = 0.0
        self.obj_xy = np.asarray(self.object_xy, float).reshape(2).copy()
        self.object_R = np.eye(3)
        self._lift_z = None
        self.log = {k: [] for k in (
            "t", "phase", "opening", "lam_true", "lam_meas", "force_target",
            "ik_err", "ee_z", "obj_z", "held", "hold_force",
        )}
        self._record()

    def object_center(self) -> np.ndarray:
        return np.array([self.obj_xy[0], self.obj_xy[1], self.obj_z])

    def held(self) -> bool:
        return bool(self.grasp.state.held)

    def _ee_object_center(self, T: np.ndarray | None = None) -> np.ndarray:
        T = self.T_ee if T is None else T
        c = self.mech.contact_local
        return T[:3, :3] @ np.array([c[0], 0.0, c[2]]) + T[:3, 3]

    def _at_pose(self, T_des: np.ndarray) -> bool:
        e = pose_error(self.T_ee, T_des)
        return float(np.linalg.norm(e[:3])) < self.ik_tol and float(np.linalg.norm(e[3:])) < 0.06

    def _ik_toward(self, T_des: np.ndarray) -> bool:
        self.q, self.ik_err = self.arm.ik_step(
            self.q, T_des,
            damping=self.ik_damping,
            max_dq=self.ik_max_dq,
            q_pref=self.q_home,
        )
        return self._at_pose(T_des)

    def _update_object(self, dt: float) -> None:
        table = self.table_z + 0.5 * self.mech.object_size[2]
        if self.held() and self.phase in ("lift", "hold"):
            c = self._ee_object_center()
            self.obj_xy = c[:2].copy()
            self.obj_z = float(c[2])
            self.obj_vz = 0.0
            self.object_R = self.T_ee[:3, :3].copy()
            return
        self.obj_vz -= 9.80665 * dt
        self.obj_z += self.obj_vz * dt
        if self.obj_z <= table:
            self.obj_z = table
            self.obj_vz = 0.0
        if self.phase == "hold" and not self.held():
            self.phase = "drop"

    def _record(self) -> None:
        s = self.grasp.state
        self.log["t"].append(self.t)
        self.log["phase"].append(self.phase)
        self.log["opening"].append(s.opening)
        self.log["lam_true"].append(s.lam_true)
        self.log["lam_meas"].append(s.lam_meas)
        self.log["force_target"].append(self.grasp.force_target)
        self.log["ik_err"].append(self.ik_err)
        self.log["ee_z"].append(float(self.T_ee[2, 3]))
        self.log["obj_z"].append(self.obj_z)
        self.log["held"].append(float(self.held()))
        self.log["hold_force"].append(self.mech.hold_force)

    def step(self, dt: float | None = None) -> None:
        dt = self.dt if dt is None else float(dt)
        self.t += dt
        g = self.grasp
        g.T_ee = self.T_ee
        # marker-free (nn) estimator renders the scene: tell it where the object is
        g.estimator.set_scene(self.object_center(), self.object_R)

        if self.phase == "home":
            self.phase = "reach"
        elif self.phase == "reach":
            if self._ik_toward(self.T_pre):
                self.phase = "approach_ee"
        elif self.phase == "approach_ee":
            if self._ik_toward(self.T_grasp):
                g.state.phase = "approach"
                self.phase = "squeeze"
        elif self.phase == "squeeze":
            g.step(dt)
            if g.state.phase == "ready_lift":
                self.phase = "lift"
                self._lift_z = float(self.T_grasp[2, 3])
        elif self.phase == "lift":
            g.step(dt)
            z0 = float(self.T_grasp[2, 3])
            z1 = float(self.T_lift[2, 3])
            if self._lift_z is None:
                self._lift_z = z0
            # Cartesian-rate lift (uses gripper.lift_speed) — not a snatch to T_lift.
            self._lift_z = min(z1, self._lift_z + float(g.lift_speed) * dt)
            T_des = self.T_lift.copy()
            T_des[2, 3] = self._lift_z
            at = self._ik_toward(T_des)
            if at and self._lift_z >= z1 - 1e-4:
                self.phase = "hold" if self.held() else "drop"
                g.state.phase = self.phase
        elif self.phase in ("hold", "drop"):
            if g.state.phase not in ("hold", "drop"):
                # keep regulating force while holding
                g.step(dt)
                g.state.phase = self.phase

        self.ik_err = float(np.linalg.norm(pose_error(self.T_ee, self._target_T())))
        self._update_object(dt)
        self._record()

    def _target_T(self) -> np.ndarray:
        if self.phase == "lift":
            T = self.T_lift.copy()
            if self._lift_z is not None:
                T[2, 3] = self._lift_z
            return T
        return {
            "home": self.arm.fk(self.q_home),
            "reach": self.T_pre,
            "approach_ee": self.T_grasp,
            "squeeze": self.T_grasp,
            "hold": self.T_lift,
            "drop": self.T_lift,
        }.get(self.phase, self.T_grasp)

    def run(self, t_max: float = 12.0) -> dict:
        while self.t < t_max and self.phase not in ("hold", "drop"):
            self.step()
        t_end = self.t + 0.4
        while self.t < t_end:
            self.step()
        return {k: (np.asarray(v) if k != "phase" else list(v)) for k, v in self.log.items()}

    def success(self) -> bool:
        table = self.table_z + 0.5 * self.mech.object_size[2]
        lifted = self.obj_z >= table + 0.5 * self.grasp.lift_height
        return self.phase == "hold" and self.held() and lifted


def build_arm_grasp_sim(grasp: GraspSim, cfg: Mapping[str, Any] | None = None) -> ArmGraspSim:
    """Attach a 7-DoF arm to an existing (A/B) ``GraspSim``."""
    ac = dict(cfg or {})
    arm = SerialArm7(
        d_base=float(ac.get("d_base", 0.14)),
        L_upper=float(ac.get("L_upper", 0.24)),
        L_fore=float(ac.get("L_fore", 0.20)),
        L_flange=float(ac.get("L_flange", 0.07)),
    )
    xy = np.asarray(ac.get("object_xy", [0.32, 0.0]), float).reshape(2)
    table_z = float(ac.get("table_z", 0.0))
    half = 0.5 * grasp.mech.object_size[2]
    p_obj = np.array([xy[0], xy[1], table_z + half])
    T_g = grasp_T(grasp.mech.contact_local, p_obj)
    pre = np.asarray(ac.get("pregrasp_offset", [-0.07, 0.0, 0.04]), float)
    T_pre = T_g.copy()
    T_pre[:3, 3] = T_g[:3, 3] + T_g[:3, :3] @ pre
    T_lift = T_g.copy()
    T_lift[:3, 3] = T_g[:3, 3] + np.array([0.0, 0.0, float(grasp.lift_height)])
    seeds = arm.planar_seeds(T_pre)
    q_home = seeds[0] if seeds else np.array([0.0, 0.4, 0.0, 1.2, 0.0, 0.5, 0.0])
    if "q_home" in ac:
        q_home = np.asarray(ac["q_home"], float)
    return ArmGraspSim(
        grasp=grasp,
        arm=arm,
        q_home=q_home,
        T_pre=T_pre,
        T_grasp=T_g,
        T_lift=T_lift,
        object_xy=xy,
        table_z=table_z,
        ik_damping=float(ac.get("ik_damping", 2e-3)),
        ik_max_dq=float(ac.get("ik_max_dq", 0.05)),
        ik_tol=float(ac.get("ik_tol", 2.5e-3)),
    )
