"""Time-stepping grasp: approach → squeeze (force control) → lift → hold/drop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .estimator import GraspEstimator
from .mechanics import GraspMechanics


@dataclass
class GraspState:
    t: float = 0.0
    opening: float = 0.0
    lift: float = 0.0
    lam_true: float = 0.0
    lam_meas: float = 0.0
    obj_z: float = 0.0
    obj_vz: float = 0.0
    held: bool = False
    phase: str = "approach"
    integral: float = 0.0
    settle_t: float = 0.0
    dwell_left: float = 0.0
    preload_dwell_done: bool = False


@dataclass
class GraspSim:
    mech: GraspMechanics
    estimator: GraspEstimator
    opening_start: float
    opening_min: float
    close_speed: float
    lift_height: float
    lift_speed: float
    force_target: float
    force_tol: float
    kp: float
    ki: float
    settle_s: float
    dt: float
    # After reaching preload: hold jaws so Kalman can average q before PI.
    post_contact_hold_s: float = 0.0
    # While unknown-contact is not locked, close slowly (do not PI on λ̂=0).
    unlock_squeeze_speed: float = 0.001
    # Light true force at which to pause and average q (unknown-contact).
    preload_force: float = 1.2
    state: GraspState = field(default_factory=GraspState)
    log: dict[str, list] = field(default_factory=dict)
    force_lift: bool = False  # user request to start lifting
    meas_filter: float = 0.0
    filter_alpha: float = 0.2
    # Stage C: the arm owns the EE pose. Squeeze still runs here; lift is
    # signalled as phase ``ready_lift`` and object z is not integrated.
    external_ee: bool = False
    T_ee: np.ndarray | None = None  # Stage D: world-camera measure() uses FK pose

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        half_h = 0.5 * self.mech.object_size[2]
        self.state = GraspState(
            opening=self.opening_start,
            obj_z=half_h,
            phase="approach",
        )
        self.force_lift = False
        self.meas_filter = 0.0
        self.estimator.reset_filter()
        self.log = {k: [] for k in (
            "t", "opening", "lift", "lam_true", "lam_meas", "force_target",
            "obj_z", "held", "phase", "hold_force",
        )}
        self._record()

    def set_mode(self, mode: str) -> None:
        self.estimator.mode = str(mode).lower()

    def _vision_unlocked(self) -> bool:
        """True when unknown-contact force is still gated (do not PI on λ̂=0)."""
        est = self.estimator
        return bool(est.unknown_contact and getattr(est, "_locked_contact", None) is None)

    def _preload_from_estimate(self) -> bool:
        """Preload is detected from λ̂ at the config contact (needs ``prelock_known_force``);
        without it the pre-lock λ̂ is zero and the legacy true-force trigger is kept."""
        est = self.estimator
        return bool(est.unknown_contact and est.prelock_known_force)

    def _record(self) -> None:
        s = self.state
        self.log["t"].append(s.t)
        self.log["opening"].append(s.opening)
        self.log["lift"].append(s.lift)
        self.log["lam_true"].append(s.lam_true)
        self.log["lam_meas"].append(s.lam_meas)
        self.log["force_target"].append(self.force_target)
        self.log["obj_z"].append(s.obj_z)
        self.log["held"].append(float(s.held))
        self.log["phase"].append(s.phase)
        self.log["hold_force"].append(self.mech.hold_force)

    def step(self, dt: float | None = None) -> GraspState:
        dt = self.dt if dt is None else float(dt)
        s = self.state
        s.t += dt
        R = None if self.T_ee is None else np.asarray(self.T_ee, float)[:3, :3]
        s.lam_true = self.mech.force_from_opening(s.opening, R)
        u = self.mech.displacement(s.lam_true, R)  # contact load + self-weight sag
        if self._preload_from_estimate():
            self.estimator.lock_votes_open = s.preload_dwell_done
        raw = self.estimator.measure(u, s.lam_true, T_ee=self.T_ee, opening=s.opening)
        # Vision path already Kalman-filters q. Extra EMA on force would double-lag.
        if self.estimator.kf is not None:
            self.meas_filter = raw
        elif s.lam_true <= 1e-9:
            self.meas_filter = raw
        else:
            self.meas_filter = (1.0 - self.filter_alpha) * self.meas_filter + self.filter_alpha * raw
        s.lam_meas = self.meas_filter

        if s.phase == "approach":
            s.opening = max(self.opening_min, s.opening - self.close_speed * dt)
            if s.lam_true > 1e-6:
                s.phase = "squeeze"
                s.integral = 0.0
                s.settle_t = 0.0
                s.dwell_left = 0.0
                s.preload_dwell_done = False
                # Known-contact / GT: optional short hold is enough at first touch.
                if not self._vision_unlocked():
                    s.dwell_left = float(self.post_contact_hold_s)
        elif s.phase == "squeeze":
            unlocked = self._vision_unlocked()

            # Unknown-contact: creep to preload, then hold to average q / lock contact.
            preload_lam = s.lam_meas if self._preload_from_estimate() else s.lam_true
            if unlocked and not s.preload_dwell_done and preload_lam >= self.preload_force:
                s.dwell_left = float(self.post_contact_hold_s)
                s.preload_dwell_done = True

            if s.dwell_left > 0.0:
                s.dwell_left = max(0.0, s.dwell_left - dt)
                if not unlocked:
                    s.dwell_left = 0.0
            elif unlocked:
                # Creep only — never run PI on λ̂=0 (that caused the blind GT ramp).
                s.opening = float(np.clip(
                    s.opening - self.unlock_squeeze_speed * dt,
                    self.opening_min, self.opening_start,
                ))
                s.integral = 0.0
                s.settle_t = 0.0
            else:
                err = self.force_target - s.lam_meas
                s.integral = float(np.clip(s.integral + err * dt, -20.0, 20.0))
                s.opening = float(np.clip(
                    s.opening - (self.kp * err + self.ki * s.integral) * dt,
                    self.opening_min, self.opening_start,
                ))
                if abs(err) < self.force_tol:
                    s.settle_t += dt
                else:
                    s.settle_t = 0.0
                if self.force_lift or s.settle_t >= self.settle_s:
                    s.phase = "ready_lift" if self.external_ee else "lift"
        elif s.phase in ("lift", "ready_lift"):
            if s.phase == "lift":
                s.lift = min(self.lift_height, s.lift + self.lift_speed * dt)
            err = self.force_target - s.lam_meas
            s.opening = float(np.clip(
                s.opening - self.kp * err * dt,
                self.opening_min, self.opening_start,
            ))
            if s.phase == "lift" and s.lift >= self.lift_height - 1e-9:
                s.phase = "hold" if s.held else "drop"
        elif s.phase in ("hold", "drop"):
            pass

        # object vertical dynamics (A/B). Stage C integrates the object on the arm.
        s.held = self.mech.can_hold(s.lam_true) and s.lam_true > 1e-9
        if not self.external_ee:
            table = 0.5 * self.mech.object_size[2]
            if s.held and s.lift > 1e-6:
                s.obj_z = table + s.lift
                s.obj_vz = 0.0
            else:
                s.obj_vz -= 9.80665 * dt
                s.obj_z += s.obj_vz * dt
                if s.obj_z <= table:
                    s.obj_z = table
                    s.obj_vz = 0.0
                if s.phase == "hold" and not s.held:
                    s.phase = "drop"

        self._record()
        return s

    def run(self, t_max: float = 6.0) -> dict:
        while self.state.t < t_max and self.state.phase not in ("hold", "drop"):
            self.step()
        # a little extra so hold/drop is visible
        t_end = self.state.t + 0.4
        while self.state.t < t_end:
            self.step()
        return {k: (np.asarray(v) if k != "phase" else list(v)) for k, v in self.log.items()}

    def success(self) -> bool:
        return self.state.phase == "hold" and self.state.held and self.state.lift >= 0.5 * self.lift_height
