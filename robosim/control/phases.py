"""Phase-based Trajectory — high-level motion sequencing for Scene API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robosim.scene.handles import RobotHandle


# ── Phase ─────────────────────────────────────────────────────────────────────

@dataclass
class Phase:
    """A single named trajectory phase.

    Parameters
    ----------
    name     : human-readable label (used for debugging and ``in_phase()``)
    target_q : desired joint configuration at phase end
    duration : nominal duration in seconds
    until    : optional early-exit condition — ``() -> bool``
               Phase ends as soon as this returns True (even before ``duration``)
    after    : optional wait-to-start condition — ``() -> bool``
               Phase does not begin advancing until this returns True
    """
    name:     str
    target_q: np.ndarray
    duration: float
    until:    Callable[[], bool] | None = None
    after:    Callable[[], bool] | None = None
    _started: bool = field(default=False, init=False, repr=False)


# ── Trajectory ────────────────────────────────────────────────────────────────

class Trajectory:
    """Phase-based joint-space trajectory bound to a :class:`RobotHandle`.

    Phases are added sequentially; each phase smoothly interpolates from the
    robot's actual joint state at phase entry to ``target_q``.  The runner
    calls ``_step(dt)`` every simulation tick.

    Usage::

        traj = robot.trajectory()
        traj.phase("HOME",     target=HOME_Q,     duration=1.5)
        traj.phase("APPROACH", target=APPROACH_Q, duration=2.5)
        traj.phase("CLOSE",    target=CLOSE_Q,    duration=1.5,
                   until=lambda: box.bottom_z > 0.1)
        traj.phase("LIFT",     target=LIFT_Q,     duration=3.0,
                   after=traj.in_phase("CLOSE"))   # (example, not needed here)
    """

    def __init__(self, robot: "RobotHandle"):
        self._robot   = robot
        self._phases: list[Phase] = []
        self._idx:    int   = 0
        self._t_ph:   float = 0.0     # elapsed time inside current phase
        self._q0:     np.ndarray | None = None   # start q of current phase
        self._done:   bool  = False

    # ── Builder ───────────────────────────────────────────────────────────────

    def phase(
        self,
        name: str,
        target: np.ndarray | list,
        duration: float,
        until: Callable[[], bool] | None = None,
        after: Callable[[], bool] | None = None,
    ) -> "Trajectory":
        """Append a phase.  Returns *self* for chaining."""
        self._phases.append(Phase(
            name=name,
            target_q=np.asarray(target, dtype=float),
            duration=max(duration, 1e-6),
            until=until,
            after=after,
        ))
        return self

    # ── Query ─────────────────────────────────────────────────────────────────

    def in_phase(self, name: str) -> Callable[[], bool]:
        """Return a callable that is ``True`` while this Trajectory is in *name*.

        Useful as the ``after=`` argument of another trajectory's phase::

            traj_b.phase("RECEIVE", target=..., duration=2.0,
                         after=traj_a.in_phase("LIFT"))
        """
        return lambda: (
            not self._done
            and self._idx < len(self._phases)
            and self._phases[self._idx].name == name
        )

    @property
    def current_phase(self) -> str | None:
        if self._done or self._idx >= len(self._phases):
            return None
        return self._phases[self._idx].name

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def phase_elapsed(self) -> float:
        return self._t_ph

    # ── Runner interface ──────────────────────────────────────────────────────

    def _step(self, dt: float) -> None:
        """Advance by *dt* and update the robot's PD target.  Called by runner."""
        if self._done or not self._phases:
            return

        ph = self._phases[self._idx]

        # Wait for `after` condition before entering this phase
        if ph.after is not None and not ph.after():
            return

        # On first tick: snapshot actual joint state as interpolation start
        if self._q0 is None:
            self._q0  = self._robot._model.q.copy()
            self._t_ph = 0.0

        self._t_ph += dt

        # Quintic smooth-step interpolation (zero velocity AND acceleration at endpoints)
        s = min(1.0, self._t_ph / ph.duration)
        s3 = s * s * s
        w = s3 * (10.0 - 15.0 * s + 6.0 * s * s)   # 5th-order Bezier
        q_des = self._q0 + w * (ph.target_q - self._q0)

        # Push target to PD controller
        ctrl = self._robot._controller
        if ctrl is not None:
            ctrl.target_q = q_des

        # Phase transition: time expired or condition met
        if self._t_ph >= ph.duration or (ph.until is not None and ph.until()):
            self._idx  += 1
            self._q0    = None
            self._t_ph  = 0.0
            if self._idx >= len(self._phases):
                self._done = True

    def reset(self) -> None:
        self._idx   = 0
        self._t_ph  = 0.0
        self._q0    = None
        self._done  = False

    def __repr__(self) -> str:
        if self._done:
            status = "DONE"
        elif self._idx < len(self._phases):
            ph = self._phases[self._idx]
            status = f"{ph.name!r}  {self._t_ph:.2f}s / {ph.duration:.1f}s"
        else:
            status = "IDLE"
        n = len(self._phases)
        return f"Trajectory[{self._robot.name}]  {status}  ({n} phases)"
