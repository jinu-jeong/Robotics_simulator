"""Constraint-based contact solver with full Coulomb cone (static + kinetic).

Companion to :class:`PenaltyContactSolver`. Reuses all of the
broadphase / narrow-phase / SDF / pair-bookkeeping infrastructure;
the *only* thing that differs is the per-contact friction model:

* Penalty solver  → regularised kinetic Coulomb (sliding force vanishes
  as ``v_t → 0``). Cannot hold a grasped object against gravity by
  friction alone — needs ``Scene.grip(...)`` + PD lock.

* Constraint solver → stick/slip Coulomb. When the impulse needed to
  null tangential velocity within one timestep stays inside the
  friction disk (``|f_required| ≤ μ·f_n``) the force is set to that
  exact value (stick); otherwise it saturates at the disk boundary
  opposing motion (slip). Provides static friction, so a grasp can be
  held purely by finger contact — no kinematic lock required.

This is the cheapest credible upgrade to the MuJoCo / PGS family of
constraint contact solvers. A future iteration will add coupled-PGS
iterations across simultaneous contacts; for the single-contact and
finger-pair cases this implementation already gives stable grasping.
"""

from __future__ import annotations

import numpy as np

from robosim.physics.contact.detection import GroundPlane
from robosim.physics.contact.response import ContactParams, compute_contact_force
from robosim.physics.contact.solver import PenaltyContactSolver


class ConstraintContactSolver(PenaltyContactSolver):
    """Coulomb-cone (stick/slip) friction.

    Inherits all RBD / FEM / pair / cross-filter bookkeeping from
    :class:`PenaltyContactSolver`. The only override is the
    per-contact force computation, which routes through
    :func:`compute_contact_force` with ``friction_model="stick_slip"``
    and the solver's stored ``dt``.
    """

    def __init__(
        self,
        ground: GroundPlane | None = None,
        params: ContactParams | None = None,
        dt: float = 1e-3,
        hard_normal: bool = True,
        max_pen_keep: float = 5e-4,
    ):
        """Constraint-mode contact solver.

        ``hard_normal=True`` enables a position-projection post-pass that
        eliminates visible penetration: any free-floating RBD body
        registered via ``register_rbd(is_free_body=True)`` is shifted
        out of an articulated link by the detected ``penetration``, and
        its inward normal velocity is clamped to zero. Articulated
        bodies are *not* moved (their joint positions are the
        controller's responsibility). Set ``hard_normal=False`` to fall
        back to the pure penalty + Coulomb stick/slip behaviour.

        This is the cheapest credible step toward LCP/PGS hard contact.
        It does not do simultaneous multi-contact resolution or accurate
        articulated effective mass; for those, a full constraint solver
        (Sequential Impulse / PGS over all contacts at once) is needed.
        """
        super().__init__(ground=ground, params=params)
        self.dt = float(dt)
        self.hard_normal = bool(hard_normal)
        # Penetration threshold: only project when pen exceeds this. Below
        # the threshold the penalty spring keeps a residual pen ↔ f_n
        # relationship intact, which the friction stick model needs.
        self.max_pen_keep = float(max_pen_keep)

    def set_dt(self, dt: float) -> None:
        self.dt = float(dt)

    def _contact_force(self, contact, v_a, v_b, effective_mass=None):
        return compute_contact_force(
            contact, v_a, v_b, self.params,
            effective_mass=effective_mass,
            friction_model="stick_slip",
            dt=self.dt,
        )

    # ── Hard-normal position projection ────────────────────────────────
    #
    # CAVEAT: simple position projection conflicts with penalty-based
    # normal force. If pen is driven to 0, the penalty spring delivers
    # f_n = 0, which destroys the friction force needed to hold a
    # grasped object. A proper hard-normal solver replaces the penalty
    # spring with an LCP/PGS impulse that simultaneously enforces
    # pen ≥ 0 *and* delivers the correct f_n for friction; that requires
    # solving a coupled constraint system (Sequential Impulse / PGS over
    # all contacts) and is left for future work.
    #
    # The implementation below is exposed for non-grasp scenarios where a
    # free body should bounce cleanly off an articulated link without
    # the small residual penetration penalty leaves behind. It is *not*
    # called from the runner by default (would break grasp). Tests
    # exercise it directly. See README "Contact" section for status.
    def project_free_body_penetrations(self, max_pen_keep: float = 0.0) -> int:
        """Push penetrating free bodies out by *(pen − max_pen_keep)* along normal.

        Aggregates contacts per (free_body, articulated_link) pair so a
        4-corner box face manifold becomes one correction (using the
        deepest pen + averaged normal). Articulated bodies are not
        moved; the controller owns their joint config.

        Use ``max_pen_keep > 0`` to leave a residual penetration that
        the penalty spring keeps loaded — necessary when downstream
        friction relies on f_n = k·pen.
        """
        if not self.hard_normal or not self._free_body_ids:
            return 0
        # Aggregate: (free_rid, articulated_rid, articulated_link_idx) -> (pen_max, n_sum, count)
        groups: dict = {}
        for cp, bid_a, bid_b in self.detector.detect_all():
            if cp.penetration <= max_pen_keep or bid_b < 0:
                continue
            rid_a, link_a = self._body_map.get(bid_a, (None, None))
            rid_b, link_b = self._body_map.get(bid_b, (None, None))
            a_free = rid_a in self._free_body_ids
            b_free = rid_b in self._free_body_ids
            if a_free == b_free:
                continue
            if a_free:
                key = (rid_a, rid_b, link_b); push_dir = cp.normal
            else:
                key = (rid_b, rid_a, link_a); push_dir = -cp.normal
            slot = groups.setdefault(key, [0.0, np.zeros(3), 0])
            if cp.penetration > slot[0]:
                slot[0] = cp.penetration
            slot[1] += push_dir
            slot[2] += 1

        for (free_rid, _, _), (pen, n_sum, _) in groups.items():
            free_solver = self._robot_solvers.get(free_rid)
            if free_solver is None:
                continue
            n_norm = float(np.linalg.norm(n_sum))
            if n_norm < 1e-12:
                continue
            push = n_sum / n_norm
            shift = pen - max_pen_keep
            self._project_free_body(free_solver, push, shift)
        return len(groups)

    @staticmethod
    def _project_free_body(rbd_solver, push_dir: np.ndarray, shift: float) -> None:
        """Move free body q[:3] by shift·push_dir; zero inward qd[:3] component."""
        robot = rbd_solver.robot
        # q for free-floating box: q[0:3] = world translation, q[3:6] = euler.
        # qd[0:3] = world linear velocity (per the create_free_box convention).
        robot.q[0:3] += shift * push_dir
        v_n = float(np.dot(robot.qd[0:3], push_dir))
        if v_n < 0.0:
            robot.qd[0:3] -= v_n * push_dir
