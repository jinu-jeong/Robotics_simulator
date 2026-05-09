"""Contact response: penalty-based normal forces and Coulomb friction.

Two friction models are exposed (selected per call):

* ``"kinetic"`` — regularised kinetic Coulomb. ``f_t = −μ·f_n · v_t/|v_t|``
  smoothed by ``v_t/eps`` near zero. **No static friction**: at
  ``v_t → 0`` the force scales to zero, so a held object slides under
  gravity. Used by :class:`PenaltyContactSolver`.

* ``"stick_slip"`` — full Coulomb cone with stick/slip transition.
  When tangential velocity is small enough that the impulse needed to
  null it within one timestep stays inside the friction disk
  (``|f_required| ≤ μ·f_n``), apply that exact force (stick). Otherwise
  saturate at ``μ·f_n`` opposing motion (slip). Provides static friction
  → grasp held purely by finger force, no kinematic lock needed. Used by
  :class:`ConstraintContactSolver`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robosim.physics.contact.sdf import ContactPoint


@dataclass
class ContactParams:
    """Tunable contact parameters."""

    stiffness: float = 1e5      # penalty spring (N/m)
    damping: float = 1e3        # contact damping (N·s/m)
    friction_mu: float = 0.5    # Coulomb friction coefficient
    friction_eps: float = 1e-3  # regularisation velocity (m/s)
    max_penetration: float = 0.01  # cap depth for penalty force (m)


@dataclass
class ContactForce:
    """Computed contact force at a contact point."""

    point: np.ndarray       # (3,) world-frame application point
    force: np.ndarray       # (3,) world-frame force on body A
    normal_force: float     # magnitude of normal component


def compute_contact_force(
    contact: ContactPoint,
    velocity_a: np.ndarray,
    velocity_b: np.ndarray,
    params: ContactParams,
    effective_mass: float | None = None,
    friction_model: str = "kinetic",
    dt: float | None = None,
) -> ContactForce | None:
    """Compute penalty + friction force for a single contact.

    Parameters
    ----------
    contact : narrow-phase contact result
    velocity_a : (3,) velocity of contact point on body A
    velocity_b : (3,) velocity of contact point on body B (0 for ground)
    params : contact parameters
    effective_mass : if given, damping is capped at critical damping for
                     this mass to prevent energy injection on light nodes.
                     Also required for ``friction_model="stick_slip"`` to
                     size the impulse that nulls ``v_t`` within one step.
    friction_model : ``"kinetic"`` (regularised kinetic Coulomb, default)
                     or ``"stick_slip"`` (Coulomb cone with static friction).
    dt : timestep, required when ``friction_model="stick_slip"``.

    Returns
    -------
    ContactForce or None if the contact is separating fast enough.
    """
    if contact.penetration <= 0:
        return None

    n = contact.normal  # from B → A
    v_rel = velocity_a - velocity_b
    v_n = float(np.dot(v_rel, n))

    # Clamp penetration depth to prevent force explosion
    d = min(contact.penetration, params.max_penetration)

    # Scale damping: for light FEM nodes the global damping may be
    # wildly overcritical, injecting energy instead of dissipating it.
    c = params.damping
    if effective_mass is not None and effective_mass > 0:
        c_crit = 2.0 * np.sqrt(params.stiffness * effective_mass)
        c = min(c, c_crit)

    # Penalty normal force: f_n = k * d - c * v_n  (repulsive only)
    fn_mag = params.stiffness * d - c * v_n
    if fn_mag <= 0:
        return None  # separating fast enough, no force

    f_normal = fn_mag * n

    # Coulomb friction
    v_t = v_rel - v_n * n
    v_t_norm = np.linalg.norm(v_t)
    f_max = params.friction_mu * fn_mag

    if friction_model == "stick_slip":
        # Stick/slip: force needed to null v_t within dt is m_eff·v_t/dt.
        # If that fits inside the Coulomb disk (|·| ≤ μ·f_n) → stick;
        # otherwise saturate at the disk boundary opposing v_t (slip).
        if dt is None or dt <= 0.0 or effective_mass is None or effective_mass <= 0.0:
            # Insufficient info for stick model — fall through to kinetic.
            pass
        else:
            if v_t_norm > 1e-12:
                f_required = -(effective_mass / dt) * v_t          # full stick force
                f_req_mag  = np.linalg.norm(f_required)
                if f_req_mag <= f_max:
                    f_friction = f_required                        # static stick
                else:
                    f_friction = -f_max * (v_t / v_t_norm)         # kinetic slip
            else:
                f_friction = np.zeros(3)                           # already at rest
            return ContactForce(
                point=contact.point_a.copy(),
                force=f_normal + f_friction,
                normal_force=fn_mag,
            )

    # Kinetic-only Coulomb (regularised). Default model.
    if v_t_norm > 1e-12:
        # Smooth clamp: mu * N * v_t / max(|v_t|, eps)
        scale = min(1.0, v_t_norm / params.friction_eps)
        f_friction = -f_max * scale * (v_t / v_t_norm)
    else:
        f_friction = np.zeros(3)

    f_total = f_normal + f_friction

    return ContactForce(
        point=contact.point_a.copy(),
        force=f_total,
        normal_force=fn_mag,
    )
