"""Contact response: penalty-based normal forces and Coulomb friction."""

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

    # Coulomb friction (smoothed)
    v_t = v_rel - v_n * n
    v_t_norm = np.linalg.norm(v_t)
    if v_t_norm > 1e-12:
        friction_mag = params.friction_mu * fn_mag
        # Smooth clamp: mu * N * v_t / max(|v_t|, eps)
        scale = min(1.0, v_t_norm / params.friction_eps)
        f_friction = -friction_mag * scale * (v_t / v_t_norm)
    else:
        f_friction = np.zeros(3)

    f_total = f_normal + f_friction

    return ContactForce(
        point=contact.point_a.copy(),
        force=f_total,
        normal_force=fn_mag,
    )
