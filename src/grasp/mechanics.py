"""Quasi-static two-finger squeeze + Coulomb lift of a rigid object.

Each finger is the project's linear FEM. A unit normal load at the inner-face
contact gives the indentation compliance ``k = λ / δ``. Closing the jaws by
``w − g`` (object width minus inner-face gap) is shared equally, so

    δ = max(0, (w − g) / 2),    λ = k δ

A deformable object is a lumped spring across its grasped width,
``Δw = λ / k_obj``, in series with the two fingers:

    w − g + 2 δ_g = 2 λ / k + λ / k_obj

(``k_obj = ∞`` is the rigid object). The finger FEM, and therefore the vision
estimator, is unaffected by ``k_obj``; only the opening ↔ force map changes.

It lifts only if the two frictional contacts beat weight:

    2 μ λ ≥ m g
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..fem.finger_model import FingerFEMModel
from .world import ParallelJawWorld

G = 9.80665


@dataclass
class GraspMechanics:
    model: FingerFEMModel
    world: ParallelJawWorld
    contact_local: np.ndarray
    stiffness: float  # λ / indentation [N/m]
    object_width: float
    object_size: np.ndarray
    mass: float
    mu: float
    contact: object = None  # SurfaceContact of the unit-load solve
    extra: dict = field(default_factory=dict)
    object_stiffness: float = math.inf  # λ / width compression [N/m]; inf = rigid

    @classmethod
    def build(
        cls,
        model: FingerFEMModel,
        contact_rel: tuple | list,
        object_size,
        mass: float,
        mu: float,
        object_stiffness: float | None = None,
    ) -> "GraspMechanics":
        model.restrict_contact_surface("side_pos_y")
        g = model.geometry
        rel = np.asarray(contact_rel, float)
        rel[1] = 1.0  # inner face
        p = g.point_from_relative(rel)
        res, contact = model.solve_normal_contact(p, 1.0)
        u_c = contact.barycentric @ res.u[contact.node_indices]
        delta = float(-contact.normal @ u_c)
        if delta <= 1e-12:
            raise RuntimeError(f"non-positive indentation under 1 N ({delta}); check contact face")
        size = np.asarray(object_size, float).reshape(3)
        return cls(
            model=model,
            world=ParallelJawWorld(g),
            contact_local=contact.position.copy(),
            stiffness=1.0 / delta,
            object_width=float(size[1]),
            object_size=size,
            mass=float(mass),
            mu=float(mu),
            contact=contact,
            extra={"unit_indent_m": delta, "hold_force_N": float(mass) * G / (2.0 * float(mu))},
            object_stiffness=math.inf if object_stiffness is None else float(object_stiffness),
        )

    @property
    def weight(self) -> float:
        return self.mass * G

    @property
    def hold_force(self) -> float:
        """Minimum per-finger normal force that can lift the object [N]."""
        return self.weight / (2.0 * self.mu)

    # ------------------------------------------------------------ self-weight
    @property
    def has_gravity(self) -> bool:
        return bool(self.model.has_gravity)

    def gravity_displacement(self, R_world_from_local=None) -> np.ndarray:
        """(N, 3) self-weight sag of one finger in its local frame (zero if disabled)."""
        return self.model.gravity_displacement(R_world_from_local)

    def gravity_indent(self, R_world_from_local=None) -> float:
        """Self-weight motion of the contact point *into* the object [m] (along −n).

        Positive when gravity pushes the inner face toward the object (adds to
        the squeeze), negative when it pulls it away. Zero for an upright finger.
        """
        if not self.has_gravity or self.contact is None:
            return 0.0
        ug = self.gravity_displacement(R_world_from_local)
        u_c = self.contact.barycentric @ ug[self.contact.node_indices]
        return float(-np.asarray(self.contact.normal, float) @ u_c)

    # ------------------------------------------------------------ deformable object
    @property
    def is_rigid(self) -> bool:
        return math.isinf(self.object_stiffness)

    @property
    def closure_compliance(self) -> float:
        """Jaw closure per unit contact force, ``2/k + 1/k_obj`` [m/N]."""
        return 2.0 / self.stiffness + 1.0 / self.object_stiffness

    def object_compression(self, lam: float) -> float:
        """Width lost by the object under contact force ``lam`` [m]."""
        return max(0.0, float(lam)) / self.object_stiffness

    def object_size_at(self, lam: float) -> np.ndarray:
        size = self.object_size.copy()
        size[1] -= self.object_compression(lam)
        return size

    def force_from_opening(self, opening: float, R_world_from_local=None) -> float:
        """Contact force of one finger for an inner-face gap ``opening``.

        Finger indentation plus object compression must take up the geometric
        closure: ``w − g + 2 δ_g = λ (2/k + 1/k_obj)`` (rigid: ``λ = k (δ_geom + δ_g)``).
        """
        closure = self.object_width - float(opening) + 2.0 * self.gravity_indent(R_world_from_local)
        return max(0.0, closure) / self.closure_compliance

    def opening_from_force(self, lam: float, R_world_from_local=None) -> float:
        lam = max(0.0, float(lam))
        return self.object_width + 2.0 * self.gravity_indent(R_world_from_local) - lam * self.closure_compliance

    def estimate_object_compliance(self, opening: float, lam: float, R_world_from_local=None) -> float:
        """Object compliance ``1/k_obj`` [m/N] from the jaw opening and a (measured) force.

        Whatever closure the finger indentation ``2λ/k`` does not explain was
        taken up by the object. Returned as compliance because it is linear in
        the force error and well defined for a rigid object (→ 0).
        """
        lam = float(lam)
        if lam <= 1e-9:
            return float("nan")
        closure = self.object_width - float(opening) + 2.0 * self.gravity_indent(R_world_from_local)
        return (closure - 2.0 * lam / self.stiffness) / lam

    def displacement(self, lam: float, R_world_from_local=None) -> np.ndarray:
        """Local-frame nodal displacement of one finger: contact load + self-weight sag."""
        u = self.gravity_displacement(R_world_from_local).copy()
        if lam > 0.0:
            res, _ = self.model.solve_normal_contact(self.contact_local, float(lam))
            u += res.u
        return u

    def can_hold(self, lam: float) -> bool:
        return float(lam) + 1e-12 >= self.hold_force
