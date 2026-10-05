"""Galerkin-reduced linear mechanics on a POD basis.

    u ≈ Φ q,    K_r = Φᵀ K Φ  (r×r, SPD),    B_r(c) = Φᵀ B(c)  (r×1 or r×3)
    forward :  K_r q = B_r λ         ->  q = K_r⁻¹ B_r λ,  u_rom = Φ q
    inverse :  given q̂ (projection of a measured field, or – later – the output
               of a vision model) recover λ for a known contact c:

        displacement-space   g_r = K_r⁻¹ B_r d,   λ̂ = max(0, g_rᵀ q̂ / g_rᵀ g_r)
        force-space          f_r = K_r q̂,          λ̂ = max(0, B_rᵀ f_r / B_rᵀ B_r)

Because Φ has orthonormal columns, least squares in q equals least squares on
Φq in the full space, so the displacement-space estimator is the ROM version
of Milestone 3's. The force-space estimator lives in the r-dimensional
subspace: high-frequency noise components orthogonal to Φ are discarded by
the projection, which tames the noise amplification of ``K``.

``K_r`` is built once for the largest basis; :meth:`truncate` reuses its leading
block, so sweeping ``r`` is free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contact.contact_mapping import PointContactMapping, SurfaceContact
from ..contact.inverse_force import ForceEstimate, _scalar_ls
from ..fem.solver import LinearFEM
from .pod import PODBasis


@dataclass
class ReducedModel:
    basis: PODBasis
    K_r: np.ndarray  # (r, r)
    mapping: PointContactMapping
    fem: LinearFEM | None = None

    # ------------------------------------------------------------ build
    @classmethod
    def from_fem(cls, fem: LinearFEM, basis: PODBasis, mapping: PointContactMapping) -> "ReducedModel":
        if basis.n_dof != fem.n_dof:
            raise ValueError("basis / FEM DOF mismatch")
        KPhi = fem.K @ basis.Phi  # (3N, r) sparse @ dense
        K_r = basis.Phi.T @ KPhi
        K_r = 0.5 * (K_r + K_r.T)
        return cls(basis, np.ascontiguousarray(K_r), mapping, fem)

    def truncate(self, r: int) -> "ReducedModel":
        return ReducedModel(self.basis.truncate(r), np.ascontiguousarray(self.K_r[:r, :r]), self.mapping, self.fem)

    @property
    def r(self) -> int:
        return self.basis.r

    # ------------------------------------------------------------ operators
    def B_r(self, contact: SurfaceContact, direction) -> np.ndarray:
        """``Φᵀ B(c)`` for a unit force along ``direction`` -> (r,)."""
        B = self.mapping.operator(contact, direction)
        return np.asarray(B.T @ self.basis.Phi).reshape(-1)

    def B_r_xyz(self, contact: SurfaceContact) -> np.ndarray:
        """``Φᵀ B_xyz(c)`` -> (r, 3)."""
        B = self.mapping.vector_operator(contact)
        return np.asarray((B.T @ self.basis.Phi).T)

    # ------------------------------------------------------------ forward
    def solve_q(self, contact: SurfaceContact, force_vector) -> np.ndarray:
        F = np.asarray(force_vector, float).reshape(3)
        return np.linalg.solve(self.K_r, self.B_r_xyz(contact) @ F)

    def solve(self, contact: SurfaceContact, force_vector) -> np.ndarray:
        """Reduced forward solve -> (N, 3) displacement."""
        return self.basis.reconstruct(self.solve_q(contact, force_vector))

    # ------------------------------------------------------------ inverse
    def estimate_force(self, q: np.ndarray, contact: SurfaceContact, method: str = "displacement",
                       mode: str = "normal", nonneg: bool = True) -> ForceEstimate:
        q = np.asarray(q, float).reshape(-1)
        if q.shape[0] != self.r:
            raise ValueError(f"q must have length r = {self.r}")
        d = -contact.normal
        Bx = self.B_r_xyz(contact)  # (r, 3)
        if method == "displacement":
            G = np.linalg.solve(self.K_r, Bx)  # (r, 3) reduced influence fields
            target = q
        elif method == "force":
            G = Bx
            target = self.K_r @ q  # reduced internal forces
        else:
            raise ValueError(f"unknown method {method!r}")
        if mode == "normal":
            g = G @ d
            lam, clipped = _scalar_ls(g, target, nonneg)
            F = lam * d
            model = g * lam
        elif mode == "vector":
            F, *_ = np.linalg.lstsq(G, target, rcond=None)
            lam, clipped = float(np.linalg.norm(F)), False
            model = G @ F
        else:
            raise ValueError(f"unknown mode {mode!r}")
        res = np.linalg.norm(model - target) / max(np.linalg.norm(target), 1e-300)
        return ForceEstimate(F, float(lam), f"rom-{method}", mode, float(res), clipped, extra={"r": self.r})

    def estimate_force_from_field(self, u: np.ndarray, contact: SurfaceContact, **kw) -> ForceEstimate:
        """Convenience: project a full field (q = Φᵀu) and estimate."""
        return self.estimate_force(self.basis.project(u), contact, **kw)
