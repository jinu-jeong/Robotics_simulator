"""Inverse problem of Milestone 3: known deformation + known contact location -> force.

Forward model (linear FEM, clamped root, single point contact):

    K_ff u_f = B_f(c) λ                         (scalar normal force λ along d = -n)
    K_ff u_f = B_xyz,f(c) F                     (full force vector F ∈ R^3)

Two least-squares formulations are implemented; both are linear and closed-form.

1. ``method="force"`` – residual in *force space* (the formulation of the project brief)

       λ̂ = argmin_{λ≥0} ‖B_f λ − K_ff u_f‖²  =  max(0, B_fᵀ K_ff u_f / B_fᵀ B_f)

   Needs the complete displacement field (``K u`` couples every node to its
   neighbours) and amplifies measurement noise: ``K`` maps small high-frequency
   errors in ``u`` to large nodal forces.

2. ``method="displacement"`` – residual in *displacement/observation space*

       g(c) = K_ff⁻¹ B_f(c)          influence field (one LU back-substitution per contact)
       λ̂    = argmin_{λ≥0} ‖H g λ − H u‖²  =  max(0, gᵀHᵀH u / gᵀHᵀH g)

   Works with any observation operator ``H`` (full field, surface nodes, camera
   markers, ...) and averages noise over all observed DOFs. The influence fields
   are cached per contact so evaluating many force levels is cheap.

The vector mode replaces the single column ``g`` by the 3-column ``G = K_ff⁻¹ B_xyz``
and solves an unconstrained 3x3 least-squares problem for ``F``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..fem.solver import LinearFEM
from .contact_mapping import PointContactMapping, SurfaceContact
from .observation import DofSelection, full_observation

Method = Literal["force", "displacement"]
Mode = Literal["normal", "vector"]


@dataclass
class ForceEstimate:
    """Result of one inverse solve. ``force_vector`` acts *on the finger* [N]."""

    force_vector: np.ndarray
    magnitude: float  # λ̂ in normal mode, |F̂| in vector mode
    method: str
    mode: str
    residual_rel: float  # ‖model − data‖ / ‖data‖ in the space of the residual
    clipped: bool = False  # non-negativity constraint became active
    extra: dict = field(default_factory=dict)

    @property
    def direction(self) -> np.ndarray:
        n = np.linalg.norm(self.force_vector)
        return self.force_vector / n if n > 0 else np.zeros(3)


def add_displacement_noise(u: np.ndarray, sigma: float, rng: np.random.Generator, free_dofs: np.ndarray | None = None) -> np.ndarray:
    """``u + ε`` with i.i.d. Gaussian ε ~ N(0, σ²) [m] on every (free) DOF."""
    u = np.asarray(u, dtype=float)
    flat = u.reshape(-1).copy()
    if sigma > 0.0:
        if free_dofs is None:
            flat += rng.normal(0.0, sigma, size=flat.shape)
        else:
            flat[free_dofs] += rng.normal(0.0, sigma, size=len(free_dofs))
    return flat.reshape(u.shape)


class InverseForceSolver:
    """Recovers the contact force from a displacement field for a *known* contact."""

    def __init__(self, fem: LinearFEM, contact: PointContactMapping, observation: DofSelection | None = None) -> None:
        self.fem = fem
        self.mapping = contact
        self.partition = fem.partition
        self.observation = observation or full_observation(self.partition)
        fem.factorize()
        self._cache: dict[tuple, np.ndarray] = {}
        # position of each observed dof inside the free-dof vector
        self._obs_in_free = np.searchsorted(self.partition.free_dofs, self.observation.dofs)
        if not np.array_equal(self.partition.free_dofs[self._obs_in_free], self.observation.dofs):
            raise ValueError("observation must select free DOFs only")

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _key(contact: SurfaceContact) -> tuple:
        return (int(contact.face_index), *np.round(contact.barycentric, 9).tolist())

    def influence_field(self, contact: SurfaceContact) -> np.ndarray:
        """``G = K_ff⁻¹ B_xyz,f(c)`` as a full ``(3N, 3)`` array (fixed DOFs zero). Cached."""
        key = self._key(contact)
        if key not in self._cache:
            Bxyz = self.mapping.vector_operator(contact)
            G = np.zeros((self.fem.n_dof, 3))
            for k in range(3):
                b = np.asarray(Bxyz[:, k].todense()).reshape(-1)
                G[self.partition.free_dofs, k] = self.fem.solve_free(b)
            self._cache[key] = G
        return self._cache[key]

    def observed_influence(self, contact: SurfaceContact) -> np.ndarray:
        """``H G`` (n_obs, 3)."""
        return self.observation.apply(self.influence_field(contact))

    def predicted_displacement(self, contact: SurfaceContact, force_vector) -> np.ndarray:
        """Forward model ``u = G F`` as (N, 3) – for viewer overlays of the estimate."""
        return (self.influence_field(contact) @ np.asarray(force_vector, float)).reshape(-1, 3)

    # ------------------------------------------------------------ estimators
    def estimate(self, u: np.ndarray, contact: SurfaceContact, method: Method = "displacement",
                 mode: Mode = "normal", nonneg: bool = True) -> ForceEstimate:
        if method == "displacement":
            return self._estimate_displacement_space(u, contact, mode, nonneg)
        if method == "force":
            return self._estimate_force_space(u, contact, mode, nonneg)
        raise ValueError(f"unknown method {method!r}")

    def _estimate_displacement_space(self, u, contact, mode, nonneg) -> ForceEstimate:
        y = self.observation.apply(u)
        HG = self.observed_influence(contact)
        d = -contact.normal
        if mode == "normal":
            g = HG @ d
            lam, clipped = _scalar_ls(g, y, nonneg)
            F = lam * d
            model = g * lam
        else:
            F, *_ = np.linalg.lstsq(HG, y, rcond=None)
            lam, clipped = float(np.linalg.norm(F)), False
            model = HG @ F
        res = np.linalg.norm(model - y) / max(np.linalg.norm(y), 1e-300)
        return ForceEstimate(F, float(lam), "displacement", mode, float(res), clipped,
                             extra={"observation": self.observation.name, "n_obs": self.observation.n_obs})

    def _estimate_force_space(self, u, contact, mode, nonneg) -> ForceEstimate:
        if self.observation.n_obs != self.partition.n_free:
            raise ValueError("the force-space residual needs the full displacement field (observation='full')")
        f_int = self.fem.internal_forces(u)[self.partition.free_dofs]  # (Tᵀ K u)_f: internal forces on free DOFs
        Bxyz = self.mapping.vector_operator(contact).tocsr()[self.partition.free_dofs]
        Bd = np.asarray(Bxyz.todense())  # (n_free, 3) – tiny number of nonzeros, dense is fine
        d = -contact.normal
        if mode == "normal":
            b = Bd @ d
            lam, clipped = _scalar_ls(b, f_int, nonneg)
            F = lam * d
            model = b * lam
        else:
            F, *_ = np.linalg.lstsq(Bd, f_int, rcond=None)
            lam, clipped = float(np.linalg.norm(F)), False
            model = Bd @ F
        # Residual is dominated by the *other* nodes where f_int should be ~0; report
        # it relative to the applied load so that exact data gives ~ discretisation noise.
        res = np.linalg.norm(model - f_int) / max(np.linalg.norm(model), 1e-300)
        return ForceEstimate(F, float(lam), "force", mode, float(res), clipped)


def _scalar_ls(g: np.ndarray, y: np.ndarray, nonneg: bool) -> tuple[float, bool]:
    gg = float(g @ g)
    if gg <= 0.0:
        return 0.0, False
    lam = float(g @ y) / gg
    if nonneg and lam < 0.0:
        return 0.0, True
    return lam, False


# ---------------------------------------------------------------- metrics
def force_error_metrics(true_mag: np.ndarray, est_mag: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / relative / max error and the y = x fit for scalar magnitudes."""
    t = np.asarray(true_mag, float).reshape(-1)
    e = np.asarray(est_mag, float).reshape(-1)
    err = e - t
    rel = np.abs(err) / np.maximum(np.abs(t), 1e-300)
    slope, intercept = np.polyfit(t, e, 1) if len(t) > 1 else (np.nan, np.nan)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((t - t.mean()) ** 2)) if len(t) > 1 else np.nan
    return {
        "n": int(len(t)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs_error": float(np.max(np.abs(err))),
        "mean_rel_error": float(np.mean(rel)),
        "max_rel_error": float(np.max(rel)),
        "bias": float(np.mean(err)),
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot else np.nan,
    }


def direction_error_deg(true_vec: np.ndarray, est_vec: np.ndarray) -> np.ndarray:
    """Angle between true and estimated force vectors [deg] (rows)."""
    t = np.asarray(true_vec, float).reshape(-1, 3)
    e = np.asarray(est_vec, float).reshape(-1, 3)
    tn = np.linalg.norm(t, axis=1)
    en = np.linalg.norm(e, axis=1)
    cos = np.einsum("ij,ij->i", t, e) / np.maximum(tn * en, 1e-300)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
