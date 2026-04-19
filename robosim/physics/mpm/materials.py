"""Constitutive models for MPM particles.

Elastic models return the Kirchhoff-style stress ``τ = P · F^T`` (with
``P`` the first Piola-Kirchhoff tensor), which is the quantity MLS-MPM
consumes when folding stress into the APIC affine momentum.

Plastic models additionally expose ``project(F_trial) → F_E`` — a return
mapping from the trial elastic deformation gradient to the closest state on
the yield surface. The solver treats ``particles.F`` as the *elastic* part
``F_E`` of the multiplicative split ``F = F_E · F_P``; plastic history is
encoded implicitly by the accumulated projections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _kirchhoff_neo_hookean(
    F: np.ndarray, mu: float, lam: float, d: np.ndarray | None = None
) -> np.ndarray:
    """τ = μ (F F^T − I) + λ · log(J) · I for a batch of F's.

    When ``d`` (per-particle damage ∈ [0, 1]) is provided, the stress is
    degraded by ``(1 − d)²`` — a standard quasi-brittle softening
    (Miehe et al. 2010). A fully damaged particle (d=1) contributes no
    stress, which is what lets an MPM blob separate into fragments.
    """
    J = np.linalg.det(F)
    FFt = np.einsum("pab,pcb->pac", F, F)
    I3 = np.eye(3)
    tau = mu * (FFt - I3) + (lam * np.log(J))[:, None, None] * I3
    if d is not None:
        tau = ((1.0 - d) ** 2)[:, None, None] * tau
    return tau


@dataclass
class NeoHookean:
    """Compressible Neo-Hookean hyperelasticity.

    Energy density::

        ψ(F) = (μ/2)(I_C − 3) − μ · log(J) + (λ/2)(log(J))²

    with ``I_C = tr(F F^T)`` and ``J = det(F)``. The first Piola-Kirchhoff
    stress is ``P = μ(F − F^{-T}) + λ · log(J) · F^{-T}``, which yields the
    closed-form Kirchhoff stress

        τ = P · F^T = μ (F F^T − I) + λ · log(J) · I.
    """

    young: float = 1e5
    poisson: float = 0.3

    @property
    def mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def lam(self) -> float:
        return self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))

    def kirchhoff_stress(self, F: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
        return _kirchhoff_neo_hookean(F, self.mu, self.lam, d)


@dataclass
class VonMisesPlastic:
    """Neo-Hookean elasticity + rate-independent J2 (Von Mises) plasticity.

    Return mapping is performed in Hencky (logarithmic) principal strain
    space on the SVD ``F_trial = U Σ V^T``:

        ε = log(Σ),    ε_dev = ε − (1/3) tr(ε) · I

    The Von Mises yield condition reads (in principal stress space)

        √(6) · μ · ‖ε_dev‖ ≤ σ_Y.

    When exceeded, ``ε_dev`` is rescaled to the yield surface (radial return);
    the volumetric part ``tr(ε)`` is untouched so the flow is isochoric —
    plastic deformation preserves volume, matching the behaviour of ductile
    metals.
    """

    young: float = 1e5
    poisson: float = 0.3
    yield_stress: float = 1e4

    @property
    def mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def lam(self) -> float:
        return self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))

    def kirchhoff_stress(self, F_E: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
        return _kirchhoff_neo_hookean(F_E, self.mu, self.lam, d)

    def project(self, F_trial: np.ndarray) -> np.ndarray:
        """Radial-return projection to the Von Mises yield surface."""
        U, sigma, Vt = np.linalg.svd(F_trial)                # batched SVD
        eps = np.log(np.maximum(sigma, 1e-12))               # (P, 3)
        eps_mean = eps.mean(axis=-1, keepdims=True)          # (P, 1)
        eps_dev = eps - eps_mean

        norm_dev = np.linalg.norm(eps_dev, axis=-1)          # (P,)
        yield_strain = self.yield_stress / (np.sqrt(6.0) * self.mu)

        # Scale deviatoric strains back onto the yield surface where needed.
        scale = np.where(
            norm_dev > yield_strain,
            yield_strain / np.maximum(norm_dev, 1e-16),
            1.0,
        )                                                    # (P,)
        eps_dev_new = eps_dev * scale[:, None]
        eps_new = eps_dev_new + eps_mean
        sigma_new = np.exp(eps_new)                          # (P, 3)

        # F_E_new = U · diag(σ_new) · V^T
        return np.einsum("pab,pb,pbc->pac", U, sigma_new, Vt)


@dataclass
class DruckerPragerPlastic:
    """Neo-Hookean elasticity + Drucker-Prager plasticity (Klár et al. 2016).

    Models granular/cohesion-less materials (sand, dry soil) or, with small
    friction, something that flows like wet clay. Return mapping uses the
    same Hencky-strain SVD approach as :class:`VonMisesPlastic`, with three
    cases:

    * **I (tension)** — ``tr(ε) > 0``: grains separate. ``F_E`` collapses
      to ``U V^T`` (rotation only; no stored elastic deformation).
    * **II (compression, on cone)** — radial return onto the Drucker-Prager
      cone in Hencky space.
    * **III (elastic core)** — inside the cone, leave ``F_E`` unchanged.

    ``friction_angle`` is the Coulomb internal friction angle ``φ`` in
    radians. The Drucker-Prager slope is
    ``α = √(2/3) · 2 sin φ / (3 − sin φ)``.
    """

    young: float = 1e5
    poisson: float = 0.3
    friction_angle: float = np.deg2rad(30.0)

    @property
    def mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def lam(self) -> float:
        return self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))

    @property
    def alpha(self) -> float:
        s = np.sin(self.friction_angle)
        return np.sqrt(2.0 / 3.0) * 2.0 * s / (3.0 - s)

    def kirchhoff_stress(self, F_E: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
        return _kirchhoff_neo_hookean(F_E, self.mu, self.lam, d)

    def project(self, F_trial: np.ndarray) -> np.ndarray:
        U, sigma, Vt = np.linalg.svd(F_trial)
        eps = np.log(np.maximum(sigma, 1e-12))                  # (P, 3)
        trace = eps.sum(axis=-1, keepdims=True)                 # (P, 1)
        eps_hat = eps - trace / 3.0                             # (P, 3) deviatoric
        norm_hat = np.linalg.norm(eps_hat, axis=-1)             # (P,)

        coef = (3.0 * self.lam + 2.0 * self.mu) / (2.0 * self.mu)
        dgamma = norm_hat + coef * trace[:, 0] * self.alpha      # (P,)

        # Case I: tension ⇒ F_E := U V^T  ⇒  ε_new = 0.
        mask_tens = trace[:, 0] > 0.0
        # Case II: compression, outside cone ⇒ radial return in dev direction.
        mask_cone = (~mask_tens) & (dgamma > 0.0)
        # Case III: inside cone ⇒ leave ε unchanged.

        safe_norm = np.where(norm_hat > 1e-16, norm_hat, 1.0)
        scale = dgamma / safe_norm                              # (P,)

        eps_new = eps.copy()
        eps_new[mask_tens] = 0.0
        # eps - (dgamma / ||ε̂||) · ε̂  , applied only where mask_cone is True.
        returned = eps - scale[:, None] * eps_hat
        eps_new[mask_cone] = returned[mask_cone]

        sigma_new = np.exp(eps_new)
        return np.einsum("pab,pb,pbc->pac", U, sigma_new, Vt)


@dataclass
class DamagedNeoHookean:
    """Neo-Hookean elasticity with monotonic local tensile damage.

    Each particle carries a damage scalar ``d ∈ [0, 1]`` that degrades the
    stress by ``(1 − d)²``. Damage grows irreversibly when the maximum
    principal stretch ``λ_max = max σᵢ(F)`` exceeds a critical stretch
    ``stretch_c`` (e.g. 1.3 for a 30 % tensile limit), following

        d_new = max(d, 1 − (stretch_c / λ_max)^softening).

    Once ``d → 1``, the particle is effectively "broken" — it still carries
    momentum and mass, but contributes no stress, so the grid loses the
    bridge that held its neighbours together. That is how an MPM blob
    tears apart without any explicit crack tracking (CD-MPM, Wolper 2019).
    """

    young: float = 1e5
    poisson: float = 0.3
    stretch_c: float = 1.25          # critical principal stretch
    softening: float = 2.0           # exponent controlling post-peak rate

    @property
    def mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def lam(self) -> float:
        return self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))

    def kirchhoff_stress(self, F: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
        return _kirchhoff_neo_hookean(F, self.mu, self.lam, d)

    def update_damage(self, F: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Grow per-particle damage from the current F. Monotonic."""
        sigma = np.linalg.svd(F, compute_uv=False)               # (P, 3)
        lam_max = sigma.max(axis=-1)                             # (P,)
        ratio = np.maximum(lam_max / self.stretch_c, 1.0)        # (P,)
        # ratio == 1 inside the elastic envelope → d_trial = 0.
        d_trial = 1.0 - ratio ** (-self.softening)
        return np.minimum(np.maximum(d, d_trial), 1.0)
