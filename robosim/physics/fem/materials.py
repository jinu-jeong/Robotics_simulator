"""Constitutive models for FEM elements.

Implements:
- Corotational linear elasticity (handles large rotations, linear in rotated frame)
- Neo-Hookean hyperelasticity (fully nonlinear, large deformations)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robosim.physics.fem.elements import polar_decomposition


@dataclass
class CorotationalElastic:
    """Corotational linear elasticity.

    Extracts rotation via polar decomposition F = R @ S, then computes
    stress in the rotated (corotational) frame using linear elasticity:
    P = R @ (2*mu*(S - I) + lambda*tr(S - I)*I)

    This handles large rotations correctly while remaining linear in strain.

    Parameters
    ----------
    young : Young's modulus (Pa)
    poisson : Poisson's ratio (dimensionless, 0 < nu < 0.5)
    """

    young: float = 1e6
    poisson: float = 0.3

    @property
    def mu(self) -> float:
        """Shear modulus (Lamé's second parameter)."""
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def lam(self) -> float:
        """Lamé's first parameter."""
        return self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))

    def compute_stress(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute first Piola-Kirchhoff stress P and rotation R.

        Parameters
        ----------
        F : (3,3) deformation gradient

        Returns
        -------
        P : (3,3) first Piola-Kirchhoff stress
        R : (3,3) rotation from polar decomposition
        """
        R, S = polar_decomposition(F)
        mu = self.mu
        lam = self.lam

        # Strain in rotated frame: epsilon = S - I
        eps = S - np.eye(3)
        trace_eps = np.trace(eps)

        # Cauchy-like stress in rotated frame (linear elasticity)
        T = 2.0 * mu * eps + lam * trace_eps * np.eye(3)

        # First Piola-Kirchhoff: P = R @ T
        P = R @ T
        return P, R

    def compute_element_stiffness(self, F: np.ndarray, dN: np.ndarray,
                                    vol: float) -> np.ndarray:
        """Compute 12x12 element stiffness matrix.

        Uses the corotational approximation: K_e ≈ R @ K_linear @ R^T
        applied to each node pair block.

        Parameters
        ----------
        F : (3,3) deformation gradient
        dN : (4,3) shape function gradients (reference config)
        vol : element volume

        Returns
        -------
        Ke : (12, 12) element stiffness matrix
        """
        R, S = polar_decomposition(F)
        mu = self.mu
        lam = self.lam

        # Linear stiffness in material frame
        Ke = np.zeros((12, 12))
        for a in range(4):
            for b in range(4):
                # K_ab = vol * (mu * (dN_a . dN_b) * I + mu * dN_b (outer) dN_a
                #               + lam * dN_a (outer) dN_b)
                dNa = dN[a]  # (3,)
                dNb = dN[b]  # (3,)

                K_block = (
                    mu * np.dot(dNa, dNb) * np.eye(3)
                    + mu * np.outer(dNb, dNa)
                    + lam * np.outer(dNa, dNb)
                )
                K_block *= vol

                # Rotate to world frame
                K_block = R @ K_block @ R.T

                i0, i1 = a * 3, a * 3 + 3
                j0, j1 = b * 3, b * 3 + 3
                Ke[i0:i1, j0:j1] = K_block

        return Ke


@dataclass
class CorotationalPlastic:
    """Corotational small-strain J2 (von Mises) plasticity.

    Same elastic skeleton as :class:`CorotationalElastic` (rotation
    extracted via polar decomposition, linear elasticity in the corotated
    frame), with a textbook Simo & Hughes radial-return mapping bolted on.

    Per element the body must persist a 3×3 symmetric *plastic strain*
    tensor ``eps_p``; the constitutive update reads it, projects the
    trial stress to the yield surface if needed, and returns the new
    ``eps_p`` for the body to write back. Fully elastic step ⇒
    ``eps_p`` unchanged ⇒ behaviour identical to ``CorotationalElastic``.

    Yield surface (Mises): ``‖dev(σ_trial)‖_F ≤ √(2/3) σ_Y_eff`` where
    ``σ_Y_eff = σ_Y + H · ε_p_eq`` with linear isotropic hardening
    modulus ``H`` (default 0 = perfect plasticity).

    Parameters
    ----------
    young, poisson : same as :class:`CorotationalElastic`.
    yield_stress   : initial yield stress σ_Y (Pa).
    hardening      : linear isotropic hardening modulus H (Pa).
                     0 → perfect plasticity (yield surface fixed).
    """

    young: float = 1e6
    poisson: float = 0.3
    yield_stress: float = 1e4
    hardening: float = 0.0

    @property
    def mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def lam(self) -> float:
        return self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))

    def compute_stress(
        self, F: np.ndarray, eps_p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single-element stress + plastic-strain update.

        Returns ``(P, R, eps_p_new)``: first Piola-Kirchhoff stress,
        rotation from polar decomposition, and the updated plastic strain
        (which the caller writes back to its per-element store).
        """
        R, S = polar_decomposition(F)
        I = np.eye(3)
        # Trial elastic strain in corotated frame.
        eps_e_trial = 0.5 * (S + S.T) - I - eps_p
        tr_e = np.trace(eps_e_trial)
        sigma_trial = 2.0 * self.mu * eps_e_trial + self.lam * tr_e * I

        # Deviatoric part: (1/3) tr-removed.
        dev = sigma_trial - (np.trace(sigma_trial) / 3.0) * I
        norm_dev = np.linalg.norm(dev)
        # Equivalent accumulated plastic strain (∝ ‖eps_p‖_F under J2).
        eps_p_eq = np.sqrt(2.0 / 3.0) * np.linalg.norm(eps_p)
        sigma_Y_eff = self.yield_stress + self.hardening * eps_p_eq
        f = norm_dev - np.sqrt(2.0 / 3.0) * sigma_Y_eff

        sigma_new = sigma_trial
        eps_p_new = eps_p
        if f > 0.0 and norm_dev > 1e-16:
            # Closed-form radial return for J2 (linear isotropic hardening).
            dgamma = f / (2.0 * self.mu + 2.0 * self.hardening / 3.0)
            n = dev / norm_dev
            sigma_new = sigma_trial - 2.0 * self.mu * dgamma * n
            eps_p_new = eps_p + dgamma * n

        P = R @ sigma_new
        return P, R, eps_p_new


@dataclass
class NeoHookean:
    """Neo-Hookean hyperelasticity.

    Strain energy density:
    W = (mu/2)(I1 - 3) - mu*ln(J) + (lambda/2)(ln J)^2

    where I1 = tr(F^T F), J = det(F).

    Parameters
    ----------
    young : Young's modulus (Pa)
    poisson : Poisson's ratio
    """

    young: float = 1e6
    poisson: float = 0.3

    @property
    def mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def lam(self) -> float:
        return self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))

    def compute_stress(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute first Piola-Kirchhoff stress.

        P = mu * (F - F^{-T}) + lambda * ln(J) * F^{-T}

        Returns
        -------
        P : (3,3) first Piola-Kirchhoff stress
        R : (3,3) rotation (from polar decomposition, for visualization)
        """
        mu = self.mu
        lam = self.lam

        J = np.linalg.det(F)
        if J < 1e-10:
            J = 1e-10  # prevent log(0)

        F_inv_T = np.linalg.inv(F).T
        P = mu * (F - F_inv_T) + lam * np.log(J) * F_inv_T

        R, _ = polar_decomposition(F)
        return P, R
