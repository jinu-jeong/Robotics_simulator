"""Isotropic linear-elastic material (small strain).

Voigt notation used throughout the FEM package
-----------------------------------------------
strain  ε = [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz]   with engineering shear γ_ij = 2 ε_ij
stress  σ = [σ_xx, σ_yy, σ_zz, τ_xy,  τ_yz,  τ_xz]

σ = C ε with

    C = | λ+2μ   λ     λ    0  0  0 |
        |  λ    λ+2μ   λ    0  0  0 |
        |  λ     λ    λ+2μ  0  0  0 |
        |  0     0     0    μ  0  0 |
        |  0     0     0    0  μ  0 |
        |  0     0     0    0  0  μ |

λ = E ν / ((1+ν)(1-2ν)),  μ = G = E / (2(1+ν)).  Units: Pa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class LinearElasticMaterial:
    youngs_modulus: float  # E [Pa]
    poisson_ratio: float  # ν [-]
    density: float = 0.0  # ρ [kg/m^3] (self-weight body force; inertia is not modelled)
    name: str = "linear elastic"

    def __post_init__(self) -> None:
        if self.youngs_modulus <= 0:
            raise ValueError("Young's modulus must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("Poisson ratio must be in (-1, 0.5) for a stable isotropic solid")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "LinearElasticMaterial":
        m = cfg.get("material", cfg)
        return cls(
            youngs_modulus=float(m["youngs_modulus"]),
            poisson_ratio=float(m["poisson_ratio"]),
            density=float(m.get("density", 0.0)),
            name=str(m.get("name", "linear elastic")),
        )

    # ------------------------------------------------------------- moduli
    @property
    def E(self) -> float:
        return self.youngs_modulus

    @property
    def nu(self) -> float:
        return self.poisson_ratio

    @property
    def lame_lambda(self) -> float:
        E, nu = self.E, self.nu
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def lame_mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def shear_modulus(self) -> float:
        return self.lame_mu

    @property
    def bulk_modulus(self) -> float:
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    def stiffness_matrix(self) -> np.ndarray:
        """6x6 Voigt elasticity matrix C (engineering shear convention)."""
        lam, mu = self.lame_lambda, self.lame_mu
        C = np.zeros((6, 6))
        C[:3, :3] = lam
        C[np.arange(3), np.arange(3)] = lam + 2.0 * mu
        C[3:, 3:] = np.eye(3) * mu
        return C

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "youngs_modulus": self.youngs_modulus,
            "poisson_ratio": self.poisson_ratio,
            "density": self.density,
        }
