"""Analytical cantilever references used for FEM verification.

Rectangular cross-section ``W x H`` (width along y, thickness along z),
length ``L`` along x, clamped at x = 0, transverse tip force ``F`` along the
bending ``axis`` – ``"z"`` (top-face press, thickness H) or ``"y"`` (lateral /
inner-face press as in the grasp, thickness W).

Euler–Bernoulli (slender beam):
    I      = W H^3 / 12   (axis z)      H W^3 / 12   (axis y)
    δ_EB   = F L^3 / (3 E I)
    w(x)   = F x^2 (3L - x) / (6 E I)

Timoshenko (adds transverse shear deformation, relevant for L/H ~ 10):
    δ_T    = δ_EB + F L / (κ G A),   A = W H,   G = E / (2(1+ν))
    κ      = 10 (1+ν) / (12 + 11 ν)          (Cowper, rectangular section)

Both neglect the extra stiffness of a fully clamped root (suppressed Poisson
contraction) and any local deformation under the load.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CantileverReference:
    length: float
    width: float
    height: float
    youngs_modulus: float
    poisson_ratio: float
    force: float  # transverse tip force magnitude [N] (positive)
    axis: str = "z"  # bending direction: "z" (thickness H) or "y" (lateral, thickness W)

    def __post_init__(self) -> None:
        if self.axis not in ("y", "z"):
            raise ValueError(f"axis must be 'y' or 'z', got {self.axis!r}")

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def bending_thickness(self) -> float:
        """Section dimension along the load direction (sets I and the slenderness)."""
        return self.height if self.axis == "z" else self.width

    @property
    def second_moment(self) -> float:
        other = self.width if self.axis == "z" else self.height
        return other * self.bending_thickness**3 / 12.0

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def cowper_kappa(self) -> float:
        nu = self.poisson_ratio
        return 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)

    @property
    def tip_deflection_euler_bernoulli(self) -> float:
        return self.force * self.length**3 / (3.0 * self.youngs_modulus * self.second_moment)

    @property
    def tip_deflection_timoshenko(self) -> float:
        shear = self.force * self.length / (self.cowper_kappa * self.shear_modulus * self.area)
        return self.tip_deflection_euler_bernoulli + shear

    def deflection_curve(self, x: np.ndarray) -> np.ndarray:
        """Euler–Bernoulli deflection magnitude w(x) along the beam."""
        x = np.asarray(x, dtype=float)
        EI = self.youngs_modulus * self.second_moment
        return self.force * x**2 * (3.0 * self.length - x) / (6.0 * EI)

    def summary(self) -> dict[str, float]:
        return {
            "I": self.second_moment,
            "delta_euler_bernoulli": self.tip_deflection_euler_bernoulli,
            "delta_timoshenko": self.tip_deflection_timoshenko,
            "shear_fraction": self.tip_deflection_timoshenko / self.tip_deflection_euler_bernoulli - 1.0,
            "slenderness_L_over_H": self.length / self.bending_thickness,
            "bending_axis": self.axis,
        }


def rectangular_torsion_constant(a: float, b: float) -> float:
    """Saint-Venant torsion constant J of a solid rectangle with sides a >= b.

    Roark's approximation ``J = a b^3 [1/3 - 0.21 (b/a) (1 - b^4 / (12 a^4))]``
    (error < 0.5 % for all aspect ratios). For a square: 0.1406 a^4; for
    a/b = 2: 0.229 a b^3.
    """
    a, b = max(a, b), min(a, b)
    return a * b**3 * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - b**4 / (12.0 * a**4)))


@dataclass(frozen=True)
class PointLoadCantilever:
    """Cantilever with a transverse point load at ``x = a`` and lateral eccentricity ``e``.

    Geometry / material as :class:`CantileverReference`; the load is
    ``F`` (signed, along +z; a pressing contact on the top surface has F < 0)
    applied at ``(a, W/2 + e, H)``.

    Bending (Euler–Bernoulli, x <= a and x > a branches):
        w(x) = F x^2 (3a - x) / (6EI)             x <= a
        w(x) = F a^2 (3x - a) / (6EI)             x >  a
    Timoshenko shear addition:
        w_s(x) = F min(x, a) / (κ G A)
    Saint-Venant torsion from the torque T_x = F e about the beam axis:
        θ(x) = T_x min(x, a) / (G J),   u_z(y) = θ (y - W/2),   u_y(z) = -θ (z - H/2)

    Kinematic displacement field (used as "beam theory" reference in the viewer):
        u_x = -(z - H/2) w'(x)          (bending, plane sections)
        u_y = -θ(x) (z - H/2)
        u_z =  w(x) + θ(x) (y - W/2)
    Warping and the clamped-root Poisson effect are neglected.
    """

    length: float
    width: float
    height: float
    youngs_modulus: float
    poisson_ratio: float
    force_z: float  # signed [N]
    a: float  # load position along x [m]
    e: float = 0.0  # lateral eccentricity y - W/2 [m]

    @property
    def EI(self) -> float:
        return self.youngs_modulus * self.width * self.height**3 / 12.0

    @property
    def G(self) -> float:
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def kappa(self) -> float:
        nu = self.poisson_ratio
        return 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)

    @property
    def J(self) -> float:
        return rectangular_torsion_constant(self.width, self.height)

    @property
    def torque(self) -> float:
        return self.force_z * self.e  # T_x = (r x F)_x = r_y F_z - r_z F_y with r = (0, e, 0), F = (0, 0, F_z)

    def bending_deflection(self, x, shear: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        F, a, EI = self.force_z, self.a, self.EI
        w = np.where(x <= a, F * x**2 * (3 * a - x) / (6 * EI), F * a**2 * (3 * x - a) / (6 * EI))
        if shear:
            w = w + F * np.minimum(x, a) / (self.kappa * self.G * self.width * self.height)
        return w

    def bending_slope(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        F, a, EI = self.force_z, self.a, self.EI
        return np.where(x <= a, F * x * (2 * a - x) / (2 * EI), F * a**2 / (2 * EI))

    def twist(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.torque * np.minimum(x, self.a) / (self.G * self.J)

    def tip_deflection(self, shear: bool = True) -> float:
        return float(self.bending_deflection(self.length, shear))

    def tip_twist(self) -> float:
        return float(self.twist(self.length))

    def displacement_field(self, nodes: np.ndarray, shear: bool = True) -> np.ndarray:
        """Kinematic beam displacement (N, 3) evaluated at FEM nodes."""
        x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
        yc, zc = 0.5 * self.width, 0.5 * self.height
        w = self.bending_deflection(x, shear)
        th = self.twist(x)
        u = np.zeros_like(nodes, dtype=float)
        u[:, 0] = -(z - zc) * self.bending_slope(x)
        u[:, 1] = -th * (z - zc)
        u[:, 2] = w + th * (y - yc)
        return u


def section_fit(nodes: np.ndarray, u: np.ndarray, width: float, height: float, tol: float = 1e-9):
    """Per x-station least-squares fit ``u_z = c0 + c1 (y - W/2) + c2 (z - H/2)``.

    Returns ``(x_stations, mean_deflection c0, twist c1)``. ``c1`` is the
    section rotation about the beam axis, comparable to Saint-Venant θ(x).
    """
    xs = np.unique(np.round(nodes[:, 0], 12))
    c0 = np.zeros(len(xs))
    c1 = np.zeros(len(xs))
    for i, xv in enumerate(xs):
        m = np.abs(nodes[:, 0] - xv) <= tol
        A = np.stack([np.ones(m.sum()), nodes[m, 1] - 0.5 * width, nodes[m, 2] - 0.5 * height], axis=1)
        coef, *_ = np.linalg.lstsq(A, u[m, 2], rcond=None)
        c0[i], c1[i] = coef[0], coef[1]
    return xs, c0, c1
