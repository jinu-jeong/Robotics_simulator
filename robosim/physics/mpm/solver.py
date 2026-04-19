"""Explicit MLS-MPM solver (Hu et al. 2018).

Single-pass P2G that folds the constitutive stress into the APIC affine
momentum. Per substep:

    1. Compute Kirchhoff stress τ_p from F_p.
    2. P2G: scatter mass and ``m·v + A·(x_i − x_p)`` with
       ``A = m·C − dt · (4/dx²) · V₀ · τ``.
    3. Normalise grid momentum to velocity.
    4. Apply external forces (gravity) and boundary conditions.
    5. G2P: gather velocity and recover the APIC affine matrix C.
    6. Update deformation gradient ``F ← (I + dt · C) · F``.
    7. Advect particles ``x ← x + dt · v``.

Stress is evaluated from the *current* F — i.e. the F at the start of the
step. This is the standard explicit "update-stress-first" scheme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from robosim.physics.mpm.boundary import apply_box_bc, apply_gravity
from robosim.physics.mpm.grid import Grid
from robosim.physics.mpm.kernel import (
    quadratic_bspline_weights,
    tensor_product_weights,
)
from robosim.physics.mpm.materials import NeoHookean
from robosim.physics.mpm.particles import Particles


@dataclass
class BoxBC:
    """Axis-aligned wall BC applied every substep."""
    lower: np.ndarray
    upper: np.ndarray
    mode: str = "slip"


@dataclass
class MPMSolver:
    particles: Particles
    grid: Grid
    material: NeoHookean = field(default_factory=NeoHookean)
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    bcs: list[BoxBC] = field(default_factory=list)
    colliders: list = field(default_factory=list)

    def step(self, dt: float) -> None:
        self._p2g_with_stress(dt)
        self._normalise_grid()
        apply_gravity(self.grid, dt, self.gravity)
        for bc in self.bcs:
            apply_box_bc(self.grid, bc.lower, bc.upper, mode=bc.mode)
        for col in self.colliders:
            col.apply(self.grid, dt)
        self._g2p(dt)
        self._update_F_and_advect(dt)

    # ── Internals ────────────────────────────────────────────────────────────

    def _p2g_with_stress(self, dt: float) -> None:
        p = self.particles
        g = self.grid
        dx = g.dx
        D_inv = 4.0 / (dx * dx)

        base, w, dw = quadratic_bspline_weights(p.x, dx=dx, origin=g.origin)
        W, _ = tensor_product_weights(w, dw)

        tau = self.material.kirchhoff_stress(p.F, p.d)        # (P, 3, 3)
        # A_p = m_p · C_p  −  dt · (4/dx²) · V₀_p · τ_p   — MLS-MPM affine.
        affine = (
            p.m[:, None, None] * p.C
            - (dt * D_inv) * p.V0[:, None, None] * tau
        )

        g.reset()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    idx = base + np.array([i, j, k], dtype=np.int64)
                    ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
                    W_ijk = W[:, i, j, k]

                    x_node = g.origin + idx * dx
                    dpos = x_node - p.x                       # (P, 3)

                    affine_dpos = np.einsum("pab,pb->pa", affine, dpos)
                    mom_contrib = W_ijk[:, None] * (
                        p.m[:, None] * p.v + affine_dpos
                    )
                    mass_contrib = W_ijk * p.m

                    np.add.at(g.m, (ix, iy, iz), mass_contrib)
                    np.add.at(g.v, (ix, iy, iz), mom_contrib)

    def _normalise_grid(self) -> None:
        g = self.grid
        mask = g.m > 0.0
        g.v[mask] /= g.m[mask][:, None]

    def _g2p(self, dt: float) -> None:
        p = self.particles
        g = self.grid
        dx = g.dx
        D_inv = 4.0 / (dx * dx)

        base, w, dw = quadratic_bspline_weights(p.x, dx=dx, origin=g.origin)
        W, _ = tensor_product_weights(w, dw)

        v_new = np.zeros_like(p.v)
        C_new = np.zeros_like(p.C)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    idx = base + np.array([i, j, k], dtype=np.int64)
                    ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
                    W_ijk = W[:, i, j, k]

                    v_i = g.v[ix, iy, iz]
                    x_node = g.origin + idx * dx
                    dpos = x_node - p.x

                    v_new += W_ijk[:, None] * v_i
                    C_new += (
                        D_inv
                        * W_ijk[:, None, None]
                        * np.einsum("pa,pb->pab", v_i, dpos)
                    )

        p.v = v_new
        p.C = C_new

    def _update_F_and_advect(self, dt: float) -> None:
        p = self.particles
        I3 = np.eye(3)
        F_trial = np.einsum(
            "pab,pbc->pac",
            I3 + dt * p.C,
            p.F,
        )
        # Plastic materials project the trial elastic gradient back to the
        # yield surface; elastic materials just keep F_trial.
        project = getattr(self.material, "project", None)
        p.F = project(F_trial) if project is not None else F_trial
        update_damage = getattr(self.material, "update_damage", None)
        if update_damage is not None:
            p.d = update_damage(p.F, p.d)
        p.x = p.x + dt * p.v


# ── Utility constructors ─────────────────────────────────────────────────────

def sample_box_particles(
    lower: Sequence[float],
    upper: Sequence[float],
    n_per_axis: int,
    density: float,
    jitter: float = 0.0,
    seed: int | None = None,
) -> Particles:
    """Seed a jelly-cube worth of MPM particles inside an AABB.

    ``n_per_axis`` lattice points per axis → ``n_per_axis³`` particles, each
    carrying an equal share of the box mass and reference volume.
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    xs = np.linspace(lower[0], upper[0], n_per_axis)
    ys = np.linspace(lower[1], upper[1], n_per_axis)
    zs = np.linspace(lower[2], upper[2], n_per_axis)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        spacing = (upper - lower) / max(n_per_axis - 1, 1)
        x = x + jitter * spacing * (rng.random(x.shape) - 0.5)

    P = x.shape[0]
    total_volume = float(np.prod(upper - lower))
    V0_p = total_volume / P
    total_mass = density * total_volume
    m_p = total_mass / P

    return Particles(
        x=x,
        v=np.zeros_like(x),
        m=np.full(P, m_p),
        V0=np.full(P, V0_p),
    )
