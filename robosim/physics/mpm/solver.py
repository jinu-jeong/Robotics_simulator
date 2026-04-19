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

        base, w, _ = quadratic_bspline_weights(p.x, dx=dx, origin=g.origin)
        W, _ = tensor_product_weights(w)                       # (P, 3, 3, 3)

        tau = self.material.kirchhoff_stress(p.F, p.d)         # (P, 3, 3)
        # A_p = m_p · C_p  −  dt · (4/dx²) · V₀_p · τ_p — MLS-MPM affine.
        affine = (
            p.m[:, None, None] * p.C
            - (dt * D_inv) * p.V0[:, None, None] * tau
        )

        nx, ny, nz = g.shape
        n_cells = nx * ny * nz
        P = p.x.shape[0]

        # Build the full 27-stencil contribution in one go. Flat shape is
        # (P * 27,) — one row per (particle, stencil-offset) pair.
        # np.bincount is ~10× faster than 27 × np.add.at calls.
        offsets = np.mgrid[0:3, 0:3, 0:3].reshape(3, -1).T     # (27, 3)
        idx_all = base[:, None, :] + offsets[None, :, :]       # (P, 27, 3)
        idx_flat = idx_all.reshape(-1, 3)                      # (P*27, 3)
        flat_lin = (idx_flat[:, 0] * (ny * nz)
                    + idx_flat[:, 1] * nz
                    + idx_flat[:, 2])                          # (P*27,)

        W_flat = W.reshape(P, -1)                              # (P, 27)
        W_rep = W_flat.ravel()                                 # (P*27,)

        # Mass contribution:   m_p * W_ijk
        m_rep = np.repeat(p.m, 27) * W_rep                     # (P*27,)

        # Momentum contribution:   W_ijk * (m_p * v_p + affine_p · dpos_p,ijk)
        #   dpos[p, 27, 3] = origin + idx*dx − x_p
        x_nodes = g.origin + idx_all * dx                      # (P, 27, 3)
        dpos = x_nodes - p.x[:, None, :]                       # (P, 27, 3)
        affine_dpos = np.einsum("pab,peb->pea", affine, dpos)  # (P, 27, 3)
        mom = (p.m[:, None, None] * p.v[:, None, :]
               + affine_dpos) * W_flat[:, :, None]             # (P, 27, 3)
        mom_flat = mom.reshape(-1, 3)                          # (P*27, 3)

        g.reset()
        g.m.reshape(-1)[:] = np.bincount(
            flat_lin, weights=m_rep, minlength=n_cells,
        )
        gv_flat = g.v.reshape(-1, 3)
        for d in range(3):
            gv_flat[:, d] = np.bincount(
                flat_lin, weights=mom_flat[:, d], minlength=n_cells,
            )

    def _normalise_grid(self) -> None:
        g = self.grid
        mask = g.m > 0.0
        g.v[mask] /= g.m[mask][:, None]

    def _g2p(self, dt: float) -> None:
        p = self.particles
        g = self.grid
        dx = g.dx
        D_inv = 4.0 / (dx * dx)

        base, w, _ = quadratic_bspline_weights(p.x, dx=dx, origin=g.origin)
        W, _ = tensor_product_weights(w)                       # (P, 3, 3, 3)
        P = p.x.shape[0]

        # Batched 27-stencil gather in one shot (no Python loop).
        offsets = np.mgrid[0:3, 0:3, 0:3].reshape(3, -1).T     # (27, 3)
        idx_all = base[:, None, :] + offsets[None, :, :]       # (P, 27, 3)
        v_all = g.v[idx_all[..., 0], idx_all[..., 1], idx_all[..., 2]]  # (P, 27, 3)
        x_nodes = g.origin + idx_all * dx                      # (P, 27, 3)
        dpos = x_nodes - p.x[:, None, :]                       # (P, 27, 3)
        W_flat = W.reshape(P, -1)                              # (P, 27)

        # v_new = Σ_ijk W · v_i ;  C_new = D_inv · Σ_ijk W · v_i ⊗ dpos
        p.v = np.einsum("pe,pea->pa", W_flat, v_all)
        p.C = D_inv * np.einsum("pe,pea,peb->pab", W_flat, v_all, dpos)

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
