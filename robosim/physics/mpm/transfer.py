"""APIC particle ↔ grid transfer (Jiang et al. 2015).

Affine Particle-In-Cell stores a per-particle 3×3 affine matrix ``C`` that
encodes the local velocity field. Compared to plain PIC, APIC preserves
angular momentum and dramatically reduces numerical dissipation; compared to
FLIP, it stays stable without instability-suppressing blends.

For the quadratic B-spline kernel used here, the APIC inverse inertia is
``D^{-1} = (4/dx²) · I``, so the affine recovery is

    C_p = (4/dx²) · Σ_i W_ip · v_i · (x_i - x_p)^T.
"""

from __future__ import annotations

import numpy as np

from robosim.physics.mpm.grid import Grid
from robosim.physics.mpm.kernel import (
    quadratic_bspline_weights,
    tensor_product_weights,
)
from robosim.physics.mpm.particles import Particles


def p2g(particles: Particles, grid: Grid) -> None:
    """Scatter mass and (affine-augmented) momentum from particles to grid.

    After this call ``grid.v`` holds *velocity* (momentum normalised by mass);
    nodes with no incident particle mass are left at zero.
    """
    dx = grid.dx
    base, w, dw = quadratic_bspline_weights(particles.x, dx=dx, origin=grid.origin)
    W, _ = tensor_product_weights(w, dw)                     # (P, 3, 3, 3)

    grid.reset()

    # Iterate over the 3×3×3 stencil (27 passes, each vectorised over P).
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = base + np.array([i, j, k], dtype=np.int64)  # (P, 3)
                ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
                W_ijk = W[:, i, j, k]                              # (P,)

                x_node = grid.origin + idx * dx                    # (P, 3)
                dx_ip = x_node - particles.x                       # (P, 3)

                # APIC augmented particle velocity: v_p + C_p · (x_i − x_p).
                Cdx = np.einsum("pab,pb->pa", particles.C, dx_ip)  # (P, 3)
                v_aug = particles.v + Cdx

                mass_contrib = W_ijk * particles.m                 # (P,)
                mom_contrib = mass_contrib[:, None] * v_aug        # (P, 3)

                np.add.at(grid.m, (ix, iy, iz), mass_contrib)
                np.add.at(grid.v, (ix, iy, iz), mom_contrib)

    # Normalise momentum → velocity. Leave zero-mass nodes at zero.
    m = grid.m
    mask = m > 0.0
    grid.v[mask] /= m[mask][:, None]


def g2p(particles: Particles, grid: Grid, dt: float) -> None:
    """Gather velocity and affine matrix from grid, then advect particles."""
    dx = grid.dx
    base, w, dw = quadratic_bspline_weights(particles.x, dx=dx, origin=grid.origin)
    W, _ = tensor_product_weights(w, dw)

    Dinv = 4.0 / (dx * dx)                                    # quadratic APIC

    v_new = np.zeros_like(particles.v)
    C_new = np.zeros_like(particles.C)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = base + np.array([i, j, k], dtype=np.int64)
                ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
                W_ijk = W[:, i, j, k]                          # (P,)

                v_i = grid.v[ix, iy, iz]                       # (P, 3)
                x_node = grid.origin + idx * dx                # (P, 3)
                dx_ip = x_node - particles.x                   # (P, 3)

                v_new += W_ijk[:, None] * v_i
                C_new += (
                    Dinv
                    * W_ijk[:, None, None]
                    * np.einsum("pa,pb->pab", v_i, dx_ip)
                )

    particles.v = v_new
    particles.C = C_new
    particles.x = particles.x + dt * v_new
