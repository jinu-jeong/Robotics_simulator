"""Grid-level external forces and boundary conditions for MPM.

All operations act on the grid **after** P2G has normalised momentum to
velocity, and **before** G2P reads it back to the particles. Only nodes with
non-zero mass participate (zero-mass nodes are ignored — they contribute
nothing to the G2P gather either).
"""

from __future__ import annotations

import numpy as np

from robosim.physics.mpm.grid import Grid


def apply_gravity(grid: Grid, dt: float, gravity: np.ndarray) -> None:
    """Add gravity impulse to every node that carries mass."""
    g = np.asarray(gravity, dtype=np.float64).reshape(3)
    mask = grid.m > 0.0
    grid.v[mask] += dt * g


def apply_box_bc(
    grid: Grid,
    lower: np.ndarray,
    upper: np.ndarray,
    mode: str = "slip",
) -> None:
    """Enforce axis-aligned walls at ``lower`` and ``upper``.

    ``mode="sticky"`` zeroes all three velocity components at boundary nodes;
    ``mode="slip"`` only zeroes the **inward** normal component (tangential
    flow is free).
    """
    if mode not in ("slip", "sticky"):
        raise ValueError(f"mode must be 'slip' or 'sticky', got {mode!r}")

    lower = np.asarray(lower, dtype=np.float64).reshape(3)
    upper = np.asarray(upper, dtype=np.float64).reshape(3)

    nx, ny, nz = grid.shape
    xs = grid.origin[0] + np.arange(nx) * grid.dx
    ys = grid.origin[1] + np.arange(ny) * grid.dx
    zs = grid.origin[2] + np.arange(nz) * grid.dx

    lo_x = xs <= lower[0]
    hi_x = xs >= upper[0]
    lo_y = ys <= lower[1]
    hi_y = ys >= upper[1]
    lo_z = zs <= lower[2]
    hi_z = zs >= upper[2]

    if mode == "sticky":
        # Any node outside the box along any axis gets clamped to zero velocity.
        any_x = (lo_x | hi_x)[:, None, None]
        any_y = (lo_y | hi_y)[None, :, None]
        any_z = (lo_z | hi_z)[None, None, :]
        wall = np.broadcast_to(any_x | any_y | any_z, grid.shape)
        grid.v[wall] = 0.0
        return

    # Slip: clamp inward normal components.
    vx = grid.v[..., 0]
    vy = grid.v[..., 1]
    vz = grid.v[..., 2]

    vx[lo_x, :, :] = np.maximum(vx[lo_x, :, :], 0.0)
    vx[hi_x, :, :] = np.minimum(vx[hi_x, :, :], 0.0)
    vy[:, lo_y, :] = np.maximum(vy[:, lo_y, :], 0.0)
    vy[:, hi_y, :] = np.minimum(vy[:, hi_y, :], 0.0)
    vz[:, :, lo_z] = np.maximum(vz[:, :, lo_z], 0.0)
    vz[:, :, hi_z] = np.minimum(vz[:, :, hi_z], 0.0)
