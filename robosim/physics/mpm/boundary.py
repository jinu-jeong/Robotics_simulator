"""Grid-level external forces and boundary conditions for MPM.

All operations act on the grid **after** P2G has normalised momentum to
velocity, and **before** G2P reads it back to the particles. Only nodes with
non-zero mass participate (zero-mass nodes are ignored — they contribute
nothing to the G2P gather either).
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class KinematicBoxCollider:
    """Axis-aligned rigid box that plows through the MPM grid.

    Unlike :func:`apply_box_bc`, which treats an AABB as a *container*
    (solid outside, fluid inside), this collider is a *solid obstacle*
    embedded in the MPM domain — everything outside is MPM, the interior
    is off-limits. The standard "snow-solver" rigid-collider projection
    (Stomakhin et al. 2013) is used:

        v_rel = v_grid − v_obj
        n     = outward normal of the nearest face
        if v_rel · n < 0:   v_rel ← v_rel − (v_rel · n) n     (slip)
        v_grid = v_rel + v_obj

    Pose, extent, and velocity can be updated between substeps to sweep
    the collider through the grid.
    """

    center: np.ndarray       # (3,)
    half_extent: np.ndarray  # (3,)
    velocity: np.ndarray     # (3,) — translational velocity of the box

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.half_extent = np.asarray(self.half_extent, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)

    def apply(self, grid: Grid, dt: float) -> None:
        # Fast AABB rejection + slice: compute the index range of grid
        # nodes that could lie inside the box, and only process that
        # sub-block. For a small collider in a large grid this avoids
        # touching 99 % of the cells every step.
        lo = self.center - self.half_extent
        hi = self.center + self.half_extent
        inv_dx = 1.0 / grid.dx
        ix_lo = int(np.floor((lo[0] - grid.origin[0]) * inv_dx))
        ix_hi = int(np.ceil ((hi[0] - grid.origin[0]) * inv_dx)) + 1
        iy_lo = int(np.floor((lo[1] - grid.origin[1]) * inv_dx))
        iy_hi = int(np.ceil ((hi[1] - grid.origin[1]) * inv_dx)) + 1
        iz_lo = int(np.floor((lo[2] - grid.origin[2]) * inv_dx))
        iz_hi = int(np.ceil ((hi[2] - grid.origin[2]) * inv_dx)) + 1

        nx, ny, nz = grid.shape
        ix_lo = max(ix_lo, 0); ix_hi = min(ix_hi, nx)
        iy_lo = max(iy_lo, 0); iy_hi = min(iy_hi, ny)
        iz_lo = max(iz_lo, 0); iz_hi = min(iz_hi, nz)
        if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
            return  # box is entirely outside the grid → nothing to do

        xs = grid.origin[0] + np.arange(ix_lo, ix_hi) * grid.dx
        ys = grid.origin[1] + np.arange(iy_lo, iy_hi) * grid.dx
        zs = grid.origin[2] + np.arange(iz_lo, iz_hi) * grid.dx
        dx = xs[:, None, None] - self.center[0]
        dy = ys[None, :, None] - self.center[1]
        dz = zs[None, None, :] - self.center[2]
        shp = (ix_hi - ix_lo, iy_hi - iy_lo, iz_hi - iz_lo)
        inside = (
            (np.abs(dx) <= self.half_extent[0])
            & (np.abs(dy) <= self.half_extent[1])
            & (np.abs(dz) <= self.half_extent[2])
        )
        if not inside.any():
            return

        # Penetration depth per axis; nearest face = smallest penetration.
        pen = np.stack(
            [
                np.broadcast_to(self.half_extent[0] - np.abs(dx), shp),
                np.broadcast_to(self.half_extent[1] - np.abs(dy), shp),
                np.broadcast_to(self.half_extent[2] - np.abs(dz), shp),
            ],
            axis=-1,
        )
        axis = np.argmin(pen, axis=-1)
        disp = np.stack([
            np.broadcast_to(dx, shp),
            np.broadcast_to(dy, shp),
            np.broadcast_to(dz, shp),
        ], axis=-1)

        # Per-node outward normal (unit axis, sign of the displacement).
        n = np.zeros(shp + (3,))
        for a in range(3):
            mask_a = inside & (axis == a)
            sgn = np.sign(disp[..., a])
            sgn = np.where(sgn == 0, 1.0, sgn)
            n[..., a] = np.where(mask_a, sgn, n[..., a])

        v_slice = grid.v[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
        v_rel = v_slice - self.velocity
        v_n = (v_rel * n).sum(axis=-1, keepdims=True)
        push_out = inside[..., None] & (v_n < 0.0)
        grid.v[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi] = np.where(
            push_out, v_slice - v_n * n, v_slice,
        )
