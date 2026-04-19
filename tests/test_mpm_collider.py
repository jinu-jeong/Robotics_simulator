"""Rigid kinematic collider tests (KinematicBoxCollider)."""

from __future__ import annotations

import numpy as np
import pytest

from robosim.physics.mpm.boundary import KinematicBoxCollider
from robosim.physics.mpm.grid import Grid
from robosim.physics.mpm.materials import NeoHookean
from robosim.physics.mpm.particles import Particles
from robosim.physics.mpm.solver import BoxBC, MPMSolver, sample_box_particles


# ── Unit tests on KinematicBoxCollider.apply ─────────────────────────────────


def test_no_nodes_inside_is_noop():
    grid = Grid.from_bounds(lower=np.zeros(3), upper=np.ones(3), dx=0.05, pad=2)
    grid.m[:] = 1.0
    grid.v[:] = [1.0, 2.0, 3.0]
    v_before = grid.v.copy()

    col = KinematicBoxCollider(
        center=np.array([5.0, 5.0, 5.0]),   # far outside
        half_extent=np.array([0.1, 0.1, 0.1]),
        velocity=np.zeros(3),
    )
    col.apply(grid, dt=1e-3)
    np.testing.assert_array_equal(grid.v, v_before)


def test_static_collider_zeros_inward_normal():
    """Nodes inside a static box with velocity pointed into the nearest
    face get that component removed; tangential stays untouched."""
    grid = Grid.from_bounds(lower=np.zeros(3), upper=np.ones(3), dx=0.05, pad=0)
    grid.m[:] = 1.0
    grid.v[:] = [0.0, 0.0, -2.0]           # moving −z (into the floor)

    col = KinematicBoxCollider(
        center=np.array([0.5, 0.5, 0.1]),
        half_extent=np.array([0.5, 0.5, 0.1]),
        velocity=np.zeros(3),
    )
    col.apply(grid, dt=1e-3)
    # Nodes well inside the collider should have v_z clamped — those near
    # the top face have normal +z, v_rel · n = −2 < 0 ⇒ v_z → 0.
    # Pick node near top of collider: z ≈ 0.2, x=y=0.5.
    ix = int(round((0.5 - grid.origin[0]) / grid.dx))
    iy = int(round((0.5 - grid.origin[1]) / grid.dx))
    iz = int(round((0.15 - grid.origin[2]) / grid.dx))
    np.testing.assert_allclose(grid.v[ix, iy, iz, 2], 0.0, atol=1e-10)


def test_collider_imparts_velocity():
    """A moving collider should drag MPM nodes along its motion (no
    penetration ⇒ node's inward-normal component must equal the box's)."""
    grid = Grid.from_bounds(lower=np.zeros(3), upper=np.ones(3), dx=0.05, pad=0)
    grid.m[:] = 1.0
    grid.v[:] = 0.0

    col = KinematicBoxCollider(
        center=np.array([0.5, 0.5, 0.5]),
        half_extent=np.array([0.2, 0.2, 0.2]),
        velocity=np.array([0.0, 0.0, 1.0]),   # pushing up
    )
    col.apply(grid, dt=1e-3)
    # Node near top face (normal +z): v_rel_z = 0 − 1 = −1 < 0 ⇒ remove:
    #   v_rel_z → 0, so v_z = 0 + 1 = 1.
    ix = int(round((0.5 - grid.origin[0]) / grid.dx))
    iy = int(round((0.5 - grid.origin[1]) / grid.dx))
    iz = int(round((0.65 - grid.origin[2]) / grid.dx))
    np.testing.assert_allclose(grid.v[ix, iy, iz, 2], 1.0, atol=1e-10)


# ── Integrated: block falling onto clay slab ─────────────────────────────────


def test_falling_block_deforms_mpm():
    """A rigid block held still above an MPM slab, then pressed down,
    should dent the slab (max z of top surface drops)."""
    pts = sample_box_particles(
        lower=(0.30, 0.30, 0.10), upper=(0.70, 0.70, 0.30),
        n_per_axis=7, density=1000.0,
    )
    grid = Grid.from_bounds(lower=np.zeros(3), upper=np.ones(3), dx=0.04, pad=3)

    col = KinematicBoxCollider(
        center=np.array([0.5, 0.5, 0.45]),
        half_extent=np.array([0.08, 0.08, 0.08]),
        velocity=np.array([0.0, 0.0, -0.8]),
    )
    solver = MPMSolver(
        particles=pts, grid=grid,
        material=NeoHookean(young=5e4, poisson=0.3),
        gravity=np.array([0.0, 0.0, -9.81]),
        bcs=[BoxBC(lower=np.zeros(3),
                   upper=np.array([1.0, 1.0, 10.0]), mode="slip")],
        colliders=[col],
    )

    dt = 5e-4
    z_top_initial = pts.x[:, 2].max()
    for _ in range(400):
        col.center = col.center + dt * col.velocity
        solver.step(dt)

    # Collider descended; clay slab should be indented below initial top.
    # Under the collider footprint, pick the particles closest in xy and
    # check their z dropped.
    mask_under = (
        (np.abs(pts.x[:, 0] - 0.5) < 0.1) & (np.abs(pts.x[:, 1] - 0.5) < 0.1)
    )
    assert pts.x[mask_under, 2].max() < z_top_initial - 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
