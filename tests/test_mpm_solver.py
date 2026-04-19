"""End-to-end MPM solver checks: gravity, walls, elastic restitution."""

from __future__ import annotations

import numpy as np
import pytest

from robosim.physics.mpm.boundary import apply_box_bc, apply_gravity
from robosim.physics.mpm.grid import Grid
from robosim.physics.mpm.materials import NeoHookean
from robosim.physics.mpm.solver import BoxBC, MPMSolver, sample_box_particles


# ── Boundary-layer unit tests ────────────────────────────────────────────────


def test_apply_gravity_only_on_mass_nodes():
    g = Grid.from_bounds(lower=np.zeros(3), upper=np.full(3, 0.1), dx=0.05, pad=2)
    g.m[1, 1, 1] = 1.0
    apply_gravity(g, dt=0.01, gravity=np.array([0.0, 0.0, -9.81]))
    np.testing.assert_allclose(g.v[1, 1, 1], [0.0, 0.0, -0.0981], atol=1e-12)
    # All other nodes remain untouched.
    g.v[1, 1, 1] = 0.0
    np.testing.assert_array_equal(g.v, 0.0)


def test_apply_box_bc_slip_clamps_normal_component():
    g = Grid.from_bounds(lower=np.zeros(3), upper=np.full(3, 0.1), dx=0.05, pad=2)
    # Seed every node with a velocity so BCs are visible.
    g.v[..., 2] = -1.0
    g.v[..., 0] = 0.3
    apply_box_bc(g, lower=np.array([0.0, 0.0, 0.0]), upper=np.array([0.1, 0.1, 0.1]), mode="slip")

    # Floor nodes (world z <= 0) should have v_z clamped ≥ 0 ⇒ 0 here.
    zs = g.origin[2] + np.arange(g.shape[2]) * g.dx
    floor_idx = np.where(zs <= 0.0)[0]
    assert floor_idx.size > 0
    for kz in floor_idx:
        np.testing.assert_array_equal(g.v[:, :, kz, 2], 0.0)
    # Tangential (x) component is untouched at floor nodes that are NOT also
    # sitting on the left/right x walls — those walls separately clamp v_x.
    xs = g.origin[0] + np.arange(g.shape[0]) * g.dx
    interior_x = np.where((xs > 0.0) & (xs < 0.1))[0]
    assert interior_x.size > 0
    for kz in floor_idx:
        np.testing.assert_array_equal(g.v[interior_x, :, kz, 0], 0.3)


# ── Integrated solver tests ──────────────────────────────────────────────────


def test_uniform_freefall_matches_analytic():
    """A cube released at rest under gravity (no BC, no internal deformation)
    translates rigidly; its centre of mass must follow 0.5 g t²."""
    pts = sample_box_particles(
        lower=(0.30, 0.30, 0.30),
        upper=(0.40, 0.40, 0.40),
        n_per_axis=5,
        density=1000.0,
    )
    grid = Grid.from_bounds(lower=np.zeros(3), upper=np.full(3, 0.7), dx=0.02, pad=3)
    solver = MPMSolver(
        particles=pts,
        grid=grid,
        material=NeoHookean(young=1e4, poisson=0.3),
        gravity=np.array([0.0, 0.0, -9.81]),
        bcs=[],  # no walls
    )

    x0_com = pts.x.mean(axis=0)
    dt = 1e-3
    n_steps = 50
    for _ in range(n_steps):
        solver.step(dt)
    t = n_steps * dt

    com = pts.x.mean(axis=0)
    # Symplectic Euler gives exactly 0.5·N·(N+1)·dt²·g for the displacement;
    # analytic 0.5·g·t² is the dt→0 limit. Match the discrete expectation.
    expected_disp = 0.5 * n_steps * (n_steps + 1) * dt**2 * solver.gravity
    np.testing.assert_allclose(com - x0_com, expected_disp, atol=1e-10)

    v_com = (pts.m[:, None] * pts.v).sum(axis=0) / pts.m.sum()
    np.testing.assert_allclose(v_com, solver.gravity * t, atol=1e-12)

    # F stays at identity under rigid translation.
    np.testing.assert_allclose(pts.F, np.broadcast_to(np.eye(3), pts.F.shape), atol=1e-10)


def test_elastic_bounce_off_floor():
    """A jello cube dropped onto a floor should rebound: after the impact
    phase, its centre-of-mass velocity must turn positive again."""
    pts = sample_box_particles(
        lower=(0.30, 0.30, 0.20),
        upper=(0.40, 0.40, 0.30),
        n_per_axis=5,
        density=1000.0,
    )
    grid = Grid.from_bounds(lower=np.zeros(3), upper=np.full(3, 0.7), dx=0.02, pad=3)
    solver = MPMSolver(
        particles=pts,
        grid=grid,
        material=NeoHookean(young=5e4, poisson=0.3),
        gravity=np.array([0.0, 0.0, -9.81]),
        bcs=[BoxBC(lower=np.array([0.0, 0.0, 0.0]),
                   upper=np.array([0.7, 0.7, 10.0]),
                   mode="slip")],
    )

    dt = 5e-4
    v_com_z_history = []
    for _ in range(800):
        solver.step(dt)
        v_com = (pts.m[:, None] * pts.v).sum(axis=0) / pts.m.sum()
        v_com_z_history.append(v_com[2])

    v_hist = np.array(v_com_z_history)
    # Falls first → min v_z < 0 somewhere.
    assert v_hist.min() < -0.1
    # Then bounces → max v_z > 0 later.
    assert v_hist.max() > 0.05
    # No blow-up: nothing ends up above ~10× drop speed.
    assert np.abs(v_hist).max() < 5.0
    # Particles stay above the floor (within a small kernel tolerance).
    assert pts.x[:, 2].min() > -3 * grid.dx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
