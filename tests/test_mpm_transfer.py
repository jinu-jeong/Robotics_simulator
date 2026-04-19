"""Verify APIC P2G/G2P round-trip properties (mass/momentum conservation,
rigid-translation exactness, linear-velocity-field recovery)."""

from __future__ import annotations

import numpy as np
import pytest

from robosim.physics.mpm.grid import Grid
from robosim.physics.mpm.particles import Particles
from robosim.physics.mpm.transfer import g2p, p2g


RNG = np.random.default_rng(seed=1)


def _make_jello(n_per_axis: int = 6, low=0.05, high=0.15, dx: float = 0.02):
    """A dense cube of particles inside a grid that comfortably contains them."""
    xs = np.linspace(low, high, n_per_axis)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    P = x.shape[0]
    v = np.zeros_like(x)
    m = np.full(P, 1e-3)
    V0 = np.full(P, (xs[1] - xs[0]) ** 3)
    pts = Particles(x=x, v=v, m=m, V0=V0)
    grid = Grid.from_bounds(lower=np.zeros(3), upper=np.full(3, 0.2), dx=dx, pad=3)
    return pts, grid


def test_mass_conservation_after_p2g():
    pts, grid = _make_jello()
    total_particle_mass = pts.m.sum()
    p2g(pts, grid)
    np.testing.assert_allclose(grid.m.sum(), total_particle_mass, atol=1e-12)


def test_momentum_conservation_after_p2g_with_zero_C():
    """With C=0, P2G must transfer particle momentum to the grid exactly."""
    pts, grid = _make_jello()
    pts.v[:] = RNG.standard_normal(pts.v.shape) * 0.3
    # C defaults to zero.
    momentum_particles = (pts.m[:, None] * pts.v).sum(axis=0)
    p2g(pts, grid)
    momentum_grid = (grid.m[..., None] * grid.v).sum(axis=(0, 1, 2))
    np.testing.assert_allclose(momentum_grid, momentum_particles, atol=1e-12)


def test_rigid_translation_roundtrip():
    """A rigid-body translation (uniform v, zero C) survives P2G → G2P."""
    pts, grid = _make_jello()
    v_const = np.array([0.7, -1.3, 0.4])
    pts.v[:] = v_const
    x_before = pts.x.copy()
    dt = 1e-3

    p2g(pts, grid)
    g2p(pts, grid, dt=dt)

    expected_v = np.broadcast_to(v_const, pts.v.shape)
    np.testing.assert_allclose(pts.v, expected_v, atol=1e-12)
    np.testing.assert_allclose(pts.x, x_before + dt * v_const, atol=1e-12)
    # Uniform-velocity field has zero gradient — affine matrix should return to 0.
    np.testing.assert_allclose(pts.C, 0.0, atol=1e-12)


def test_linear_velocity_field_roundtrip():
    """For v(x) = A·x + b the APIC round-trip must recover the same linear
    field on the particles (the quadratic kernel reproduces linear fields)."""
    pts, grid = _make_jello()
    A = np.array(
        [[0.2, -0.1, 0.05],
         [0.0, 0.15, -0.2],
         [0.1, 0.0, 0.3]]
    )
    b = np.array([0.01, -0.02, 0.03])
    pts.v[:] = pts.x @ A.T + b
    # Initial particle affine matrix matches the field gradient A.
    pts.C[:] = A

    p2g(pts, grid)
    g2p(pts, grid, dt=0.0)   # dt=0 isolates the transfer from advection

    expected_v = pts.x @ A.T + b
    np.testing.assert_allclose(pts.v, expected_v, atol=1e-10)
    # Each particle's recovered affine matrix should be A.
    for p in range(pts.n):
        np.testing.assert_allclose(pts.C[p], A, atol=1e-10)


def test_momentum_conservation_roundtrip_linear_field():
    """Total momentum survives a full round-trip even with a non-trivial
    affine field (the APIC augmentation is what makes this work)."""
    pts, grid = _make_jello()
    A = 0.1 * RNG.standard_normal((3, 3))
    b = 0.02 * RNG.standard_normal(3)
    pts.v[:] = pts.x @ A.T + b
    pts.C[:] = A

    p0 = (pts.m[:, None] * pts.v).sum(axis=0)

    p2g(pts, grid)
    g2p(pts, grid, dt=0.0)

    p1 = (pts.m[:, None] * pts.v).sum(axis=0)
    np.testing.assert_allclose(p1, p0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
