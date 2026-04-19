"""Local-stretch damage tests for DamagedNeoHookean (CD-MPM-style)."""

from __future__ import annotations

import numpy as np
import pytest

from robosim.physics.mpm.grid import Grid
from robosim.physics.mpm.materials import DamagedNeoHookean, NeoHookean
from robosim.physics.mpm.particles import Particles
from robosim.physics.mpm.solver import BoxBC, MPMSolver, sample_box_particles


# ── Unit tests on update_damage / kirchhoff_stress ───────────────────────────


def test_damage_inside_envelope_is_zero():
    """F inside the critical stretch must leave damage at zero."""
    mat = DamagedNeoHookean(young=1e5, poisson=0.3, stretch_c=1.3)
    F = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    F[1] = np.diag([1.1, 1.0, 1.0])           # below stretch_c
    d0 = np.zeros(4)
    d1 = mat.update_damage(F, d0)
    np.testing.assert_allclose(d1, 0.0, atol=1e-12)


def test_damage_grows_past_stretch_c():
    """Above stretch_c, damage becomes strictly positive and ∈ (0, 1)."""
    mat = DamagedNeoHookean(young=1e5, poisson=0.3, stretch_c=1.2, softening=2.0)
    F = np.diag([1.5, 1.0, 1.0])[None]        # λ_max = 1.5 > 1.2
    d = mat.update_damage(F, np.zeros(1))
    expected = 1.0 - (1.2 / 1.5) ** 2
    np.testing.assert_allclose(d[0], expected, rtol=1e-12)
    assert 0.0 < d[0] < 1.0


def test_damage_is_monotonic():
    """update_damage must never decrease d (irreversible softening)."""
    mat = DamagedNeoHookean(young=1e5, poisson=0.3, stretch_c=1.1)
    F_high = np.diag([1.6, 1.0, 1.0])[None]
    F_low  = np.diag([1.0, 1.0, 1.0])[None]    # would give d_trial = 0
    d = mat.update_damage(F_high, np.zeros(1))
    d_after = mat.update_damage(F_low, d)
    np.testing.assert_array_equal(d_after, d)


def test_damage_clamped_to_one():
    mat = DamagedNeoHookean(young=1e5, poisson=0.3, stretch_c=1.01, softening=8.0)
    F = np.diag([10.0, 1.0, 1.0])[None]
    d = mat.update_damage(F, np.zeros(1))
    assert 0.0 < d[0] <= 1.0


def test_stress_degrades_with_damage():
    """(1-d)^2 factor: d=0 matches NeoHookean, d=1 kills the stress."""
    dmat = DamagedNeoHookean(young=1e5, poisson=0.3)
    nh = NeoHookean(young=1e5, poisson=0.3)
    F = np.diag([1.05, 0.97, 1.0])[None]

    tau_elastic = nh.kirchhoff_stress(F)
    tau_d0 = dmat.kirchhoff_stress(F, np.zeros(1))
    tau_d1 = dmat.kirchhoff_stress(F, np.ones(1))
    tau_half = dmat.kirchhoff_stress(F, np.full(1, 0.5))

    np.testing.assert_allclose(tau_d0, tau_elastic, atol=1e-12)
    np.testing.assert_allclose(tau_d1, 0.0, atol=1e-12)
    np.testing.assert_allclose(tau_half, 0.25 * tau_elastic, atol=1e-12)


# ── Integrated: tearing under stretch ────────────────────────────────────────


def test_mpm_blob_separates_under_stretch():
    """A bar pulled hard at both ends: with damage on, middle particles
    accumulate damage and the two halves drift apart further than the
    pristine elastic run."""

    def run(material, seed_v=None):
        # Bar ~0.2 × 0.04 × 0.04 with ~2 particles per grid cell in every axis.
        from robosim.physics.mpm.particles import Particles
        lower = np.array([0.40, 0.48, 0.48])
        upper = np.array([0.60, 0.52, 0.52])
        dx = 0.02
        n = np.array([21, 5, 5])
        xs = [np.linspace(lower[k], upper[k], n[k]) for k in range(3)]
        X, Y, Z = np.meshgrid(*xs, indexing="ij")
        x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        P = x.shape[0]
        V = float(np.prod(upper - lower))
        pts = Particles(
            x=x,
            v=np.zeros_like(x),
            m=np.full(P, 1000.0 * V / P),
            V0=np.full(P, V / P),
        )
        if seed_v is not None:
            pts.v[:] = seed_v(pts.x)
        grid = Grid.from_bounds(lower=np.zeros(3), upper=np.ones(3), dx=dx, pad=3)
        solver = MPMSolver(
            particles=pts,
            grid=grid,
            material=material,
            gravity=np.zeros(3),
            bcs=[],
        )
        for _ in range(200):
            solver.step(2e-4)
        return solver

    def pull_apart(x):
        # Uniform tensile strain rate: ∂v_x/∂x = 15 /s in x.
        vx = 15.0 * (x[:, 0] - 0.5)
        return np.stack([vx, np.zeros_like(vx), np.zeros_like(vx)], axis=-1)

    dmat = DamagedNeoHookean(young=1e4, poisson=0.3, stretch_c=1.05, softening=3.0)
    nh = NeoHookean(young=1e4, poisson=0.3)

    sd = run(dmat, seed_v=pull_apart)
    se = run(nh,   seed_v=pull_apart)

    # Damage should have accumulated somewhere.
    assert sd.particles.d.max() > 0.2

    # The damaged bar should end up longer (less stiffness holding halves).
    ext_d = float(np.ptp(sd.particles.x[:, 0]))
    ext_e = float(np.ptp(se.particles.x[:, 0]))
    assert ext_d > ext_e + 0.01, (
        f"damaged bar should stretch further (damaged={ext_d}, elastic={ext_e})"
    )


def test_particles_default_damage_zero():
    """Fresh particles start undamaged and with the expected dtype/shape."""
    pts = sample_box_particles(lower=(0, 0, 0), upper=(0.1, 0.1, 0.1),
                               n_per_axis=3, density=1000.0)
    assert pts.d.shape == (pts.n,)
    assert pts.d.dtype == np.float64
    np.testing.assert_array_equal(pts.d, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
