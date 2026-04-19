"""Plasticity return-mapping tests (Von Mises / J2)."""

from __future__ import annotations

import numpy as np
import pytest

from robosim.physics.mpm.grid import Grid
from robosim.physics.mpm.materials import (
    DruckerPragerPlastic,
    NeoHookean,
    VonMisesPlastic,
)
from robosim.physics.mpm.solver import BoxBC, MPMSolver, sample_box_particles


RNG = np.random.default_rng(seed=2)


# ── Unit tests for the return mapping ────────────────────────────────────────


def test_vm_elastic_regime_is_identity():
    """A small stretch (Hencky strain below yield) is returned unchanged."""
    mat = VonMisesPlastic(young=1e5, poisson=0.3, yield_stress=1e4)
    # Nearly-identity F batch — well below yield.
    F = np.broadcast_to(np.eye(3), (8, 3, 3)).copy()
    F[0] = np.diag([1.001, 1.0, 0.999])          # tiny deviatoric stretch
    F[1] = np.diag([1.01, 0.995, 1.0])
    F_proj = mat.project(F)
    np.testing.assert_allclose(F_proj, F, atol=1e-10)


def test_vm_plastic_regime_clips_dev_strain():
    """A large deviatoric stretch gets clipped to the yield surface."""
    mat = VonMisesPlastic(young=1e5, poisson=0.3, yield_stress=1e3)
    # Strongly deviatoric but isochoric (det=1): stretch in x, compress in yz.
    s = 1.5
    F = np.diag([s, 1.0 / np.sqrt(s), 1.0 / np.sqrt(s)])[None]
    F_proj = mat.project(F)

    # After projection, ‖ε_dev‖ must equal yield_strain (within tiny slack).
    _, sigma_p, _ = np.linalg.svd(F_proj)
    eps_p = np.log(sigma_p)
    eps_dev_p = eps_p - eps_p.mean(axis=-1, keepdims=True)
    yield_strain = mat.yield_stress / (np.sqrt(6.0) * mat.mu)
    np.testing.assert_allclose(np.linalg.norm(eps_dev_p, axis=-1)[0], yield_strain, rtol=1e-10)


def test_vm_plasticity_is_isochoric():
    """J2 plastic flow preserves volume: det(F_E) after projection must
    equal det(F_trial)."""
    mat = VonMisesPlastic(young=1e5, poisson=0.3, yield_stress=5e2)
    F = np.empty((16, 3, 3))
    for i in range(16):
        # Random F with det > 0 (guarantee via diagonally-dominant + id).
        A = RNG.standard_normal((3, 3)) * 0.3
        F[i] = np.eye(3) + A
    det_before = np.linalg.det(F)
    F_proj = mat.project(F)
    det_after = np.linalg.det(F_proj)
    np.testing.assert_allclose(det_after, det_before, rtol=1e-10)


def test_vm_infinite_yield_matches_elastic():
    """yield_stress → ∞ recovers the purely elastic behaviour (no projection)."""
    mat = VonMisesPlastic(young=1e5, poisson=0.3, yield_stress=1e20)
    F = np.empty((8, 3, 3))
    for i in range(8):
        A = RNG.standard_normal((3, 3)) * 0.2
        F[i] = np.eye(3) + A
    F_proj = mat.project(F)
    np.testing.assert_allclose(F_proj, F, atol=1e-10)


def test_vm_stress_matches_neo_hookean_on_F_E():
    """VonMises uses the same elastic law on F_E as plain NeoHookean on F."""
    vm = VonMisesPlastic(young=1e5, poisson=0.3, yield_stress=1e3)
    nh = NeoHookean(young=1e5, poisson=0.3)
    F = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    F[1] = np.diag([1.05, 0.97, 1.0])
    F[2] = np.diag([0.9, 1.1, 1.01])
    np.testing.assert_allclose(vm.kirchhoff_stress(F), nh.kirchhoff_stress(F), atol=1e-12)


# ── Integrated solver tests (plasticity inside an MPM step) ──────────────────


def test_plastic_vs_elastic_energy_loss():
    """Same drop, two materials: the plastic one dissipates more energy
    (lower rebound height) than the purely elastic one, and its F_E stays
    inside the yield envelope throughout."""

    def run(material):
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
            material=material,
            gravity=np.array([0.0, 0.0, -9.81]),
            bcs=[BoxBC(lower=np.zeros(3),
                       upper=np.array([0.7, 0.7, 10.0]),
                       mode="slip")],
        )
        dt = 5e-4
        z_com_peak_after_bounce = -np.inf
        saw_bottom = False
        for _ in range(800):
            solver.step(dt)
            com_z = pts.x[:, 2].mean()
            if com_z < 0.12:
                saw_bottom = True
            if saw_bottom and com_z > z_com_peak_after_bounce:
                z_com_peak_after_bounce = com_z
        return solver, z_com_peak_after_bounce

    vm = VonMisesPlastic(young=5e4, poisson=0.3, yield_stress=1e3)
    nh = NeoHookean(young=5e4, poisson=0.3)

    solver_vm, rebound_vm = run(vm)
    _, rebound_nh = run(nh)

    # Plastic drop rebounds less high than the elastic one.
    assert rebound_vm < rebound_nh - 0.01, (
        f"expected plastic rebound < elastic (plastic={rebound_vm}, elastic={rebound_nh})"
    )

    # Throughout, the plastic F_E never exceeds the yield envelope.
    _, sigma, _ = np.linalg.svd(solver_vm.particles.F)
    eps = np.log(np.maximum(sigma, 1e-12))
    eps_dev = eps - eps.mean(axis=-1, keepdims=True)
    yield_strain = vm.yield_stress / (np.sqrt(6.0) * vm.mu)
    assert np.linalg.norm(eps_dev, axis=-1).max() <= yield_strain + 1e-8


# ── Drucker-Prager tests ─────────────────────────────────────────────────────


def test_dp_elastic_core_leaves_small_F_untouched():
    mat = DruckerPragerPlastic(young=1e5, poisson=0.3, friction_angle=np.deg2rad(30.0))
    F = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    # Small compression inside the cone (trace slightly negative, tiny dev).
    F[0] = np.diag([0.999, 0.999, 0.999])
    F[1] = np.diag([0.998, 1.0, 0.999])
    F_proj = mat.project(F)
    np.testing.assert_allclose(F_proj, F, atol=1e-10)


def test_dp_tension_collapses_to_rotation_only():
    """With ``tr(ε) > 0`` (volumetric tension), F_E must become U V^T —
    i.e. all singular values are 1 after projection."""
    mat = DruckerPragerPlastic(young=1e5, poisson=0.3, friction_angle=np.deg2rad(30.0))
    F = np.diag([1.2, 1.1, 1.05])[None]           # pure stretch ⇒ tr(log Σ) > 0
    F_proj = mat.project(F)
    _, sigma_p, _ = np.linalg.svd(F_proj)
    np.testing.assert_allclose(sigma_p[0], [1.0, 1.0, 1.0], atol=1e-10)


def test_dp_compression_projected_onto_cone():
    """For a heavily sheared compressive trial state, the projected point
    lies on the yield cone: ‖ε̂_new‖ = −α · coef · tr(ε_new)."""
    mat = DruckerPragerPlastic(young=1e5, poisson=0.3, friction_angle=np.deg2rad(30.0))
    # Compression in z + deviatoric shear ⇒ trace < 0, large dev.
    F = np.diag([1.1, 1.1, 0.6])[None]
    F_proj = mat.project(F)

    _, sigma_p, _ = np.linalg.svd(F_proj)
    eps = np.log(sigma_p)
    trace_new = eps.sum(axis=-1, keepdims=True)
    eps_hat_new = eps - trace_new / 3.0
    norm_hat_new = np.linalg.norm(eps_hat_new, axis=-1)

    coef = (3.0 * mat.lam + 2.0 * mat.mu) / (2.0 * mat.mu)
    # On the cone: ‖ε̂‖ + α · coef · tr(ε) = 0   ⇒  ‖ε̂‖ = −α · coef · tr(ε).
    expected_norm = -mat.alpha * coef * trace_new[:, 0]
    np.testing.assert_allclose(norm_hat_new, expected_norm, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
