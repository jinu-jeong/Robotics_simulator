"""Unit tests for the corotational small-strain J2 plastic material.

Tests cover:
- Elastic-regime equivalence with CorotationalElastic (no yielding ⇒ same stress).
- Plastic activation: stress capped near yield surface, plastic strain grows.
- Hardening: yield surface expands ⇒ residual plastic strain better preserved.
- Rotation invariance: rigid-body rotation ⇒ no plastic flow, R correctly extracted.
- Volumetric incompressibility of plastic flow: tr(eps_p) ≈ 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from robosim.physics.fem.materials import (
    CorotationalElastic,
    CorotationalPlastic,
)


def _zero_eps_p() -> np.ndarray:
    return np.zeros((3, 3))


def test_below_yield_matches_elastic():
    """A tiny stretch produces stress identical to CorotationalElastic and
    leaves the plastic strain at zero."""
    F = np.diag([1.0005, 1.0, 1.0])
    elastic = CorotationalElastic(young=1e6, poisson=0.3)
    plastic = CorotationalPlastic(young=1e6, poisson=0.3, yield_stress=1e4)

    P_e, _ = elastic.compute_stress(F)
    P_p, _, eps_p_new = plastic.compute_stress(F, _zero_eps_p())

    np.testing.assert_allclose(P_p, P_e, rtol=1e-12)
    assert np.linalg.norm(eps_p_new) == 0.0


def test_above_yield_caps_stress_and_grows_plastic_strain():
    F = np.diag([1.05, 1.0, 1.0])
    elastic = CorotationalElastic(young=1e6, poisson=0.3)
    plastic = CorotationalPlastic(young=1e6, poisson=0.3, yield_stress=1e3)

    P_e, _ = elastic.compute_stress(F)
    P_p, _, eps_p_new = plastic.compute_stress(F, _zero_eps_p())

    # Plastic stress strictly smaller magnitude than the unyielded elastic.
    assert np.abs(P_p[0, 0]) < np.abs(P_e[0, 0])
    # Plastic flow direction is deviatoric ⇒ trace is zero.
    np.testing.assert_allclose(np.trace(eps_p_new), 0.0, atol=1e-12)
    # Plastic strain magnitude is non-trivial.
    assert np.linalg.norm(eps_p_new) > 1e-3


def test_perfect_plasticity_obeys_yield_surface():
    """Under perfect plasticity (H=0) the deviatoric stress norm should
    sit on the yield surface (within float tolerance) after loading."""
    plastic = CorotationalPlastic(young=1e6, poisson=0.3, yield_stress=2e3)
    F = np.diag([1.10, 1.0, 1.0])
    sigma, R, _ = plastic.compute_stress(F, _zero_eps_p())
    # Stress in corotated frame: T = R^T · P
    T = R.T @ sigma
    dev = T - (np.trace(T) / 3.0) * np.eye(3)
    norm_dev = np.linalg.norm(dev)
    yield_radius = np.sqrt(2.0 / 3.0) * plastic.yield_stress
    np.testing.assert_allclose(norm_dev, yield_radius, rtol=1e-10)


def test_hardening_retains_more_plastic_strain_on_unload():
    F_load = np.diag([1.05, 1.0, 1.0])
    F_unload = np.eye(3)

    perfect = CorotationalPlastic(young=1e6, poisson=0.3,
                                  yield_stress=1e3, hardening=0.0)
    hardening = CorotationalPlastic(young=1e6, poisson=0.3,
                                    yield_stress=1e3, hardening=1e5)

    _, _, eps_p_perf_loaded = perfect.compute_stress(F_load, _zero_eps_p())
    _, _, eps_p_perf_unloaded = perfect.compute_stress(F_unload, eps_p_perf_loaded)

    _, _, eps_p_hard_loaded = hardening.compute_stress(F_load, _zero_eps_p())
    _, _, eps_p_hard_unloaded = hardening.compute_stress(F_unload, eps_p_hard_loaded)

    perfect_retention = np.linalg.norm(eps_p_perf_unloaded) / np.linalg.norm(eps_p_perf_loaded)
    hardening_retention = np.linalg.norm(eps_p_hard_unloaded) / np.linalg.norm(eps_p_hard_loaded)

    # Hardening keeps more residual plastic strain than perfect plasticity.
    assert hardening_retention > perfect_retention
    # And it should be a meaningful fraction (sanity).
    assert hardening_retention > 0.10


def test_pure_rotation_produces_no_plastic_flow():
    """A rigid-body rotation has F=R, S=I → zero strain → no yielding."""
    theta = 0.7  # radians
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    plastic = CorotationalPlastic(young=1e6, poisson=0.3, yield_stress=1.0)
    P, R_out, eps_p_new = plastic.compute_stress(R, _zero_eps_p())

    np.testing.assert_allclose(R_out, R, atol=1e-12)
    # Polar decomposition residual leaks a tiny stress; permit float noise.
    np.testing.assert_allclose(P, np.zeros((3, 3)), atol=1e-8)
    assert np.linalg.norm(eps_p_new) == 0.0


def test_batch_matches_single_element():
    """The vectorised _batch_corotational_plastic_stress must agree
    element-wise with CorotationalPlastic.compute_stress under a mix of
    elastic and yielded states."""
    from robosim.physics.fem.assembly import _batch_corotational_plastic_stress
    from robosim.physics.fem.elements import polar_decomposition

    plastic = CorotationalPlastic(young=1e6, poisson=0.3,
                                  yield_stress=2e3, hardening=5e4)

    rng = np.random.default_rng(7)
    ne = 12
    # Random small-strain F with stretches in [0.97, 1.06] and rotations.
    Fs = []
    for _ in range(ne):
        # random rotation via QR
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        stretches = 1.0 + (rng.random(3) - 0.5) * 0.18
        Fs.append(Q @ np.diag(stretches))
    F_all = np.stack(Fs)

    # Random pre-existing plastic strain (deviatoric).
    eps_p_in = rng.standard_normal((ne, 3, 3)) * 0.005
    eps_p_in = 0.5 * (eps_p_in + np.transpose(eps_p_in, (0, 2, 1)))
    eps_p_in -= np.trace(eps_p_in, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)

    # Batch path
    R_all = np.stack([polar_decomposition(F)[0] for F in F_all])
    S_all = np.stack([polar_decomposition(F)[1] for F in F_all])
    P_batch, eps_p_batch = _batch_corotational_plastic_stress(
        F_all, R_all, S_all, eps_p_in,
        plastic.mu, plastic.lam, plastic.yield_stress, plastic.hardening,
    )

    # Loop path
    P_loop = np.zeros_like(F_all)
    eps_p_loop = np.zeros_like(eps_p_in)
    for e in range(ne):
        P_e, _, eps_p_e = plastic.compute_stress(F_all[e], eps_p_in[e])
        P_loop[e] = P_e
        eps_p_loop[e] = eps_p_e

    np.testing.assert_allclose(P_batch, P_loop, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(eps_p_batch, eps_p_loop, atol=1e-12, rtol=1e-9)


def test_plastic_strain_is_isochoric():
    """tr(eps_p) ≈ 0 should hold across multiple loading directions."""
    plastic = CorotationalPlastic(young=1e6, poisson=0.3, yield_stress=1e3)
    F_set = [
        np.diag([1.05, 1.0, 1.0]),
        np.diag([1.0, 1.04, 1.0]),
        np.diag([1.03, 1.02, 1.0]),
        np.diag([1.04, 1.04, 0.92]),
    ]
    eps_p = _zero_eps_p()
    for F in F_set:
        _, _, eps_p = plastic.compute_stress(F, eps_p)
        np.testing.assert_allclose(np.trace(eps_p), 0.0, atol=1e-10)
