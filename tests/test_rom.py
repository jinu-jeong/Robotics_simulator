"""Milestone 4 – POD basis and Galerkin-reduced mechanics."""

import numpy as np
import pytest

from src.fem.finger_model import FingerFEMModel
from src.rom.pod import PODBasis, compute_pod, storage_rank_tolerance
from src.rom.reduced_mechanics import ReducedModel
from src.utils.config import load_config


@pytest.fixture(scope="module")
def model():
    m = FingerFEMModel.from_config(load_config("fem"), nx=16, ny=4, nz=4)
    m.restrict_contact_surface("top")
    return m


@pytest.fixture(scope="module")
def snapshots(model):
    """12 contacts (4 along x, 3 across y) x 2 force levels -> physical rank 12."""
    g = model.geometry
    U, contacts, F = [], [], []
    for xr in (0.4, 0.6, 0.8, 0.95):
        for yr in (0.25, 0.5, 0.75):
            for f in (0.5, 1.7):  # ratio not a power of two -> float32 rounding breaks exact collinearity
                res, c = model.solve_normal_contact(g.point_from_relative([xr, yr, 1.0]), f)
                U.append(res.u.reshape(-1))
                contacts.append(c)
                F.append(f)
    return np.array(U).T, contacts, np.array(F)


@pytest.fixture(scope="module")
def basis(snapshots):
    return compute_pod(snapshots[0])


# ------------------------------------------------------------------ POD
def test_pod_rank_and_orthonormality(basis, snapshots):
    U = snapshots[0]
    assert basis.meta["numerical_rank"] == 12  # force levels are collinear
    assert basis.r == 12
    assert np.allclose(basis.Phi.T @ basis.Phi, np.eye(basis.r), atol=1e-10)
    # full-rank basis reproduces every snapshot
    for k in range(U.shape[1]):
        rmse, rel = basis.reconstruction_error(U[:, k])
        assert rel < 1e-9


def test_pod_energy_and_truncation_monotone(basis, snapshots):
    U = snapshots[0]
    energies = [basis.energy_fraction(r) for r in range(1, basis.r + 1)]
    assert np.all(np.diff(energies) >= -1e-15) and energies[-1] == pytest.approx(1.0)
    assert basis.modes_for_energy(0.999) <= basis.modes_for_energy(0.99999)
    errs = [basis.truncate(r).reconstruction_error(U[:, 5])[1] for r in (1, 2, 4, 8, 12)]
    assert np.all(np.diff(errs) <= 1e-12)
    assert errs[0] < 0.2  # first mode already captures the bulk of a bending field
    with pytest.raises(ValueError):
        basis.truncate(basis.r + 1)


def test_pod_fixed_dofs_are_zero_and_project_shapes(basis, model):
    fixed = model.partition.fixed_dofs
    assert np.all(np.abs(basis.Phi[fixed]) < 1e-12)
    u = basis.mode(0)
    assert u.shape == (model.mesh.n_nodes, 3)
    q = basis.project(u)
    assert q.shape == (basis.r,)
    assert np.allclose(q, np.eye(basis.r)[0], atol=1e-12)


def test_pod_float32_noise_floor_handling(snapshots):
    U = snapshots[0]
    U32 = U.astype(np.float32).astype(np.float64)
    strict = compute_pod(U32)  # 1e-12 tolerance counts quantisation noise as rank
    tolerant = compute_pod(U32, rel_tol=storage_rank_tolerance(np.float32))
    assert strict.meta["numerical_rank"] > 12
    assert tolerant.meta["numerical_rank"] == 12


def test_pod_save_load_roundtrip(basis, tmp_path):
    p = basis.save(tmp_path / "basis.npz")
    b2 = PODBasis.load(p)
    assert np.allclose(b2.Phi, basis.Phi) and np.allclose(b2.singular_values, basis.singular_values)
    assert b2.meta["numerical_rank"] == basis.meta["numerical_rank"]


# ------------------------------------------------------------------ reduced mechanics
def test_reduced_stiffness_spd_and_galerkin_consistency(model, basis, snapshots):
    rom = ReducedModel.from_fem(model.fem, basis, model.contact)
    assert np.allclose(rom.K_r, rom.K_r.T)
    assert np.all(np.linalg.eigvalsh(rom.K_r) > 0)
    # a snapshot lies in the span -> reduced forward solve reproduces the full solve
    U, contacts, F = snapshots
    k = 7
    u_rom = rom.solve(contacts[k], -F[k] * contacts[k].normal)
    assert np.allclose(u_rom.reshape(-1), U[:, k], rtol=1e-8, atol=1e-14)
    # truncation reuses the leading block
    r4 = rom.truncate(4)
    assert np.allclose(r4.K_r, rom.K_r[:4, :4]) and r4.r == 4


def test_reduced_force_recovery_in_span(model, basis, snapshots):
    U, contacts, F = snapshots
    rom = ReducedModel.from_fem(model.fem, basis, model.contact)
    for k in (0, 5, 11, 20):
        q = basis.project(U[:, k])
        for method in ("displacement", "force"):
            est = rom.estimate_force(q, contacts[k], method=method)
            assert est.magnitude == pytest.approx(F[k], rel=1e-7), (k, method)
        vec = rom.estimate_force(q, contacts[k], mode="vector")
        assert np.allclose(vec.force_vector, -F[k] * contacts[k].normal, atol=1e-8)


def test_reduced_displacement_space_estimator_is_stable_for_unseen_contact(model, basis):
    """Contact not in the snapshot set: displacement-space stays within a few %, force-space degrades."""
    g = model.geometry
    res, c = model.solve_normal_contact(g.point_from_relative([0.7, 0.4, 1.0]), 1.0)
    rom = ReducedModel.from_fem(model.fem, basis, model.contact)
    errs = {}
    for method in ("displacement", "force"):
        errs[method] = [abs(rom.truncate(r).estimate_force_from_field(res.u, c, method=method).magnitude - 1.0) for r in (2, 4, 8, 12)]
    assert max(errs["displacement"]) < 0.05
    assert errs["force"][-1] > errs["displacement"][-1]


def test_reduced_estimator_filters_noise_at_small_r(model, basis, snapshots):
    U, contacts, F = snapshots
    rng = np.random.default_rng(0)
    k = 3
    sigma = 1e-2 * np.abs(U[:, k]).max()
    un = U[:, k] + rng.normal(0.0, sigma, size=U.shape[0]) * (np.abs(basis.Phi).sum(axis=1) > 0)  # noise on free dofs only
    rom = ReducedModel.from_fem(model.fem, basis, model.contact)
    e_disp = abs(rom.truncate(4).estimate_force_from_field(un, contacts[k]).magnitude - F[k]) / F[k]
    e_force_r1 = abs(rom.truncate(1).estimate_force_from_field(un, contacts[k], method="force").magnitude - F[k]) / F[k]
    e_force_r12 = abs(rom.truncate(12).estimate_force_from_field(un, contacts[k], method="force", nonneg=False).magnitude - F[k]) / F[k]
    assert e_disp < 0.05
    assert e_force_r1 < 0.1
    assert e_force_r12 > e_force_r1  # K_r amplification returns as more modes admit noise


def test_reduced_model_rejects_wrong_sizes(model, basis, snapshots):
    rom = ReducedModel.from_fem(model.fem, basis, model.contact)
    with pytest.raises(ValueError):
        rom.estimate_force(np.zeros(basis.r + 1), snapshots[1][0])
    with pytest.raises(ValueError):
        rom.estimate_force(np.zeros(basis.r), snapshots[1][0], method="nope")
