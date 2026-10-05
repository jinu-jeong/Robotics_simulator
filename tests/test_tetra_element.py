"""Element-level tests: material matrix, B matrix, K_e properties."""

import numpy as np
import pytest

from src.fem.material import LinearElasticMaterial
from src.fem.tetra_element import (
    element_dof_indices,
    element_stiffness_matrices,
    shape_gradients,
    strain_displacement_matrices,
)

MAT = LinearElasticMaterial(youngs_modulus=1.0e6, poisson_ratio=0.3)
UNIT_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
RNG = np.random.default_rng(0)


def _random_tets(n=5):
    """Random positively-oriented tets."""
    nodes = RNG.normal(size=(4 * n, 3))
    tets = np.arange(4 * n).reshape(n, 4)
    _, vol = shape_gradients(nodes, tets)
    neg = vol < 0
    tets[neg] = tets[neg][:, [0, 1, 3, 2]]
    return nodes, tets


def test_material_matrix_symmetric_positive_definite():
    C = MAT.stiffness_matrix()
    assert np.allclose(C, C.T)
    assert np.all(np.linalg.eigvalsh(C) > 0)
    lam, mu = MAT.lame_lambda, MAT.lame_mu
    assert C[0, 0] == pytest.approx(lam + 2 * mu)
    assert C[0, 1] == pytest.approx(lam)
    assert C[3, 3] == pytest.approx(mu)
    assert MAT.shear_modulus == pytest.approx(1e6 / 2.6)
    with pytest.raises(ValueError):
        LinearElasticMaterial(1e6, 0.5)


def test_shape_gradients_unit_tet_and_partition_of_unity():
    grads, vol = shape_gradients(UNIT_TET, np.array([[0, 1, 2, 3]]))
    assert vol[0] == pytest.approx(1 / 6)
    assert np.allclose(grads[0, 1:], np.eye(3))
    assert np.allclose(grads[0, 0], [-1, -1, -1])
    assert np.allclose(grads.sum(axis=1), 0.0)  # Σ ∇N_i = 0


def test_gradients_reproduce_linear_field():
    nodes, tets = _random_tets(4)
    grads, _ = shape_gradients(nodes, tets)
    a = np.array([0.3, -1.2, 0.7])
    phi = nodes @ a + 2.0  # linear field
    grad_fem = np.einsum("mi,mij->mj", phi[tets], grads)
    assert np.allclose(grad_fem, a)


def test_B_matrix_gives_exact_constant_strain():
    nodes, tets = _random_tets(3)
    grads, _ = shape_gradients(nodes, tets)
    B = strain_displacement_matrices(grads)
    # u = G x with a symmetric gradient G -> strain = G (no rotation part)
    G = np.array([[1e-3, 2e-4, 0], [2e-4, -5e-4, 3e-4], [0, 3e-4, 4e-4]])
    u = nodes @ G.T
    ue = u.reshape(-1)[element_dof_indices(tets)]
    eps = np.einsum("mij,mj->mi", B, ue)
    expected = [G[0, 0], G[1, 1], G[2, 2], 2 * G[0, 1], 2 * G[1, 2], 2 * G[0, 2]]
    assert np.allclose(eps, expected)


def test_element_stiffness_symmetric_psd_with_six_rigid_modes():
    nodes, tets = _random_tets(6)
    Ke, _, vol = element_stiffness_matrices(nodes, tets, MAT.stiffness_matrix())
    assert np.all(vol > 0)
    for K in Ke:
        assert np.allclose(K, K.T, atol=1e-9 * np.abs(K).max())
        w = np.linalg.eigvalsh(K)
        scale = np.abs(K).max()
        assert w.min() > -1e-9 * scale  # positive semi-definite
        assert np.sum(np.abs(w) < 1e-8 * scale) == 6  # exactly 6 zero-energy (rigid) modes


def test_rigid_body_motion_produces_zero_force():
    nodes, tets = _random_tets(3)
    Ke, _, _ = element_stiffness_matrices(nodes, tets, MAT.stiffness_matrix())
    t = np.array([0.1, -0.2, 0.3])
    omega = np.array([1e-3, -2e-3, 5e-4])  # small rotation
    for e in range(len(tets)):
        x = nodes[tets[e]]
        u = t + np.cross(omega, x)
        f = Ke[e] @ u.reshape(-1)
        assert np.allclose(f, 0.0, atol=1e-9 * np.abs(Ke[e]).max())


def test_negative_orientation_rejected():
    tets = np.array([[0, 1, 3, 2]])
    with pytest.raises(ValueError):
        element_stiffness_matrices(UNIT_TET, tets, MAT.stiffness_matrix())


def test_element_dof_indices():
    d = element_dof_indices(np.array([[2, 0, 5, 1]]))
    assert d.tolist() == [[6, 7, 8, 0, 1, 2, 15, 16, 17, 3, 4, 5]]
