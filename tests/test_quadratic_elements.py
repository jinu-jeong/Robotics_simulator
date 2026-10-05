"""Element-level tests for the quadratic T10 / H20 library and the hanging-node ties."""

import numpy as np
import pytest

from src.fem.boundary import clamp_nodes
from src.fem.elements import HEX20, TET10, element_stiffness, element_volumes, h20_shape, interpolate, t10_shape
from src.fem.loads import total_force_on_plane
from src.fem.material import LinearElasticMaterial
from src.fem.solver import LinearFEM
from src.geometry.finger import FingerGeometry, make_rectangular_finger_mesh

MAT = LinearElasticMaterial(1.0e6, 0.3, 1000.0)
GEOM = FingerGeometry(0.1, 0.02, 0.01)
RNG = np.random.default_rng(1)


def _unit_elements():
    tet_c = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    hex_c = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    out = []
    for et, c in ((TET10, tet_c), (HEX20, hex_c)):
        mids = 0.5 * (c[et.edges[:, 0]] + c[et.edges[:, 1]])
        out.append((et, np.concatenate([c, mids])))
    return out


@pytest.mark.parametrize("et,shape", [(TET10, t10_shape), (HEX20, h20_shape)])
def test_partition_of_unity_and_kronecker(et, shape):
    xi = RNG.uniform(0, 0.3, (7, 3)) if et is TET10 else RNG.uniform(-1, 1, (7, 3))
    N, dN = shape(xi)
    assert np.allclose(N.sum(axis=1), 1.0)
    assert np.allclose(dN.sum(axis=1), 0.0)
    # N_i(node_j) = δ_ij using the element's own node coordinates in natural space
    for _, nodes in _unit_elements():
        if len(nodes) != et.n_nodes:
            continue
        nat = nodes if et is TET10 else 2.0 * nodes - 1.0
        Nn, _ = shape(nat)
        assert np.allclose(Nn, np.eye(et.n_nodes), atol=1e-12)


@pytest.mark.parametrize("idx", [0, 1])
def test_element_stiffness_rigid_modes_and_mass(idx):
    et, nodes = _unit_elements()[idx]
    # random affine distortion keeps the element valid and the quadrature exact
    A = np.eye(3) + 0.2 * RNG.normal(size=(3, 3))
    if np.linalg.det(A) < 0:
        A[:, 0] *= -1
    x = nodes @ A.T + 0.3
    conn = np.arange(et.n_nodes)[None, :]
    Ke, Bbar, V, m = element_stiffness(x, conn, MAT.stiffness_matrix())
    vol_ref = (1.0 / 6.0 if et is TET10 else 1.0) * abs(np.linalg.det(A))
    assert V[0] == pytest.approx(vol_ref, rel=1e-12)
    assert m.sum() == pytest.approx(vol_ref, rel=1e-12)
    assert element_volumes(x, conn)[0] == pytest.approx(vol_ref, rel=1e-12)
    K = Ke[0]
    assert np.allclose(K, K.T, atol=1e-9 * abs(K).max())
    w = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(w) < 1e-9 * w.max()) == 6  # 3 translations + 3 rotations
    rot = np.cross([0.3, -0.7, 0.2], x).reshape(-1)
    assert np.allclose(K @ rot, 0.0, atol=1e-9 * abs(K).max())
    # linear field -> exact constant strain from the mean B
    F = 1e-3 * RNG.normal(size=(3, 3))
    u = (x @ F.T).reshape(-1)
    eps = 0.5 * (F + F.T)
    voigt = np.array([eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[0, 1], 2 * eps[1, 2], 2 * eps[0, 2]])
    assert np.allclose(Bbar[0] @ u, voigt, atol=1e-15)
    # interpolation reproduces the geometry (isoparametric) and a quadratic field at a point
    xi = et.quad_points[0]
    xq = interpolate(conn[0], et, xi, x)
    fq = interpolate(conn[0], et, xi, (x**2).sum(axis=1)[:, None])
    assert np.allclose(fq[0], (xq**2).sum())


def test_element_stiffness_rejects_inverted_elements():
    et, nodes = _unit_elements()[0]
    bad = nodes[[0, 2, 1, 3, 6, 5, 4, 7, 9, 8]]  # swap corners 1,2 (and their edges)
    with pytest.raises(ValueError):
        element_stiffness(bad, np.arange(10)[None, :], MAT.stiffness_matrix())


def test_tied_mesh_patch_test_uniform_strain():
    """Hanging-node ties must transmit a linear displacement field exactly (patch test)."""
    mesh = make_rectangular_finger_mesh(GEOM, 10, 2, 2, tet_fraction=0.3)
    assert mesh.n_ties > 0
    part = clamp_nodes(mesh.n_nodes, mesh.nodes_on_plane(0, 0.0))
    fem = LinearFEM(mesh, MAT, part)
    # ties as a linear map: slaves are exactly interpolated from their masters
    u_lin = fem.T @ np.tile(np.array([1e-3, -2e-3, 0.5e-3]), mesh.n_nodes)
    assert np.allclose(u_lin.reshape(-1, 3), [1e-3, -2e-3, 0.5e-3])
    # axial pull: constant stress away from the clamped root; ties satisfied to machine precision
    F = 1.0
    res = fem.solve(total_force_on_plane(mesh, 0, GEOM.length, [F, 0, 0]))
    us = res.u[mesh.tie_slaves]
    um = np.einsum("sk,skj->sj", mesh.tie_weights, res.u[mesh.tie_masters])
    assert np.allclose(us, um, atol=1e-15)
    s = fem.stresses(res)
    sigma = F / (GEOM.width * GEOM.height)
    far = mesh.nodes[mesh.tets[:, :4]].mean(axis=1)[:, 0] > 0.5 * GEOM.length  # tet region (tip)
    assert np.allclose(s["stress"][: mesh.n_tets][far, 0], sigma, rtol=2e-2)
    assert res.equilibrium_residual() < 1e-10
    assert res.reaction_resultant()[0] == pytest.approx(-F, rel=1e-10)


def test_body_force_uses_consistent_mass_shares():
    mesh = make_rectangular_finger_mesh(GEOM, 6, 2, 2)
    fem = LinearFEM(mesh, MAT, clamp_nodes(mesh.n_nodes, mesh.nodes_on_plane(0, 0.0)))
    f = fem.body_force_vector([0, 0, -9.81]).reshape(-1, 3)
    assert f.sum(axis=0) == pytest.approx([0, 0, -9.81 * MAT.density * GEOM.length * GEOM.width * GEOM.height])
    # consistent (not lumped) shares: mid-edge nodes carry positive mass, corners negative
    n_corner = 7 * 3 * 3
    assert np.all(f[n_corner:, 2] < 0)
    assert np.all(f[:n_corner, 2] > 0)
