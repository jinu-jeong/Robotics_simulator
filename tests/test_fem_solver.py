"""System-level FEM tests: assembly, boundary conditions, solver, loads, stress."""

import numpy as np
import pytest

from src.fem.assembly import assemble_global_stiffness, symmetry_error
from src.fem.beam_theory import CantileverReference
from src.fem.boundary import DofPartition, clamp_nodes, clamp_plane, combine, node_dofs
from src.fem.loads import (
    distributed_node_load,
    faces_on_plane,
    point_load,
    resultant,
    total_force_on_faces,
    total_force_on_plane,
    uniform_traction_load,
)
from src.fem.material import LinearElasticMaterial
from src.fem.solver import LinearFEM, solve_linear_fem
from src.fem.stress import von_mises
from src.geometry.finger import FingerGeometry, make_rectangular_finger_mesh, root_fixed_nodes
from src.geometry.mesh_utils import triangle_areas

GEOM = FingerGeometry(0.1, 0.02, 0.01)
MAT = LinearElasticMaterial(5.0e7, 0.3)


@pytest.fixture(scope="module")
def mesh():
    return make_rectangular_finger_mesh(GEOM, 8, 2, 2)


@pytest.fixture(scope="module")
def clamped(mesh):
    return LinearFEM(mesh, MAT, clamp_nodes(mesh.n_nodes, root_fixed_nodes(mesh)))


# ------------------------------------------------------------------ assembly
def test_global_stiffness_symmetric_and_rigid_translation_free(mesh):
    K = assemble_global_stiffness(mesh.nodes, mesh.element_blocks, MAT)
    assert K.shape == (mesh.n_dofs, mesh.n_dofs)
    assert symmetry_error(K) < 1e-12
    for comp in range(3):
        u = np.zeros((mesh.n_nodes, 3))
        u[:, comp] = 1.0
        assert np.allclose(K @ u.reshape(-1), 0.0, atol=1e-9 * abs(K).max())


def test_global_stiffness_psd(mesh):
    K = assemble_global_stiffness(mesh.nodes, mesh.element_blocks, MAT).toarray()
    w = np.linalg.eigvalsh(K)
    assert w.min() > -1e-9 * np.abs(K).max()
    assert np.sum(np.abs(w) < 1e-8 * np.abs(K).max()) == 6


# ------------------------------------------------------------------ boundary
def test_dof_partition_bookkeeping(mesh):
    idx = root_fixed_nodes(mesh)
    part = clamp_nodes(mesh.n_nodes, idx)
    assert part.n_fixed == 3 * len(idx)
    assert part.n_free + part.n_fixed == mesh.n_dofs
    assert set(part.fully_fixed_nodes()) == set(idx)
    assert node_dofs([2], (2,)).tolist() == [8]
    with pytest.raises(ValueError):
        DofPartition(mesh.n_dofs, [mesh.n_dofs])
    partial = clamp_plane(mesh.nodes, 0, 0.0, components=(0,))
    assert partial.n_fixed == len(idx)
    assert len(partial.fully_fixed_nodes()) == 0


def test_fixed_dofs_enforced_exactly(clamped):
    mesh = clamped.mesh
    f = point_load(mesh.n_nodes, mesh.n_nodes - 1, [0.3, -0.2, -1.0])
    r = clamped.solve(f)
    assert np.allclose(r.u[r.fixed_nodes], 0.0)
    assert r.u.shape == (mesh.n_nodes, 3)


def test_prescribed_displacement(mesh):
    """Uniform axial stretch imposed as Dirichlet data -> exact constant strain."""
    root = root_fixed_nodes(mesh)
    tip = mesh.nodes_on_plane(0, GEOM.length)
    fixed = np.concatenate([node_dofs(root, (0,)), node_dofs(tip, (0,))])
    prescribed = np.concatenate([np.zeros(len(root)), np.full(len(tip), 1e-4)])
    # also fix lateral rigid modes (rollers on y = 0 and z = 0)
    lateral = np.concatenate([node_dofs(mesh.nodes_on_plane(1, 0.0), (1,)), node_dofs(mesh.nodes_on_plane(2, 0.0), (2,))])
    part = DofPartition(mesh.n_dofs, np.concatenate([fixed, lateral]), np.concatenate([prescribed, np.zeros(len(lateral))]))
    r = LinearFEM(mesh, MAT, part).solve(np.zeros(mesh.n_dofs))
    assert np.allclose(r.u[:, 0], 1e-4 * mesh.nodes[:, 0] / GEOM.length, atol=1e-12)
    assert r.reactions[tip, 0].sum() == pytest.approx(MAT.E * 1e-4 / GEOM.length * GEOM.width * GEOM.height, rel=1e-9)


# ------------------------------------------------------------------ solver
def test_zero_force_zero_displacement(clamped):
    r = clamped.solve(np.zeros(clamped.n_dof))
    assert np.allclose(r.u, 0.0)
    assert np.allclose(r.reactions, 0.0)


def test_linearity_force_scaling(clamped):
    mesh = clamped.mesh
    f = point_load(mesh.n_nodes, int(np.argmax(mesh.nodes[:, 0] + mesh.nodes[:, 2])), [0, 0, -1.0])
    u1 = clamped.solve(f).u
    u2 = clamped.solve(2.0 * f).u
    assert np.allclose(u2, 2.0 * u1, rtol=1e-10, atol=1e-16)
    assert u1[:, 2].min() < 0  # bends in the load direction


def test_superposition(clamped):
    mesh = clamped.mesh
    fa = point_load(mesh.n_nodes, 10, [0, 0, -1.0])
    fb = point_load(mesh.n_nodes, mesh.n_nodes - 1, [0.5, 0, 0])
    ua, ub, uab = (clamped.solve(x).u for x in (fa, fb, fa + fb))
    assert np.allclose(uab, ua + ub, rtol=1e-10, atol=1e-16)


def test_equilibrium_reactions_balance_applied_load(clamped):
    mesh = clamped.mesh
    f = distributed_node_load(mesh.n_nodes, mesh.nodes_on_plane(0, GEOM.length), [0.2, 0.1, -1.0])
    r = clamped.solve(f)
    assert np.allclose(r.applied_resultant(), [0.2, 0.1, -1.0])
    assert r.equilibrium_residual() < 1e-10
    # reactions only at constrained dofs
    free_nodes = np.setdiff1d(np.arange(mesh.n_nodes), r.fixed_nodes)
    assert np.allclose(r.reactions[free_nodes], 0.0)


def test_uniaxial_tension_is_exact():
    """Constant-strain problem: H20 + tied T10 reproduce σ = F/A, ε = σ/E exactly."""
    mesh = make_rectangular_finger_mesh(GEOM, 5, 2, 2, tet_fraction=0.4)
    assert mesh.n_ties > 0
    part = combine(
        clamp_plane(mesh.nodes, 0, 0.0, (0,)),
        clamp_plane(mesh.nodes, 1, 0.0, (1,)),
        clamp_plane(mesh.nodes, 2, 0.0, (2,)),
    )
    F = 2.0
    faces = faces_on_plane(mesh.nodes, mesh.surface_faces, 0, GEOM.length)
    assert triangle_areas(mesh.nodes, faces).sum() == pytest.approx(GEOM.width * GEOM.height)
    f = total_force_on_plane(mesh, 0, GEOM.length, [F, 0, 0])  # consistent quad8 traction
    assert np.allclose(resultant(f), [F, 0, 0])
    assert np.allclose(resultant(total_force_on_faces(mesh.nodes, faces, [F, 0, 0])), [F, 0, 0])
    fem = LinearFEM(mesh, MAT, part)
    r = fem.solve(f)
    sigma = F / (GEOM.width * GEOM.height)
    assert np.allclose(r.u[:, 0], sigma / MAT.E * mesh.nodes[:, 0], rtol=1e-9, atol=1e-15)
    assert np.allclose(r.u[:, 1], -MAT.nu * sigma / MAT.E * mesh.nodes[:, 1], rtol=1e-9, atol=1e-15)
    s = fem.stresses(r)
    assert np.allclose(s["stress"][:, 0], sigma, rtol=1e-9)
    assert np.allclose(s["stress"][:, 1:], 0.0, atol=1e-9 * sigma)
    assert np.allclose(s["von_mises"], sigma, rtol=1e-9)
    assert np.allclose(s["von_mises_nodal"], sigma, rtol=1e-9)


def test_cantilever_converges_towards_beam_theory():
    """Quadratic elements: a few cells through the thickness already give ~1 % vs Timoshenko."""
    F = 1.0
    ref = CantileverReference(GEOM.length, GEOM.width, GEOM.height, MAT.E, MAT.nu, F)
    errs = []
    for res in [(5, 1, 1), (10, 2, 2), (20, 4, 4)]:
        m = make_rectangular_finger_mesh(GEOM, *res)
        f = total_force_on_plane(m, 0, GEOM.length, [0, 0, -F])
        r = solve_linear_fem(m, MAT, clamp_nodes(m.n_nodes, root_fixed_nodes(m)), f)
        tip = -r.u[m.nodes_on_plane(0, GEOM.length), 2].mean()
        errs.append(tip / ref.tip_deflection_timoshenko - 1.0)
    assert all(e < 0 for e in errs)  # stiffer than theory
    assert errs[0] < errs[1] < errs[2]  # monotone convergence
    assert abs(errs[-1]) < 0.03
    assert abs(errs[0]) < 0.15  # a single H20 through the thickness is already within 15 %


def test_uniform_traction_consistent_loads(mesh):
    faces = faces_on_plane(mesh.nodes, mesh.surface_faces, 2, GEOM.height)
    f = uniform_traction_load(mesh.nodes, faces, [0, 0, -1000.0])
    assert resultant(f)[2] == pytest.approx(-1000.0 * GEOM.length * GEOM.width)
    loaded = np.nonzero(np.abs(f.reshape(-1, 3)[:, 2]) > 0)[0]
    assert np.allclose(mesh.nodes[loaded, 2], GEOM.height)


def test_von_mises_pure_shear_and_hydrostatic():
    assert von_mises(np.array([[0, 0, 0, 3.0, 0, 0]]))[0] == pytest.approx(3.0 * np.sqrt(3))
    assert von_mises(np.array([[5.0, 5.0, 5.0, 0, 0, 0]]))[0] == pytest.approx(0.0)


def test_result_to_visualization_state(clamped):
    mesh = clamped.mesh
    r = clamped.solve(point_load(mesh.n_nodes, mesh.n_nodes - 1, [0, 0, -1.0]))
    st = r.to_visualization_state(node_scalar=clamped.stresses(r)["von_mises_nodal"], node_scalar_name="von Mises [Pa]")
    assert st.n_nodes == mesh.n_nodes
    assert np.allclose(st.displacement, r.u)
    assert set(st.fixed_nodes) == set(root_fixed_nodes(mesh))
    assert "equilibrium residual" in st.info
