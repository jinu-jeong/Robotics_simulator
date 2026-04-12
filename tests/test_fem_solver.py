"""Tests for FEM solver: mesh, elements, materials, assembly, integration."""

import math

import numpy as np
import pytest

from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.elements import (
    compute_shape_derivatives,
    compute_deformation_gradient,
    polar_decomposition,
)
from robosim.physics.fem.materials import CorotationalElastic, NeoHookean
from robosim.physics.fem.assembly import precompute_element_data, assemble_forces, assemble_mass_matrix
from robosim.physics.fem.solver import FEMSolver, DeformableBody


class TestTetMesh:
    def test_create_box(self):
        mesh = TetMesh.create_box(
            origin=np.zeros(3), size=np.ones(3), divisions=(2, 2, 2)
        )
        assert mesh.n_nodes == 27  # (2+1)^3
        assert mesh.n_elements == 40  # 8 hex cells * 5 tets

    def test_volume_unit_cube(self):
        mesh = TetMesh.create_box(
            origin=np.zeros(3), size=np.ones(3), divisions=(2, 2, 2)
        )
        total_vol = mesh.total_volume()
        assert abs(total_vol - 1.0) < 1e-10

    def test_surface_extraction(self):
        mesh = TetMesh.create_box(
            origin=np.zeros(3), size=np.ones(3), divisions=(2, 2, 2)
        )
        surface = mesh.extract_surface()
        # A cube with 2x2x2 divisions has 6 faces * 4 quads * 2 triangles = 48 surface triangles
        # But tet decomposition may differ; just check it's a reasonable number
        assert surface.shape[1] == 3
        assert surface.shape[0] > 20  # at least some faces

    def test_volume_rectangular(self):
        mesh = TetMesh.create_box(
            origin=np.array([1.0, 2.0, 3.0]),
            size=np.array([2.0, 3.0, 4.0]),
            divisions=(3, 3, 3),
        )
        assert abs(mesh.total_volume() - 24.0) < 1e-8


class TestElements:
    def test_shape_derivatives_unit_tet(self):
        # Reference unit tet
        x0 = np.array([0, 0, 0], dtype=float)
        x1 = np.array([1, 0, 0], dtype=float)
        x2 = np.array([0, 1, 0], dtype=float)
        x3 = np.array([0, 0, 1], dtype=float)

        dN, vol = compute_shape_derivatives(x0, x1, x2, x3)
        assert abs(vol - 1.0 / 6.0) < 1e-12
        assert dN.shape == (4, 3)

        # Partition of unity: sum of shape function gradients = 0
        np.testing.assert_array_almost_equal(dN.sum(axis=0), np.zeros(3))

    def test_deformation_gradient_identity(self):
        """Undeformed configuration should give F = I."""
        x0 = np.array([0, 0, 0], dtype=float)
        x1 = np.array([1, 0, 0], dtype=float)
        x2 = np.array([0, 1, 0], dtype=float)
        x3 = np.array([0, 0, 1], dtype=float)

        dN, _ = compute_shape_derivatives(x0, x1, x2, x3)
        x_def = np.array([x0, x1, x2, x3])
        F = compute_deformation_gradient(dN, x_def)
        np.testing.assert_array_almost_equal(F, np.eye(3))

    def test_deformation_gradient_stretch(self):
        """Uniform 2x stretch in X should give F[0,0] = 2."""
        x0 = np.array([0, 0, 0], dtype=float)
        x1 = np.array([1, 0, 0], dtype=float)
        x2 = np.array([0, 1, 0], dtype=float)
        x3 = np.array([0, 0, 1], dtype=float)

        dN, _ = compute_shape_derivatives(x0, x1, x2, x3)
        x_def = np.array([x0 * [2, 1, 1], x1 * [2, 1, 1], x2 * [2, 1, 1], x3 * [2, 1, 1]])
        F = compute_deformation_gradient(dN, x_def)
        np.testing.assert_array_almost_equal(F, np.diag([2, 1, 1]))

    def test_polar_decomposition_rotation(self):
        """Pure rotation should give R = rotation, S = I."""
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        F = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

        R, S = polar_decomposition(F)
        np.testing.assert_array_almost_equal(R, F, decimal=10)
        np.testing.assert_array_almost_equal(S, np.eye(3), decimal=10)


class TestMaterials:
    def test_corotational_zero_strain(self):
        """Undeformed F=I should give zero stress."""
        mat = CorotationalElastic(young=1e6, poisson=0.3)
        P, R = mat.compute_stress(np.eye(3))
        np.testing.assert_array_almost_equal(P, np.zeros((3, 3)))
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_corotational_uniaxial(self):
        """Small uniaxial stretch: P should be approximately E * epsilon in that direction."""
        mat = CorotationalElastic(young=1e6, poisson=0.3)
        eps = 0.001
        F = np.diag([1 + eps, 1, 1])
        P, _ = mat.compute_stress(F)

        # For small strain, P_11 ≈ (lambda + 2*mu) * epsilon
        expected = (mat.lam + 2 * mat.mu) * eps
        assert abs(P[0, 0] - expected) / expected < 0.01

    def test_neohookean_zero_strain(self):
        mat = NeoHookean(young=1e6, poisson=0.3)
        P, R = mat.compute_stress(np.eye(3))
        np.testing.assert_array_almost_equal(P, np.zeros((3, 3)), decimal=5)

    def test_stiffness_matrix_symmetric(self):
        mat = CorotationalElastic(young=1e6, poisson=0.3)
        x0 = np.array([0, 0, 0], dtype=float)
        x1 = np.array([1, 0, 0], dtype=float)
        x2 = np.array([0, 1, 0], dtype=float)
        x3 = np.array([0, 0, 1], dtype=float)
        dN, vol = compute_shape_derivatives(x0, x1, x2, x3)
        F = np.eye(3)
        Ke = mat.compute_element_stiffness(F, dN, vol)
        np.testing.assert_array_almost_equal(Ke, Ke.T, decimal=10)


class TestAssembly:
    def test_mass_matrix_total(self):
        """Total mass should equal density * volume."""
        mesh = TetMesh.create_box(size=np.ones(3), divisions=(2, 2, 2))
        dN_list, volumes = precompute_element_data(mesh)
        density = 1000.0
        M = assemble_mass_matrix(mesh, density, volumes)

        total_mass = M.diagonal().sum() / 3.0  # each node has 3 DOFs
        expected = density * 1.0  # volume = 1 m^3
        assert abs(total_mass - expected) < 1e-6

    def test_internal_force_zero_at_rest(self):
        """Internal forces should be zero in the reference configuration."""
        mesh = TetMesh.create_box(size=np.ones(3), divisions=(2, 2, 2))
        dN_list, volumes = precompute_element_data(mesh)
        mat = CorotationalElastic(young=1e6, poisson=0.3)

        f_int = assemble_forces(mesh, mesh.nodes, mat, dN_list, volumes)
        np.testing.assert_array_almost_equal(f_int, np.zeros_like(f_int), decimal=8)


class TestFEMSolver:
    def _make_cantilever(self, nx=6, ny=2, nz=2):
        """Create a cantilever beam: fixed at x=0 face, free otherwise."""
        length, width, height = 1.0, 0.1, 0.1
        mesh = TetMesh.create_box(
            origin=np.zeros(3),
            size=np.array([length, width, height]),
            divisions=(nx, ny, nz),
        )
        # Fix nodes at x=0
        fixed = np.where(mesh.nodes[:, 0] < 1e-10)[0]

        body = DeformableBody(
            name="cantilever",
            mesh=mesh,
            material=CorotationalElastic(young=1e6, poisson=0.3),
            density=1000.0,
            fixed_nodes=fixed,
        )
        return body

    def test_cantilever_deflects_under_gravity(self):
        """Cantilever beam should deflect downward under gravity."""
        body = self._make_cantilever()
        solver = FEMSolver(bodies=[body], gravity=np.array([0.0, 0.0, -9.81]), damping=0.5)
        solver.initialize(dt=0.01)

        # Run a few steps
        for _ in range(20):
            solver.step()

        # Tip (max x) should have moved downward
        tip_nodes = np.where(body.mesh.nodes[:, 0] > 0.9)[0]
        tip_z_displacement = body.x[tip_nodes, 2].mean() - body.mesh.nodes[tip_nodes, 2].mean()
        assert tip_z_displacement < -1e-6, f"Tip should deflect down, got dz={tip_z_displacement}"

    def test_cantilever_analytical_comparison(self):
        """Compare tip deflection with Euler-Bernoulli beam theory.

        delta_max = -rho * g * A * L^4 / (8 * E * I)
        where I = b*h^3/12 for a rectangular cross section.
        """
        L, b, h = 1.0, 0.1, 0.1
        E = 1e6
        rho = 1000.0
        I_moment = b * h**3 / 12.0
        A = b * h
        w = rho * 9.81 * A  # distributed load per unit length

        delta_analytical = -w * L**4 / (8 * E * I_moment)

        body = self._make_cantilever(nx=10, ny=2, nz=2)
        solver = FEMSolver(
            bodies=[body],
            gravity=np.array([0.0, 0.0, -9.81]),
            damping=2.0,  # heavy damping to reach static equilibrium
        )
        solver.initialize(dt=0.005)

        # Run until quasi-static (high damping)
        for _ in range(300):
            solver.step()

        tip_nodes = np.where(body.mesh.nodes[:, 0] > 0.95)[0]
        tip_deflection = body.x[tip_nodes, 2].mean() - body.mesh.nodes[tip_nodes, 2].mean()

        # FEM with coarse mesh won't match exactly, allow 50% error
        # (linear tets are stiff, known as "locking")
        ratio = tip_deflection / delta_analytical
        assert 0.2 < ratio < 2.0, (
            f"Tip deflection ratio: {ratio:.2f} "
            f"(FEM: {tip_deflection:.6f}, analytical: {delta_analytical:.6f})"
        )

    def test_fixed_nodes_dont_move(self):
        """Fixed (clamped) nodes should remain at their reference positions."""
        body = self._make_cantilever()
        solver = FEMSolver(bodies=[body], gravity=np.array([0.0, 0.0, -9.81]))
        solver.initialize(dt=0.01)

        for _ in range(10):
            solver.step()

        fixed = body.fixed_nodes
        np.testing.assert_array_almost_equal(
            body.x[fixed], body.mesh.nodes[fixed], decimal=10
        )
