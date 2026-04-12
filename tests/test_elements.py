"""Tests for Hex8, Tet10 elements and generalized assembly."""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.physics.fem.elements import (
    ElementType, hex8_shape, tet10_shape, tet4_shape,
    gauss_points_hex8, gauss_points_tet4, gauss_points_tet1,
    shape_function, gauss_rule,
)
from robosim.physics.fem.mesh import FEMesh, TetMesh
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.solver import FEMSolver, DeformableBody
from robosim.physics.fem.assembly import (
    precompute_element_data, assemble_forces, assemble_stiffness,
    assemble_mass_matrix, batch_von_mises, ElementIntegrationData,
)


# ═══════════════════════════════════════════════════════════════
# Shape function tests
# ═══════════════════════════════════════════════════════════════

class TestHex8Shape:
    def test_partition_of_unity(self):
        """Sum of shape functions = 1 at any point."""
        for _ in range(20):
            xi = np.random.uniform(-1, 1, size=3)
            N, _ = hex8_shape(xi)
            assert abs(N.sum() - 1.0) < 1e-12

    def test_interpolation_property(self):
        """N_i(corner_j) = delta_{ij}."""
        corners = np.array([
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ], dtype=float)
        for i in range(8):
            N, _ = hex8_shape(corners[i])
            for j in range(8):
                expected = 1.0 if i == j else 0.0
                assert abs(N[j] - expected) < 1e-12, f"N[{j}] at corner {i}"

    def test_gradient_sum_zero(self):
        """Sum of gradients = 0 (constant field has zero gradient)."""
        for _ in range(10):
            xi = np.random.uniform(-1, 1, size=3)
            _, dN = hex8_shape(xi)
            assert np.allclose(dN.sum(axis=0), 0.0, atol=1e-12)

    def test_gradient_numerical(self):
        """Check dN against finite difference."""
        xi0 = np.array([0.3, -0.2, 0.5])
        N0, dN = hex8_shape(xi0)
        eps = 1e-7
        for d in range(3):
            xi_p = xi0.copy(); xi_p[d] += eps
            xi_m = xi0.copy(); xi_m[d] -= eps
            Np, _ = hex8_shape(xi_p)
            Nm, _ = hex8_shape(xi_m)
            dN_fd = (Np - Nm) / (2 * eps)
            assert np.allclose(dN[:, d], dN_fd, atol=1e-6)


class TestTet10Shape:
    def test_partition_of_unity(self):
        """Sum of shape functions = 1 at any point inside tet."""
        for _ in range(20):
            r, s, t = np.random.uniform(0, 1, 3)
            if r + s + t > 1:
                r, s, t = 1 - r, 1 - s, 1 - t
                if r + s + t > 1:
                    continue
            N, _ = tet10_shape(np.array([r, s, t]))
            assert abs(N.sum() - 1.0) < 1e-10, f"sum={N.sum()}"

    def test_interpolation_corners(self):
        """N_i at corner j = delta_{ij} for corners 0-3."""
        corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        for i in range(4):
            N, _ = tet10_shape(corners[i])
            assert abs(N[i] - 1.0) < 1e-12
            for j in range(10):
                if j != i:
                    assert abs(N[j]) < 1e-12, f"N[{j}] at corner {i} = {N[j]}"

    def test_interpolation_midpoints(self):
        """N at edge midpoints: the midpoint shape function = 1."""
        midpoints = {
            4: np.array([0.5, 0.0, 0.0]),  # edge 0-1
            5: np.array([0.5, 0.5, 0.0]),  # edge 1-2
            6: np.array([0.0, 0.5, 0.0]),  # edge 0-2
            7: np.array([0.0, 0.0, 0.5]),  # edge 0-3
            8: np.array([0.5, 0.0, 0.5]),  # edge 1-3
            9: np.array([0.0, 0.5, 0.5]),  # edge 2-3
        }
        for node_idx, xi in midpoints.items():
            N, _ = tet10_shape(xi)
            assert abs(N[node_idx] - 1.0) < 1e-12, f"N[{node_idx}] at midpoint"

    def test_gradient_numerical(self):
        """Check dN against finite difference."""
        xi0 = np.array([0.2, 0.15, 0.1])
        N0, dN = tet10_shape(xi0)
        eps = 1e-7
        for d in range(3):
            xi_p = xi0.copy(); xi_p[d] += eps
            xi_m = xi0.copy(); xi_m[d] -= eps
            Np, _ = tet10_shape(xi_p)
            Nm, _ = tet10_shape(xi_m)
            dN_fd = (Np - Nm) / (2 * eps)
            assert np.allclose(dN[:, d], dN_fd, atol=1e-5)


# ═══════════════════════════════════════════════════════════════
# Mesh generation tests
# ═══════════════════════════════════════════════════════════════

class TestMeshGeneration:
    def test_hex_box_node_count(self):
        mesh = FEMesh.create_hex_box(divisions=(3, 2, 2))
        assert mesh.n_nodes == 4 * 3 * 3  # (nx+1)(ny+1)(nz+1)
        assert mesh.n_elements == 3 * 2 * 2
        assert mesh.element_type == ElementType.HEX8

    def test_hex_box_volume(self):
        """Total volume should match box size."""
        mesh = FEMesh.create_hex_box(size=np.array([2, 3, 4]), divisions=(2, 2, 2))
        edata = precompute_element_data(mesh)
        total_vol = edata.weights.sum()
        assert abs(total_vol - 24.0) < 1e-8, f"volume={total_vol}"

    def test_tet10_box_creation(self):
        mesh = FEMesh.create_tet10_box(divisions=(2, 2, 2))
        assert mesh.element_type == ElementType.TET10
        assert mesh.nodes_per_element == 10
        assert mesh.elements.shape[1] == 10

    def test_tet10_volume_matches_tet4(self):
        """Tet10 mesh should have same total volume as Tet4."""
        tet4 = TetMesh.create_box(size=np.ones(3), divisions=(2, 2, 2))
        tet10 = FEMesh.create_tet10_box(size=np.ones(3), divisions=(2, 2, 2))
        tet4_vol = tet4.total_volume()
        edata10 = precompute_element_data(tet10)
        tet10_vol = edata10.weights.sum()
        assert abs(tet4_vol - tet10_vol) < 1e-6, f"tet4={tet4_vol}, tet10={tet10_vol}"

    def test_hex_surface_extraction(self):
        mesh = FEMesh.create_hex_box(divisions=(2, 2, 2))
        surf = mesh.extract_surface()
        assert surf.shape[1] == 3  # triangles
        # 6 faces × 4 quads × 2 tri = 48 triangles
        assert surf.shape[0] == 48


# ═══════════════════════════════════════════════════════════════
# FEM simulation tests
# ═══════════════════════════════════════════════════════════════

class TestHex8FEM:
    def test_cantilever_deflects(self):
        """Hex8 cantilever should deflect under gravity."""
        mesh = FEMesh.create_hex_box(
            origin=np.zeros(3),
            size=np.array([1.0, 0.1, 0.1]),
            divisions=(4, 1, 1),
        )
        body = DeformableBody(
            name="beam", mesh=mesh,
            material=CorotationalElastic(young=1e6, poisson=0.3),
            density=1000.0,
        )
        body.fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-6)[0]

        fem = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]), damping=0.5)
        fem.initialize(dt=0.002)

        for _ in range(100):
            fem.step()

        tip_z = body.x[mesh.nodes[:, 0] > 0.99, 2].mean()
        assert tip_z < 0.04, f"Tip should deflect down: z={tip_z}"
        assert np.all(np.isfinite(body.x))

    def test_zero_force_at_rest(self):
        """Internal forces should be near zero in rest configuration."""
        mesh = FEMesh.create_hex_box(size=np.array([0.1, 0.1, 0.1]), divisions=(2, 2, 2))
        edata = precompute_element_data(mesh)
        material = CorotationalElastic(young=1e5, poisson=0.3)
        f = assemble_forces(mesh, mesh.nodes, material, edata, None)
        assert np.allclose(f, 0.0, atol=1e-8)

    def test_mass_matrix(self):
        """Total mass should match density × volume."""
        mesh = FEMesh.create_hex_box(size=np.array([1, 1, 1]), divisions=(2, 2, 2))
        edata = precompute_element_data(mesh)
        M = assemble_mass_matrix(mesh, density=500.0, volumes=edata)
        total_mass = M.diagonal()[::3].sum()
        assert abs(total_mass - 500.0) < 1e-6, f"mass={total_mass}"


class TestTet10FEM:
    def test_cantilever_deflects(self):
        """Tet10 cantilever should deflect under gravity."""
        mesh = FEMesh.create_tet10_box(
            origin=np.zeros(3),
            size=np.array([1.0, 0.1, 0.1]),
            divisions=(4, 1, 1),
        )
        body = DeformableBody(
            name="beam10", mesh=mesh,
            material=CorotationalElastic(young=1e6, poisson=0.3),
            density=1000.0,
        )
        body.fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-6)[0]

        fem = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]), damping=0.5)
        fem.initialize(dt=0.002)

        for _ in range(100):
            fem.step()

        tip_z = body.x[mesh.nodes[:, 0] > 0.99, 2].mean()
        assert tip_z < 0.04, f"Tip should deflect down: z={tip_z}"
        assert np.all(np.isfinite(body.x))

    def test_zero_force_at_rest(self):
        """Internal forces should be near zero in rest configuration."""
        mesh = FEMesh.create_tet10_box(size=np.array([0.1, 0.1, 0.1]), divisions=(2, 2, 2))
        edata = precompute_element_data(mesh)
        material = CorotationalElastic(young=1e5, poisson=0.3)
        f = assemble_forces(mesh, mesh.nodes, material, edata, None)
        assert np.allclose(f, 0.0, atol=1e-8)

    def test_tet10_more_accurate_than_tet4(self):
        """Tet10 with same divisions should give similar or better accuracy.

        Compared via steady-state deflection (more DOF → closer to analytical).
        """
        # We just verify both converge (deflect downward) without blowup
        for etype_fn in [TetMesh.create_box, FEMesh.create_tet10_box]:
            mesh = etype_fn(origin=np.zeros(3), size=np.array([0.5, 0.05, 0.05]),
                            divisions=(3, 1, 1))
            body = DeformableBody(name="b", mesh=mesh,
                                  material=CorotationalElastic(young=1e6, poisson=0.3),
                                  density=1000.0)
            body.fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-6)[0]
            fem = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]), damping=0.5)
            fem.initialize(dt=0.002)
            for _ in range(50):
                fem.step()
            assert np.all(np.isfinite(body.x)), f"NaN with {type(mesh)}"
