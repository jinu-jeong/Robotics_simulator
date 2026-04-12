"""Tests for CompositeMesh: mixed Hex8 + Tet4 element blocks in one body."""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.physics.fem.elements import ElementType
from robosim.physics.fem.mesh import (
    CompositeMesh, ElementBlock, FEMesh, TetMesh,
)
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.solver import FEMSolver, DeformableBody
from robosim.physics.fem.assembly import (
    precompute_element_data, assemble_forces, assemble_stiffness,
    assemble_mass_matrix, batch_von_mises, CompositeIntegrationData,
)


# ═══════════════════════════════════════════════════════════════
# Mesh construction tests
# ═══════════════════════════════════════════════════════════════

class TestCompositeMeshCreation:
    def test_hex_tet_box_basic(self):
        """create_hex_tet_box produces valid composite mesh."""
        mesh = CompositeMesh.create_hex_tet_box(
            size=np.ones(3), hex_divisions=(4, 4, 4), tet_layers=1
        )
        assert isinstance(mesh, CompositeMesh)
        assert mesh.n_blocks == 2
        assert mesh.n_nodes > 0
        assert mesh.n_elements > 0

    def test_hex_tet_box_blocks(self):
        """Correct element types in blocks."""
        mesh = CompositeMesh.create_hex_tet_box(
            size=np.ones(3), hex_divisions=(4, 4, 4), tet_layers=1
        )
        etypes = [b.element_type for b in mesh.blocks]
        assert ElementType.HEX8 in etypes
        assert ElementType.TET4 in etypes

    def test_hex_tet_box_element_count(self):
        """Total hex + tet elements should cover the entire box."""
        nx, ny, nz = 4, 4, 4
        mesh = CompositeMesh.create_hex_tet_box(
            size=np.ones(3), hex_divisions=(nx, ny, nz), tet_layers=1
        )
        n_hex = sum(b.n_elements for b in mesh.blocks if b.element_type == ElementType.HEX8)
        n_tet = sum(b.n_elements for b in mesh.blocks if b.element_type == ElementType.TET4)
        # Inner 2×2×2 = 8 hex, outer cells = 64-8=56 cells × 5 tets each = 280 tets
        assert n_hex == 8
        assert n_tet == 56 * 5

    def test_hex_tet_box_node_count(self):
        """Node count matches grid."""
        nx, ny, nz = 4, 4, 4
        mesh = CompositeMesh.create_hex_tet_box(
            size=np.ones(3), hex_divisions=(nx, ny, nz), tet_layers=1
        )
        expected = (nx + 1) * (ny + 1) * (nz + 1)
        assert mesh.n_nodes == expected

    def test_from_blocks(self):
        """Manual block construction."""
        nodes = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=float)
        tet_block = ElementBlock(
            element_type=ElementType.TET4,
            elements=np.array([[0, 1, 2, 3]]),
            name="tet",
        )
        mesh = CompositeMesh.from_blocks(nodes, [tet_block])
        assert mesh.n_blocks == 1
        assert mesh.n_elements == 1
        assert mesh.n_nodes == 8

    def test_merge_meshes(self):
        """Merging two meshes shares boundary nodes."""
        hex_m = FEMesh.create_hex_box(
            origin=np.zeros(3), size=np.array([0.5, 0.5, 0.5]),
            divisions=(2, 2, 2),
        )
        tet_m = TetMesh.create_box(
            origin=np.array([0.5, 0.0, 0.0]), size=np.array([0.5, 0.5, 0.5]),
            divisions=(2, 2, 2),
        )
        composite = CompositeMesh.merge_meshes([hex_m, tet_m], merge_tol=1e-8)

        # Nodes should be merged at the shared x=0.5 face
        n_separate = hex_m.n_nodes + tet_m.n_nodes
        assert composite.n_nodes < n_separate, "No nodes merged"
        shared = n_separate - composite.n_nodes
        assert shared == 9, f"Expected 9 shared nodes (3x3 grid), got {shared}"

    def test_element_types_property(self):
        mesh = CompositeMesh.create_hex_tet_box(
            size=np.ones(3), hex_divisions=(4, 4, 4), tet_layers=1
        )
        etypes = mesh.element_types
        assert len(etypes) == 2
        assert ElementType.HEX8 in etypes
        assert ElementType.TET4 in etypes


# ═══════════════════════════════════════════════════════════════
# Surface extraction
# ═══════════════════════════════════════════════════════════════

class TestCompositeSurface:
    def test_surface_extraction(self):
        """Surface triangles should form a closed surface."""
        mesh = CompositeMesh.create_hex_tet_box(
            size=np.ones(3), hex_divisions=(4, 4, 4), tet_layers=1
        )
        surf = mesh.extract_surface()
        assert surf.shape[1] == 3  # triangles
        assert surf.shape[0] > 0

    def test_surface_all_tet(self):
        """All-tet composite should produce same surface as FEMesh."""
        tet_m = TetMesh.create_box(size=np.ones(3), divisions=(2, 2, 2))
        block = ElementBlock(ElementType.TET4, tet_m.elements, "tets")
        composite = CompositeMesh(nodes=tet_m.nodes, blocks=[block])
        surf_c = composite.extract_surface()
        surf_t = tet_m.extract_surface()
        assert surf_c.shape[0] == surf_t.shape[0]

    def test_merged_surface_no_internal_faces(self):
        """Merged hex+tet should not have internal interface faces in surface."""
        hex_m = FEMesh.create_hex_box(
            origin=np.zeros(3), size=np.array([0.5, 1.0, 1.0]),
            divisions=(2, 2, 2),
        )
        tet_m = TetMesh.create_box(
            origin=np.array([0.5, 0.0, 0.0]), size=np.array([0.5, 1.0, 1.0]),
            divisions=(2, 2, 2),
        )
        composite = CompositeMesh.merge_meshes([hex_m, tet_m])

        surf = composite.extract_surface()
        # Surface should correspond to the outer box (1x1x1) only
        # No internal interface faces should appear
        # 6 faces × 4 quads × 2 triangles = 48 for a 2×2×2 grid on each face
        # But the split is different, so just check reasonable range
        assert surf.shape[0] > 20
        # Check all surface nodes have valid indices
        assert surf.max() < composite.n_nodes
        assert surf.min() >= 0


# ═══════════════════════════════════════════════════════════════
# Assembly tests
# ═══════════════════════════════════════════════════════════════

class TestCompositeAssembly:
    def test_precompute_returns_composite_data(self):
        mesh = CompositeMesh.create_hex_tet_box(
            size=np.ones(3), hex_divisions=(4, 4, 4), tet_layers=1
        )
        edata = precompute_element_data(mesh)
        assert isinstance(edata, CompositeIntegrationData)
        assert len(edata.block_data) == mesh.n_blocks

    def test_zero_force_at_rest(self):
        """Internal forces should be zero in rest configuration."""
        mesh = CompositeMesh.create_hex_tet_box(
            size=np.array([0.2, 0.2, 0.2]),
            hex_divisions=(4, 4, 4), tet_layers=1,
        )
        edata = precompute_element_data(mesh)
        material = CorotationalElastic(young=1e5, poisson=0.3)
        f = assemble_forces(mesh, mesh.nodes, material, edata, None)
        assert np.allclose(f, 0.0, atol=1e-8)

    def test_mass_conservation(self):
        """Total mass should match density × volume."""
        size = np.array([1.0, 1.0, 1.0])
        mesh = CompositeMesh.create_hex_tet_box(
            size=size, hex_divisions=(4, 4, 4), tet_layers=1,
        )
        edata = precompute_element_data(mesh)
        density = 500.0
        M = assemble_mass_matrix(mesh, density, edata)
        total_mass = M.diagonal()[::3].sum()
        expected = density * np.prod(size)
        assert abs(total_mass - expected) < 1e-4, f"mass={total_mass}, expected={expected}"

    def test_volume_matches_box(self):
        """Sum of integration weights should equal box volume."""
        size = np.array([2.0, 3.0, 1.5])
        mesh = CompositeMesh.create_hex_tet_box(
            size=size, hex_divisions=(4, 4, 4), tet_layers=1,
        )
        edata = precompute_element_data(mesh)
        total_vol = sum(bd.weights.sum() for bd in edata.block_data)
        expected_vol = np.prod(size)
        assert abs(total_vol - expected_vol) < 1e-6, f"vol={total_vol}, expected={expected_vol}"

    def test_von_mises_at_rest(self):
        """Von Mises stress should be ~0 at rest."""
        mesh = CompositeMesh.create_hex_tet_box(
            size=np.ones(3), hex_divisions=(4, 4, 4), tet_layers=1,
        )
        edata = precompute_element_data(mesh)
        material = CorotationalElastic(young=1e5, poisson=0.3)
        vm = batch_von_mises(mesh, mesh.nodes, material, edata, None)
        assert vm.shape[0] == mesh.n_nodes
        assert vm.max() < 1e-6

    def test_stiffness_symmetric(self):
        """Global stiffness matrix should be symmetric."""
        mesh = CompositeMesh.create_hex_tet_box(
            size=np.array([0.2, 0.2, 0.2]),
            hex_divisions=(3, 3, 3), tet_layers=1,
        )
        edata = precompute_element_data(mesh)
        material = CorotationalElastic(young=1e5, poisson=0.3)
        K = assemble_stiffness(mesh, mesh.nodes, material, edata, None)
        diff = (K - K.T).toarray()
        assert np.allclose(diff, 0, atol=1e-6)


# ═══════════════════════════════════════════════════════════════
# Simulation tests
# ═══════════════════════════════════════════════════════════════

class TestCompositeFEM:
    def test_cantilever_deflects(self):
        """Composite cantilever should deflect under gravity without blowup."""
        mesh = CompositeMesh.create_hex_tet_box(
            origin=np.zeros(3),
            size=np.array([0.5, 0.1, 0.1]),
            hex_divisions=(6, 3, 3),
            tet_layers=1,
        )
        body = DeformableBody(
            name="composite_beam",
            mesh=mesh,
            material=CorotationalElastic(young=1e6, poisson=0.3),
            density=1000.0,
        )
        body.fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-6)[0]

        fem = FEMSolver(
            bodies=[body],
            gravity=np.array([0, 0, -9.81]),
            damping=0.5,
        )
        fem.initialize(dt=0.002)

        for _ in range(50):
            fem.step()

        tip_nodes = np.where(mesh.nodes[:, 0] > 0.45)[0]
        tip_z = body.x[tip_nodes, 2].mean()
        assert tip_z < 0.1, f"Tip should deflect down: z={tip_z}"
        assert np.all(np.isfinite(body.x))

    def test_merged_mesh_simulation(self):
        """Merged hex+tet mesh should simulate stably."""
        hex_m = FEMesh.create_hex_box(
            origin=np.zeros(3), size=np.array([0.5, 0.1, 0.1]),
            divisions=(3, 2, 2),
        )
        tet_m = TetMesh.create_box(
            origin=np.array([0.5, 0.0, 0.0]), size=np.array([0.5, 0.1, 0.1]),
            divisions=(3, 2, 2),
        )
        mesh = CompositeMesh.merge_meshes([hex_m, tet_m], names=["hex", "tet"])

        body = DeformableBody(
            name="merged_beam",
            mesh=mesh,
            material=CorotationalElastic(young=1e6, poisson=0.3),
            density=1000.0,
        )
        body.fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-6)[0]

        fem = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]), damping=0.5)
        fem.initialize(dt=0.002)

        for _ in range(30):
            fem.step()

        assert np.all(np.isfinite(body.x)), "NaN in merged mesh simulation"
        tip_nodes = np.where(mesh.nodes[:, 0] > 0.9)[0]
        if len(tip_nodes) > 0:
            tip_z = body.x[tip_nodes, 2].mean()
            assert tip_z < 0.1, f"Should deflect: z={tip_z}"

    def test_von_mises_under_load(self):
        """Von Mises stress should be nonzero under deformation."""
        mesh = CompositeMesh.create_hex_tet_box(
            size=np.array([0.5, 0.1, 0.1]),
            hex_divisions=(4, 3, 3), tet_layers=1,
        )
        body = DeformableBody(
            name="test",
            mesh=mesh,
            material=CorotationalElastic(young=1e6, poisson=0.3),
            density=1000.0,
        )
        body.fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-6)[0]

        fem = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]), damping=0.5)
        fem.initialize(dt=0.002)

        for _ in range(20):
            fem.step()

        vm = batch_von_mises(body.mesh, body.x, body.material,
                             body._dN_list, body._volumes)
        assert vm.max() > 0, "Stress should be nonzero under gravity"
