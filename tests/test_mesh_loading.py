"""Tests for mesh loading and URDF mesh integration."""

import numpy as np
import pytest
from pathlib import Path

from robosim.model.geometry import Geometry, GeometryType, _resolve_mesh_path


PANDA_DIR = Path(__file__).parent.parent / "examples" / "urdf" / "panda"
HAS_PANDA = (PANDA_DIR / "meshes" / "collision" / "link0.stl").exists()


class TestGeometryMeshLoading:
    """Test Geometry.load_mesh and path resolution."""

    @pytest.mark.skipif(not HAS_PANDA, reason="Panda meshes not available")
    def test_load_stl(self):
        g = Geometry.mesh("meshes/collision/link0.stl")
        assert g.load_mesh(PANDA_DIR)
        assert g.mesh_loaded
        assert g.mesh_vertices.shape[1] == 3
        assert g.mesh_faces.shape[1] == 3
        assert g.mesh_vertices.shape[0] > 0
        assert g.mesh_faces.shape[0] > 0

    @pytest.mark.skipif(not HAS_PANDA, reason="Panda meshes not available")
    def test_load_dae(self):
        g = Geometry.mesh("meshes/visual/link0.dae")
        assert g.load_mesh(PANDA_DIR)
        assert g.mesh_loaded
        assert g.mesh_vertices.shape[0] > 100  # DAE has more detail

    @pytest.mark.skipif(not HAS_PANDA, reason="Panda meshes not available")
    def test_mesh_scale(self):
        g1 = Geometry.mesh("meshes/collision/link0.stl")
        g1.load_mesh(PANDA_DIR)
        g2 = Geometry.mesh("meshes/collision/link0.stl", scale=np.array([2.0, 2.0, 2.0]))
        g2.load_mesh(PANDA_DIR)
        # Scaled mesh should be 2x larger
        extent1 = g1.mesh_vertices.max(axis=0) - g1.mesh_vertices.min(axis=0)
        extent2 = g2.mesh_vertices.max(axis=0) - g2.mesh_vertices.min(axis=0)
        np.testing.assert_allclose(extent2, extent1 * 2, rtol=0.01)

    def test_load_missing_file(self):
        g = Geometry.mesh("nonexistent/mesh.stl")
        assert not g.load_mesh("/tmp")
        assert not g.mesh_loaded

    def test_load_non_mesh_geometry(self):
        g = Geometry.box(1, 1, 1)
        assert not g.load_mesh("/tmp")

    def test_mesh_loaded_property(self):
        g = Geometry.mesh("test.stl")
        assert not g.mesh_loaded
        g.mesh_vertices = np.zeros((10, 3))
        g.mesh_faces = np.zeros((5, 3), dtype=np.int32)
        assert g.mesh_loaded


class TestPathResolution:
    """Test _resolve_mesh_path for various path formats."""

    def test_relative_path(self, tmp_path):
        (tmp_path / "meshes").mkdir()
        (tmp_path / "meshes" / "test.stl").touch()
        result = _resolve_mesh_path("meshes/test.stl", tmp_path)
        assert result is not None
        assert result.exists()

    def test_absolute_path(self, tmp_path):
        f = tmp_path / "test.stl"
        f.touch()
        result = _resolve_mesh_path(str(f), None)
        assert result == f

    def test_package_uri(self, tmp_path):
        # Create package structure: tmp_path/my_pkg/meshes/test.stl
        pkg = tmp_path / "my_pkg" / "meshes"
        pkg.mkdir(parents=True)
        (pkg / "test.stl").touch()
        # Search from inside the package
        result = _resolve_mesh_path("package://my_pkg/meshes/test.stl", tmp_path)
        assert result is not None
        assert result.exists()

    def test_package_uri_missing(self):
        result = _resolve_mesh_path("package://nonexistent/mesh.stl", Path("/tmp"))
        assert result is None or not result.exists()


class TestURDFMeshIntegration:
    """Test mesh loading through URDF parser."""

    @pytest.mark.skipif(not HAS_PANDA, reason="Panda meshes not available")
    def test_parse_panda_urdf(self):
        from robosim.model.urdf_parser import parse_urdf
        robot = parse_urdf(str(PANDA_DIR / "panda.urdf"))
        assert robot.name == "panda"
        assert len(robot.links) == 12
        assert robot.n_dof == 9

    @pytest.mark.skipif(not HAS_PANDA, reason="Panda meshes not available")
    def test_all_meshes_loaded(self):
        from robosim.model.urdf_parser import parse_urdf
        robot = parse_urdf(str(PANDA_DIR / "panda.urdf"))
        for link in robot.links:
            for v in link.visuals:
                if v.geometry.geometry_type == GeometryType.MESH:
                    assert v.geometry.mesh_loaded, f"Visual mesh not loaded for {link.name}"
            for c in link.collisions:
                if c.geometry.geometry_type == GeometryType.MESH:
                    assert c.geometry.mesh_loaded, f"Collision mesh not loaded for {link.name}"

    @pytest.mark.skipif(not HAS_PANDA, reason="Panda meshes not available")
    def test_forward_kinematics(self):
        from robosim.model.urdf_parser import parse_urdf
        robot = parse_urdf(str(PANDA_DIR / "panda.urdf"))
        robot.q = np.zeros(robot.n_dof)
        fk = robot.forward_kinematics()
        # Should have transforms for all links
        assert len(fk) == len(robot.links)
        # Base should be at origin
        np.testing.assert_allclose(fk[0].translation, [0, 0, 0], atol=1e-10)

    @pytest.mark.skipif(not HAS_PANDA, reason="Panda meshes not available")
    def test_mesh_data_types(self):
        from robosim.model.urdf_parser import parse_urdf
        robot = parse_urdf(str(PANDA_DIR / "panda.urdf"))
        link0 = robot.links[0]
        v = link0.visuals[0].geometry
        assert v.mesh_vertices.dtype == np.float64
        assert v.mesh_faces.dtype == np.int32
