"""Collision and visual geometry primitives."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GeometryType(Enum):
    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    MESH = "mesh"


@dataclass
class Geometry:
    """A geometry primitive used for collision detection or visual rendering."""

    geometry_type: GeometryType

    # Box: half-extents (x, y, z)
    size: Optional[np.ndarray] = None

    # Sphere: radius
    radius: Optional[float] = None

    # Cylinder: radius and length (along Z axis)
    length: Optional[float] = None

    # Mesh: file path
    mesh_path: Optional[str] = None
    mesh_scale: np.ndarray | None = None

    # Cached mesh data (populated by load_mesh)
    mesh_vertices: np.ndarray | None = field(default=None, repr=False)
    mesh_faces: np.ndarray | None = field(default=None, repr=False)

    @staticmethod
    def box(size_x: float, size_y: float, size_z: float) -> Geometry:
        return Geometry(
            geometry_type=GeometryType.BOX,
            size=np.array([size_x, size_y, size_z]),
        )

    @staticmethod
    def sphere(radius: float) -> Geometry:
        return Geometry(geometry_type=GeometryType.SPHERE, radius=radius)

    @staticmethod
    def cylinder(radius: float, length: float) -> Geometry:
        return Geometry(
            geometry_type=GeometryType.CYLINDER, radius=radius, length=length
        )

    @staticmethod
    def mesh(path: str, scale: np.ndarray | None = None) -> Geometry:
        if scale is None:
            scale = np.ones(3)
        return Geometry(
            geometry_type=GeometryType.MESH,
            mesh_path=path,
            mesh_scale=np.asarray(scale, dtype=np.float64),
        )

    # ── Mesh loading ──

    def load_mesh(self, base_dir: Path | str | None = None) -> bool:
        """Load mesh file and cache vertices/faces.

        Resolves the mesh_path relative to *base_dir*, handling both plain
        relative paths and ``package://pkg_name/...`` URIs.

        Returns True if loading succeeded.
        """
        if self.geometry_type != GeometryType.MESH or not self.mesh_path:
            return False

        resolved = _resolve_mesh_path(self.mesh_path, base_dir)
        if resolved is None or not resolved.exists():
            logger.warning("Mesh file not found: %s (base_dir=%s)", self.mesh_path, base_dir)
            return False

        try:
            import trimesh
            tm = trimesh.load(str(resolved), force="mesh")
            # Handle Scene objects (multi-body meshes)
            if isinstance(tm, trimesh.Scene):
                tm = tm.dump(concatenate=True)
            if self.mesh_scale is not None:
                tm.apply_scale(self.mesh_scale)
            self.mesh_vertices = np.array(tm.vertices, dtype=np.float64)
            self.mesh_faces = np.array(tm.faces, dtype=np.int32)
            return True
        except Exception as e:
            logger.warning("Failed to load mesh %s: %s", resolved, e)
            return False

    @property
    def mesh_loaded(self) -> bool:
        return self.mesh_vertices is not None and self.mesh_faces is not None

    def __repr__(self) -> str:
        if self.geometry_type == GeometryType.BOX:
            return f"Geometry.box({self.size})"
        elif self.geometry_type == GeometryType.SPHERE:
            return f"Geometry.sphere(r={self.radius})"
        elif self.geometry_type == GeometryType.CYLINDER:
            return f"Geometry.cylinder(r={self.radius}, l={self.length})"
        else:
            loaded = " [loaded]" if self.mesh_loaded else ""
            return f"Geometry.mesh({self.mesh_path}{loaded})"


def _resolve_mesh_path(raw_path: str, base_dir: Path | str | None) -> Path | None:
    """Resolve a URDF mesh path to an absolute filesystem path.

    Handles:
    - ``package://pkg_name/meshes/foo.stl`` → walk up from base_dir to find pkg_name
    - Absolute paths
    - Relative paths (resolved against base_dir)
    """
    if base_dir is not None:
        base_dir = Path(base_dir)

    # package:// URI
    if raw_path.startswith("package://"):
        remainder = raw_path[len("package://"):]
        parts = remainder.split("/", 1)
        pkg_name = parts[0]
        rel_path = parts[1] if len(parts) > 1 else ""

        if base_dir is not None:
            # Walk up from base_dir looking for a directory named pkg_name
            search = base_dir
            for _ in range(10):  # max depth
                candidate = search / pkg_name / rel_path
                if candidate.exists():
                    return candidate
                # Also check if base_dir IS inside the package
                candidate2 = search / rel_path
                if candidate2.exists():
                    return candidate2
                if search.parent == search:
                    break
                search = search.parent

        # Fallback: try just the relative part
        if base_dir is not None:
            fallback = base_dir / rel_path
            if fallback.exists():
                return fallback

        return None

    path = Path(raw_path)

    # Absolute path
    if path.is_absolute():
        return path

    # Relative path
    if base_dir is not None:
        return base_dir / path

    return path
