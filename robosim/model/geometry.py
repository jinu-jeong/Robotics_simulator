"""Collision and visual geometry primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


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

    def __repr__(self) -> str:
        name = self.geometry_type.value
        if self.geometry_type == GeometryType.BOX:
            return f"Geometry.box({self.size})"
        elif self.geometry_type == GeometryType.SPHERE:
            return f"Geometry.sphere(r={self.radius})"
        elif self.geometry_type == GeometryType.CYLINDER:
            return f"Geometry.cylinder(r={self.radius}, l={self.length})"
        else:
            return f"Geometry.mesh({self.mesh_path})"
