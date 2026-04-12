"""Link (rigid body) in an articulated robot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from robosim.math.spatial import SpatialInertia
from robosim.math.transforms import Transform
from robosim.model.geometry import Geometry


@dataclass
class Visual:
    """Visual representation of a link."""

    geometry: Geometry
    origin: Transform = field(default_factory=Transform.identity)
    color: Optional[np.ndarray] = None  # RGBA [0,1]

    def __post_init__(self):
        if self.color is not None:
            self.color = np.asarray(self.color, dtype=np.float64)


@dataclass
class Collision:
    """Collision geometry of a link."""

    geometry: Geometry
    origin: Transform = field(default_factory=Transform.identity)


@dataclass
class Link:
    """A rigid link in a robot model.

    Attributes
    ----------
    name : Link identifier
    inertial : Spatial inertia of the link (mass, CoM, rotational inertia)
    inertial_origin : Transform from link frame to inertial frame
    visuals : Visual geometries for rendering
    collisions : Collision geometries for contact detection
    """

    name: str
    inertial: SpatialInertia = field(
        default_factory=lambda: SpatialInertia(
            mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3))
        )
    )
    inertial_origin: Transform = field(default_factory=Transform.identity)
    visuals: list[Visual] = field(default_factory=list)
    collisions: list[Collision] = field(default_factory=list)

    @property
    def mass(self) -> float:
        return self.inertial.mass

    def __repr__(self) -> str:
        return f"Link('{self.name}', mass={self.mass:.4f})"
