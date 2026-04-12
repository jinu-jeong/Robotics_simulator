"""Joint types for articulated robots."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from robosim.math.transforms import Transform


class JointType(Enum):
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"  # revolute without limits
    PRISMATIC = "prismatic"
    FIXED = "fixed"
    FLOATING = "floating"


@dataclass
class JointLimits:
    """Position, velocity, and effort limits for a joint."""

    lower: float = -np.inf
    upper: float = np.inf
    velocity: float = np.inf
    effort: float = np.inf


@dataclass
class Joint:
    """A joint connecting a parent link to a child link.

    Attributes
    ----------
    name : Joint identifier
    joint_type : Type of joint (revolute, prismatic, fixed, ...)
    axis : (3,) joint axis in the child link's frame
    origin : Transform from parent link frame to joint frame
    limits : Optional position/velocity/effort limits
    damping : Viscous damping coefficient
    friction : Coulomb friction coefficient
    parent_link : Name of the parent link
    child_link : Name of the child link
    """

    name: str
    joint_type: JointType
    parent_link: str
    child_link: str
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    origin: Transform = field(default_factory=Transform.identity)
    limits: Optional[JointLimits] = None
    damping: float = 0.0
    friction: float = 0.0

    # Mimic joint: follower mirrors a leader joint
    mimic_joint: Optional[str] = None       # name of the leader joint
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0

    def __post_init__(self):
        self.axis = np.asarray(self.axis, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(self.axis)
        if norm > 1e-12:
            self.axis = self.axis / norm

    @property
    def num_dof(self) -> int:
        if self.joint_type in (JointType.FIXED,):
            return 0
        elif self.joint_type in (
            JointType.REVOLUTE,
            JointType.CONTINUOUS,
            JointType.PRISMATIC,
        ):
            return 1
        elif self.joint_type == JointType.FLOATING:
            return 6
        return 0

    @property
    def is_actuated(self) -> bool:
        return self.joint_type not in (JointType.FIXED,)

    def __repr__(self) -> str:
        return (
            f"Joint('{self.name}', type={self.joint_type.value}, "
            f"parent='{self.parent_link}', child='{self.child_link}')"
        )
