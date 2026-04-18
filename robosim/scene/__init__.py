"""Scene API — high-level entry point for RoboSim simulations."""

from robosim.scene.scene import Scene
from robosim.scene.objects import Robot, Box, Ground, FEM, CB, Rigid
from robosim.scene.handles import (
    RobotHandle,
    FEMBodyHandle,
    CBBodyHandle,
    RigidBodyHandle,
)

__all__ = [
    "Scene",
    "Robot", "Box", "Ground",
    "FEM", "CB", "Rigid",
    "RobotHandle", "FEMBodyHandle", "CBBodyHandle", "RigidBodyHandle",
]
