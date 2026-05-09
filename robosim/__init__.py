"""RoboSim: 3D Robotics Simulator with hybrid RBD and FEM."""

__version__ = "0.1.0"

# ── Scene API (high-level) ────────────────────────────────────────────────────
from robosim.scene.scene   import Scene
from robosim.scene.objects import (
    Robot, Box, Ground, FEM, FEMPlastic, HybridPlastic, CB, Rigid,
)
from robosim.scene.handles import (
    RobotHandle,
    FEMBodyHandle,
    CBBodyHandle,
    RigidBodyHandle,
    HybridPlasticBodyHandle,
)
from robosim.control.pd     import JointPD
from robosim.control.phases import Trajectory

__all__ = [
    # scene
    "Scene",
    "Robot", "Box", "Ground",
    "FEM", "FEMPlastic", "HybridPlastic", "CB", "Rigid",
    # handles
    "RobotHandle", "FEMBodyHandle", "CBBodyHandle", "RigidBodyHandle",
    "HybridPlasticBodyHandle",
    # control
    "JointPD", "Trajectory",
]
