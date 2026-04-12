"""Tests for RBD-FEM penalty coupling."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.math.spatial import SpatialInertia
from robosim.math.transforms import Transform
from robosim.model.geometry import Geometry
from robosim.model.joint import Joint, JointType
from robosim.model.link import Collision, Link, Visual
from robosim.model.robot import Robot
from robosim.physics.rbd.solver import RBDSolver
from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.solver import FEMSolver, DeformableBody
from robosim.physics.coupling.interface import (
    BoundaryMap, create_boundary_map, select_face_nodes,
)
from robosim.physics.coupling.penalty import PenaltyCoupling


def _make_arm_robot(arm_len=0.5):
    """Single revolute-joint arm: base(fixed) → revolute Y → arm link."""
    base = Link(
        name="base",
        inertial=SpatialInertia(mass=0, com=np.zeros(3), inertia=np.zeros((3, 3))),
    )
    arm = Link(
        name="arm",
        inertial=SpatialInertia(
            mass=1.0,
            com=np.array([arm_len / 2, 0, 0]),
            inertia=np.eye(3) * 0.01,
        ),
        collisions=[Collision(
            geometry=Geometry.box(arm_len, 0.05, 0.05),
            origin=Transform.from_translation(np.array([arm_len / 2, 0, 0])),
        )],
    )
    joint = Joint(
        name="shoulder",
        joint_type=JointType.REVOLUTE,
        parent_link="base",
        child_link="arm",
        axis=np.array([0.0, 1.0, 0.0]),
        origin=Transform.from_translation(np.array([0, 0, 1.0])),
    )
    robot = Robot(name="arm", links=[base, arm], joints=[joint])
    robot.gravity = np.array([0.0, 0.0, -9.81])
    robot.build()
    return robot


def _make_fem_block(origin, size=0.1, divisions=(2, 2, 2), young=1e5):
    """Small FEM block."""
    mesh = TetMesh.create_box(
        origin=origin,
        size=np.array([size, size, size]),
        divisions=divisions,
    )
    return DeformableBody(
        name="block",
        mesh=mesh,
        material=CorotationalElastic(young=young, poisson=0.3),
        density=500.0,
    )


class TestBoundaryMap:
    def test_select_face_nodes(self):
        mesh = TetMesh.create_box(
            origin=np.zeros(3),
            size=np.array([1.0, 1.0, 1.0]),
            divisions=(2, 2, 2),
        )
        top = select_face_nodes(mesh.nodes, axis=2, side="max")
        assert len(top) > 0
        assert np.allclose(mesh.nodes[top, 2], 1.0)

        bot = select_face_nodes(mesh.nodes, axis=2, side="min")
        assert len(bot) > 0
        assert np.allclose(mesh.nodes[bot, 2], 0.0)

    def test_create_boundary_map(self):
        robot = _make_arm_robot()
        robot.q = np.array([0.0])
        robot.qd = np.array([0.0])

        # Place FEM block at arm tip
        arm_len = 0.5
        origin = np.array([arm_len, -0.05, 0.95])
        body = _make_fem_block(origin)

        fem = FEMSolver(bodies=[body])
        fem.initialize(dt=0.001)

        # Bottom face nodes → glue to arm link (index 1)
        bottom = select_face_nodes(body.x, axis=2, side="min")
        bmap = create_boundary_map(robot, link_idx=1, fem_body=body,
                                   fem_body_idx=0, boundary_nodes=bottom)

        assert bmap.rigid_link_idx == 1
        assert len(bmap.fem_boundary_nodes) == len(bottom)
        assert bmap.local_positions.shape == (len(bottom), 3)


class TestPenaltyCoupling:
    def test_static_equilibrium(self):
        """With no gravity, coupled system at rest should stay at rest."""
        robot = _make_arm_robot()
        robot.gravity = np.zeros(3)
        robot.q = np.array([0.0])
        robot.qd = np.array([0.0])

        origin = np.array([0.5, -0.05, 0.95])
        body = _make_fem_block(origin)

        rbd = RBDSolver(robot=robot)
        rbd.initialize(dt=0.001)
        fem = FEMSolver(bodies=[body], gravity=np.zeros(3))
        fem.initialize(dt=0.001)

        bottom = select_face_nodes(body.x, axis=2, side="min")
        bmap = create_boundary_map(robot, 1, body, 0, bottom)

        coupling = PenaltyCoupling(stiffness=1e4, damping=100)
        coupling.setup(rbd, fem, [bmap])

        # Run 100 steps
        pos_0 = body.x.copy()
        for _ in range(100):
            rbd_w, fem_f = coupling.compute_forces()
            rbd.clear_external_forces()
            for li, w in rbd_w.items():
                rbd.set_external_force(li, w)
            rbd.step()
            fem.step(extra_forces=fem_f if fem_f else None)

        # Positions should barely change (no gravity, no initial velocity)
        drift = np.linalg.norm(body.x - pos_0, axis=1).max()
        assert drift < 0.001, f"FEM drifted {drift:.4f}m with no gravity"

    def test_coupling_forces_nonzero(self):
        """If FEM nodes are displaced from target, coupling force should appear."""
        robot = _make_arm_robot()
        robot.q = np.array([0.0])
        robot.qd = np.array([0.0])

        origin = np.array([0.5, -0.05, 0.95])
        body = _make_fem_block(origin)

        rbd = RBDSolver(robot=robot)
        rbd.initialize(dt=0.001)
        fem = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]))
        fem.initialize(dt=0.001)

        bottom = select_face_nodes(body.x, axis=2, side="min")
        bmap = create_boundary_map(robot, 1, body, 0, bottom)

        coupling = PenaltyCoupling(stiffness=1e4, damping=100)
        coupling.setup(rbd, fem, [bmap])

        # Displace FEM body downward
        body.x[:, 2] -= 0.01

        rbd_w, fem_f = coupling.compute_forces()

        # FEM force should push nodes back up (positive Z)
        f_z = fem_f[0].reshape(-1, 3)[:, 2]
        assert f_z[bottom].sum() > 0, "Coupling should pull FEM nodes toward link"

        # RBD wrench should pull link down (reaction)
        assert 1 in rbd_w
        assert rbd_w[1][5] < 0, "Reaction should pull link toward FEM"

    def test_coupled_motion(self):
        """Arm swings under gravity → FEM block should follow."""
        robot = _make_arm_robot()
        robot.gravity = np.array([0.0, 0.0, -9.81])
        robot.q = np.array([0.3])  # slight angle
        robot.qd = np.array([0.0])

        arm_len = 0.5
        fk = robot.forward_kinematics()
        tip = fk[1].apply_point(np.array([arm_len, 0, 0]))
        origin = tip + np.array([-0.05, -0.05, -0.1])
        body = _make_fem_block(origin, divisions=(3, 3, 3), young=1e6)

        rbd = RBDSolver(robot=robot)
        rbd.initialize(dt=0.001)
        fem = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]),
                        damping=0.1)
        fem.initialize(dt=0.001)

        top = select_face_nodes(body.x, axis=2, side="max")
        bmap = create_boundary_map(robot, 1, body, 0, top)

        coupling = PenaltyCoupling(stiffness=1e4, damping=200, max_force=500)
        coupling.setup(rbd, fem, [bmap])

        com_0 = body.x.mean(axis=0).copy()

        for _ in range(200):
            rbd_w, fem_f = coupling.compute_forces()
            rbd.clear_external_forces()
            for li, w in rbd_w.items():
                rbd.set_external_force(li, w)
            rbd.step()
            fem.step(extra_forces=fem_f if fem_f else None)

        com_final = body.x.mean(axis=0)
        displacement = np.linalg.norm(com_final - com_0)

        # The block should have moved (arm swung)
        assert displacement > 0.01, f"Block didn't follow arm: d={displacement:.4f}"
        # And no NaN
        assert np.all(np.isfinite(body.x)), "NaN in FEM positions"
        assert np.all(np.isfinite(body.v)), "NaN in FEM velocities"

    def test_boundary_drift(self):
        """Boundary FEM nodes should stay close to their RBD target positions.

        Uses zero gravity and slow constant arm rotation to isolate coupling
        tracking from extreme arm dynamics.
        """
        robot = _make_arm_robot()
        robot.gravity = np.zeros(3)
        robot.q = np.array([0.0])
        robot.qd = np.array([0.5])  # slow rotation

        origin = np.array([0.5, -0.05, 0.95])
        body = _make_fem_block(origin, divisions=(3, 3, 3), young=1e6)

        rbd = RBDSolver(robot=robot)
        rbd.initialize(dt=0.001)
        fem = FEMSolver(bodies=[body], gravity=np.zeros(3))
        fem.initialize(dt=0.001)

        top = select_face_nodes(body.x, axis=2, side="max")
        bmap = create_boundary_map(robot, 1, body, 0, top)

        coupling = PenaltyCoupling(stiffness=1e4, damping=200, max_force=500)
        coupling.setup(rbd, fem, [bmap])

        max_drift = 0.0
        for _ in range(200):
            rbd_w, fem_f = coupling.compute_forces()
            rbd.clear_external_forces()
            for li, w in rbd_w.items():
                rbd.set_external_force(li, w)
            rbd.step()
            fem.step(extra_forces=fem_f if fem_f else None)

            # Measure boundary drift
            fk = robot.forward_kinematics()
            T = fk[bmap.rigid_link_idx]
            targets = np.array([T.apply_point(p) for p in bmap.local_positions])
            actual = body.x[bmap.fem_boundary_nodes]
            drift = np.linalg.norm(targets - actual, axis=1).max()
            max_drift = max(max_drift, drift)

        assert max_drift < 0.02, f"Boundary drift too large: {max_drift:.4f}m"
