"""Tests for URDF parsing and Robot model."""

import math

import numpy as np
import pytest

from robosim.model.urdf_parser import parse_urdf
from robosim.model.joint import JointType


class TestURDFParser:
    def test_load_simple_arm(self, simple_arm_urdf):
        robot = parse_urdf(simple_arm_urdf)
        assert robot.name == "simple_3dof_arm"
        assert robot.n_links == 4
        assert robot.n_joints == 3
        assert robot.n_dof == 3

    def test_link_names(self, simple_arm_urdf):
        robot = parse_urdf(simple_arm_urdf)
        names = [link.name for link in robot.links]
        assert "base_link" in names
        assert "upper_arm" in names
        assert "forearm" in names
        assert "hand" in names

    def test_joint_types(self, simple_arm_urdf):
        robot = parse_urdf(simple_arm_urdf)
        for joint in robot.joints:
            assert joint.joint_type == JointType.REVOLUTE

    def test_joint_parent_child(self, simple_arm_urdf):
        robot = parse_urdf(simple_arm_urdf)
        j0 = robot.joints[0]
        assert j0.parent_link == "base_link"
        assert j0.child_link == "upper_arm"

    def test_link_mass(self, simple_arm_urdf):
        robot = parse_urdf(simple_arm_urdf)
        base = robot.links[robot.link_index("base_link")]
        assert abs(base.mass - 5.0) < 1e-10

    def test_joint_limits(self, simple_arm_urdf):
        robot = parse_urdf(simple_arm_urdf)
        j = robot.joints[robot.joint_index("shoulder_yaw")]
        assert j.limits is not None
        assert abs(j.limits.lower - (-3.14)) < 1e-10
        assert abs(j.limits.upper - 3.14) < 1e-10

    def test_visual_geometry(self, simple_arm_urdf):
        robot = parse_urdf(simple_arm_urdf)
        hand = robot.links[robot.link_index("hand")]
        assert len(hand.visuals) == 1
        assert hand.visuals[0].geometry.geometry_type.value == "box"

    def test_visual_color(self, simple_arm_urdf):
        robot = parse_urdf(simple_arm_urdf)
        hand = robot.links[robot.link_index("hand")]
        color = hand.visuals[0].color
        assert color is not None
        np.testing.assert_array_almost_equal(color, [0.8, 0.2, 0.2, 1.0])


class TestURDFFromString:
    def test_parse_minimal_urdf(self):
        urdf = """<?xml version="1.0"?>
        <robot name="test">
          <link name="base"/>
          <link name="child">
            <inertial>
              <mass value="1.0"/>
              <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
            </inertial>
          </link>
          <joint name="j1" type="revolute">
            <parent link="base"/>
            <child link="child"/>
            <axis xyz="0 0 1"/>
            <limit lower="-1" upper="1" velocity="1" effort="10"/>
          </joint>
        </robot>
        """
        robot = parse_urdf(urdf)
        assert robot.name == "test"
        assert robot.n_links == 2
        assert robot.n_dof == 1


class TestRobotKinematics:
    def test_forward_kinematics_zero_config(self, simple_arm_urdf):
        robot = parse_urdf(simple_arm_urdf)
        # Zero configuration
        fk = robot.forward_kinematics()
        assert len(fk) == robot.n_links

        # Base link should be at identity
        np.testing.assert_array_almost_equal(
            fk[robot.link_index("base_link")].to_matrix(), np.eye(4)
        )

        # Hand should be at the sum of all joint z-offsets
        # base->upper: z=0.1, upper->forearm: z=0.4, forearm->hand: z=0.3
        hand_t = fk[robot.link_index("hand")].translation
        assert abs(hand_t[2] - 0.8) < 1e-10  # 0.1 + 0.4 + 0.3

    def test_forward_kinematics_rotated(self, simple_arm_urdf):
        robot = parse_urdf(simple_arm_urdf)
        # Rotate shoulder 90 degrees
        robot.set_joint_q(robot.joint_index("shoulder_yaw"), math.pi / 2)
        fk = robot.forward_kinematics()

        # Upper arm should be rotated 90 degrees about Z
        upper = fk[robot.link_index("upper_arm")]
        # X axis of upper arm should point in Y direction
        x_axis = upper.apply_vector(np.array([1, 0, 0]))
        np.testing.assert_array_almost_equal(x_axis, [0, 1, 0], decimal=10)

    def test_kinematic_tree_structure(self, simple_arm_urdf):
        robot = parse_urdf(simple_arm_urdf)
        base_idx = robot.link_index("base_link")
        assert robot.parent_index(base_idx) == -1  # root has no parent

        hand_idx = robot.link_index("hand")
        forearm_idx = robot.link_index("forearm")
        assert robot.parent_index(hand_idx) == forearm_idx

    def test_print_tree(self, simple_arm_urdf, capsys):
        robot = parse_urdf(simple_arm_urdf)
        robot.print_tree()
        captured = capsys.readouterr()
        assert "base_link" in captured.out
        assert "shoulder_yaw" in captured.out
        assert "hand" in captured.out
