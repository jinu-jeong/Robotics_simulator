"""Factory functions for creating common robot/object configurations."""

from __future__ import annotations

import numpy as np

from robosim.math.spatial import SpatialInertia
from robosim.math.transforms import Transform
from robosim.model.geometry import Geometry
from robosim.model.joint import Joint, JointLimits, JointType
from robosim.model.link import Collision, Link, Visual
from robosim.model.robot import Robot


def create_free_box(
    name: str = "free_box",
    size: tuple[float, float, float] = (0.05, 0.05, 0.05),
    mass: float = 0.5,
    position: np.ndarray | None = None,
    color: np.ndarray | None = None,
) -> Robot:
    """Create a 6-DOF free-floating rigid box.

    Uses 6 chained 1-DOF joints (3 prismatic + 3 revolute) so the
    existing single-DOF algorithms (ABA, RNEA, integrator) work without
    modification.

    Kinematic chain:
        base (fixed) -> tx -> ty -> tz -> rx -> ry -> rz -> box_link

    Parameters
    ----------
    name : Robot name
    size : (sx, sy, sz) box dimensions in meters
    mass : Total mass in kg
    position : (3,) initial position [x, y, z]
    color : (4,) RGBA color

    Returns
    -------
    Robot with 6 DOF and initial q set so the box starts at *position*.
    """
    if position is None:
        position = np.array([0.0, 0.0, 0.1])
    if color is None:
        color = np.array([0.9, 0.7, 0.2, 1.0])

    sx, sy, sz = size

    # Inertia of a uniform box
    ixx = mass / 12.0 * (sy**2 + sz**2)
    iyy = mass / 12.0 * (sx**2 + sz**2)
    izz = mass / 12.0 * (sx**2 + sy**2)

    # Virtual massless links for translation DOFs
    zero_inertia = SpatialInertia(mass=0, com=np.zeros(3), inertia=np.zeros((3, 3)))

    base = Link(name=f"{name}_base", inertial=zero_inertia)
    tx_link = Link(name=f"{name}_tx", inertial=zero_inertia)
    ty_link = Link(name=f"{name}_ty", inertial=zero_inertia)
    tz_link = Link(name=f"{name}_tz", inertial=zero_inertia)
    rx_link = Link(name=f"{name}_rx", inertial=zero_inertia)
    ry_link = Link(name=f"{name}_ry", inertial=zero_inertia)

    box_link = Link(
        name=f"{name}_body",
        inertial=SpatialInertia(
            mass=mass,
            com=np.zeros(3),
            inertia=np.diag([ixx, iyy, izz]),
        ),
        visuals=[Visual(
            geometry=Geometry.box(sx, sy, sz),
            origin=Transform.identity(),
            color=color,
        )],
        collisions=[Collision(
            geometry=Geometry.box(sx, sy, sz),
            origin=Transform.identity(),
        )],
    )

    links = [base, tx_link, ty_link, tz_link, rx_link, ry_link, box_link]

    # Big limits for translation, ±2π for rotation
    t_lim = JointLimits(lower=-10.0, upper=10.0, velocity=10.0, effort=1000.0)
    r_lim = JointLimits(lower=-2 * np.pi, upper=2 * np.pi, velocity=20.0, effort=100.0)

    joints = [
        Joint(name=f"{name}_tx", joint_type=JointType.PRISMATIC,
              parent_link=base.name, child_link=tx_link.name,
              axis=np.array([1, 0, 0]), origin=Transform.identity(), limits=t_lim),
        Joint(name=f"{name}_ty", joint_type=JointType.PRISMATIC,
              parent_link=tx_link.name, child_link=ty_link.name,
              axis=np.array([0, 1, 0]), origin=Transform.identity(), limits=t_lim),
        Joint(name=f"{name}_tz", joint_type=JointType.PRISMATIC,
              parent_link=ty_link.name, child_link=tz_link.name,
              axis=np.array([0, 0, 1]), origin=Transform.identity(), limits=t_lim),
        Joint(name=f"{name}_rx", joint_type=JointType.REVOLUTE,
              parent_link=tz_link.name, child_link=rx_link.name,
              axis=np.array([1, 0, 0]), origin=Transform.identity(), limits=r_lim),
        Joint(name=f"{name}_ry", joint_type=JointType.REVOLUTE,
              parent_link=rx_link.name, child_link=ry_link.name,
              axis=np.array([0, 1, 0]), origin=Transform.identity(), limits=r_lim),
        Joint(name=f"{name}_rz", joint_type=JointType.REVOLUTE,
              parent_link=ry_link.name, child_link=box_link.name,
              axis=np.array([0, 0, 1]), origin=Transform.identity(), limits=r_lim),
    ]

    robot = Robot(name=name, links=links, joints=joints)
    robot.gravity = np.array([0.0, 0.0, -9.81])
    robot.build()

    # Set initial position via the translation joints
    robot.q[0] = position[0]  # tx
    robot.q[1] = position[1]  # ty
    robot.q[2] = position[2]  # tz

    return robot


def create_free_sphere(
    name: str = "free_sphere",
    radius: float = 0.05,
    mass: float = 0.5,
    position: np.ndarray | None = None,
    color: np.ndarray | None = None,
) -> Robot:
    """Create a 6-DOF free-floating rigid sphere.

    Same kinematic structure as create_free_box but with sphere geometry.
    """
    if position is None:
        position = np.array([0.0, 0.0, 0.1])
    if color is None:
        color = np.array([0.9, 0.4, 0.2, 1.0])

    I_sphere = 2.0 / 5.0 * mass * radius**2
    zero_inertia = SpatialInertia(mass=0, com=np.zeros(3), inertia=np.zeros((3, 3)))

    base = Link(name=f"{name}_base", inertial=zero_inertia)
    tx_link = Link(name=f"{name}_tx", inertial=zero_inertia)
    ty_link = Link(name=f"{name}_ty", inertial=zero_inertia)
    tz_link = Link(name=f"{name}_tz", inertial=zero_inertia)
    rx_link = Link(name=f"{name}_rx", inertial=zero_inertia)
    ry_link = Link(name=f"{name}_ry", inertial=zero_inertia)
    sphere_link = Link(
        name=f"{name}_body",
        inertial=SpatialInertia(
            mass=mass, com=np.zeros(3),
            inertia=np.diag([I_sphere, I_sphere, I_sphere]),
        ),
        visuals=[Visual(
            geometry=Geometry.sphere(radius),
            origin=Transform.identity(),
            color=color,
        )],
        collisions=[Collision(
            geometry=Geometry.sphere(radius),
            origin=Transform.identity(),
        )],
    )

    links = [base, tx_link, ty_link, tz_link, rx_link, ry_link, sphere_link]

    t_lim = JointLimits(lower=-10.0, upper=10.0, velocity=10.0, effort=1000.0)
    r_lim = JointLimits(lower=-2 * np.pi, upper=2 * np.pi, velocity=20.0, effort=100.0)

    joints = [
        Joint(name=f"{name}_tx", joint_type=JointType.PRISMATIC,
              parent_link=base.name, child_link=tx_link.name,
              axis=np.array([1, 0, 0]), origin=Transform.identity(), limits=t_lim),
        Joint(name=f"{name}_ty", joint_type=JointType.PRISMATIC,
              parent_link=tx_link.name, child_link=ty_link.name,
              axis=np.array([0, 1, 0]), origin=Transform.identity(), limits=t_lim),
        Joint(name=f"{name}_tz", joint_type=JointType.PRISMATIC,
              parent_link=ty_link.name, child_link=tz_link.name,
              axis=np.array([0, 0, 1]), origin=Transform.identity(), limits=t_lim),
        Joint(name=f"{name}_rx", joint_type=JointType.REVOLUTE,
              parent_link=tz_link.name, child_link=rx_link.name,
              axis=np.array([1, 0, 0]), origin=Transform.identity(), limits=r_lim),
        Joint(name=f"{name}_ry", joint_type=JointType.REVOLUTE,
              parent_link=rx_link.name, child_link=ry_link.name,
              axis=np.array([0, 1, 0]), origin=Transform.identity(), limits=r_lim),
        Joint(name=f"{name}_rz", joint_type=JointType.REVOLUTE,
              parent_link=ry_link.name, child_link=sphere_link.name,
              axis=np.array([0, 0, 1]), origin=Transform.identity(), limits=r_lim),
    ]

    robot = Robot(name=name, links=links, joints=joints)
    robot.gravity = np.array([0.0, 0.0, -9.81])
    robot.build()

    robot.q[0] = position[0]
    robot.q[1] = position[1]
    robot.q[2] = position[2]

    return robot
