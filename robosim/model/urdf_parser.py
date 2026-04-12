"""URDF (Unified Robot Description Format) parser.

Parses a URDF XML file into a Robot model with links, joints,
and kinematic tree structure.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

from robosim.math.spatial import SpatialInertia
from robosim.math.transforms import Transform
from robosim.model.geometry import Geometry
from robosim.model.joint import Joint, JointLimits, JointType
from robosim.model.link import Collision, Link, Visual
from robosim.model.robot import Robot


def parse_urdf(source: str | Path) -> Robot:
    """Parse a URDF file or XML string into a Robot model.

    Parameters
    ----------
    source : file path or XML string

    Returns
    -------
    Robot with links, joints, and built kinematic tree.
    """
    source_str = str(source)
    if source_str.endswith(".urdf") or source_str.endswith(".xml") or Path(source_str).exists():
        tree = ET.parse(source_str)
        root = tree.getroot()
    else:
        root = ET.fromstring(source_str)

    robot_name = root.attrib.get("name", "unnamed_robot")
    links = []
    joints = []

    for link_elem in root.findall("link"):
        links.append(_parse_link(link_elem))

    for joint_elem in root.findall("joint"):
        joints.append(_parse_joint(joint_elem))

    robot = Robot(name=robot_name, links=links, joints=joints)
    robot.build()
    return robot


def _parse_link(elem: ET.Element) -> Link:
    """Parse a <link> element."""
    name = elem.attrib["name"]
    link = Link(name=name)

    # Inertial
    inertial_elem = elem.find("inertial")
    if inertial_elem is not None:
        link.inertial_origin = _parse_origin(inertial_elem.find("origin"))

        mass_elem = inertial_elem.find("mass")
        mass = float(mass_elem.attrib.get("value", "0")) if mass_elem is not None else 0.0

        inertia_elem = inertial_elem.find("inertia")
        if inertia_elem is not None:
            ixx = float(inertia_elem.attrib.get("ixx", "0"))
            ixy = float(inertia_elem.attrib.get("ixy", "0"))
            ixz = float(inertia_elem.attrib.get("ixz", "0"))
            iyy = float(inertia_elem.attrib.get("iyy", "0"))
            iyz = float(inertia_elem.attrib.get("iyz", "0"))
            izz = float(inertia_elem.attrib.get("izz", "0"))
            inertia_mat = np.array([
                [ixx, ixy, ixz],
                [ixy, iyy, iyz],
                [ixz, iyz, izz],
            ])
        else:
            inertia_mat = np.zeros((3, 3))

        # URDF specifies inertia at the inertial origin frame.
        # We store CoM as the translation of the inertial origin.
        com = link.inertial_origin.translation.copy()
        link.inertial = SpatialInertia(mass=mass, com=com, inertia=inertia_mat)

    # Visuals
    for visual_elem in elem.findall("visual"):
        geom = _parse_geometry(visual_elem.find("geometry"))
        if geom is not None:
            origin = _parse_origin(visual_elem.find("origin"))
            color = _parse_color(visual_elem.find("material"))
            link.visuals.append(Visual(geometry=geom, origin=origin, color=color))

    # Collisions
    for collision_elem in elem.findall("collision"):
        geom = _parse_geometry(collision_elem.find("geometry"))
        if geom is not None:
            origin = _parse_origin(collision_elem.find("origin"))
            link.collisions.append(Collision(geometry=geom, origin=origin))

    return link


def _parse_joint(elem: ET.Element) -> Joint:
    """Parse a <joint> element."""
    name = elem.attrib["name"]
    joint_type_str = elem.attrib.get("type", "fixed")

    type_map = {
        "revolute": JointType.REVOLUTE,
        "continuous": JointType.CONTINUOUS,
        "prismatic": JointType.PRISMATIC,
        "fixed": JointType.FIXED,
        "floating": JointType.FLOATING,
    }
    joint_type = type_map.get(joint_type_str, JointType.FIXED)

    parent_elem = elem.find("parent")
    child_elem = elem.find("child")
    parent_link = parent_elem.attrib["link"] if parent_elem is not None else ""
    child_link = child_elem.attrib["link"] if child_elem is not None else ""

    origin = _parse_origin(elem.find("origin"))

    axis_elem = elem.find("axis")
    if axis_elem is not None:
        axis = _parse_vec3(axis_elem.attrib.get("xyz", "0 0 1"))
    else:
        axis = np.array([0.0, 0.0, 1.0])

    # Limits
    limits = None
    limit_elem = elem.find("limit")
    if limit_elem is not None:
        limits = JointLimits(
            lower=float(limit_elem.attrib.get("lower", "-inf")),
            upper=float(limit_elem.attrib.get("upper", "inf")),
            velocity=float(limit_elem.attrib.get("velocity", "inf")),
            effort=float(limit_elem.attrib.get("effort", "inf")),
        )

    # Dynamics
    dynamics_elem = elem.find("dynamics")
    damping = 0.0
    friction = 0.0
    if dynamics_elem is not None:
        damping = float(dynamics_elem.attrib.get("damping", "0"))
        friction = float(dynamics_elem.attrib.get("friction", "0"))

    return Joint(
        name=name,
        joint_type=joint_type,
        parent_link=parent_link,
        child_link=child_link,
        axis=axis,
        origin=origin,
        limits=limits,
        damping=damping,
        friction=friction,
    )


def _parse_origin(elem: Optional[ET.Element]) -> Transform:
    """Parse an <origin> element to a Transform."""
    if elem is None:
        return Transform.identity()

    xyz = _parse_vec3(elem.attrib.get("xyz", "0 0 0"))
    rpy_str = elem.attrib.get("rpy", "0 0 0")
    rpy = _parse_vec3(rpy_str)

    T = Transform.from_rpy(rpy[0], rpy[1], rpy[2])
    T.translation = xyz
    return T


def _parse_geometry(elem: Optional[ET.Element]) -> Optional[Geometry]:
    """Parse a <geometry> element."""
    if elem is None:
        return None

    box = elem.find("box")
    if box is not None:
        size = _parse_vec3(box.attrib.get("size", "1 1 1"))
        return Geometry.box(size[0], size[1], size[2])

    sphere = elem.find("sphere")
    if sphere is not None:
        r = float(sphere.attrib.get("radius", "0.5"))
        return Geometry.sphere(r)

    cylinder = elem.find("cylinder")
    if cylinder is not None:
        r = float(cylinder.attrib.get("radius", "0.5"))
        l = float(cylinder.attrib.get("length", "1.0"))
        return Geometry.cylinder(r, l)

    mesh = elem.find("mesh")
    if mesh is not None:
        filename = mesh.attrib.get("filename", "")
        scale_str = mesh.attrib.get("scale", None)
        scale = _parse_vec3(scale_str) if scale_str else None
        return Geometry.mesh(filename, scale)

    return None


def _parse_color(elem: Optional[ET.Element]) -> Optional[np.ndarray]:
    """Parse a <material> element for color."""
    if elem is None:
        return None
    color_elem = elem.find("color")
    if color_elem is not None:
        rgba = color_elem.attrib.get("rgba", "0.5 0.5 0.5 1.0")
        return np.array([float(x) for x in rgba.split()])
    return None


def _parse_vec3(s: str) -> np.ndarray:
    """Parse a space-separated string of 3 floats."""
    return np.array([float(x) for x in s.split()], dtype=np.float64)
