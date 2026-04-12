"""Scene renderer: bridges Robot model with SimViewer."""

from __future__ import annotations

import numpy as np

from robosim.math.transforms import Transform
from robosim.model.geometry import Geometry, GeometryType
from robosim.model.robot import Robot
from robosim.viz.viewer import SimViewer, geometry_to_trimesh


class RobotRenderer:
    """Renders a Robot model in the SimViewer.

    Extracts visual geometries from the robot and manages their
    transforms each frame.
    """

    def __init__(self, robot: Robot, viewer: SimViewer):
        self.robot = robot
        self.viewer = viewer
        self._visual_link_names: list[str] = []
        self._visual_geoms: list[tuple[Geometry, Transform]] = []
        self._visual_colors: list[np.ndarray | None] = []

    def setup(self):
        """Extract visual geometries from robot and add to viewer."""
        for link in self.robot.links:
            if link.visuals:
                vis = link.visuals[0]  # Use first visual
                self._visual_link_names.append(link.name)
                self._visual_geoms.append((vis.geometry, vis.origin))
                self._visual_colors.append(vis.color)
            else:
                # Create a default small sphere for links without visuals
                default_geom = Geometry.sphere(0.015)
                self._visual_link_names.append(link.name)
                self._visual_geoms.append((default_geom, Transform.identity()))
                self._visual_colors.append(np.array([0.5, 0.5, 0.5, 0.3]))

        self.viewer.add_robot_meshes(
            self.robot.name,
            self._visual_link_names,
            self._visual_geoms,
            self._visual_colors,
        )

    def update(self):
        """Update all link transforms from current robot state."""
        fk = self.robot.forward_kinematics()

        transforms = []
        for link_name in self._visual_link_names:
            idx = self.robot.link_index(link_name)
            transforms.append(fk[idx])

        self.viewer.update_link_transforms(
            self.robot.name,
            self._visual_link_names,
            transforms,
            self._visual_geoms,
            self._visual_colors,
        )


def create_joint_markers(robot: Robot, viewer: SimViewer) -> None:
    """Add small sphere markers at joint locations."""
    fk = robot.forward_kinematics()
    for j_idx, joint in enumerate(robot.joints):
        child_idx = robot.link_index(joint.child_link)
        pos = fk[child_idx].translation
        verts, faces = geometry_to_trimesh(Geometry.sphere(0.012))
        verts = verts + pos
        viewer.add_mesh(
            f"joint_marker/{joint.name}",
            verts, faces,
            color=np.array([0.9, 0.9, 0.2]),
        )
