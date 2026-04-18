"""Scene renderer: bridges Robot model with SimViewer."""

from __future__ import annotations

import numpy as np

from robosim.math.transforms import Transform
from robosim.model.geometry import Geometry, GeometryType
from robosim.model.robot import Robot
from robosim.viz.viewer import SimViewer, geometry_to_trimesh


class RobotRenderer:
    """Renders a Robot model in the SimViewer.

    Extracts visual **and** collision geometries from the robot.
    By default visual meshes are shown; collision meshes are hidden.
    Use :meth:`show_collision` to toggle.
    """

    def __init__(self, robot: Robot, viewer: SimViewer):
        self.robot = robot
        self.viewer = viewer

        # Visual layer
        self._visual_link_names: list[str] = []
        self._visual_geoms: list[tuple[Geometry, Transform]] = []
        self._visual_colors: list[np.ndarray | None] = []

        # Collision layer
        self._collision_link_names: list[str] = []
        self._collision_geoms: list[tuple[Geometry, Transform]] = []
        self._collision_colors: list[np.ndarray | None] = []

        self._show_collision = False
        self._contact_links: list[str] | None = None  # None = show all

    # ── Visual mesh prefix ──
    def _vis_prefix(self) -> str:
        return f"{self.robot.name}/"

    def _col_prefix(self) -> str:
        return f"{self.robot.name}_col/"

    def setup(self):
        """Extract visual and collision geometries from robot and add to viewer."""
        # ── Visual meshes ──
        for link in self.robot.links:
            if link.visuals:
                vis = link.visuals[0]
                self._visual_link_names.append(link.name)
                self._visual_geoms.append((vis.geometry, vis.origin))
                self._visual_colors.append(vis.color)
            # Links without explicit visual geometry are skipped entirely
            # (no fallback sphere) to avoid cluttering the scene with
            # virtual/massless intermediate links (e.g. free-body chains).

        self.viewer.add_robot_meshes(
            self.robot.name,
            self._visual_link_names,
            self._visual_geoms,
            self._visual_colors,
        )

        # ── Collision meshes (added hidden) ──
        col_color = np.array([0.2, 0.8, 0.3, 0.7])  # translucent green
        for link in self.robot.links:
            if link.collisions:
                col = link.collisions[0]
                self._collision_link_names.append(link.name)
                self._collision_geoms.append((col.geometry, col.origin))
                self._collision_colors.append(col_color.copy())

        if self._collision_link_names:
            self.viewer.add_robot_meshes(
                f"{self.robot.name}_col",
                self._collision_link_names,
                self._collision_geoms,
                self._collision_colors,
            )
            # Hide collision meshes by default, render as wireframe
            self.viewer.set_meshes_visible_by_prefix(
                self._col_prefix(), False)
            self.viewer.set_meshes_wireframe_by_prefix(
                self._col_prefix(), True)

    @property
    def collision_visible(self) -> bool:
        return self._show_collision

    def set_contact_links(self, link_names: list[str]):
        """Limit which links' collision meshes are shown in collision mode.

        Only the specified links will be visible when :meth:`show_collision`
        is called with ``show=True``.  Pass ``None`` to restore the default
        (show all links with collision geometry).
        """
        self._contact_links = list(link_names) if link_names is not None else None

    def show_collision(self, show: bool):
        """Toggle collision mesh visibility (hides visual meshes when on)."""
        if show == self._show_collision:
            return
        self._show_collision = show
        self.viewer.set_meshes_visible_by_prefix(
            self._vis_prefix(), not show)

        if self._contact_links is None:
            # No filter: show/hide all collision meshes
            self.viewer.set_meshes_visible_by_prefix(
                self._col_prefix(), show)
        else:
            # Filter: hide all first, then reveal only contact-relevant links
            self.viewer.set_meshes_visible_by_prefix(
                self._col_prefix(), False)
            if show:
                for link_name in self._contact_links:
                    if link_name in self._collision_link_names:
                        self.viewer.set_mesh_visible(
                            f"{self._col_prefix()}{link_name}", True)

    def update(self, fk: list | None = None):
        """Update all link transforms from current robot state.

        Parameters
        ----------
        fk : optional pre-computed FK to avoid redundant recomputation.
        """
        if fk is None:
            fk = self.robot.forward_kinematics()

        # Update the active layer (visual or collision)
        if not self._show_collision:
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
        else:
            transforms = []
            for link_name in self._collision_link_names:
                idx = self.robot.link_index(link_name)
                transforms.append(fk[idx])

            self.viewer.update_link_transforms(
                f"{self.robot.name}_col",
                self._collision_link_names,
                transforms,
                self._collision_geoms,
                self._collision_colors,
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
