"""Visual overlays for the simulation viewer.

Provides contact force arrows, joint axis indicators, bounding boxes,
and wireframe rendering as lightweight triangle meshes injected into
the Taichi scene.
"""

from __future__ import annotations

import numpy as np

from robosim.math.transforms import Transform
from robosim.physics.contact.response import ContactForce


# ── Arrow geometry ──────────────────────────────────────────────

def _arrow_mesh(
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    shaft_radius: float = 0.003,
    head_radius: float = 0.008,
    head_fraction: float = 0.25,
    n_seg: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 3D arrow mesh (shaft + cone head).

    Returns (vertices, faces) as numpy arrays.
    """
    d = direction / (np.linalg.norm(direction) + 1e-15)

    # Build orthonormal frame
    if abs(d[2]) < 0.9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = np.array([1.0, 0.0, 0.0])
    u = np.cross(d, up)
    u /= np.linalg.norm(u) + 1e-15
    v = np.cross(d, u)

    shaft_len = length * (1.0 - head_fraction)
    head_len = length * head_fraction

    verts = []
    faces = []

    # Shaft: cylinder from origin along d
    for i in range(n_seg):
        angle = 2 * np.pi * i / n_seg
        offset = shaft_radius * (np.cos(angle) * u + np.sin(angle) * v)
        verts.append(origin + offset)            # bottom ring
        verts.append(origin + shaft_len * d + offset)  # top ring

    # Shaft faces
    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        b0, t0 = 2 * i, 2 * i + 1
        b1, t1 = 2 * i_next, 2 * i_next + 1
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])

    # Cone head: base ring at shaft_len, tip at full length
    base_offset = len(verts)
    cone_base = origin + shaft_len * d
    cone_tip = origin + length * d

    for i in range(n_seg):
        angle = 2 * np.pi * i / n_seg
        offset = head_radius * (np.cos(angle) * u + np.sin(angle) * v)
        verts.append(cone_base + offset)

    tip_idx = len(verts)
    verts.append(cone_tip)

    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        faces.append([base_offset + i, base_offset + i_next, tip_idx])

    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


# ── Contact force overlay ───────────────────────────────────────

class ContactForceOverlay:
    """Renders contact forces as colored arrows in the viewer.

    Usage::

        overlay = ContactForceOverlay(viewer)
        # Each frame:
        overlay.update(contact_solver.last_forces)
    """

    def __init__(
        self,
        viewer,
        scale: float = 0.001,
        max_arrows: int = 50,
        color: np.ndarray = np.array([1.0, 0.3, 0.1]),
        min_force: float = 1.0,
    ):
        self.viewer = viewer
        self.scale = scale
        self.max_arrows = max_arrows
        self.color = color.copy()
        self.min_force = min_force
        self._mesh_name = "__contact_arrows__"
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, val: bool):
        self._active = val
        if not val:
            # Remove mesh when deactivated
            if self._mesh_name in self.viewer._meshes:
                del self.viewer._meshes[self._mesh_name]
                self.viewer._dirty = True

    def update(self, forces: list[ContactForce]) -> None:
        """Update arrow meshes from current contact forces."""
        if not self._active or not forces:
            if self._mesh_name in self.viewer._meshes:
                del self.viewer._meshes[self._mesh_name]
                self.viewer._dirty = True
            return

        all_verts = []
        all_faces = []
        vert_offset = 0

        count = 0
        for cf in forces:
            if cf.normal_force < self.min_force:
                continue
            if count >= self.max_arrows:
                break

            mag = np.linalg.norm(cf.force)
            if mag < 1e-6:
                continue

            direction = cf.force / mag
            arrow_len = mag * self.scale
            arrow_len = np.clip(arrow_len, 0.01, 0.3)

            verts, faces = _arrow_mesh(
                cf.point, direction, arrow_len,
                shaft_radius=0.002, head_radius=0.005,
            )
            all_verts.append(verts)
            all_faces.append(faces + vert_offset)
            vert_offset += len(verts)
            count += 1

        if not all_verts:
            if self._mesh_name in self.viewer._meshes:
                del self.viewer._meshes[self._mesh_name]
                self.viewer._dirty = True
            return

        combined_verts = np.concatenate(all_verts)
        combined_faces = np.concatenate(all_faces)

        self.viewer.add_mesh(
            self._mesh_name,
            combined_verts,
            combined_faces,
            color=self.color,
        )


# ── Joint axis overlay ──────────────────────────────────────────

class JointAxisOverlay:
    """Renders joint rotation/translation axes as colored arrows.

    Usage::

        overlay = JointAxisOverlay(viewer)
        # Each frame:
        fk = robot.forward_kinematics()
        overlay.update(robot, fk)
    """

    def __init__(
        self,
        viewer,
        length: float = 0.08,
        revolute_color: np.ndarray = np.array([0.2, 0.6, 1.0]),
        prismatic_color: np.ndarray = np.array([0.2, 1.0, 0.6]),
    ):
        self.viewer = viewer
        self.length = length
        self.revolute_color = revolute_color.copy()
        self.prismatic_color = prismatic_color.copy()
        self._mesh_name = "__joint_axes__"
        self._active = False  # off by default

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, val: bool):
        self._active = val
        if not val and self._mesh_name in self.viewer._meshes:
            del self.viewer._meshes[self._mesh_name]
            self.viewer._dirty = True

    def update(self, robot, fk: list[Transform]) -> None:
        """Rebuild axis arrows from current FK."""
        from robosim.model.joint import JointType

        if not self._active:
            return

        all_verts = []
        all_faces = []
        all_colors = []
        vert_offset = 0

        for j_idx, joint in enumerate(robot.joints):
            if joint.joint_type == JointType.FIXED:
                continue

            child_idx = robot.link_index(joint.child)
            if child_idx is None:
                continue

            T = fk[child_idx]
            origin = T.translation
            axis_world = T.rotation @ joint.axis

            is_revolute = joint.joint_type in (
                JointType.REVOLUTE, JointType.CONTINUOUS)
            color = self.revolute_color if is_revolute else self.prismatic_color

            verts, faces = _arrow_mesh(
                origin, axis_world, self.length,
                shaft_radius=0.002, head_radius=0.004,
            )

            nv = len(verts)
            all_verts.append(verts)
            all_faces.append(faces + vert_offset)
            all_colors.append(np.broadcast_to(color, (nv, 3)).copy())
            vert_offset += nv

        if not all_verts:
            return

        combined_verts = np.concatenate(all_verts)
        combined_faces = np.concatenate(all_faces)
        combined_colors = np.concatenate(all_colors)

        self.viewer.add_mesh(
            self._mesh_name,
            combined_verts,
            combined_faces,
            color=np.array([0.5, 0.5, 0.5]),  # fallback
            per_vertex_color=combined_colors,
        )


# ── AABB wireframe overlay ──────────────────────────────────────

def aabb_wireframe_mesh(
    min_pt: np.ndarray,
    max_pt: np.ndarray,
    thickness: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate thin box edges as triangle mesh to visualize an AABB.

    Returns (vertices, faces).
    """
    # 12 edges of a box, each as a thin rectangular prism
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
    ], dtype=np.float64)

    edges = [
        (0,1), (1,2), (2,3), (3,0),  # bottom
        (4,5), (5,6), (6,7), (7,4),  # top
        (0,4), (1,5), (2,6), (3,7),  # verticals
    ]

    all_verts = []
    all_faces = []
    offset = 0
    t = thickness

    for i0, i1 in edges:
        p0, p1 = corners[i0], corners[i1]
        d = p1 - p0
        length = np.linalg.norm(d)
        if length < 1e-10:
            continue

        d_hat = d / length
        if abs(d_hat[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([1.0, 0.0, 0.0])
        u = np.cross(d_hat, up)
        u /= np.linalg.norm(u) + 1e-15
        v = np.cross(d_hat, u)

        # 4 verts at each end of the edge → 8 verts per edge
        offsets_2d = [(-t, -t), (t, -t), (t, t), (-t, t)]
        for end_pt in [p0, p1]:
            for du, dv in offsets_2d:
                all_verts.append(end_pt + du * u + dv * v)

        # 6 faces per edge (rectangular prism sides)
        # Bottom quad: 0123, Top quad: 4567
        idx = [offset + k for k in range(8)]
        # 4 side faces (2 triangles each)
        for s in range(4):
            s_next = (s + 1) % 4
            all_faces.append([idx[s], idx[s_next], idx[4 + s_next]])
            all_faces.append([idx[s], idx[4 + s_next], idx[4 + s]])
        offset += 8

    if not all_verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

    return np.array(all_verts, dtype=np.float64), np.array(all_faces, dtype=np.int32)


class CollisionAABBOverlay:
    """Renders collision body AABBs as wireframe boxes.

    Useful for debugging broad-phase collision detection.
    """

    def __init__(
        self,
        viewer,
        color: np.ndarray = np.array([0.0, 1.0, 0.0]),
    ):
        self.viewer = viewer
        self.color = color.copy()
        self._mesh_name = "__collision_aabb__"
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, val: bool):
        self._active = val
        if not val and self._mesh_name in self.viewer._meshes:
            del self.viewer._meshes[self._mesh_name]
            self.viewer._dirty = True

    def update(self, contact_solver) -> None:
        """Rebuild AABB wireframes from contact detector."""
        if not self._active:
            return

        from robosim.physics.contact.detection import compute_aabb

        all_verts = []
        all_faces = []
        offset = 0

        for body in contact_solver.detector.bodies:
            aabb = compute_aabb(body.geometry, body.transform)
            verts, faces = aabb_wireframe_mesh(aabb.min_pt, aabb.max_pt)
            if len(verts) == 0:
                continue
            all_verts.append(verts)
            all_faces.append(faces + offset)
            offset += len(verts)

        if not all_verts:
            return

        self.viewer.add_mesh(
            self._mesh_name,
            np.concatenate(all_verts),
            np.concatenate(all_faces),
            color=self.color,
        )
