"""Contact detection: AABB broad phase + analytical narrow phase."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosim.model.geometry import Geometry, GeometryType
from robosim.math.transforms import Transform
from robosim.physics.contact.sdf import (
    ContactPoint,
    sphere_ground,
    box_ground,
    cylinder_ground,
    sphere_sphere,
    box_box,
    box_sphere,
    mesh_box,
    mesh_ground,
    points_ground,
)


@dataclass
class AABB:
    """Axis-aligned bounding box."""

    min_pt: np.ndarray  # (3,)
    max_pt: np.ndarray  # (3,)

    def overlaps(self, other: AABB) -> bool:
        return bool(np.all(self.min_pt <= other.max_pt) and
                    np.all(other.min_pt <= self.max_pt))

    def expand(self, margin: float) -> AABB:
        return AABB(self.min_pt - margin, self.max_pt + margin)


def compute_aabb(geometry: Geometry, transform: Transform) -> AABB:
    """Compute world-space AABB for a geometry."""
    R = transform.rotation
    t = transform.translation

    if geometry.geometry_type == GeometryType.SPHERE:
        r = geometry.radius
        return AABB(t - r, t + r)

    elif geometry.geometry_type == GeometryType.BOX:
        he = geometry.size / 2.0
        # Rotated AABB: project half-extents onto each world axis
        extent = np.abs(R) @ he
        return AABB(t - extent, t + extent)

    elif geometry.geometry_type == GeometryType.CYLINDER:
        r, hl = geometry.radius, geometry.length / 2.0
        # Conservative: bounding sphere of cylinder
        bound = np.sqrt(r**2 + hl**2)
        return AABB(t - bound, t + bound)

    elif geometry.geometry_type == GeometryType.MESH:
        if geometry.mesh_loaded:
            # Transform mesh vertices to world space, compute AABB
            world_verts = (R @ geometry.mesh_vertices.T).T + t
            return AABB(world_verts.min(axis=0), world_verts.max(axis=0))
        return AABB(t - 0.1, t + 0.1)

    else:
        # Fallback: small sphere
        return AABB(t - 0.1, t + 0.1)


@dataclass
class CollisionBody:
    """A body in the collision world."""

    body_id: int
    geometry: Geometry
    transform: Transform        # current world-frame pose
    body_type: str = "rbd_link"  # "rbd_link" or "fem_surface"
    owner_index: int = 0        # link_idx for RBD, body_idx for FEM


@dataclass
class GroundPlane:
    """Infinite ground plane."""

    height: float = 0.0
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


class ContactDetector:
    """Manages collision bodies and detects contacts."""

    def __init__(self, ground: GroundPlane | None = None):
        self.ground = ground
        self.bodies: list[CollisionBody] = []
        self._next_id = 0
        self._filter_pairs: set[tuple[int, int]] = set()  # body_id pairs to skip

    def add_body(self, geometry: Geometry, transform: Transform,
                 body_type: str = "rbd_link", owner_index: int = 0) -> int:
        """Register a collision body, return its id."""
        bid = self._next_id
        self._next_id += 1
        self.bodies.append(CollisionBody(
            body_id=bid, geometry=geometry, transform=transform,
            body_type=body_type, owner_index=owner_index,
        ))
        return bid

    def update_transform(self, body_id: int, transform: Transform):
        """Update a body's world-frame pose."""
        self.bodies[body_id].transform = transform

    def add_filter(self, body_id_a: int, body_id_b: int):
        """Add a pair of body IDs to skip during collision detection."""
        pair = (min(body_id_a, body_id_b), max(body_id_a, body_id_b))
        self._filter_pairs.add(pair)

    def detect_all(self) -> list[tuple[ContactPoint, int, int]]:
        """Detect all contacts.

        Returns list of (ContactPoint, body_id_a, body_id_b).
        body_id_b = -1 for ground contacts.
        """
        results: list[tuple[ContactPoint, int, int]] = []

        # ── Ground contacts ──
        if self.ground is not None:
            gh = self.ground.height
            gn = self.ground.normal
            for body in self.bodies:
                contacts = self._narrow_ground(body, gh, gn)
                for cp in contacts:
                    results.append((cp, body.body_id, -1))

        # ── Body-body contacts (broad + narrow) ──
        n = len(self.bodies)
        if n > 1:
            aabbs = [compute_aabb(b.geometry, b.transform) for b in self.bodies]
            for i in range(n):
                for j in range(i + 1, n):
                    # Check collision filter
                    bi, bj = self.bodies[i].body_id, self.bodies[j].body_id
                    pair = (min(bi, bj), max(bi, bj))
                    if pair in self._filter_pairs:
                        continue
                    if aabbs[i].overlaps(aabbs[j]):
                        contacts = self._narrow_pair(self.bodies[i], self.bodies[j])
                        for cp in contacts:
                            results.append((cp, bi, bj))

        return results

    def _narrow_ground(self, body: CollisionBody,
                       gh: float, gn: np.ndarray) -> list[ContactPoint]:
        """Narrow phase: single body vs ground."""
        g = body.geometry
        T = body.transform
        center = T.translation
        R = T.rotation

        if g.geometry_type == GeometryType.SPHERE:
            cp = sphere_ground(center, g.radius, gh, gn)
            return [cp] if cp is not None else []

        elif g.geometry_type == GeometryType.BOX:
            return box_ground(center, R, g.size / 2.0, gh, gn)

        elif g.geometry_type == GeometryType.CYLINDER:
            return cylinder_ground(center, R, g.radius, g.length / 2.0, gh, gn)

        elif g.geometry_type == GeometryType.MESH:
            if g.mesh_loaded:
                return mesh_ground(g.mesh_vertices, R, center, gh, gn)

        return []

    def _narrow_pair(self, a: CollisionBody, b: CollisionBody) -> list[ContactPoint]:
        """Narrow phase: body vs body."""
        ga, gb = a.geometry, b.geometry
        ta, tb = a.transform, b.transform
        type_a, type_b = ga.geometry_type, gb.geometry_type

        # Sphere-sphere
        if type_a == GeometryType.SPHERE and type_b == GeometryType.SPHERE:
            cp = sphere_sphere(ta.translation, ga.radius,
                               tb.translation, gb.radius)
            return [cp] if cp is not None else []

        # Box-box
        if type_a == GeometryType.BOX and type_b == GeometryType.BOX:
            return box_box(
                ta.translation, ta.rotation, ga.size / 2.0,
                tb.translation, tb.rotation, gb.size / 2.0,
            )

        # Box-sphere / Sphere-box
        if type_a == GeometryType.BOX and type_b == GeometryType.SPHERE:
            cp = box_sphere(ta.translation, ta.rotation, ga.size / 2.0,
                            tb.translation, gb.radius)
            return [cp] if cp is not None else []

        if type_a == GeometryType.SPHERE and type_b == GeometryType.BOX:
            cp = box_sphere(tb.translation, tb.rotation, gb.size / 2.0,
                            ta.translation, ga.radius)
            if cp is not None:
                # Flip normal direction (box_sphere returns normal from box→sphere)
                cp.normal = -cp.normal
                cp.point_a, cp.point_b = cp.point_b, cp.point_a
            return [cp] if cp is not None else []

        # Mesh-box / Box-mesh
        if type_a == GeometryType.MESH and type_b == GeometryType.BOX:
            if ga.mesh_loaded:
                world_verts = (ta.rotation @ ga.mesh_vertices.T).T + ta.translation
                return mesh_box(world_verts, tb.translation, tb.rotation, gb.size / 2.0)
            return []

        if type_a == GeometryType.BOX and type_b == GeometryType.MESH:
            if gb.mesh_loaded:
                world_verts = (tb.rotation @ gb.mesh_vertices.T).T + tb.translation
                contacts = mesh_box(world_verts, ta.translation, ta.rotation, ga.size / 2.0)
                # Flip: mesh_box returns normal from box→mesh, we need B→A
                for cp in contacts:
                    cp.normal = -cp.normal
                    cp.point_a, cp.point_b = cp.point_b, cp.point_a
                return contacts
            return []

        return []
