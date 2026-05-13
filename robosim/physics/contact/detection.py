"""Contact detection: AABB broad phase + analytical narrow phase.

Issue 4 fix : tight cylinder AABB (OBB-projection, not bounding sphere)
Issue 5 add : BVH broad phase for O(N log N) body-body pair culling
"""

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
    _USE_CPP_CONTACT,
    _cpp_mod,
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


# ═══════════════════════════════════════════════════════════════
# BVH (Bounding Volume Hierarchy) — Issue 5
# ═══════════════════════════════════════════════════════════════

class BVHNode:
    """Node in a binary BVH tree built over CollisionBody AABBs."""

    __slots__ = ("aabb", "left", "right", "leaf_index")

    def __init__(self) -> None:
        self.aabb: AABB | None = None
        self.left: BVHNode | None = None
        self.right: BVHNode | None = None
        self.leaf_index: int = -1   # >= 0 only for leaf nodes

    @property
    def is_leaf(self) -> bool:
        return self.left is None


def _merge_aabbs(aabbs: list[AABB]) -> AABB:
    min_pt = aabbs[0].min_pt.copy()
    max_pt = aabbs[0].max_pt.copy()
    for a in aabbs[1:]:
        np.minimum(min_pt, a.min_pt, out=min_pt)
        np.maximum(max_pt, a.max_pt, out=max_pt)
    return AABB(min_pt, max_pt)


def _aabb_volume(aabb: AABB) -> float:
    ext = aabb.max_pt - aabb.min_pt
    return float(ext[0] * ext[1] * ext[2])


def _build_bvh(indices: list[int], aabbs: list[AABB]) -> BVHNode:
    """Recursively build a BVH over the given body indices.

    Splits along the longest AABB axis at the median centroid.
    """
    node = BVHNode()

    if len(indices) == 1:
        node.aabb = aabbs[indices[0]]
        node.leaf_index = indices[0]
        return node

    # Merged AABB for all bodies in this subtree
    node.aabb = _merge_aabbs([aabbs[i] for i in indices])

    # Split along the longest axis
    extent = node.aabb.max_pt - node.aabb.min_pt
    axis = int(np.argmax(extent))

    # Sort by centroid on that axis
    centroids = np.array(
        [(aabbs[i].min_pt[axis] + aabbs[i].max_pt[axis]) * 0.5 for i in indices]
    )
    order = np.argsort(centroids)
    sorted_idx = [indices[k] for k in order]
    mid = max(1, len(sorted_idx) // 2)

    node.left = _build_bvh(sorted_idx[:mid], aabbs)
    node.right = _build_bvh(sorted_idx[mid:], aabbs)
    return node


def _bvh_cross_pairs(
    node_a: BVHNode,
    node_b: BVHNode,
    filter_pairs: set[tuple[int, int]],
    body_ids: list[int],
    results: list[tuple[int, int]],
) -> None:
    """Append overlapping (body_index_a, body_index_b) pairs to *results*.

    One body comes from node_a's subtree and one from node_b's subtree.
    """
    if not node_a.aabb.overlaps(node_b.aabb):
        return

    if node_a.is_leaf and node_b.is_leaf:
        i, j = node_a.leaf_index, node_b.leaf_index
        bi, bj = body_ids[i], body_ids[j]
        pair = (min(bi, bj), max(bi, bj))
        if pair not in filter_pairs:
            results.append((i, j))
        return

    # Expand the node with the larger AABB volume to limit recursion depth
    if node_a.is_leaf or (
        not node_b.is_leaf and _aabb_volume(node_a.aabb) < _aabb_volume(node_b.aabb)
    ):
        _bvh_cross_pairs(node_a, node_b.left, filter_pairs, body_ids, results)
        _bvh_cross_pairs(node_a, node_b.right, filter_pairs, body_ids, results)
    else:
        _bvh_cross_pairs(node_a.left, node_b, filter_pairs, body_ids, results)
        _bvh_cross_pairs(node_a.right, node_b, filter_pairs, body_ids, results)


def _bvh_self_pairs(
    node: BVHNode,
    filter_pairs: set[tuple[int, int]],
    body_ids: list[int],
    results: list[tuple[int, int]],
) -> None:
    """Find all overlapping pairs within a BVH subtree (self-collision)."""
    if node.is_leaf:
        return
    _bvh_self_pairs(node.left, filter_pairs, body_ids, results)
    _bvh_self_pairs(node.right, filter_pairs, body_ids, results)
    _bvh_cross_pairs(node.left, node.right, filter_pairs, body_ids, results)


def compute_aabb(geometry: Geometry, transform: Transform) -> AABB:
    """Compute world-space AABB for a geometry."""
    R = transform.rotation
    t = transform.translation

    if geometry.geometry_type == GeometryType.SPHERE:
        if _USE_CPP_CONTACT:
            buf = _cpp_mod.contact.aabb_sphere(t, float(geometry.radius))
            return AABB(np.ascontiguousarray(buf[0]), np.ascontiguousarray(buf[1]))
        r = geometry.radius
        return AABB(t - r, t + r)

    elif geometry.geometry_type == GeometryType.BOX:
        if _USE_CPP_CONTACT:
            buf = _cpp_mod.contact.aabb_box(R, t, geometry.size / 2.0)
            return AABB(np.ascontiguousarray(buf[0]), np.ascontiguousarray(buf[1]))
        he = geometry.size / 2.0
        # Rotated AABB: project half-extents onto each world axis
        extent = np.abs(R) @ he
        return AABB(t - extent, t + extent)

    elif geometry.geometry_type == GeometryType.CYLINDER:
        if _USE_CPP_CONTACT:
            buf = _cpp_mod.contact.aabb_cylinder(
                R, t, float(geometry.radius), float(geometry.length))
            return AABB(np.ascontiguousarray(buf[0]), np.ascontiguousarray(buf[1]))
        r, hl = geometry.radius, geometry.length / 2.0
        # Tight AABB: project OBB cylinder onto each world axis.
        # For axis unit-vector u and world axis e_i:
        #   half-extent_i = hl * |u·e_i| + r * sqrt(1 - (u·e_i)^2)
        axis = R[:, 2]          # cylinder symmetry axis in world frame
        cos2 = axis ** 2        # (cos θ_i)^2 for each world axis
        extent = hl * np.abs(axis) + r * np.sqrt(np.maximum(0.0, 1.0 - cos2))
        return AABB(t - extent, t + extent)

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

        # AABBs are needed by both ground-skip and body-body broad phase.
        # Compute once and share.
        aabbs = [compute_aabb(b.geometry, b.transform) for b in self.bodies]

        # ── Ground contacts ──
        if self.ground is not None:
            gh = self.ground.height
            gn = self.ground.normal
            # Axis-aligned-up ground: skip bodies whose AABB minimum Z is
            # above ground.  Saves O(8) corner checks per body when nothing
            # near the floor (typical during free flight / grasping).
            up_z = (gn[0] == 0.0 and gn[1] == 0.0 and gn[2] > 0.0)
            for k, body in enumerate(self.bodies):
                if up_z and aabbs[k].min_pt[2] > gh:
                    continue
                contacts = self._narrow_ground(body, gh, gn)
                for cp in contacts:
                    results.append((cp, body.body_id, -1))

        # ── Body-body contacts (BVH broad + narrow) ──
        n = len(self.bodies)
        if n > 1:
            body_ids = [b.body_id for b in self.bodies]

            if n <= 16:
                # For small scenes brute force is faster than BVH construction.
                # For grasping setups (≤16 collision bodies) this branch is hit
                # and the per-substep BVH rebuild cost vanishes.
                candidate_pairs: list[tuple[int, int]] = []
                for i in range(n):
                    aabb_i_min = aabbs[i].min_pt
                    aabb_i_max = aabbs[i].max_pt
                    for j in range(i + 1, n):
                        bi, bj = body_ids[i], body_ids[j]
                        fp = (min(bi, bj), max(bi, bj))
                        if fp in self._filter_pairs:
                            continue
                        # Inlined AABB overlap (avoids method dispatch + np.all).
                        aabb_j_min = aabbs[j].min_pt
                        aabb_j_max = aabbs[j].max_pt
                        if (aabb_i_min[0] <= aabb_j_max[0] and aabb_j_min[0] <= aabb_i_max[0] and
                            aabb_i_min[1] <= aabb_j_max[1] and aabb_j_min[1] <= aabb_i_max[1] and
                            aabb_i_min[2] <= aabb_j_max[2] and aabb_j_min[2] <= aabb_i_max[2]):
                            candidate_pairs.append((i, j))
            else:
                # BVH traversal — O(N log N) average
                root = _build_bvh(list(range(n)), aabbs)
                candidate_pairs = []
                _bvh_self_pairs(root, self._filter_pairs, body_ids, candidate_pairs)

            for i, j in candidate_pairs:
                bi, bj = body_ids[i], body_ids[j]
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
