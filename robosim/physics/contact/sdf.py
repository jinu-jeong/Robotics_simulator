"""Signed distance functions and contact point queries for geometry primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContactPoint:
    """Result of a narrow-phase contact query."""

    point_a: np.ndarray       # (3,) contact point on body A (world frame)
    point_b: np.ndarray       # (3,) contact point on body B (world frame)
    normal: np.ndarray        # (3,) unit normal pointing from B → A
    penetration: float        # positive = penetrating


# ── Ground plane contacts ────────────────────────────────────────

def sphere_ground(
    center: np.ndarray, radius: float,
    ground_height: float = 0.0, ground_normal: np.ndarray | None = None,
) -> ContactPoint | None:
    """Sphere vs infinite ground plane."""
    if ground_normal is None:
        ground_normal = np.array([0.0, 0.0, 1.0])
    n = ground_normal / np.linalg.norm(ground_normal)

    dist = np.dot(center, n) - ground_height  # signed distance from plane
    penetration = radius - dist

    if penetration <= 0:
        return None

    point_a = center - radius * n          # lowest point on sphere
    point_b = center - dist * n            # projection onto plane
    return ContactPoint(point_a=point_a, point_b=point_b,
                        normal=n, penetration=penetration)


def box_ground(
    center: np.ndarray, rotation: np.ndarray, half_extents: np.ndarray,
    ground_height: float = 0.0, ground_normal: np.ndarray | None = None,
) -> list[ContactPoint]:
    """Box vs ground plane — returns contact for each penetrating vertex."""
    if ground_normal is None:
        ground_normal = np.array([0.0, 0.0, 1.0])
    n = ground_normal / np.linalg.norm(ground_normal)
    hx, hy, hz = half_extents

    # 8 corner vertices in local frame
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float64)
    local_verts = signs * half_extents  # (8, 3)

    # Transform to world
    world_verts = (rotation @ local_verts.T).T + center  # (8, 3)

    contacts = []
    for v in world_verts:
        dist = np.dot(v, n) - ground_height
        if dist < 0:
            point_a = v.copy()
            point_b = v - dist * n
            contacts.append(ContactPoint(
                point_a=point_a, point_b=point_b,
                normal=n, penetration=-dist,
            ))
    return contacts


def cylinder_ground(
    center: np.ndarray, rotation: np.ndarray,
    radius: float, half_length: float,
    ground_height: float = 0.0, ground_normal: np.ndarray | None = None,
    n_ring: int = 8,
) -> list[ContactPoint]:
    """Cylinder vs ground — sample points on bottom/top ring edges."""
    if ground_normal is None:
        ground_normal = np.array([0.0, 0.0, 1.0])
    n = ground_normal / np.linalg.norm(ground_normal)

    contacts = []
    for z_sign in [-1.0, 1.0]:
        for i in range(n_ring):
            angle = 2 * np.pi * i / n_ring
            local = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                z_sign * half_length,
            ])
            world = rotation @ local + center
            dist = np.dot(world, n) - ground_height
            if dist < 0:
                contacts.append(ContactPoint(
                    point_a=world.copy(),
                    point_b=world - dist * n,
                    normal=n,
                    penetration=-dist,
                ))
    return contacts


# ── Body–body contacts ───────────────────────────────────────────

def sphere_sphere(
    c1: np.ndarray, r1: float,
    c2: np.ndarray, r2: float,
) -> ContactPoint | None:
    """Sphere vs sphere contact."""
    diff = c1 - c2
    dist = np.linalg.norm(diff)
    penetration = r1 + r2 - dist

    if penetration <= 0 or dist < 1e-12:
        return None

    n = diff / dist
    point_a = c1 - r1 * n
    point_b = c2 + r2 * n
    return ContactPoint(point_a=point_a, point_b=point_b,
                        normal=n, penetration=penetration)


# ── Box–Box (SAT) & Box–Sphere contacts ─────────────────────────

def box_box(
    center_a: np.ndarray, rot_a: np.ndarray, half_a: np.ndarray,
    center_b: np.ndarray, rot_b: np.ndarray, half_b: np.ndarray,
) -> list[ContactPoint]:
    """OBB vs OBB using Separating Axis Theorem (SAT).

    Tests 15 potential separating axes (3+3 face normals + 9 edge-edge).
    Returns contact points at the penetrating vertices of the incident face.
    """
    # Axes of each box in world frame
    ax_a = rot_a.T  # (3, 3) — rows are the 3 local axes of A in world
    ax_b = rot_b.T

    d = center_b - center_a  # vector from A center to B center

    # --- Collect all 15 candidate separating axes ---
    axes: list[np.ndarray] = []

    # Face normals of A (3)
    for i in range(3):
        axes.append(ax_a[i])

    # Face normals of B (3)
    for i in range(3):
        axes.append(ax_b[i])

    # Edge-edge cross products (9)
    for i in range(3):
        for j in range(3):
            c = np.cross(ax_a[i], ax_b[j])
            n = np.linalg.norm(c)
            if n > 1e-8:
                axes.append(c / n)

    # --- Find axis of minimum penetration ---
    min_pen = np.inf
    min_axis = None

    for axis in axes:
        # Project half-extents onto axis
        proj_a = sum(half_a[i] * abs(np.dot(ax_a[i], axis)) for i in range(3))
        proj_b = sum(half_b[i] * abs(np.dot(ax_b[i], axis)) for i in range(3))
        dist = abs(np.dot(d, axis))
        pen = proj_a + proj_b - dist

        if pen <= 0:
            return []  # separating axis found — no collision

        if pen < min_pen:
            min_pen = pen
            # Ensure normal points from B to A
            min_axis = axis if np.dot(d, axis) < 0 else -axis

    if min_axis is None:
        return []

    normal = min_axis
    penetration = min_pen

    # --- Generate contact points (vertex-based) ---
    # Find vertices of B that are most penetrating into A
    contacts = []
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float64)

    # Test vertices of B against A
    verts_b = (rot_b @ (signs * half_b).T).T + center_b
    for v in verts_b:
        # Check if vertex is inside A (using local coordinates)
        v_local_a = rot_a.T @ (v - center_a)
        if (np.abs(v_local_a) <= half_a + 1e-6).all():
            # Penetrating vertex — project onto contact plane
            pen_depth = np.dot(v - center_a, normal) + _support_dist(half_a, ax_a, normal)
            if pen_depth > 0:
                point_b = v.copy()
                point_a = v + pen_depth * normal
                contacts.append(ContactPoint(
                    point_a=point_a, point_b=point_b,
                    normal=normal, penetration=pen_depth,
                ))

    # Also test vertices of A against B
    verts_a = (rot_a @ (signs * half_a).T).T + center_a
    for v in verts_a:
        v_local_b = rot_b.T @ (v - center_b)
        if (np.abs(v_local_b) <= half_b + 1e-6).all():
            pen_depth = -np.dot(v - center_b, normal) + _support_dist(half_b, ax_b, -normal)
            if pen_depth > 0:
                point_a = v.copy()
                point_b = v - pen_depth * normal
                contacts.append(ContactPoint(
                    point_a=point_a, point_b=point_b,
                    normal=normal, penetration=pen_depth,
                ))

    # If no vertex contacts found, generate a single contact at midpoint
    if not contacts:
        mid = (center_a + center_b) / 2.0
        contacts.append(ContactPoint(
            point_a=mid + normal * penetration / 2,
            point_b=mid - normal * penetration / 2,
            normal=normal, penetration=penetration,
        ))

    return contacts


def _support_dist(half_ext: np.ndarray, axes: np.ndarray, direction: np.ndarray) -> float:
    """Support distance: how far box extends along direction from its center."""
    return sum(half_ext[i] * abs(np.dot(axes[i], direction)) for i in range(3))


def box_sphere(
    box_center: np.ndarray, box_rot: np.ndarray, box_half: np.ndarray,
    sphere_center: np.ndarray, sphere_radius: float,
) -> ContactPoint | None:
    """OBB vs Sphere contact test.

    Finds closest point on OBB surface to sphere center, then checks distance.
    """
    # Transform sphere center into box local frame
    local = box_rot.T @ (sphere_center - box_center)

    # Clamp to box extents → closest point in local frame
    closest_local = np.clip(local, -box_half, box_half)

    # Back to world frame
    closest_world = box_rot @ closest_local + box_center

    diff = sphere_center - closest_world
    dist = np.linalg.norm(diff)

    if dist > sphere_radius or dist < 1e-12:
        return None

    normal = diff / dist
    penetration = sphere_radius - dist

    point_a = closest_world.copy()
    point_b = sphere_center - sphere_radius * normal
    return ContactPoint(point_a=point_a, point_b=point_b,
                        normal=normal, penetration=penetration)


def mesh_ground(
    vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
    ground_height: float = 0.0, ground_normal: np.ndarray | None = None,
) -> list[ContactPoint]:
    """Mesh vs ground plane — test each vertex against the plane."""
    if ground_normal is None:
        ground_normal = np.array([0.0, 0.0, 1.0])
    n = ground_normal / np.linalg.norm(ground_normal)

    world_verts = (rotation @ vertices.T).T + translation
    dists = world_verts @ n - ground_height

    contacts = []
    for i in np.where(dists < 0)[0]:
        v = world_verts[i]
        contacts.append(ContactPoint(
            point_a=v.copy(),
            point_b=v - dists[i] * n,
            normal=n,
            penetration=-dists[i],
        ))
    return contacts


# ── Triangle proximity queries ──────────────────────────────────

def point_triangle_distance(
    point: np.ndarray,
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Closest point on a triangle to a query point.

    Parameters
    ----------
    point : (3,) query point
    v0, v1, v2 : (3,) triangle vertices

    Returns
    -------
    closest : (3,) closest point on triangle
    distance : scalar distance (always >= 0)
    bary : (3,) barycentric coordinates of closest point
    """
    e0 = v1 - v0
    e1 = v2 - v0
    v = point - v0

    d00 = np.dot(e0, e0)
    d01 = np.dot(e0, e1)
    d11 = np.dot(e1, e1)
    d20 = np.dot(v, e0)
    d21 = np.dot(v, e1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-30:
        # Degenerate triangle
        return v0.copy(), float(np.linalg.norm(point - v0)), np.array([1.0, 0.0, 0.0])

    inv_denom = 1.0 / denom
    s = (d11 * d20 - d01 * d21) * inv_denom
    t = (d00 * d21 - d01 * d20) * inv_denom

    # Clamp to triangle: project to nearest edge/vertex if outside
    if s >= 0 and t >= 0 and s + t <= 1:
        # Inside triangle
        bary = np.array([1.0 - s - t, s, t])
        closest = v0 + s * e0 + t * e1
    else:
        # Project onto edges and pick closest
        best_dist = np.inf
        closest = v0.copy()
        bary = np.array([1.0, 0.0, 0.0])

        # Edge v0-v1
        cp, d, b = _closest_point_on_segment(point, v0, v1)
        if d < best_dist:
            best_dist, closest = d, cp
            bary = np.array([1.0 - b, b, 0.0])

        # Edge v0-v2
        cp, d, b = _closest_point_on_segment(point, v0, v2)
        if d < best_dist:
            best_dist, closest = d, cp
            bary = np.array([1.0 - b, 0.0, b])

        # Edge v1-v2
        cp, d, b = _closest_point_on_segment(point, v1, v2)
        if d < best_dist:
            best_dist, closest = d, cp
            bary = np.array([0.0, 1.0 - b, b])

    dist = float(np.linalg.norm(point - closest))
    return closest, dist, bary


def _closest_point_on_segment(
    point: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Closest point on segment ab to point. Returns (closest, dist, t)."""
    ab = b - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-30:
        return a.copy(), float(np.linalg.norm(point - a)), 0.0
    t = np.clip(np.dot(point - a, ab) / ab_sq, 0.0, 1.0)
    closest = a + t * ab
    return closest, float(np.linalg.norm(point - closest)), float(t)


def triangle_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Outward normal of a triangle (unnormalized ok, will be normalized)."""
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    if norm < 1e-20:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def sphere_triangle(
    center: np.ndarray, radius: float,
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
) -> ContactPoint | None:
    """Sphere vs triangle contact.

    Returns ContactPoint if sphere overlaps triangle (within radius distance).
    Normal points from triangle toward sphere center.
    """
    closest, dist, bary = point_triangle_distance(center, v0, v1, v2)
    penetration = radius - dist

    if penetration <= 0:
        return None

    if dist < 1e-12:
        # Center exactly on triangle — use face normal
        normal = triangle_normal(v0, v1, v2)
    else:
        normal = (center - closest)
        normal = normal / np.linalg.norm(normal)

    point_a = center - radius * normal   # point on sphere surface
    point_b = closest                     # point on triangle
    return ContactPoint(point_a=point_a, point_b=point_b,
                        normal=normal, penetration=penetration)


def points_ground(
    points: np.ndarray,
    ground_height: float = 0.0,
    ground_normal: np.ndarray | None = None,
) -> list[tuple[int, ContactPoint]]:
    """Multiple points (e.g. FEM surface nodes) vs ground plane.

    Returns list of (point_index, ContactPoint).
    """
    if ground_normal is None:
        ground_normal = np.array([0.0, 0.0, 1.0])
    n = ground_normal / np.linalg.norm(ground_normal)

    dists = points @ n - ground_height  # (n_pts,)
    below = np.where(dists < 0)[0]

    contacts = []
    for idx in below:
        pt = points[idx]
        contacts.append((int(idx), ContactPoint(
            point_a=pt.copy(),
            point_b=pt - dists[idx] * n,
            normal=n,
            penetration=-dists[idx],
        )))
    return contacts
