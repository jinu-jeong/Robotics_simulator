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
