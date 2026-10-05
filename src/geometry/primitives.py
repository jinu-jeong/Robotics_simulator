"""Triangle-mesh primitives used for visualization (objects, arrows, markers).

All functions return ``(vertices, faces)`` with ``vertices`` of shape (V, 3)
in meters and ``faces`` of shape (F, 3) with outward-oriented triangles.
Nothing here touches the physics; these shapes are display-only.
"""

from __future__ import annotations

import numpy as np


def _orthonormal_frame(direction: np.ndarray) -> np.ndarray:
    """Return a 3x3 matrix whose columns (e1, e2, d) form a right-handed frame
    with ``d`` the normalized ``direction``."""
    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-15:
        raise ValueError("direction must be non-zero")
    d = d / n
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(helper, d)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return np.stack([e1, e2, d], axis=1)


def box_mesh(center, size) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box. ``size`` = full extents (sx, sy, sz)."""
    c = np.asarray(center, dtype=float)
    h = 0.5 * np.asarray(size, dtype=float)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    verts = c + signs * h  # index = x*4 + y*2 + z
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2],  # -x
            [4, 6, 7], [4, 7, 5],  # +x
            [0, 4, 5], [0, 5, 1],  # -y
            [2, 3, 7], [2, 7, 6],  # +y
            [0, 2, 6], [0, 6, 4],  # -z
            [1, 5, 7], [1, 7, 3],  # +z
        ],
        dtype=np.int64,
    )
    return verts, faces


def sphere_mesh(center, radius: float, n_lat: int = 16, n_lon: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """UV sphere with outward triangles (poles are single vertices)."""
    c = np.asarray(center, dtype=float)
    verts = [c + np.array([0.0, 0.0, radius])]
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat
        st, ct = np.sin(theta), np.cos(theta)
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            verts.append(c + radius * np.array([st * np.cos(phi), st * np.sin(phi), ct]))
    verts.append(c + np.array([0.0, 0.0, -radius]))
    verts = np.array(verts)
    top, bottom = 0, len(verts) - 1

    def ring(i, j):  # i in 1..n_lat-1
        return 1 + (i - 1) * n_lon + (j % n_lon)

    faces = []
    for j in range(n_lon):
        faces.append([top, ring(1, j), ring(1, j + 1)])
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a, b = ring(i, j), ring(i, j + 1)
            c2, d = ring(i + 1, j), ring(i + 1, j + 1)
            faces.append([a, c2, d])
            faces.append([a, d, b])
    for j in range(n_lon):
        faces.append([bottom, ring(n_lat - 1, j + 1), ring(n_lat - 1, j)])
    return verts, np.array(faces, dtype=np.int64)


def cylinder_mesh(p0, p1, radius: float, n_seg: int = 16, caps: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Closed cylinder from ``p0`` to ``p1``."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    axis = p1 - p0
    frame = _orthonormal_frame(axis)
    e1, e2 = frame[:, 0], frame[:, 1]
    ang = 2.0 * np.pi * np.arange(n_seg) / n_seg
    ring_dir = np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2)
    bottom = p0 + radius * ring_dir
    top = p1 + radius * ring_dir
    verts = [bottom, top]
    faces = []
    for j in range(n_seg):
        a, b = j, (j + 1) % n_seg
        faces.append([a, b, n_seg + b])
        faces.append([a, n_seg + b, n_seg + a])
    if caps:
        cb, ct = 2 * n_seg, 2 * n_seg + 1
        verts.append(p0[None, :])
        verts.append(p1[None, :])
        for j in range(n_seg):
            a, b = j, (j + 1) % n_seg
            faces.append([cb, b, a])
            faces.append([ct, n_seg + a, n_seg + b])
    return np.concatenate(verts, axis=0), np.array(faces, dtype=np.int64)


def cone_mesh(base_center, tip, radius: float, n_seg: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Closed cone with base at ``base_center`` pointing to ``tip``."""
    b = np.asarray(base_center, dtype=float)
    t = np.asarray(tip, dtype=float)
    frame = _orthonormal_frame(t - b)
    e1, e2 = frame[:, 0], frame[:, 1]
    ang = 2.0 * np.pi * np.arange(n_seg) / n_seg
    ring = b + radius * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))
    verts = np.concatenate([ring, t[None, :], b[None, :]], axis=0)
    tip_i, base_i = n_seg, n_seg + 1
    faces = []
    for j in range(n_seg):
        a, c = j, (j + 1) % n_seg
        faces.append([a, c, tip_i])
        faces.append([base_i, c, a])
    return verts, np.array(faces, dtype=np.int64)


def arrow_mesh(
    origin,
    direction,
    length: float,
    shaft_radius: float,
    head_radius: float,
    head_length: float,
    n_seg: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """3D arrow (cylinder shaft + cone head) starting at ``origin``.

    ``length`` is the total tip-to-tail length. If ``head_length`` exceeds
    ``length`` the arrow degenerates to a cone of the full length.
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / max(np.linalg.norm(d), 1e-15)
    head_length = min(head_length, length)
    shaft_len = length - head_length
    tip = o + length * d
    head_base = o + shaft_len * d
    hv, hf = cone_mesh(head_base, tip, head_radius, n_seg)
    if shaft_len <= 1e-12:
        return hv, hf
    sv, sf = cylinder_mesh(o, head_base, shaft_radius, n_seg, caps=True)
    return merge_meshes([(sv, sf), (hv, hf)])


def merge_meshes(meshes) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ``[(verts, faces), ...]`` into one mesh with offset indices."""
    verts, faces, off = [], [], 0
    for v, f in meshes:
        verts.append(v)
        faces.append(np.asarray(f, dtype=np.int64) + off)
        off += len(v)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(verts, axis=0), np.concatenate(faces, axis=0)


def to_triangle_soup(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Expand an indexed mesh into (3F, 3) vertices in triangle order."""
    return np.asarray(verts, dtype=float)[np.asarray(faces, dtype=np.int64).ravel()]
