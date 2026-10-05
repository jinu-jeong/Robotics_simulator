"""Generic tetrahedral-mesh utilities (NumPy only).

Conventions
-----------
* ``nodes``  : float array of shape (N, 3), SI meters.
* ``tets``   : int array of shape (M, 4). Every tetrahedron is stored with
  *positive orientation*, i.e. ``signed_volume(a, b, c, d) > 0`` where
  ``6 V = det[b - a, c - a, d - a]``. FEM element routines rely on this.
* ``faces``  : int array of shape (F, 3), surface triangles oriented
  counter-clockwise when seen from *outside* (outward normal by the
  right-hand rule).
* ``edges``  : int array of shape (E, 2) with ``edges[:, 0] < edges[:, 1]``.
"""

from __future__ import annotations

import numpy as np

# Local face list of a tetrahedron (a, b, c, d): face i is opposite node i.
_TET_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int64)
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)


def tet_signed_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volume of every tetrahedron, ``V = det[b-a, c-a, d-a] / 6``."""
    a = nodes[tets[:, 0]]
    b = nodes[tets[:, 1]] - a
    c = nodes[tets[:, 2]] - a
    d = nodes[tets[:, 3]] - a
    return np.einsum("ij,ij->i", b, np.cross(c, d)) / 6.0


def fix_tet_orientation(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return a copy of ``tets`` where every element has positive volume.

    Negative elements are fixed by swapping their last two nodes (an odd
    permutation flips the sign of the determinant).
    """
    tets = np.array(tets, dtype=np.int64, copy=True)
    vol = tet_signed_volumes(nodes, tets)
    neg = vol < 0
    if np.any(neg):
        tets[neg, 2], tets[neg, 3] = tets[neg, 3].copy(), tets[neg, 2].copy()
    return tets


def extract_surface_triangles(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Boundary triangles of a tet mesh, oriented with outward normals.

    A face is on the boundary iff it belongs to exactly one tetrahedron.
    Orientation is fixed geometrically: the triangle normal is required to
    point away from the centroid of its owning tetrahedron. This is robust
    regardless of the node ordering of the input tets.
    """
    tets = np.asarray(tets, dtype=np.int64)
    faces = tets[:, _TET_FACES].reshape(-1, 3)                     # (4M, 3)
    owner = np.repeat(np.arange(len(tets)), 4)
    key = np.sort(faces, axis=1)
    _, first_idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    boundary_idx = first_idx[counts == 1]
    faces = faces[boundary_idx]
    owner = owner[boundary_idx]

    tet_centroid = nodes[tets[owner]].mean(axis=1)
    p0, p1, p2 = nodes[faces[:, 0]], nodes[faces[:, 1]], nodes[faces[:, 2]]
    normal = np.cross(p1 - p0, p2 - p0)
    face_centroid = (p0 + p1 + p2) / 3.0
    inward = np.einsum("ij,ij->i", normal, face_centroid - tet_centroid) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return faces


def triangle_normals(nodes: np.ndarray, faces: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Per-triangle normals (right-hand rule)."""
    p0, p1, p2 = nodes[faces[:, 0]], nodes[faces[:, 1]], nodes[faces[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    if normalize:
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    return n


def triangle_areas(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(triangle_normals(nodes, faces, normalize=False), axis=1)


def vertex_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals for surface nodes (zero for interior nodes)."""
    fn = triangle_normals(nodes, faces, normalize=False)
    vn = np.zeros_like(nodes, dtype=float)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.divide(vn, norm, out=np.zeros_like(vn), where=norm > 0)


def unique_edges(simplices: np.ndarray) -> np.ndarray:
    """Unique undirected *corner* edges of triangles (F,3), tets (M,4), T10 (M,10),
    hex8 (M,8) or H20 (M,20) connectivity (VTK ordering)."""
    simplices = np.asarray(simplices, dtype=np.int64)
    k = simplices.shape[1]
    if k in (3, 6):
        local = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    elif k in (4, 10):
        local = _TET_EDGES
    elif k in (8, 20):
        local = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int64)
    else:
        raise ValueError("unsupported connectivity width (expected 3/6, 4/10 or 8/20 columns)")
    e = simplices[:, local].reshape(-1, 2)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def feature_edges(nodes: np.ndarray, faces: np.ndarray, angle_deg: float = 30.0) -> np.ndarray:
    """Edges where the dihedral angle between adjacent triangles exceeds ``angle_deg``
    (plus open boundary edges). For a box this is its 12 outline edges."""
    faces = np.asarray(faces, dtype=np.int64)
    e = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    owner = np.repeat(np.arange(len(faces)), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, owner = key[order], owner[order]
    n = triangle_normals(nodes, faces)
    cos_thr = np.cos(np.deg2rad(angle_deg))
    out = []
    i = 0
    while i < len(key):
        j = i + 1
        while j < len(key) and np.array_equal(key[j], key[i]):
            j += 1
        if j - i == 1:
            out.append(key[i])  # boundary edge
        else:
            c = np.dot(n[owner[i]], n[owner[i + 1]])
            if c < cos_thr:
                out.append(key[i])
        i = j
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def surface_node_indices(faces: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(faces).ravel())


def coincident_node_map(query_nodes: np.ndarray, target_nodes: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """For each query node the index of the coincident node in ``target_nodes``.

    Used to sample a field computed on a nested (refined) mesh at the nodes of a
    coarser one. Raises if some query node has no counterpart within ``tol``.
    """
    from scipy.spatial import cKDTree

    d, idx = cKDTree(np.asarray(target_nodes, float)).query(np.asarray(query_nodes, float))
    if d.max() > tol:
        raise ValueError(f"{int((d > tol).sum())} query nodes have no coincident target node (max distance {d.max():.2e})")
    return np.asarray(idx, dtype=np.int64)


def select_nodes_on_plane(nodes: np.ndarray, axis: int, value: float, tol: float = 1e-9) -> np.ndarray:
    """Indices of nodes whose ``axis`` coordinate equals ``value`` within ``tol``."""
    return np.nonzero(np.abs(nodes[:, axis] - value) <= tol)[0]


def nearest_node(nodes: np.ndarray, point, candidates: np.ndarray | None = None) -> int:
    """Index (into ``nodes``) of the node closest to ``point``.

    If ``candidates`` is given the search is restricted to those indices
    (e.g. surface nodes only).
    """
    point = np.asarray(point, dtype=float)
    if candidates is None:
        d = np.linalg.norm(nodes - point, axis=1)
        return int(np.argmin(d))
    d = np.linalg.norm(nodes[candidates] - point, axis=1)
    return int(candidates[int(np.argmin(d))])


def bounding_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return pts.min(axis=0), pts.max(axis=0)


def euler_characteristic(faces: np.ndarray) -> int:
    """V - E + F of a closed triangle surface (2 for a sphere/box-like shell)."""
    v = len(surface_node_indices(faces))
    e = len(unique_edges(faces))
    return v - e + len(faces)


def is_closed_surface(faces: np.ndarray) -> bool:
    """True if every edge is shared by exactly two triangles."""
    faces = np.asarray(faces, dtype=np.int64)
    e = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))
