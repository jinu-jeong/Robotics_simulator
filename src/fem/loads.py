"""Nodal load vectors for the linear FEM.

Sign convention: a load vector ``f`` (3N,) contains forces *acting on the
body* in the global frame [N]. A contact that presses down on the top
surface therefore has a negative z component.
"""

from __future__ import annotations

import numpy as np

from ..geometry.mesh_utils import triangle_areas


def zero_load(n_nodes: int) -> np.ndarray:
    return np.zeros(3 * n_nodes)


def point_load(n_nodes: int, node: int, force) -> np.ndarray:
    """Concentrated force at one node."""
    f = zero_load(n_nodes)
    f[3 * node : 3 * node + 3] = np.asarray(force, dtype=float)
    return f


def distributed_node_load(n_nodes: int, nodes_idx, total_force) -> np.ndarray:
    """Total force split equally among the given nodes."""
    nodes_idx = np.asarray(nodes_idx, dtype=np.int64).reshape(-1)
    f = zero_load(n_nodes)
    per = np.asarray(total_force, dtype=float) / len(nodes_idx)
    for n in nodes_idx:
        f[3 * n : 3 * n + 3] += per
    return f


def uniform_traction_load(nodes: np.ndarray, faces: np.ndarray, traction) -> np.ndarray:
    """Lumped nodal loads of a uniform traction t [Pa] on (display) triangles.

    Each vertex receives ``t * A / 3``; the total force is exactly ``t * Σ A``.
    On the quadratic mesh this is a lumped approximation (local end effects);
    use :func:`total_force_on_plane` / :func:`uniform_traction_load_quadratic`
    for the consistent load on the parent faces.
    """
    traction = np.asarray(traction, dtype=float)
    areas = triangle_areas(nodes, faces)
    f = zero_load(nodes.shape[0])
    for k in range(3):
        contrib = np.outer(areas / 3.0, traction)  # (F, 3)
        np.add.at(f.reshape(-1, 3), faces[:, k], contrib)
    return f


def total_force_on_faces(nodes: np.ndarray, faces: np.ndarray, total_force) -> np.ndarray:
    """Uniform traction over ``faces`` whose resultant equals ``total_force`` [N]."""
    A = triangle_areas(nodes, faces).sum()
    return uniform_traction_load(nodes, faces, np.asarray(total_force, float) / A)


def faces_on_plane(nodes: np.ndarray, faces: np.ndarray, axis: int, value: float, tol: float = 1e-9) -> np.ndarray:
    """Surface faces (any width: display triangles, quad8, tri6) whose nodes all lie on ``x[axis] = value``."""
    on = np.abs(nodes[:, axis] - value) <= tol
    return faces[np.all(on[faces], axis=1)]


# ------------------------------------------------------------------ quadratic parent faces
# ∫ N_i dA / A for uniform traction on straight-sided quadratic faces
_QUAD8_SHARES = np.array([-1.0 / 12.0] * 4 + [1.0 / 3.0] * 4)
_TRI6_SHARES = np.array([0.0] * 3 + [1.0 / 3.0] * 3)


def _parent_face_areas(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    k = faces.shape[1]
    if k == 8:  # two corner triangles
        return triangle_areas(nodes, faces[:, [0, 1, 2]]) + triangle_areas(nodes, faces[:, [0, 2, 3]])
    if k == 6:
        return triangle_areas(nodes, faces[:, :3])
    raise ValueError("parent faces must be quad8 (8 cols) or tri6 (6 cols)")


def uniform_traction_load_quadratic(nodes: np.ndarray, faces: np.ndarray, traction) -> np.ndarray:
    """Consistent nodal loads ``f_i = t ∫ N_i dA`` of a uniform traction on quad8 / tri6 faces
    (exact for planar faces with mid-side nodes at the edge midpoints)."""
    traction = np.asarray(traction, dtype=float)
    f = zero_load(nodes.shape[0])
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return f
    shares = _QUAD8_SHARES if faces.shape[1] == 8 else _TRI6_SHARES
    A = _parent_face_areas(nodes, faces)
    fv = f.reshape(-1, 3)
    for k in range(faces.shape[1]):
        np.add.at(fv, faces[:, k], np.outer(A * shares[k], traction))
    return f


def parent_faces_on_plane(mesh, axis: int, value: float, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Boundary quad8 and tri6 parent faces of ``mesh`` lying on ``x[axis] = value``."""
    return (
        faces_on_plane(mesh.nodes, mesh.surface_quads, axis, value, tol),
        faces_on_plane(mesh.nodes, mesh.surface_tris, axis, value, tol),
    )


def total_force_on_plane(mesh, axis: int, value: float, total_force, tol: float = 1e-9) -> np.ndarray:
    """Consistent uniform traction on the boundary plane ``x[axis] = value`` with resultant ``total_force``."""
    quads, tris = parent_faces_on_plane(mesh, axis, value, tol)
    A = float(sum(_parent_face_areas(mesh.nodes, fc).sum() for fc in (quads, tris) if fc.size))
    if A <= 0.0:
        raise ValueError("no boundary faces on the requested plane")
    t = np.asarray(total_force, float) / A
    return uniform_traction_load_quadratic(mesh.nodes, quads, t) + uniform_traction_load_quadratic(mesh.nodes, tris, t)


def resultant(f: np.ndarray) -> np.ndarray:
    """Sum of nodal forces (3,)."""
    return f.reshape(-1, 3).sum(axis=0)
