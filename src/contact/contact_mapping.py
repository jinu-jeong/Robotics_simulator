"""Contact load operator ``B(c)``: surface contact -> FEM nodal force vector.

For a point contact at surface location ``c`` with unit force direction
``d`` (a vector acting *on the finger*, e.g. the inward normal ``-n`` for a
pressing contact) the nodal load is

    f_contact = B(c) λ,        B(c) ∈ R^{3N x 1},  λ ≥ 0 [N]

where ``B(c)`` distributes a unit force to the three vertices of the surface
triangle containing ``c`` with barycentric weights (consistent point load on
a linear triangle). If ``c`` is exactly at a node this reduces to a nodal
point load. Later milestones generalise this to patches and pressure fields
(``B`` with several columns).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..geometry.finger import TetMesh
from ..geometry.mesh_utils import triangle_normals


def _closest_point_on_triangles(p: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest points of ``p`` on triangles (F,3,3) and barycentric weights (F,3).

    Vectorised version of the Ericson (Real-Time Collision Detection) algorithm.
    """
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    F = len(tri)
    w = np.zeros((F, 3))
    done = np.zeros(F, dtype=bool)

    # vertex a
    m = (d1 <= 0) & (d2 <= 0)
    w[m] = [1, 0, 0]
    done |= m
    # vertex b
    m = ~done & (d3 >= 0) & (d4 <= d3)
    w[m] = [0, 1, 0]
    done |= m
    # edge ab
    vc = d1 * d4 - d3 * d2
    m = ~done & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v = np.divide(d1, d1 - d3, out=np.zeros(F), where=(d1 - d3) != 0)
    w[m, 0], w[m, 1] = 1 - v[m], v[m]
    done |= m
    # vertex c
    m = ~done & (d6 >= 0) & (d5 <= d6)
    w[m] = [0, 0, 1]
    done |= m
    # edge ac
    vb = d5 * d2 - d1 * d6
    m = ~done & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    wv = np.divide(d2, d2 - d6, out=np.zeros(F), where=(d2 - d6) != 0)
    w[m, 0], w[m, 2] = 1 - wv[m], wv[m]
    done |= m
    # edge bc
    va = d3 * d6 - d5 * d4
    m = ~done & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    wv = np.divide(d4 - d3, (d4 - d3) + (d5 - d6), out=np.zeros(F), where=((d4 - d3) + (d5 - d6)) != 0)
    w[m, 1], w[m, 2] = 1 - wv[m], wv[m]
    done |= m
    # interior
    m = ~done
    denom = va + vb + vc
    v = np.divide(vb, denom, out=np.zeros(F), where=denom != 0)
    ww = np.divide(vc, denom, out=np.zeros(F), where=denom != 0)
    w[m, 0], w[m, 1], w[m, 2] = 1 - v[m] - ww[m], v[m], ww[m]

    q = w[:, 0:1] * a + w[:, 1:2] * b + w[:, 2:3] * c
    return q, w


@dataclass
class SurfaceContact:
    """Resolved contact location on the finger surface."""

    position: np.ndarray  # closest surface point [m]
    face_index: int  # index into ``mesh.surface_faces`` (global boundary triangle id)
    barycentric: np.ndarray  # (3,) weights on the face's vertices
    normal: np.ndarray  # outward unit normal of the face
    node_indices: np.ndarray  # (3,) vertices of the face
    distance: float  # |query - position|

    @property
    def nearest_node(self) -> int:
        return int(self.node_indices[int(np.argmax(self.barycentric))])


def contact_from_face(mesh: TetMesh, face_index: int, barycentric) -> SurfaceContact:
    """Construct a :class:`SurfaceContact` on boundary triangle ``face_index`` of ``mesh``."""
    w = np.asarray(barycentric, dtype=float).reshape(3)
    if np.any(w < -1e-12) or abs(w.sum() - 1.0) > 1e-9:
        raise ValueError("barycentric weights must be non-negative and sum to one")
    face = mesh.surface_faces[int(face_index)]
    tri = mesh.nodes[face]
    pos = w @ tri
    n = triangle_normals(mesh.nodes, face[None, :])[0]
    return SurfaceContact(position=pos, face_index=int(face_index), barycentric=w, normal=n,
                          node_indices=np.asarray(face, dtype=np.int64), distance=0.0)


class PointContactMapping:
    """Builds ``B(c)`` for single point contacts on a fixed mesh surface."""

    def __init__(self, mesh: TetMesh, candidate_face_ids: np.ndarray | None = None) -> None:
        """``candidate_face_ids``: indices into ``mesh.surface_faces`` that may carry
        contact (default: all boundary triangles)."""
        self.mesh = mesh
        all_faces = mesh.surface_faces
        self.face_ids = np.arange(len(all_faces)) if candidate_face_ids is None else np.asarray(candidate_face_ids, dtype=np.int64)
        self.faces = all_faces[self.face_ids]
        self.face_normals = triangle_normals(mesh.nodes, self.faces)
        self._tri = mesh.nodes[self.faces]  # (F, 3, 3)

    def locate(self, point) -> SurfaceContact:
        """Project an arbitrary point onto the candidate surface."""
        p = np.asarray(point, dtype=float).reshape(3)
        q, w = _closest_point_on_triangles(p, self._tri)
        d = np.linalg.norm(q - p, axis=1)
        k = int(np.argmin(d))
        return SurfaceContact(
            position=q[k],
            face_index=int(self.face_ids[k]),
            barycentric=w[k],
            normal=self.face_normals[k],
            node_indices=self.faces[k],
            distance=float(d[k]),
        )

    def contact_from_face(self, face_index: int, barycentric) -> SurfaceContact:
        """Rebuild a contact from stored data (global boundary face id + weights),
        e.g. from a :class:`~src.fem.contact_dataset.ContactDataset` sample."""
        return contact_from_face(self.mesh, face_index, barycentric)

    def operator(self, contact: SurfaceContact, direction) -> sp.csc_matrix:
        """Sparse ``B(c)`` (3N x 1) for unit force along ``direction``."""
        d = np.asarray(direction, dtype=float).reshape(3)
        d = d / np.linalg.norm(d)
        return self.vector_operator(contact) @ sp.csc_matrix(d.reshape(3, 1))

    def vector_operator(self, contact: SurfaceContact) -> sp.csc_matrix:
        """Sparse ``B_xyz(c)`` (3N x 3): column ``k`` is a unit force along axis ``k``.

        ``B(c) λ`` for a scalar normal force equals ``B_xyz(c) (λ d)``; the 3-column
        form lets inverse solvers recover a full force *vector*.
        """
        rows, cols, vals = [], [], []
        for node, wgt in zip(contact.node_indices, contact.barycentric):
            if wgt <= 0.0:
                continue
            for comp in range(3):
                rows.append(3 * int(node) + comp)
                cols.append(comp)
                vals.append(wgt)
        return sp.csc_matrix((vals, (rows, cols)), shape=(self.mesh.n_dofs, 3))

    def load_vector(self, point, force_vector) -> tuple[np.ndarray, SurfaceContact]:
        """Flat nodal load (3N,) for a force vector [N] applied at ``point``."""
        fv = np.asarray(force_vector, dtype=float).reshape(3)
        mag = np.linalg.norm(fv)
        contact = self.locate(point)
        if mag == 0.0:
            return np.zeros(self.mesh.n_dofs), contact
        B = self.operator(contact, fv / mag)
        return np.asarray(B @ np.array([mag])).reshape(-1), contact

    def normal_load_vector(self, point, magnitude: float) -> tuple[np.ndarray, SurfaceContact]:
        """Pressing normal contact of ``magnitude`` [N] (force along -n)."""
        contact = self.locate(point)
        B = self.operator(contact, -contact.normal)
        return np.asarray(B @ np.array([float(magnitude)])).reshape(-1), contact
