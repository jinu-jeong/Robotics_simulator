"""Virtual surface markers (keypoints) riding on the deformed finger surface.

A marker is attached to a boundary triangle with barycentric weights, so its
world position follows the FEM displacement exactly:

    p_k(u) = Σ_i w_ki (x_i + u_i),   i ∈ vertices of face f_k

Markers are the simplest controlled "vision feature": the synthetic camera
observes their 2D projections ``(u, v)`` (plus a visibility flag), which is
what an early vision model regresses before moving to raw images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contact.contact_mapping import PointContactMapping
from ..geometry.finger import FingerGeometry, TetMesh
from ..geometry.mesh_utils import triangle_normals


def face_ids_on_plane(mesh: TetMesh, axis: int, value: float, tol: float = 1e-9) -> np.ndarray:
    """Boundary triangles whose three vertices lie on ``x[axis] == value``."""
    tri = mesh.nodes[mesh.surface_faces]
    return np.nonzero(np.all(np.abs(tri[:, :, axis] - value) < tol, axis=1))[0]


def finger_face_ids(mesh: TetMesh, geometry: FingerGeometry, face: str) -> np.ndarray:
    planes = {
        "top": (2, geometry.height), "bottom": (2, 0.0),
        "side_pos_y": (1, geometry.width), "side_neg_y": (1, 0.0),
        "tip": (0, geometry.length),
    }
    if face not in planes:
        raise KeyError(f"unknown face {face!r}; choose from {sorted(planes)}")
    return face_ids_on_plane(mesh, *planes[face])


@dataclass
class SurfaceMarkers:
    face_ids: np.ndarray  # (M,) indices into mesh.surface_faces
    barycentric: np.ndarray  # (M, 3)
    name: str = "markers"

    def __post_init__(self) -> None:
        self.face_ids = np.asarray(self.face_ids, dtype=np.int64).reshape(-1)
        self.barycentric = np.asarray(self.barycentric, dtype=float).reshape(-1, 3)
        if len(self.face_ids) != len(self.barycentric):
            raise ValueError("face_ids and barycentric must have the same length")

    def __len__(self) -> int:
        return int(len(self.face_ids))

    def positions(self, mesh: TetMesh, u: np.ndarray | None = None) -> np.ndarray:
        """(M, 3) world positions on the (deformed) surface."""
        nodes = mesh.nodes if u is None else mesh.nodes + np.asarray(u, float).reshape(-1, 3)
        tri = nodes[mesh.surface_faces[self.face_ids]]  # (M, 3, 3)
        return np.einsum("mi,mij->mj", self.barycentric, tri)

    def normals(self, mesh: TetMesh, u: np.ndarray | None = None) -> np.ndarray:
        """(M, 3) outward normals of the carrying triangles in the (deformed) configuration."""
        nodes = mesh.nodes if u is None else mesh.nodes + np.asarray(u, float).reshape(-1, 3)
        return triangle_normals(nodes, mesh.surface_faces[self.face_ids])

    def visibility(self, camera, mesh: TetMesh, u: np.ndarray | None = None, margin_px: float = 0.0) -> np.ndarray:
        """Boolean (M,) mask: marker faces the camera and projects inside the image.

        Self-occlusion by other parts of the finger is not modelled; for the
        convex box-like finger the facing test is sufficient.
        """
        p = self.positions(mesh, u)
        n = self.normals(mesh, u)
        facing = np.einsum("mj,mj->m", n, camera.position[None, :] - p) > 0.0
        uv, z = camera.project(p)
        inside = (z > 0) & (uv[:, 0] >= margin_px) & (uv[:, 0] <= camera.image_width - margin_px) \
            & (uv[:, 1] >= margin_px) & (uv[:, 1] <= camera.image_height - margin_px)
        return facing & inside

    def to_dict(self) -> dict:
        return {"face_ids": self.face_ids.tolist(), "barycentric": self.barycentric.tolist(), "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "SurfaceMarkers":
        return cls(np.asarray(d["face_ids"]), np.asarray(d["barycentric"]), str(d.get("name", "markers")))


def make_marker_grid(mesh: TetMesh, geometry: FingerGeometry, face: str = "top", nx: int = 10, ny: int = 3,
                     margin_rel: float = 0.05, name: str | None = None) -> SurfaceMarkers:
    """Regular ``nx × ny`` marker grid on a finger face, inset by ``margin_rel`` of the face size."""
    g = geometry
    ids = finger_face_ids(mesh, geometry, face)
    mapping = PointContactMapping(mesh, ids)
    if face in ("top", "bottom"):
        a_len, b_len, z = g.length, g.width, (g.height if face == "top" else 0.0)
        mk = lambda a, b: (a, b, z)  # noqa: E731
    elif face in ("side_pos_y", "side_neg_y"):
        a_len, b_len, y = g.length, g.height, (g.width if face == "side_pos_y" else 0.0)
        mk = lambda a, b: (a, y, b)  # noqa: E731
    else:  # tip
        a_len, b_len, x = g.width, g.height, g.length
        mk = lambda a, b: (x, a, b)  # noqa: E731
    a_vals = np.linspace(margin_rel, 1.0 - margin_rel, nx) * a_len
    b_vals = np.linspace(margin_rel, 1.0 - margin_rel, ny) * b_len
    face_ids, bary = [], []
    for a in a_vals:
        for b in b_vals:
            c = mapping.locate(np.array(mk(a, b)))
            face_ids.append(c.face_index)
            bary.append(c.barycentric)
    return SurfaceMarkers(np.array(face_ids), np.array(bary), name or f"{face}_{nx}x{ny}")
