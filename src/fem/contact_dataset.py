"""Contact-sweep dataset: FEM ground-truth deformations for many single contacts.

File format: a single compressed ``.npz`` containing the mesh, all samples and
a JSON metadata blob, so the dataset is self-describing and can be inspected
in the Taichi viewer without rebuilding the FEM.

Arrays (S samples, N nodes)
---------------------------
nodes (N,3) f64 | tets (Mt,10) i64 | hexes (Mh,20) i64 | surface_faces (F,3) i64
surface_edges (Es,2) i64 | element_edges (Ee,2) i64 | fixed_nodes (K,) i64
U (S,N,3)           nodal displacement [m] (float32 or float64)
contact_position (S,3), contact_normal (S,3)  [m] / outward unit normal
contact_face (S,) i64, contact_bary (S,3)     surface triangle + barycentric weights
force_magnitude (S,), force_vector (S,3)      [N], force acting on the finger
max_displacement (S,)                          [m]
meta (json string)  geometry, material, mesh resolution, config snapshot, ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ContactDataset:
    nodes: np.ndarray
    tets: np.ndarray
    surface_faces: np.ndarray
    fixed_nodes: np.ndarray
    U: np.ndarray
    contact_position: np.ndarray
    contact_normal: np.ndarray
    contact_face: np.ndarray
    contact_bary: np.ndarray
    force_magnitude: np.ndarray
    force_vector: np.ndarray
    max_displacement: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    hexes: np.ndarray = field(default_factory=lambda: np.zeros((0, 20), np.int64))
    surface_edges: np.ndarray | None = None
    element_edges: np.ndarray | None = None

    # ------------------------------------------------------------ basics
    def __len__(self) -> int:
        return int(self.U.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_dofs(self) -> int:
        return 3 * self.n_nodes

    @property
    def n_elements(self) -> int:
        return int(self.tets.shape[0] + self.hexes.shape[0])

    def displacement(self, i: int) -> np.ndarray:
        return np.asarray(self.U[i], dtype=np.float64)

    def snapshot_matrix(self, indices=None) -> np.ndarray:
        """``(3N, S)`` matrix of flattened displacements (for POD in Milestone 4)."""
        idx = np.arange(len(self)) if indices is None else np.asarray(indices)
        return np.asarray(self.U[idx], dtype=np.float64).reshape(len(idx), -1).T

    def subset(self, indices) -> "ContactDataset":
        idx = np.asarray(indices)
        return ContactDataset(
            self.nodes, self.tets, self.surface_faces, self.fixed_nodes,
            self.U[idx], self.contact_position[idx], self.contact_normal[idx], self.contact_face[idx],
            self.contact_bary[idx], self.force_magnitude[idx], self.force_vector[idx], self.max_displacement[idx],
            dict(self.meta), hexes=self.hexes, surface_edges=self.surface_edges, element_edges=self.element_edges,
        )

    # ------------------------------------------------------------ io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.surface_edges is not None:
            extra["surface_edges"] = self.surface_edges
        if self.element_edges is not None:
            extra["element_edges"] = self.element_edges
        np.savez_compressed(
            path,
            nodes=self.nodes, tets=self.tets, hexes=self.hexes, surface_faces=self.surface_faces, fixed_nodes=self.fixed_nodes,
            U=self.U, contact_position=self.contact_position, contact_normal=self.contact_normal,
            contact_face=self.contact_face, contact_bary=self.contact_bary,
            force_magnitude=self.force_magnitude, force_vector=self.force_vector,
            max_displacement=self.max_displacement,
            meta=np.array(json.dumps(self.meta)),
            **extra,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ContactDataset":
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return cls(
                nodes=z["nodes"], tets=z["tets"], surface_faces=z["surface_faces"], fixed_nodes=z["fixed_nodes"],
                U=z["U"], contact_position=z["contact_position"], contact_normal=z["contact_normal"],
                contact_face=z["contact_face"], contact_bary=z["contact_bary"],
                force_magnitude=z["force_magnitude"], force_vector=z["force_vector"],
                max_displacement=z["max_displacement"], meta=meta,
                hexes=z["hexes"] if "hexes" in z else np.zeros((0, 20), np.int64),
                surface_edges=z["surface_edges"] if "surface_edges" in z else None,
                element_edges=z["element_edges"] if "element_edges" in z else None,
            )

    # ------------------------------------------------------------ summary
    def summary(self) -> dict[str, Any]:
        return {
            "n_samples": len(self),
            "n_nodes": self.n_nodes,
            "n_tets": int(self.tets.shape[0]),
            "n_hexes": int(self.hexes.shape[0]),
            "force_range_N": [float(self.force_magnitude.min()), float(self.force_magnitude.max())],
            "n_unique_contacts": int(len(np.unique(np.round(self.contact_position, 9), axis=0))),
            "max_displacement_range_mm": [1e3 * float(self.max_displacement.min()), 1e3 * float(self.max_displacement.max())],
            "storage_dtype": str(self.U.dtype),
            "size_MB": self.U.nbytes / 1e6,
        }

    # ------------------------------------------------------------ viewer
    def to_visualization_state(self, i: int, estimated_force=None, with_object: bool = True, extra_info: dict | None = None):
        """Package sample ``i`` for the Taichi viewer."""
        from ..geometry.mesh_utils import unique_edges
        from ..visualization.state import ContactPoint, ForceVector, ObjectGeometry, VisualizationState

        i = int(i) % len(self)
        pos, n = self.contact_position[i], self.contact_normal[i]
        fv = self.force_vector[i]
        objects = []
        if with_object:
            r = 0.4 * float(self.meta.get("geometry", {}).get("height", 0.01))
            objects.append(ObjectGeometry.sphere(pos + r * n, r, name="probe", attach_to=pos))
        est = None if estimated_force is None else ForceVector(pos, estimated_force, label="F_est")
        face_nodes = self.surface_faces[self.contact_face[i]]
        nearest = int(face_nodes[int(np.argmax(self.contact_bary[i]))])
        info = {
            "sample": f"{i} / {len(self) - 1}",
            "force magnitude": f"{self.force_magnitude[i]:.4f} N",
            "contact (x, y, z)": f"({1e3 * pos[0]:.2f}, {1e3 * pos[1]:.2f}, {1e3 * pos[2]:.2f}) mm",
            "max |u|": f"{1e3 * self.max_displacement[i]:.4f} mm",
        }
        info.update(extra_info or {})
        ee = self.element_edges
        if ee is None:
            parts = [unique_edges(c) for c in (self.tets, self.hexes) if len(c)]
            ee = np.unique(np.concatenate(parts), axis=0) if parts else None
        return VisualizationState(
            nodes_original=self.nodes,
            surface_faces=self.surface_faces,
            element_edges=ee,
            surface_edges=self.surface_edges,
            displacement=self.displacement(i),
            fixed_nodes=self.fixed_nodes,
            contact_points=[ContactPoint(pos, node_index=nearest, normal=n)],
            ground_truth_force=ForceVector(pos, fv, label="F_true"),
            estimated_force=est,
            objects=objects,
            info=info,
        )
