"""Observation operator ``y = H u``: which part of the displacement field is measured.

A camera never sees every FEM node. All inverse solvers therefore work on the
observed vector ``y = H u`` rather than on ``u`` itself. For now ``H`` is a
*selection* of DOFs (rows of the identity); later milestones add marker
projections and image-based observations with the same interface.

    full ......... every free DOF (Milestone 3 baseline: "exact u")
    surface ...... all DOFs of boundary nodes (what a depth camera could see)
    named face ... DOFs of nodes on one finger face, e.g. "top" or "side_pos_y"
    components ... restrict to a subset of (x, y, z), e.g. only out-of-plane z

Fixed (clamped) DOFs are always excluded – they carry no information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fem.boundary import DofPartition
from ..geometry.finger import FingerGeometry, TetMesh
from ..geometry.mesh_utils import surface_node_indices


@dataclass(frozen=True)
class DofSelection:
    """``H`` as an index array into the flat ``(3N,)`` displacement vector."""

    dofs: np.ndarray
    n_dof: int
    name: str = "selection"

    def __post_init__(self) -> None:
        d = np.unique(np.asarray(self.dofs, dtype=np.int64).reshape(-1))
        if d.size == 0:
            raise ValueError("observation selects no DOF")
        if d.min() < 0 or d.max() >= self.n_dof:
            raise ValueError("observed dof index out of range")
        object.__setattr__(self, "dofs", d)

    @property
    def n_obs(self) -> int:
        return int(len(self.dofs))

    @property
    def nodes(self) -> np.ndarray:
        return np.unique(self.dofs // 3)

    def apply(self, u: np.ndarray) -> np.ndarray:
        """``y = H u`` for a flat (3N,) or (N,3) displacement, or (3N,k) matrix."""
        u = np.asarray(u, dtype=float)
        if u.ndim == 2 and u.shape == (self.n_dof // 3, 3):
            u = u.reshape(-1)
        return u[self.dofs]

    def restrict_components(self, components) -> "DofSelection":
        """Keep only the given displacement components (subset of 0,1,2)."""
        comps = np.asarray(components, dtype=np.int64).reshape(-1)
        keep = np.isin(self.dofs % 3, comps)
        return DofSelection(self.dofs[keep], self.n_dof, f"{self.name}[{''.join('xyz'[c] for c in comps)}]")

    def subsample(self, fraction: float, rng: np.random.Generator) -> "DofSelection":
        """Random subset of the observed *nodes* (all their components kept)."""
        nodes = self.nodes
        k = max(1, int(round(fraction * len(nodes))))
        pick = rng.choice(nodes, size=k, replace=False)
        dofs = self.dofs[np.isin(self.dofs // 3, pick)]
        return DofSelection(dofs, self.n_dof, f"{self.name}~{fraction:g}")


def _node_dofs(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.int64).reshape(-1, 1)
    return (3 * nodes + np.arange(3)[None, :]).reshape(-1)


def full_observation(partition: DofPartition) -> DofSelection:
    """Every free DOF."""
    return DofSelection(partition.free_dofs, partition.n_dof, "full")


def surface_observation(mesh: TetMesh, partition: DofPartition) -> DofSelection:
    """All DOFs of boundary nodes (minus clamped ones)."""
    dofs = _node_dofs(surface_node_indices(mesh.surface_faces))
    return DofSelection(np.intersect1d(dofs, partition.free_dofs), partition.n_dof, "surface")


def face_observation(mesh: TetMesh, partition: DofPartition, geometry: FingerGeometry, face: str, tol: float = 1e-9) -> DofSelection:
    """DOFs of nodes lying on one finger face: top | bottom | side_pos_y | side_neg_y | tip."""
    planes = {
        "top": (2, geometry.height), "bottom": (2, 0.0),
        "side_pos_y": (1, geometry.width), "side_neg_y": (1, 0.0),
        "tip": (0, geometry.length),
    }
    if face not in planes:
        raise KeyError(f"unknown face {face!r}; choose from {sorted(planes)}")
    axis, val = planes[face]
    nodes = np.nonzero(np.abs(mesh.nodes[:, axis] - val) < tol)[0]
    dofs = np.intersect1d(_node_dofs(nodes), partition.free_dofs)
    return DofSelection(dofs, partition.n_dof, face)


def observation_from_name(name: str, mesh: TetMesh, partition: DofPartition, geometry: FingerGeometry | None = None) -> DofSelection:
    """Parse ``"full"``, ``"surface"``, ``"top"``, ``"surface:z"``, ``"top:xz"`` ..."""
    base, _, comps = name.partition(":")
    if base == "full":
        obs = full_observation(partition)
    elif base == "surface":
        obs = surface_observation(mesh, partition)
    else:
        if geometry is None:
            raise ValueError("face observations need the finger geometry")
        obs = face_observation(mesh, partition, geometry, base)
    if comps:
        obs = obs.restrict_components(["xyz".index(c) for c in comps])
    return obs
