"""Boundary-condition helpers: node/DOF selection and free/constrained partition.

Conventions
-----------
* DOF index ``= 3 * node + component`` (component 0,1,2 = x,y,z).
* A *fixed* (Dirichlet) DOF has a prescribed displacement (default 0).
* ``DofPartition`` stores the free set ``f`` and constrained set ``c`` so
  the solver can form

      [K_ff K_fc] [u_f]   [f_f]
      [K_cf K_cc] [u_c] = [f_c]      ->  K_ff u_f = f_f - K_fc u_c.

The clamped finger root (all components of all nodes on ``x = 0``) is the
standard case, but component-wise constraints (symmetry planes, rollers)
are supported for verification problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def node_dofs(nodes_idx, components=(0, 1, 2)) -> np.ndarray:
    """Global DOF indices for the given nodes and components (sorted, unique)."""
    nodes_idx = np.asarray(nodes_idx, dtype=np.int64).reshape(-1)
    comps = np.asarray(components, dtype=np.int64).reshape(-1)
    dofs = (3 * nodes_idx[:, None] + comps[None, :]).reshape(-1)
    return np.unique(dofs)


def dofs_to_nodes(dofs) -> np.ndarray:
    return np.unique(np.asarray(dofs, dtype=np.int64) // 3)


@dataclass
class DofPartition:
    """Free/constrained DOF bookkeeping for ``n_dof`` degrees of freedom."""

    n_dof: int
    fixed_dofs: np.ndarray
    prescribed: np.ndarray = field(default=None)  # values for fixed dofs (same length)

    def __post_init__(self) -> None:
        raw = np.asarray(self.fixed_dofs, dtype=np.int64).reshape(-1)
        if raw.size and (raw.min() < 0 or raw.max() >= self.n_dof):
            raise ValueError("fixed dof index out of range")
        if self.prescribed is None:
            vals = np.zeros(len(raw))
        else:
            vals = np.asarray(self.prescribed, dtype=float).reshape(-1)
            if vals.shape != raw.shape:
                raise ValueError("prescribed values must match fixed_dofs")
        # Sort dofs and carry the prescribed values along; duplicates must agree.
        uniq, first = np.unique(raw, return_index=True)
        for d, v in zip(raw, vals):
            if not np.isclose(vals[first[np.searchsorted(uniq, d)]], v):
                raise ValueError(f"conflicting prescribed values for dof {d}")
        self.fixed_dofs = uniq
        self.prescribed = vals[first]
        mask = np.ones(self.n_dof, dtype=bool)
        mask[self.fixed_dofs] = False
        self.free_dofs = np.nonzero(mask)[0]

    @property
    def n_free(self) -> int:
        return int(len(self.free_dofs))

    @property
    def n_fixed(self) -> int:
        return int(len(self.fixed_dofs))

    @property
    def fixed_nodes(self) -> np.ndarray:
        return dofs_to_nodes(self.fixed_dofs)

    def fully_fixed_nodes(self) -> np.ndarray:
        """Nodes whose x, y and z DOFs are all constrained."""
        s = set(self.fixed_dofs.tolist())
        return np.array([n for n in self.fixed_nodes if all(3 * n + c in s for c in range(3))], dtype=np.int64)

    def full_vector(self, u_free: np.ndarray) -> np.ndarray:
        """Scatter free values and prescribed values into a full (n_dof,) vector."""
        u = np.zeros(self.n_dof)
        u[self.free_dofs] = u_free
        u[self.fixed_dofs] = self.prescribed
        return u


def clamp_nodes(n_nodes: int, nodes_idx) -> DofPartition:
    """Fully clamp the given nodes (u_x = u_y = u_z = 0)."""
    return DofPartition(3 * n_nodes, node_dofs(nodes_idx))


def clamp_plane(nodes: np.ndarray, axis: int, value: float, components=(0, 1, 2), tol: float = 1e-9) -> DofPartition:
    """Constrain ``components`` of all nodes lying on the plane ``x[axis] = value``."""
    idx = np.nonzero(np.abs(nodes[:, axis] - value) <= tol)[0]
    if idx.size == 0:
        raise ValueError("no nodes found on the requested plane")
    return DofPartition(3 * nodes.shape[0], node_dofs(idx, components))


def combine(*parts: DofPartition) -> DofPartition:
    """Union of several partitions (all with zero prescribed values)."""
    n_dof = parts[0].n_dof
    if any(p.n_dof != n_dof for p in parts):
        raise ValueError("partitions refer to different systems")
    if any(np.any(p.prescribed != 0) for p in parts):
        raise NotImplementedError("combine() only supports homogeneous constraints")
    return DofPartition(n_dof, np.concatenate([p.fixed_dofs for p in parts]))
