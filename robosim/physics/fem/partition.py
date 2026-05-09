"""Graph partitioning + fill-reducing ordering for FEM / CB workflows.

Two distinct uses of METIS in a FEM codebase, both wrapped here:

1. **K-way mesh partitioning** — split element/node sets into ``K`` roughly
   balanced regions with small interface size. Used for:
   * Future hybrid full-order/reduced-order plasticity (each region either
     CB-reduced or full-FEM depending on local stress state).
   * Domain decomposition for parallel solvers.

2. **Nested-dissection (ND) ordering** — fill-reducing permutation for sparse
   Cholesky/LU factorisations. Used for:
   * Craig-Bampton constraint-mode solve ``K_ii⁻¹ K_ib``.
   * ``eigsh`` shift-invert factorisations.

Both rely on ``pymetis`` (METIS 5 bindings). Import is lazy — callers that
don't touch these functions don't pay the import cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from robosim.physics.fem.mesh import FEMesh


# ── Adjacency builders ─────────────────────────────────────────────────────

def _node_adjacency_from_elements(
    n_nodes: int, elements: np.ndarray
) -> list[list[int]]:
    """Node-node adjacency: two nodes share an edge if they sit in any
    common element. Output is a per-node sorted list of distinct neighbours
    (no self-loop)."""
    adj_set: list[set[int]] = [set() for _ in range(n_nodes)]
    for elem in elements:
        for a in elem:
            adj_set[int(a)].update(int(b) for b in elem if b != a)
    return [sorted(s) for s in adj_set]


def _element_adjacency_from_mesh(elements: np.ndarray) -> list[list[int]]:
    """Element-element adjacency: two elements are adjacent if they share
    at least one node (for hex/tet meshes this implies sharing a face,
    edge, or vertex; sufficient for partitioning purposes)."""
    n_nodes = int(elements.max()) + 1 if elements.size else 0
    node_to_elems: list[list[int]] = [[] for _ in range(n_nodes)]
    for ei, elem in enumerate(elements):
        for n in elem:
            node_to_elems[int(n)].append(ei)

    n_elems = elements.shape[0]
    adj_set: list[set[int]] = [set() for _ in range(n_elems)]
    for n in range(n_nodes):
        sharing = node_to_elems[n]
        for a in sharing:
            adj_set[a].update(b for b in sharing if b != a)
    return [sorted(s) for s in adj_set]


# ── K-way partition ────────────────────────────────────────────────────────

def partition_mesh_nodes(mesh: "FEMesh", n_parts: int) -> np.ndarray:
    """Assign each node to one of ``n_parts`` groups via METIS K-way.

    Returns a length-``n_nodes`` ``int64`` array of part labels in ``[0, K)``.
    Aims to (a) balance group sizes and (b) minimise edge cuts so adjacent
    nodes tend to land in the same group — useful for any
    locality-sensitive use case (RBD-FEM contact zones, plastic regions).
    """
    if n_parts <= 0:
        raise ValueError(f"n_parts must be ≥ 1, got {n_parts}")
    if n_parts == 1:
        return np.zeros(mesh.n_nodes, dtype=np.int64)
    import pymetis
    adj = _node_adjacency_from_elements(mesh.n_nodes, np.asarray(mesh.elements))
    _ncuts, labels = pymetis.part_graph(n_parts, adjacency=adj)
    return np.asarray(labels, dtype=np.int64)


def partition_mesh_elements(mesh: "FEMesh", n_parts: int) -> np.ndarray:
    """Assign each element to one of ``n_parts`` groups via METIS K-way.

    Returns a length-``n_elements`` ``int64`` array. Use for plastic-zone
    region splitting where the unit of switching is an element."""
    if n_parts <= 0:
        raise ValueError(f"n_parts must be ≥ 1, got {n_parts}")
    if n_parts == 1:
        return np.zeros(mesh.n_elements, dtype=np.int64)
    import pymetis
    adj = _element_adjacency_from_mesh(np.asarray(mesh.elements))
    _ncuts, labels = pymetis.part_graph(n_parts, adjacency=adj)
    return np.asarray(labels, dtype=np.int64)


# ── Nested-dissection ordering for sparse factorisation ────────────────────

def nested_dissection_order(A: sp.spmatrix) -> np.ndarray:
    """Return a fill-reducing permutation ``P`` of ``range(A.shape[0])``.

    For symmetric positive-definite ``A`` (e.g. interior stiffness ``K_ii``),
    factorising ``A[P,:][:,P]`` instead of ``A`` typically reduces fill from
    O(N^{5/3}) to O(N · log N) on 3-D meshes, which is the difference
    between unusable and routine for N > a few thousand.

    Apply as::

        P = nested_dissection_order(K_ii)
        K_perm = K_ii[P, :][:, P]
        b_perm = b[P]
        x_perm = solve(K_perm, b_perm)
        x = np.empty_like(x_perm); x[P] = x_perm
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}")
    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)
    import pymetis
    A_csr = A.tocsr()
    indptr = A_csr.indptr
    indices = A_csr.indices
    adj = [
        [int(j) for j in indices[indptr[i]:indptr[i + 1]] if j != i]
        for i in range(n)
    ]
    perm, _iperm = pymetis.nested_dissection(adj)
    return np.asarray(perm, dtype=np.int64)


def fill_count_after_lu(A: sp.spmatrix, perm: np.ndarray | None = None) -> int:
    """Count nonzeros in ``L+U`` of the LU factorisation of a (possibly
    permuted) sparse matrix. Used by tests / benchmarks to verify that an
    ND permutation actually reduces fill."""
    from scipy.sparse.linalg import splu
    A_csc = A.tocsc()
    if perm is not None:
        P = np.asarray(perm)
        A_csc = A_csc[P, :][:, P].tocsc()
    lu = splu(A_csc, permc_spec="NATURAL")  # bypass scipy's reorder
    return int(lu.L.nnz + lu.U.nnz)
