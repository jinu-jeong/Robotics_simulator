"""Graph partitioning + fill-reducing ordering for FEM / CB workflows.

Two distinct uses of METIS in a FEM codebase, both wrapped here:

1. **K-way mesh partitioning** — split element/node sets into ``K`` roughly
   balanced regions with small interface size. Used for:
   * Hybrid full-order/reduced-order plasticity (each region either
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from robosim.physics.fem.mesh import FEMesh


@dataclass
class RegionPartition:
    """K-way mesh partition with element + node membership bookkeeping.

    Built once at body-construction time from
    :func:`partition_mesh_elements`; consumed by the hybrid CB-plastic
    body for per-region state-machine routing (ELASTIC ⇄ PLASTIC_ACTIVE).

    Conventions
    -----------
    * ``element_region[e] = r``  — element ``e`` belongs to region ``r``.
    * ``region_elements[r]``     — int64 array of element indices in region ``r``.
    * ``region_nodes[r]``        — sorted int64 array of *all* nodes touched by
      any element in region ``r`` (includes shared boundary nodes).
    * ``region_interior_nodes[r]`` — nodes in ``region_nodes[r]`` *not*
      appearing in any other region's element. Safe to be reduced
      (CB interior DOFs) — when this region is ELASTIC, these nodes can
      be expressed via modal coordinates.
    * ``inter_region_boundary_nodes`` — sorted int64 array of nodes shared
      by ≥ 2 regions. These are the master DOFs that *always* stay
      full-order; the hybrid solver couples regions through them.
    """

    n_regions: int
    element_region: np.ndarray             # (n_elements,) int64
    region_elements: list[np.ndarray]      # [r] -> element indices
    region_nodes: list[np.ndarray]         # [r] -> all nodes touched
    region_interior_nodes: list[np.ndarray]  # [r] -> nodes ONLY in region r
    inter_region_boundary_nodes: np.ndarray  # nodes in ≥ 2 regions


def build_region_partition(mesh: "FEMesh", n_regions: int) -> RegionPartition:
    """METIS K-way partition + per-region node bookkeeping.

    K=1 short-circuit returns a single region with all elements/nodes
    and an empty boundary set.
    """
    if n_regions <= 0:
        raise ValueError(f"n_regions must be ≥ 1, got {n_regions}")

    elements = np.asarray(mesh.elements, dtype=np.int64)
    n_elements = elements.shape[0]
    n_nodes = int(mesh.n_nodes)

    if n_regions == 1:
        all_elem = np.arange(n_elements, dtype=np.int64)
        all_node = np.arange(n_nodes, dtype=np.int64)
        return RegionPartition(
            n_regions=1,
            element_region=np.zeros(n_elements, dtype=np.int64),
            region_elements=[all_elem],
            region_nodes=[all_node],
            region_interior_nodes=[all_node],
            inter_region_boundary_nodes=np.array([], dtype=np.int64),
        )

    elem_region = partition_mesh_elements(mesh, n_regions)

    # Per-region element index lists.
    region_elements = [
        np.where(elem_region == r)[0].astype(np.int64)
        for r in range(n_regions)
    ]

    # Per-region node sets: union of all element-node memberships.
    # Also count, per node, how many distinct regions contain it.
    region_nodes: list[np.ndarray] = []
    region_membership_count = np.zeros(n_nodes, dtype=np.int64)
    region_owner_of_node = -np.ones(n_nodes, dtype=np.int64)
    region_node_seen = np.zeros((n_regions, n_nodes), dtype=bool)
    for r in range(n_regions):
        nodes_r = np.unique(elements[region_elements[r]].ravel())
        region_nodes.append(nodes_r.astype(np.int64))
        region_node_seen[r, nodes_r] = True
    region_membership_count = region_node_seen.sum(axis=0).astype(np.int64)

    # Interior nodes per region: appear in ONLY one region.
    region_interior_nodes: list[np.ndarray] = []
    for r in range(n_regions):
        in_r = region_node_seen[r]
        interior_mask = in_r & (region_membership_count == 1)
        region_interior_nodes.append(
            np.where(interior_mask)[0].astype(np.int64)
        )

    # Inter-region boundary: nodes in ≥ 2 regions.
    inter_region_boundary = np.where(region_membership_count >= 2)[0].astype(np.int64)

    return RegionPartition(
        n_regions=n_regions,
        element_region=elem_region,
        region_elements=region_elements,
        region_nodes=region_nodes,
        region_interior_nodes=region_interior_nodes,
        inter_region_boundary_nodes=inter_region_boundary,
    )


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
