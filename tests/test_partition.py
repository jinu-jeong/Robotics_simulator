"""Tests for graph partitioning + ND ordering utilities."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


pytest.importorskip("pymetis", reason="pymetis not installed")


from robosim.physics.fem.mesh import FEMesh                              # noqa: E402
from robosim.physics.fem.partition import (                              # noqa: E402
    partition_mesh_nodes,
    partition_mesh_elements,
    nested_dissection_order,
    fill_count_after_lu,
)


# ── Test mesh: 6×6×6 hex box ───────────────────────────────────────────────

@pytest.fixture
def hex_box() -> FEMesh:
    return FEMesh.create_hex_box(
        origin=np.array([0.0, 0.0, 0.0]),
        size=np.array([0.1, 0.1, 0.1]),
        divisions=(6, 6, 6),
    )


def test_partition_nodes_balanced(hex_box: FEMesh):
    """K-way node partition: each part within 1.5× of mean size."""
    K = 4
    labels = partition_mesh_nodes(hex_box, K)
    assert labels.shape == (hex_box.n_nodes,)
    assert labels.min() >= 0 and labels.max() < K
    sizes = np.bincount(labels, minlength=K)
    mean = hex_box.n_nodes / K
    assert sizes.max() < 1.5 * mean, (
        f"largest part {sizes.max()} > 1.5×mean {mean}: {sizes}"
    )


def test_partition_elements_balanced(hex_box: FEMesh):
    """K-way element partition: each part covers ≥ half of the mean."""
    K = 5
    labels = partition_mesh_elements(hex_box, K)
    assert labels.shape == (hex_box.n_elements,)
    assert labels.min() >= 0 and labels.max() < K
    sizes = np.bincount(labels, minlength=K)
    mean = hex_box.n_elements / K
    assert sizes.min() >= 0.5 * mean, (
        f"smallest part {sizes.min()} < 0.5×mean {mean}: {sizes}"
    )


def test_partition_n_parts_one_is_trivial(hex_box: FEMesh):
    """K=1 → everyone in part 0, no METIS call needed."""
    n_labels = partition_mesh_nodes(hex_box, 1)
    e_labels = partition_mesh_elements(hex_box, 1)
    assert (n_labels == 0).all()
    assert (e_labels == 0).all()


# ── ND ordering: must reduce LU fill on a 3-D grid Laplacian ───────────────

def _grid_laplacian_3d(n: int) -> sp.csc_matrix:
    """Discrete 7-point Laplacian on an n×n×n grid (Dirichlet boundaries).
    Standard test matrix for ordering algorithms — 3-D fills aggressively
    under natural ordering, much less under ND."""
    N = n * n * n

    def idx(i, j, k):
        return (i * n + j) * n + k

    rows, cols, data = [], [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                p = idx(i, j, k)
                rows.append(p); cols.append(p); data.append(6.0)
                for di, dj, dk in [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                    (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < n and 0 <= jj < n and 0 <= kk < n:
                        rows.append(p); cols.append(idx(ii, jj, kk)); data.append(-1.0)
    return sp.csc_matrix((data, (rows, cols)), shape=(N, N))


def test_nd_reduces_fill_on_3d_grid():
    """ND ordering on a 6³ grid Laplacian should produce strictly less fill
    than natural (no-permutation) ordering. This is the canonical
    sparse-direct benchmark and the single most important property of ND."""
    A = _grid_laplacian_3d(6)
    P = nested_dissection_order(A)

    fill_natural = fill_count_after_lu(A, perm=None)
    fill_nd      = fill_count_after_lu(A, perm=P)

    assert fill_nd < fill_natural, (
        f"ND fill {fill_nd} should be < natural fill {fill_natural}"
    )
    # On a 6³ grid the reduction is typically 2-3×.
    assert fill_nd < 0.7 * fill_natural, (
        f"ND should reduce fill by ≥30%; got {fill_nd}/{fill_natural} "
        f"= {fill_nd/fill_natural:.2f}"
    )


def test_nd_returns_valid_permutation():
    """``perm`` must be a permutation of ``[0, n)``."""
    A = _grid_laplacian_3d(4)
    P = nested_dissection_order(A)
    assert P.shape == (A.shape[0],)
    assert np.array_equal(np.sort(P), np.arange(A.shape[0]))


def test_nd_trivial_sizes():
    """Edge cases: 0×0 and 1×1 matrices return without crashing."""
    P0 = nested_dissection_order(sp.csc_matrix((0, 0)))
    assert P0.shape == (0,)
    P1 = nested_dissection_order(sp.csc_matrix(np.array([[2.0]])))
    assert np.array_equal(P1, np.array([0]))
