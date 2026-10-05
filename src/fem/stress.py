"""Strain / stress recovery for the quadratic FEM.

Element quantities are *element means* (volume-averaged strain matrix ``B̄``
times the element displacements). Nodal values are volume-weighted averages
of the adjacent elements and are meant for visualization (per-node scalar
colouring), not for accuracy-critical post-processing.
"""

from __future__ import annotations

import numpy as np

from .elements import element_dof_indices
from .material import LinearElasticMaterial


def element_strains(B: np.ndarray, conn: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Mean Voigt strains (M, 6) of one element block from the flat displacement ``u`` (3N,)."""
    ue = np.asarray(u).reshape(-1)[element_dof_indices(conn)]  # (M, 3n)
    return np.einsum("mij,mj->mi", B, ue)


def element_stresses(strains: np.ndarray, material: LinearElasticMaterial) -> np.ndarray:
    return strains @ material.stiffness_matrix().T


def von_mises(stress: np.ndarray) -> np.ndarray:
    """Von Mises equivalent stress from Voigt stresses (M, 6)."""
    sxx, syy, szz, txy, tyz, txz = stress.T
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (txy**2 + tyz**2 + txz**2)
    )


def element_to_nodes(values: np.ndarray, conn, volumes: np.ndarray, n_nodes: int) -> np.ndarray:
    """Volume-weighted average of element values (M,) or (M, k) at the nodes.

    ``conn`` may be one connectivity array or a list of blocks whose row counts
    add up to ``len(values)`` (mesh assembly order).
    """
    values = np.asarray(values, dtype=float)
    scalar = values.ndim == 1
    v = values[:, None] if scalar else values
    blocks = list(conn) if isinstance(conn, (list, tuple)) else [np.asarray(conn)]
    acc = np.zeros((n_nodes, v.shape[1]))
    w = np.zeros(n_nodes)
    start = 0
    for c in blocks:
        c = np.asarray(c, dtype=np.int64)
        m = c.shape[0]
        if m == 0:
            continue
        vb, volb = v[start:start + m], np.asarray(volumes, float)[start:start + m]
        for k in range(c.shape[1]):
            np.add.at(acc, c[:, k], vb * volb[:, None])
            np.add.at(w, c[:, k], volb)
        start += m
    out = acc / np.maximum(w, 1e-300)[:, None]
    return out[:, 0] if scalar else out
