"""Linear (4-node, constant-strain) tetrahedral element for 3D linear elasticity.

Formulation
-----------
Shape functions are the barycentric coordinates N_i(x), i = 0..3, which are
linear in x. With the edge matrix

    J = [x1 - x0, x2 - x0, x3 - x0]   (3x3, columns)

the physical position is x = x0 + J ξ, and the shape-function gradients are

    ∇N_i = J^{-T} e_i  (i = 1, 2, 3),      ∇N_0 = -(∇N_1 + ∇N_2 + ∇N_3).

Since the gradients are constant, the strain-displacement matrix B (6x12) is
constant over the element and the stiffness is simply

    K_e = V B^T C B,      V = det(J) / 6 > 0.

DOF ordering inside the element: [u0x u0y u0z u1x u1y u1z u2x ... u3z], i.e.
local dof = 3 * local_node + component. Globally the same rule applies with
the global node index (see :mod:`src.fem.boundary`).

Voigt strain ordering: [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz] (engineering
shear). For node i with gradient (bx, by, bz) the 6x3 block of B is

        | bx  0   0  |
        | 0   by  0  |
        | 0   0   bz |
        | by  bx  0  |
        | 0   bz  by |
        | bz  0   bx |

All routines are vectorised over elements: input ``tets`` of shape (M, 4).
"""

from __future__ import annotations

import numpy as np


def shape_gradients(nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of the 4 shape functions and element volumes.

    Returns
    -------
    grads : (M, 4, 3)  ∇N_i for every element and local node
    volumes : (M,)      signed volumes (positive for correctly oriented tets)
    """
    x = nodes[tets]  # (M, 4, 3)
    J = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)  # (M, 3, 3) columns
    detJ = np.linalg.det(J)
    volumes = detJ / 6.0
    if np.any(np.abs(volumes) < 1e-30):
        raise ValueError("degenerate tetrahedron (zero volume) encountered")
    # ξ = J^{-1}(x - x0)  =>  ∇N_i = row (i-1) of J^{-1}  (= column of J^{-T})
    Jinv = np.linalg.inv(J)  # (M, 3, 3)
    grads = np.empty((len(tets), 4, 3))
    grads[:, 1:, :] = Jinv
    grads[:, 0, :] = -Jinv.sum(axis=1)
    return grads, volumes


def strain_displacement_matrices(grads: np.ndarray) -> np.ndarray:
    """B matrices (M, 6, 12) from shape gradients (M, 4, 3)."""
    M = grads.shape[0]
    B = np.zeros((M, 6, 12))
    for i in range(4):
        bx, by, bz = grads[:, i, 0], grads[:, i, 1], grads[:, i, 2]
        c = 3 * i
        B[:, 0, c + 0] = bx
        B[:, 1, c + 1] = by
        B[:, 2, c + 2] = bz
        B[:, 3, c + 0] = by
        B[:, 3, c + 1] = bx
        B[:, 4, c + 1] = bz
        B[:, 4, c + 2] = by
        B[:, 5, c + 0] = bz
        B[:, 5, c + 2] = bx
    return B


def element_stiffness_matrices(nodes: np.ndarray, tets: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Element stiffness matrices ``K_e = V B^T C B`` for all elements.

    Returns
    -------
    Ke : (M, 12, 12)
    B : (M, 6, 12)
    volumes : (M,)
    """
    grads, volumes = shape_gradients(nodes, tets)
    if np.any(volumes <= 0):
        raise ValueError("all tetrahedra must be positively oriented (use fix_tet_orientation)")
    B = strain_displacement_matrices(grads)
    CB = np.einsum("ij,mjk->mik", C, B)  # (M, 6, 12)
    Ke = np.einsum("mji,mjk->mik", B, CB) * volumes[:, None, None]
    return Ke, B, volumes


def element_dof_indices(tets: np.ndarray) -> np.ndarray:
    """(M, 12) global DOF indices, ``3 * node + component``."""
    tets = np.asarray(tets, dtype=np.int64)
    return (3 * tets[:, :, None] + np.arange(3)[None, None, :]).reshape(len(tets), 12)
