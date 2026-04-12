"""Tet4 (linear tetrahedral) finite element.

Provides shape function derivatives, reference-to-deformed mapping,
deformation gradient computation, and element force/stiffness routines.
"""

from __future__ import annotations

import numpy as np


def compute_shape_derivatives(x0: np.ndarray, x1: np.ndarray,
                               x2: np.ndarray, x3: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute shape function derivatives for a Tet4 element.

    For linear tets, shape function gradients are constant within the element.

    Parameters
    ----------
    x0, x1, x2, x3 : (3,) node positions (reference configuration)

    Returns
    -------
    dN : (4, 3) matrix of shape function gradients, dN[i,j] = dN_i/dx_j
    vol : element volume
    """
    # Edge vectors from node 0
    d1 = x1 - x0
    d2 = x2 - x0
    d3 = x3 - x0

    # Jacobian of the reference mapping: [d1 | d2 | d3] as columns
    J = np.column_stack([d1, d2, d3])  # (3, 3)
    det_J = np.linalg.det(J)
    vol = abs(det_J) / 6.0

    if abs(det_J) < 1e-20:
        return np.zeros((4, 3)), 0.0

    J_inv = np.linalg.inv(J)  # (3, 3)

    # Shape function gradients in physical coordinates
    # For Tet4: N_0 = 1 - xi - eta - zeta, N_1 = xi, N_2 = eta, N_3 = zeta
    # dN/d(xi,eta,zeta) = [[-1,-1,-1], [1,0,0], [0,1,0], [0,0,1]]
    # dN/dx = dN/d(xi) @ J_inv
    dN_ref = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0],
    ])

    dN = dN_ref @ J_inv  # (4, 3)
    return dN, vol


def compute_deformation_gradient(dN: np.ndarray, x_def: np.ndarray) -> np.ndarray:
    """Compute deformation gradient F for a Tet4 element.

    F = dx/dX = sum_i x_i (outer) dN_i/dX

    Parameters
    ----------
    dN : (4, 3) shape function gradients in reference configuration
    x_def : (4, 3) deformed node positions

    Returns
    -------
    F : (3, 3) deformation gradient
    """
    # F_ij = sum_a x_def[a, i] * dN[a, j]
    F = x_def.T @ dN  # (3, 3)
    return F


def polar_decomposition(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute polar decomposition F = R @ S.

    Uses SVD: F = U @ Sigma @ V^T, then R = U @ V^T, S = V @ Sigma @ V^T.

    Returns
    -------
    R : (3,3) rotation matrix (proper, det=+1)
    S : (3,3) symmetric stretch tensor
    """
    U, sigma, Vt = np.linalg.svd(F)
    det_UV = np.linalg.det(U) * np.linalg.det(Vt)

    # Ensure proper rotation (det R = +1)
    if det_UV < 0:
        # Flip the smallest singular value
        U[:, 2] *= -1
        sigma[2] *= -1

    R = U @ Vt
    S = Vt.T @ np.diag(sigma) @ Vt
    return R, S
