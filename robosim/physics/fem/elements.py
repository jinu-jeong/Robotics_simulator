"""Finite element shape functions and integration rules.

Supported elements:
  - Tet4  : 4-node linear tetrahedron (1 Gauss point)
  - Tet10 : 10-node quadratic tetrahedron (4 Gauss points)
  - Hex8  : 8-node trilinear hexahedron (2×2×2 = 8 Gauss points)
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class ElementType(Enum):
    TET4 = "tet4"
    TET10 = "tet10"
    HEX8 = "hex8"


# Number of nodes per element
NODES_PER_ELEMENT = {
    ElementType.TET4: 4,
    ElementType.TET10: 10,
    ElementType.HEX8: 8,
}

# ═══════════════════════════════════════════════════════════════
# Gauss quadrature rules
# ═══════════════════════════════════════════════════════════════

def gauss_points_tet1() -> tuple[np.ndarray, np.ndarray]:
    """1-point rule for tetrahedra (exact for linear)."""
    pts = np.array([[0.25, 0.25, 0.25]])
    wts = np.array([1.0 / 6.0])
    return pts, wts


def gauss_points_tet4() -> tuple[np.ndarray, np.ndarray]:
    """4-point rule for tetrahedra (exact for quadratic).

    Barycentric coords: (a, b, b, b) and permutations,
    a = 0.5854102, b = 0.1381966.
    """
    a = 0.5854101966249685
    b = 0.1381966011250105
    pts = np.array([
        [a, b, b],
        [b, a, b],
        [b, b, a],
        [b, b, b],
    ])
    wts = np.full(4, 1.0 / 24.0)  # each = 1/4 of tet volume (1/6)
    return pts, wts


def gauss_points_hex8() -> tuple[np.ndarray, np.ndarray]:
    """2×2×2 Gauss rule for hexahedra (reference domain [-1,1]³)."""
    g = 1.0 / np.sqrt(3.0)
    pts = np.array([
        [-g, -g, -g], [+g, -g, -g], [+g, +g, -g], [-g, +g, -g],
        [-g, -g, +g], [+g, -g, +g], [+g, +g, +g], [-g, +g, +g],
    ])
    wts = np.ones(8)  # all weights = 1
    return pts, wts


# ═══════════════════════════════════════════════════════════════
# Tet4 (linear tetrahedron, 4 nodes)
# ═══════════════════════════════════════════════════════════════

def tet4_shape(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape functions and reference gradients for Tet4.

    Reference coords: xi=(ξ₁, ξ₂, ξ₃) in standard tet
      Node 0: (0,0,0), Node 1: (1,0,0), Node 2: (0,1,0), Node 3: (0,0,1)
    N₀ = 1-ξ₁-ξ₂-ξ₃, N₁ = ξ₁, N₂ = ξ₂, N₃ = ξ₃

    Returns
    -------
    N    : (4,) shape function values
    dNdxi: (4, 3) derivatives w.r.t. reference coords
    """
    x, y, z = xi
    N = np.array([1.0 - x - y - z, x, y, z])
    dNdxi = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0],
    ])
    return N, dNdxi


# ═══════════════════════════════════════════════════════════════
# Tet10 (quadratic tetrahedron, 10 nodes)
# ═══════════════════════════════════════════════════════════════

def tet10_shape(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape functions and reference gradients for Tet10.

    Node ordering (standard):
      0-3 : corner nodes (same as Tet4)
      4   : edge 0-1 midpoint
      5   : edge 1-2 midpoint
      6   : edge 0-2 midpoint
      7   : edge 0-3 midpoint
      8   : edge 1-3 midpoint
      9   : edge 2-3 midpoint

    Barycentric: L₁ = 1-ξ-η-ζ, L₂ = ξ, L₃ = η, L₄ = ζ

    Returns
    -------
    N    : (10,) shape function values
    dNdxi: (10, 3) derivatives w.r.t. (ξ, η, ζ)
    """
    r, s, t = xi
    L1 = 1.0 - r - s - t
    L2 = r
    L3 = s
    L4 = t

    N = np.array([
        L1 * (2.0 * L1 - 1.0),  # 0: corner
        L2 * (2.0 * L2 - 1.0),  # 1: corner
        L3 * (2.0 * L3 - 1.0),  # 2: corner
        L4 * (2.0 * L4 - 1.0),  # 3: corner
        4.0 * L1 * L2,          # 4: edge 0-1
        4.0 * L2 * L3,          # 5: edge 1-2
        4.0 * L1 * L3,          # 6: edge 0-2
        4.0 * L1 * L4,          # 7: edge 0-3
        4.0 * L2 * L4,          # 8: edge 1-3
        4.0 * L3 * L4,          # 9: edge 2-3
    ])

    # dL/d(r,s,t)
    # L1 = 1-r-s-t → (-1,-1,-1)
    # L2 = r       → (1, 0, 0)
    # L3 = s       → (0, 1, 0)
    # L4 = t       → (0, 0, 1)

    dNdxi = np.zeros((10, 3))

    # Corner nodes: d[Li*(2Li-1)]/dr = dLi/dr * (4Li - 1)
    dNdxi[0] = np.array([-1, -1, -1]) * (4.0 * L1 - 1.0)
    dNdxi[1] = np.array([ 1,  0,  0]) * (4.0 * L2 - 1.0)
    dNdxi[2] = np.array([ 0,  1,  0]) * (4.0 * L3 - 1.0)
    dNdxi[3] = np.array([ 0,  0,  1]) * (4.0 * L4 - 1.0)

    # Edge midpoint nodes: d[4*Li*Lj]/dr = 4*(dLi/dr*Lj + Li*dLj/dr)
    dNdxi[4] = 4.0 * (np.array([-1, -1, -1]) * L2 + L1 * np.array([1, 0, 0]))  # 0-1
    dNdxi[5] = 4.0 * (np.array([ 1,  0,  0]) * L3 + L2 * np.array([0, 1, 0]))  # 1-2
    dNdxi[6] = 4.0 * (np.array([-1, -1, -1]) * L3 + L1 * np.array([0, 1, 0]))  # 0-2
    dNdxi[7] = 4.0 * (np.array([-1, -1, -1]) * L4 + L1 * np.array([0, 0, 1]))  # 0-3
    dNdxi[8] = 4.0 * (np.array([ 1,  0,  0]) * L4 + L2 * np.array([0, 0, 1]))  # 1-3
    dNdxi[9] = 4.0 * (np.array([ 0,  1,  0]) * L4 + L3 * np.array([0, 0, 1]))  # 2-3

    return N, dNdxi


# ═══════════════════════════════════════════════════════════════
# Hex8 (trilinear hexahedron, 8 nodes)
# ═══════════════════════════════════════════════════════════════

# Corner coords in [-1,1]³ reference domain
_HEX8_CORNERS = np.array([
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
], dtype=np.float64)


def hex8_shape(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape functions and reference gradients for Hex8.

    Reference domain: [-1,1]³
    N_i = (1/8)(1 + ξ_i·ξ)(1 + η_i·η)(1 + ζ_i·ζ)

    Returns
    -------
    N    : (8,) shape function values
    dNdxi: (8, 3) derivatives w.r.t. (ξ, η, ζ)
    """
    xi_val, eta_val, zeta_val = xi

    N = np.zeros(8)
    dNdxi = np.zeros((8, 3))

    for i in range(8):
        xi_i, eta_i, zeta_i = _HEX8_CORNERS[i]
        fx = 1.0 + xi_i * xi_val
        fy = 1.0 + eta_i * eta_val
        fz = 1.0 + zeta_i * zeta_val

        N[i] = 0.125 * fx * fy * fz
        dNdxi[i, 0] = 0.125 * xi_i * fy * fz
        dNdxi[i, 1] = 0.125 * fx * eta_i * fz
        dNdxi[i, 2] = 0.125 * fx * fy * zeta_i

    return N, dNdxi


# ═══════════════════════════════════════════════════════════════
# Generic helpers
# ═══════════════════════════════════════════════════════════════

def shape_function(etype: ElementType, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch shape function evaluation by element type."""
    if etype == ElementType.TET4:
        return tet4_shape(xi)
    elif etype == ElementType.TET10:
        return tet10_shape(xi)
    elif etype == ElementType.HEX8:
        return hex8_shape(xi)
    raise ValueError(f"Unknown element type: {etype}")


def gauss_rule(etype: ElementType) -> tuple[np.ndarray, np.ndarray]:
    """Return (points, weights) for the default Gauss rule."""
    if etype == ElementType.TET4:
        return gauss_points_tet1()
    elif etype == ElementType.TET10:
        return gauss_points_tet4()
    elif etype == ElementType.HEX8:
        return gauss_points_hex8()
    raise ValueError(f"Unknown element type: {etype}")


# ═══════════════════════════════════════════════════════════════
# Legacy Tet4 API (backward compatible)
# ═══════════════════════════════════════════════════════════════

def compute_shape_derivatives(x0, x1, x2, x3):
    """Compute shape function derivatives for a Tet4 element (legacy API).

    Returns
    -------
    dN : (4, 3) shape function gradients in physical coords
    vol : element volume
    """
    d1 = x1 - x0
    d2 = x2 - x0
    d3 = x3 - x0
    J = np.column_stack([d1, d2, d3])
    det_J = np.linalg.det(J)
    vol = abs(det_J) / 6.0
    if abs(det_J) < 1e-20:
        return np.zeros((4, 3)), 0.0
    J_inv = np.linalg.inv(J)
    dN_ref = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    dN = dN_ref @ J_inv
    return dN, vol


def compute_deformation_gradient(dN, x_def):
    """F = x_def^T @ dN  (works for any element with constant dN)."""
    return x_def.T @ dN


def polar_decomposition(F):
    """Polar decomposition F = R @ S via SVD."""
    U, sigma, Vt = np.linalg.svd(F)
    det_UV = np.linalg.det(U) * np.linalg.det(Vt)
    if det_UV < 0:
        U[:, 2] *= -1
        sigma[2] *= -1
    R = U @ Vt
    S = Vt.T @ np.diag(sigma) @ Vt
    return R, S
