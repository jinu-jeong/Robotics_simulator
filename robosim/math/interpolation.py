"""FEM interpolation helpers: shape functions, natural coordinates, and field evaluation.

This module provides convenience functions for interpolating quantities
within finite elements without needing to work with the raw element
module directly.

Usage::

    from robosim.math.interpolation import interpolate_field, find_natural_coords

    # Interpolate displacement at a physical point inside a tet element
    u_point = interpolate_field(nodes, displacements, xi)

    # Find natural coordinates of a physical point
    xi = find_natural_coords_tet4(nodes, phys_point)
"""

from __future__ import annotations

import numpy as np


# ── Tet4 shape functions (convenience wrappers) ─────────────────

def tet4_shape_functions(xi: np.ndarray) -> np.ndarray:
    """Evaluate Tet4 shape functions at natural coordinate xi.

    Parameters
    ----------
    xi : (3,) natural coordinates (xi1, xi2, xi3)

    Returns
    -------
    N : (4,) shape function values
    """
    return np.array([
        1.0 - xi[0] - xi[1] - xi[2],
        xi[0],
        xi[1],
        xi[2],
    ])


def tet4_shape_gradients() -> np.ndarray:
    """Reference-space gradients of Tet4 shape functions.

    Returns
    -------
    dN_dxi : (4, 3) shape function gradients w.r.t. (xi1, xi2, xi3)
    """
    return np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0],
    ])


def tet4_physical_gradients(
    node_coords: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute physical-space shape gradients and element volume.

    Parameters
    ----------
    node_coords : (4, 3) nodal coordinates

    Returns
    -------
    dN_dx : (4, 3) shape gradients in physical space
    volume : element volume
    """
    dN_dxi = tet4_shape_gradients()
    J = node_coords[1:] - node_coords[0]  # (3, 3) Jacobian
    det_J = np.linalg.det(J)
    volume = abs(det_J) / 6.0

    J_inv = np.linalg.inv(J)
    dN_dx = dN_dxi @ J_inv
    return dN_dx, volume


# ── Natural coordinate inversion ────────────────────────────────

def find_natural_coords_tet4(
    node_coords: np.ndarray,
    phys_point: np.ndarray,
) -> np.ndarray:
    """Find natural coordinates of a physical point within a Tet4 element.

    Uses the exact linear mapping: x = x0 + J * xi
      => xi = J^{-1} (x - x0)

    Parameters
    ----------
    node_coords : (4, 3) element nodal coordinates
    phys_point : (3,) physical point

    Returns
    -------
    xi : (3,) natural coordinates (may be outside [0,1] if point is outside)
    """
    x0 = node_coords[0]
    J = (node_coords[1:] - x0).T  # (3, 3)
    xi = np.linalg.solve(J, phys_point - x0)
    return xi


def point_in_tet(
    node_coords: np.ndarray,
    phys_point: np.ndarray,
    tol: float = 1e-8,
) -> bool:
    """Check if a physical point lies inside a Tet4 element.

    Parameters
    ----------
    node_coords : (4, 3) element nodal coordinates
    phys_point : (3,) physical point
    tol : tolerance for boundary inclusion

    Returns
    -------
    True if point is inside (or on boundary within tolerance)
    """
    xi = find_natural_coords_tet4(node_coords, phys_point)
    lam0 = 1.0 - xi[0] - xi[1] - xi[2]
    return bool(
        xi[0] >= -tol and xi[1] >= -tol and xi[2] >= -tol
        and lam0 >= -tol
    )


# ── Field interpolation ────────────────────────────────────────

def interpolate_field(
    node_coords: np.ndarray,
    node_values: np.ndarray,
    xi: np.ndarray,
) -> np.ndarray:
    """Interpolate a nodal field at natural coordinates xi within a Tet4.

    Parameters
    ----------
    node_coords : (4, 3) element nodal coordinates (unused for Tet4,
                  kept for API consistency with higher-order elements)
    node_values : (4,) or (4, d) nodal field values
    xi : (3,) natural coordinates

    Returns
    -------
    Interpolated value: scalar or (d,) vector
    """
    N = tet4_shape_functions(xi)
    return N @ node_values


def interpolate_gradient(
    node_coords: np.ndarray,
    node_values: np.ndarray,
) -> np.ndarray:
    """Compute the gradient of a nodal field within a Tet4 element.

    For Tet4, the gradient is constant within the element.

    Parameters
    ----------
    node_coords : (4, 3) element nodal coordinates
    node_values : (4,) or (4, d) nodal field values

    Returns
    -------
    gradient : (3,) or (d, 3) field gradient in physical space
    """
    dN_dx, _ = tet4_physical_gradients(node_coords)
    # For scalar field: grad = sum_i (N_i,x * u_i)
    # For vector field: each component separately
    if node_values.ndim == 1:
        return dN_dx.T @ node_values  # (3,)
    else:
        return node_values.T @ dN_dx  # (d, 3)


# ── Deformation gradient ───────────────────────────────────────

def deformation_gradient_tet4(
    ref_coords: np.ndarray,
    cur_coords: np.ndarray,
) -> np.ndarray:
    """Compute deformation gradient F for a Tet4 element.

    F = dx/dX = I + du/dX where u = x - X (displacement)

    Parameters
    ----------
    ref_coords : (4, 3) reference (undeformed) nodal coordinates
    cur_coords : (4, 3) current (deformed) nodal coordinates

    Returns
    -------
    F : (3, 3) deformation gradient
    """
    dN_dX, _ = tet4_physical_gradients(ref_coords)
    # F = I + sum_i (u_i outer dN_i/dX)
    # = sum_i (x_i outer dN_i/dX) = cur_coords.T @ dN_dX
    # Wait: F_ij = dx_i/dX_j = delta_ij + du_i/dX_j
    # du_i/dX_j = sum_a (u_a_i * dN_a/dX_j)
    u = cur_coords - ref_coords
    F = np.eye(3) + u.T @ dN_dX
    return F


# ── Barycentric coordinates ────────────────────────────────────

def barycentric_coords(
    node_coords: np.ndarray,
    phys_point: np.ndarray,
) -> np.ndarray:
    """Compute barycentric coordinates of a point within a Tet4.

    Parameters
    ----------
    node_coords : (4, 3) element nodal coordinates
    phys_point : (3,) physical point

    Returns
    -------
    bary : (4,) barycentric coordinates (sum to 1.0 if inside)
    """
    xi = find_natural_coords_tet4(node_coords, phys_point)
    lam0 = 1.0 - xi[0] - xi[1] - xi[2]
    return np.array([lam0, xi[0], xi[1], xi[2]])
