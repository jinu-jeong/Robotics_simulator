"""Implicit Euler time integration with Newton-Raphson for FEM.

Solves:  M * (x_{n+1} - x_n - dt * v_n) / dt^2 = f_int(x_{n+1}) + f_ext

Rearranged as a residual to solve for x_{n+1}:
  r(x) = M * (x - x_n - dt * v_n) / dt^2 - f_int(x) - f_ext = 0

Newton step:
  (M/dt^2 + K) * dx = -r
  x <- x + alpha * dx  (with line search)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.materials import CorotationalElastic, NeoHookean
from robosim.physics.fem.assembly import (
    assemble_forces,
    assemble_stiffness,
    assemble_mass_matrix,
    precompute_element_data,
)


@dataclass
class ImplicitEulerResult:
    """Result of one implicit Euler step."""

    x_new: np.ndarray
    v_new: np.ndarray
    newton_iters: int
    converged: bool
    residual_norm: float


def implicit_euler_step(
    mesh: TetMesh,
    x: np.ndarray,
    v: np.ndarray,
    f_ext: np.ndarray,
    dt: float,
    material: CorotationalElastic | NeoHookean,
    M: sp.csr_matrix,
    dN_list: list[np.ndarray],
    volumes: np.ndarray,
    fixed_dofs: np.ndarray | None = None,
    max_newton_iters: int = 20,
    tol: float = 1e-6,
    damping: float = 0.0,
) -> ImplicitEulerResult:
    """Perform one implicit Euler time step.

    Parameters
    ----------
    mesh : reference mesh
    x : (n_nodes, 3) current positions (flat to n_dof for internal use)
    v : (n_nodes, 3) current velocities
    f_ext : (n_dof,) external forces (gravity, coupling, etc.)
    dt : time step
    material : constitutive model
    M : (n_dof, n_dof) mass matrix
    dN_list : precomputed shape function gradients
    volumes : precomputed element volumes
    fixed_dofs : indices of DOFs with Dirichlet BC (positions held fixed)
    max_newton_iters : maximum Newton iterations
    tol : convergence tolerance (relative residual)
    damping : Rayleigh mass-proportional damping coefficient

    Returns
    -------
    ImplicitEulerResult with new positions and velocities.
    """
    n_dof = mesh.n_nodes * 3
    x_flat = x.reshape(-1).copy()
    v_flat = v.reshape(-1).copy()

    # Predicted position (explicit Euler prediction)
    x_pred = x_flat + dt * v_flat

    # Start Newton from the prediction
    x_new = x_pred.copy()

    # Build free DOF mask
    if fixed_dofs is not None and len(fixed_dofs) > 0:
        free_mask = np.ones(n_dof, dtype=bool)
        free_mask[fixed_dofs] = False
        free_dofs = np.where(free_mask)[0]
    else:
        free_dofs = np.arange(n_dof)

    converged = False
    residual_norm = 0.0
    dt2_inv = 1.0 / (dt * dt)

    for iteration in range(max_newton_iters):
        x_3d = x_new.reshape(-1, 3)

        # Internal forces + intermediates for stiffness reuse
        result = assemble_forces(mesh, x_3d, material, dN_list, volumes,
                                 return_intermediates=True)
        f_int, F_all, R_all, S_all = result

        # Residual: r = M/dt^2 * (x_new - x_pred) - f_int - f_ext
        inertia_term = M @ (x_new - x_pred) * dt2_inv
        damping_term = damping * M @ (x_new - x_flat) / dt if damping > 0 else 0.0
        residual = inertia_term - f_int - f_ext + damping_term

        # Apply BCs: zero residual at fixed DOFs
        if fixed_dofs is not None and len(fixed_dofs) > 0:
            residual[fixed_dofs] = 0.0

        residual_norm = float(np.linalg.norm(residual[free_dofs]))

        # Check convergence
        if residual_norm < tol * max(1.0, np.linalg.norm(f_ext[free_dofs]) + 1.0):
            converged = True
            break

        # Tangent stiffness — reuse R_all from force assembly (skip SVD)
        K = assemble_stiffness(mesh, x_3d, material, dN_list, volumes,
                               R_all=R_all)

        # System matrix: A = M/dt^2 + K (+ damping)
        A = M * dt2_inv + K
        if damping > 0:
            A = A + damping * M / dt

        # Apply BCs: zero rows/cols for fixed DOFs
        if fixed_dofs is not None and len(fixed_dofs) > 0:
            A_ff = A[np.ix_(free_dofs, free_dofs)]
            r_f = residual[free_dofs]
            dx_f = spla.spsolve(A_ff, -r_f)
            dx = np.zeros(n_dof)
            dx[free_dofs] = dx_f
        else:
            dx = spla.spsolve(A, -residual)

        # Accept full Newton step (skip line search for small dt).
        # Line search only if residual grows.
        x_new = x_new + dx

        # Enforce fixed DOF positions
        if fixed_dofs is not None and len(fixed_dofs) > 0:
            x_new[fixed_dofs] = x_flat[fixed_dofs]

    # Velocity update
    v_new = (x_new - x_flat) / dt

    return ImplicitEulerResult(
        x_new=x_new.reshape(-1, 3),
        v_new=v_new.reshape(-1, 3),
        newton_iters=iteration + 1,
        converged=converged,
        residual_norm=residual_norm,
    )
