"""Implicit Euler time integration with Newton-Raphson for FEM.

Solves:  M * (x_{n+1} - x_n - dt * v_n) / dt^2 = f_int(x_{n+1}) + f_ext

Rearranged as a residual to solve for x_{n+1}:
  r(x) = M * (x - x_n - dt * v_n) / dt^2 - f_int(x) - f_ext = 0

Newton step:
  (M/dt^2 + K) * dx = -r
  x <- x + alpha * dx  (with line search)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from robosim import _cpp as _cpp_mod
    _HAVE_CPP_FEM = hasattr(_cpp_mod, "fem")
except ImportError:
    _cpp_mod = None
    _HAVE_CPP_FEM = False

_USE_CPP_FEM = _HAVE_CPP_FEM and not os.environ.get("ROBOSIM_NO_CPP_FEM")


def _cached_or_spsolve(A, b, factor_cache, x_warm=None):
    """Solve A x = b via the supplied cached solver. Dispatches to:

    * :class:`SparseCG`        — CG + diagonal precond + warm start
    * :class:`SparseSPDFactor` — direct LDLT with pattern reuse

    Either falls back to ``scipy.spsolve`` on error / cache miss.

    The CG path expects ``x_warm`` from the previous solve (typically
    the last Newton-iter ``dx``); empty/None warm guesses fall back
    to zero.
    """
    if factor_cache is None or not _USE_CPP_FEM:
        return _spsolve(A, b)
    A_csr = A.tocsr() if not sp.isspmatrix_csr(A) else A
    n = A_csr.shape[0]
    indptr  = A_csr.indptr.astype(np.int32)
    indices = A_csr.indices.astype(np.int32)
    data    = A_csr.data.astype(np.float64)
    sig = (n, int(indptr[-1]))
    cached_sig = getattr(factor_cache, "_robosim_sig", None)
    is_cg = type(factor_cache).__name__ == "SparseCG"
    if not factor_cache.ready() or cached_sig != sig:
        factor_cache.analyze(indptr, indices, data, int(n))
        factor_cache._robosim_sig = sig
    try:
        if is_cg:
            warm = (x_warm if x_warm is not None and x_warm.size == n
                    else np.zeros(n))
            x, iters, ok = factor_cache.solve(
                data, np.ascontiguousarray(b, dtype=np.float64),
                np.ascontiguousarray(warm, dtype=np.float64),
                1e-8, 200,
            )
            if ok:
                return x
            # Non-converged — bail to scipy direct.
            return _spsolve(A, b)
        factor_cache.factorize(data)
        return factor_cache.solve(np.ascontiguousarray(b, dtype=np.float64))
    except Exception:
        factor_cache._robosim_sig = None
        return _spsolve(A, b)


def _spsolve(A, b):
    """Dispatch sparse solve to C++ SimplicialLDLT when available.

    Falls back to ``scipy.sparse.linalg.spsolve`` (SuperLU) on shape
    or factorisation issues. Mirrors the public spsolve signature
    so call sites stay unchanged.
    """
    if _USE_CPP_FEM:
        try:
            A_csr = A.tocsr() if not sp.isspmatrix_csr(A) else A
            n = A_csr.shape[0]
            return _cpp_mod.fem.sparse_spd_solve(
                A_csr.indptr.astype(np.int32),
                A_csr.indices.astype(np.int32),
                A_csr.data.astype(np.float64),
                int(n),
                np.ascontiguousarray(b, dtype=np.float64),
            )
        except Exception:
            # SPD assumption violated or factor failure — bail to SuperLU.
            pass
    return spla.spsolve(A, b)

from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.materials import (
    CorotationalElastic, CorotationalPlastic, NeoHookean,
)
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
    # Updated per-element plastic strain (only set when material is
    # CorotationalPlastic; ``None`` for elastic materials).
    eps_p_new: np.ndarray | None = None


def implicit_euler_step(
    mesh: TetMesh,
    x: np.ndarray,
    v: np.ndarray,
    f_ext: np.ndarray,
    dt: float,
    material: CorotationalElastic | CorotationalPlastic | NeoHookean,
    M: sp.csr_matrix,
    dN_list: list[np.ndarray],
    volumes: np.ndarray,
    fixed_dofs: np.ndarray | None = None,
    max_newton_iters: int = 20,
    tol: float = 1e-6,
    damping: float = 0.0,
    eps_p: np.ndarray | None = None,
    factor_cache: "object | None" = None,
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

    # Warm-start storage for the iterative CG solver. The previous
    # Newton iter's ``dx`` is a good initial guess for the current one
    # because A shifts only a little between iterations.
    dx_warm: np.ndarray | None = None

    # Plastic state: Newton uses the *frozen* eps_p_n from the start of the
    # step (elastic-predictor / plastic-corrector pattern). After the
    # iteration converges we re-assemble once at the converged x to commit
    # eps_p_{n+1}. Inside Newton, eps_p_new returned by assemble_forces is
    # discarded — only the projected force is consumed.
    is_plastic = isinstance(material, CorotationalPlastic)
    eps_p_n = eps_p if is_plastic else None

    for iteration in range(max_newton_iters):
        x_3d = x_new.reshape(-1, 3)

        # Internal forces + intermediates for stiffness reuse
        if is_plastic:
            f_int, F_all, R_all, S_all, _eps_p_trial = assemble_forces(
                mesh, x_3d, material, dN_list, volumes,
                return_intermediates=True, eps_p=eps_p_n,
            )
        else:
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
            warm_f = (dx_warm[free_dofs] if dx_warm is not None else None)
            dx_f = _cached_or_spsolve(A_ff, -r_f, factor_cache, x_warm=warm_f)
            dx = np.zeros(n_dof)
            dx[free_dofs] = dx_f
        else:
            dx = _cached_or_spsolve(A, -residual, factor_cache, x_warm=dx_warm)
        dx_warm = dx.copy()

        # Accept full Newton step (skip line search for small dt).
        # Line search only if residual grows.
        x_new = x_new + dx

        # Enforce fixed DOF positions
        if fixed_dofs is not None and len(fixed_dofs) > 0:
            x_new[fixed_dofs] = x_flat[fixed_dofs]

    # Velocity update
    v_new = (x_new - x_flat) / dt

    # Commit plastic strain at converged x: re-evaluate the constitutive
    # model once with the converged configuration so eps_p_{n+1} reflects
    # the projection actually consistent with x_{n+1}.
    eps_p_new = None
    if is_plastic:
        f_int_final, _, _, _, eps_p_new = assemble_forces(
            mesh, x_new.reshape(-1, 3), material, dN_list, volumes,
            return_intermediates=True, eps_p=eps_p_n,
        )

    return ImplicitEulerResult(
        x_new=x_new.reshape(-1, 3),
        v_new=v_new.reshape(-1, 3),
        newton_iters=iteration + 1,
        converged=converged,
        residual_norm=residual_norm,
        eps_p_new=eps_p_new,
    )
