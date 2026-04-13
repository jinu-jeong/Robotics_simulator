"""Sparse linear algebra solver wrappers.

Provides a unified interface over SciPy sparse solvers with
optional caching (factorization reuse) for iterative FEM solves.

Usage::

    solver = SparseSolver(method="cholesky")
    x = solver.solve(A, b)           # first call: factorize
    x = solver.solve(A_updated, b2)  # re-factorize if sparsity changed
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg


class SolverMethod(str, Enum):
    """Available sparse solver methods."""

    DIRECT = "direct"       # scipy.sparse.linalg.spsolve (SuperLU)
    CHOLESKY = "cholesky"   # Cholesky factorization (SPD matrices only)
    CG = "cg"               # Conjugate gradient (SPD matrices)
    GMRES = "gmres"         # GMRES (general non-symmetric)
    MINRES = "minres"       # MINRES (symmetric indefinite)


class SparseSolver:
    """Wrapper for sparse linear system solvers.

    Supports direct and iterative methods with optional
    preconditioner caching.

    Parameters
    ----------
    method : solver method to use
    tol : convergence tolerance for iterative methods
    max_iter : maximum iterations for iterative methods
    verbose : print convergence info
    """

    def __init__(
        self,
        method: str | SolverMethod = "direct",
        tol: float = 1e-10,
        max_iter: int = 1000,
        verbose: bool = False,
    ):
        if isinstance(method, str):
            method = SolverMethod(method)
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

        self._factor = None     # cached factorization
        self._precond = None    # cached preconditioner
        self._last_shape = None
        self._last_nnz = None
        self._solve_count = 0

    def solve(
        self,
        A: sparse.spmatrix,
        b: np.ndarray,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve Ax = b.

        Parameters
        ----------
        A : sparse matrix (CSC or CSR)
        b : right-hand side vector
        x0 : initial guess for iterative solvers

        Returns
        -------
        x : solution vector
        """
        if not sparse.issparse(A):
            A = sparse.csc_matrix(A)

        self._solve_count += 1

        if self.method == SolverMethod.DIRECT:
            return self._solve_direct(A, b)
        elif self.method == SolverMethod.CHOLESKY:
            return self._solve_cholesky(A, b)
        elif self.method == SolverMethod.CG:
            return self._solve_cg(A, b, x0)
        elif self.method == SolverMethod.GMRES:
            return self._solve_gmres(A, b, x0)
        elif self.method == SolverMethod.MINRES:
            return self._solve_minres(A, b, x0)
        else:
            raise ValueError(f"Unknown solver method: {self.method}")

    def _solve_direct(self, A, b):
        """Direct solve via SuperLU."""
        A_csc = sparse.csc_matrix(A)
        return sp_linalg.spsolve(A_csc, b)

    def _solve_cholesky(self, A, b):
        """Cholesky factorization (SPD matrices).

        Falls back to direct solve if scikit-sparse is not available.
        """
        try:
            from sksparse.cholmod import cholesky as sk_cholesky

            # Reuse factorization if sparsity pattern hasn't changed
            shape_match = (self._last_shape == A.shape)
            nnz_match = (self._last_nnz == A.nnz) if shape_match else False

            if self._factor is not None and shape_match and nnz_match:
                try:
                    self._factor.cholesky_inplace(sparse.csc_matrix(A))
                except Exception:
                    self._factor = sk_cholesky(sparse.csc_matrix(A))
            else:
                self._factor = sk_cholesky(sparse.csc_matrix(A))

            self._last_shape = A.shape
            self._last_nnz = A.nnz
            return self._factor(b)
        except ImportError:
            # Fallback to direct
            return self._solve_direct(A, b)

    def _solve_cg(self, A, b, x0):
        """Conjugate gradient for SPD matrices."""
        # Build incomplete Cholesky preconditioner
        M = self._build_precond(A)
        x, info = sp_linalg.cg(A, b, x0=x0, rtol=self.tol,
                                maxiter=self.max_iter, M=M)
        if info != 0 and self.verbose:
            print(f"CG: convergence issue (info={info})")
        return x

    def _solve_gmres(self, A, b, x0):
        """GMRES for general matrices."""
        M = self._build_precond(A)
        x, info = sp_linalg.gmres(A, b, x0=x0, rtol=self.tol,
                                   maxiter=self.max_iter, M=M)
        if info != 0 and self.verbose:
            print(f"GMRES: convergence issue (info={info})")
        return x

    def _solve_minres(self, A, b, x0):
        """MINRES for symmetric indefinite matrices."""
        x, info = sp_linalg.minres(A, b, x0=x0, rtol=self.tol,
                                    maxiter=self.max_iter)
        if info != 0 and self.verbose:
            print(f"MINRES: convergence issue (info={info})")
        return x

    def _build_precond(self, A):
        """Build or reuse ILU preconditioner."""
        shape_changed = (self._last_shape != A.shape)
        nnz_changed = (self._last_nnz != A.nnz if not shape_changed else True)

        if self._precond is None or shape_changed or nnz_changed:
            try:
                ilu = sp_linalg.spilu(sparse.csc_matrix(A))
                self._precond = sp_linalg.LinearOperator(
                    A.shape, matvec=ilu.solve)
            except Exception:
                self._precond = None

            self._last_shape = A.shape
            self._last_nnz = A.nnz

        return self._precond

    @property
    def solve_count(self) -> int:
        """Number of solves performed."""
        return self._solve_count

    def reset(self) -> None:
        """Clear cached factorizations."""
        self._factor = None
        self._precond = None
        self._last_shape = None
        self._last_nnz = None


def solve_sparse(
    A: sparse.spmatrix,
    b: np.ndarray,
    method: str = "direct",
) -> np.ndarray:
    """Convenience function: solve Ax = b with a one-shot solver.

    For repeated solves, use SparseSolver for factorization reuse.
    """
    solver = SparseSolver(method=method)
    return solver.solve(A, b)


def sparse_eigenvalues(
    A: sparse.spmatrix,
    k: int = 6,
    which: str = "SM",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute k smallest eigenvalues of a sparse SPD matrix.

    Useful for modal analysis of stiffness matrices.

    Parameters
    ----------
    A : sparse matrix (n x n)
    k : number of eigenvalues
    which : "SM" = smallest magnitude, "LM" = largest

    Returns
    -------
    eigenvalues : (k,) array
    eigenvectors : (n, k) array
    """
    vals, vecs = sp_linalg.eigsh(A, k=k, which=which)
    return vals, vecs


def condition_number_estimate(A: sparse.spmatrix) -> float:
    """Estimate condition number of a sparse matrix.

    Uses the ratio of largest to smallest singular value estimate.
    """
    try:
        sv_max = sp_linalg.svds(A, k=1, which="LM", return_singular_vectors=False)
        sv_min = sp_linalg.svds(A, k=1, which="SM", return_singular_vectors=False)
        return float(sv_max[0] / max(sv_min[0], 1e-15))
    except Exception:
        return float("inf")
