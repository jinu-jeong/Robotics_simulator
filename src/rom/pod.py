"""Proper orthogonal decomposition (POD / PCA / SVD) of FEM displacement snapshots.

    U = [u¹ … uˢ] ∈ R^{3N×S},   U = Φ Σ Vᵀ,   u ≈ Φ_r q,   q = Φ_rᵀ u

The basis is *not* mean-centred: the FEM is linear and homogeneous (u = 0 for
λ = 0), and the reduced mechanics ``K_r q = B_r λ`` needs the subspace to
contain the origin. Clamped DOFs are zero in every snapshot, so the
corresponding rows of Φ vanish automatically.

Energy criterion: the fraction of snapshot energy captured by ``r`` modes is
``Σ_{i≤r} σ_i² / Σ σ_i²``. For a linear model with S_c distinct contact
locations the snapshot matrix has rank ≤ S_c regardless of how many force
levels were sampled – evaluate generalisation on held-out *contacts*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PODBasis:
    Phi: np.ndarray  # (3N, r) orthonormal columns
    singular_values: np.ndarray  # all singular values of the snapshot matrix (len >= r)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ basics
    @property
    def r(self) -> int:
        return int(self.Phi.shape[1])

    @property
    def n_dof(self) -> int:
        return int(self.Phi.shape[0])

    def truncate(self, r: int) -> "PODBasis":
        if r < 1 or r > self.r:
            raise ValueError(f"r must be in [1, {self.r}]")
        return PODBasis(self.Phi[:, :r], self.singular_values, dict(self.meta))

    def energy_fraction(self, r: int | None = None) -> float:
        s2 = self.singular_values**2
        r = self.r if r is None else r
        return float(s2[:r].sum() / s2.sum())

    def modes_for_energy(self, fraction: float) -> int:
        s2 = np.cumsum(self.singular_values**2) / np.sum(self.singular_values**2)
        return int(np.searchsorted(s2, fraction) + 1)

    def mode(self, k: int) -> np.ndarray:
        """Mode ``k`` as an (N, 3) displacement field."""
        return self.Phi[:, k].reshape(-1, 3)

    # ------------------------------------------------------------ project / reconstruct
    def project(self, u: np.ndarray) -> np.ndarray:
        """``q = Φᵀ u`` for a flat (3N,) / (N,3) field or a (3N, k) matrix."""
        u = np.asarray(u, dtype=float)
        if u.ndim == 2 and u.shape == (self.n_dof // 3, 3):
            u = u.reshape(-1)
        return self.Phi.T @ u

    def reconstruct(self, q: np.ndarray) -> np.ndarray:
        """``u = Φ q`` as an (N, 3) field (or (N, 3, k) for a (r, k) matrix)."""
        q = np.asarray(q, dtype=float)
        u = self.Phi @ q
        return u.reshape(-1, 3) if q.ndim == 1 else u.reshape(-1, 3, q.shape[1])

    def reconstruction_error(self, u: np.ndarray) -> tuple[float, float]:
        """(RMSE [m], relative L2 error) of ``Φ Φᵀ u`` versus ``u``."""
        u = np.asarray(u, dtype=float).reshape(-1)
        d = u - self.Phi @ (self.Phi.T @ u)
        return float(np.sqrt(np.mean(d**2))), float(np.linalg.norm(d) / max(np.linalg.norm(u), 1e-300))

    # ------------------------------------------------------------ io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, Phi=self.Phi, singular_values=self.singular_values, meta=np.array(json.dumps(self.meta)))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PODBasis":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(Phi=z["Phi"], singular_values=z["singular_values"], meta=json.loads(str(z["meta"])))


def compute_pod(snapshots: np.ndarray, r_max: int | None = None, rel_tol: float = 1e-12, meta: dict | None = None) -> PODBasis:
    """Thin SVD of the (3N, S) snapshot matrix; keep the first ``r_max`` modes.

    Singular values below ``rel_tol · σ₁`` are treated as numerical zero and the
    basis is truncated there. With a linear model the physical rank equals the
    number of distinct contacts; snapshots stored in float32 add a noise floor
    around ``1e-8 … 1e-9 · σ₁``, so pass ``rel_tol ≈ 1e-6`` for such data
    (:func:`storage_rank_tolerance`).
    """
    U = np.asarray(snapshots, dtype=float)
    if U.ndim != 2:
        raise ValueError("snapshots must be a (3N, S) matrix")
    Phi, s, _ = np.linalg.svd(U, full_matrices=False)
    # DOFs that are identically zero in every snapshot (clamped root) stay exactly zero
    # in the modes (LAPACK leaves ~1e-12 round-off there).
    Phi[~np.any(U != 0.0, axis=1)] = 0.0
    rank = int(np.sum(s > rel_tol * s[0]))
    r = rank if r_max is None else min(int(r_max), rank)
    info = {"n_snapshots": int(U.shape[1]), "numerical_rank": rank, "rank_rel_tol": rel_tol, "r": r, **(meta or {})}
    return PODBasis(np.ascontiguousarray(Phi[:, :r]), s, info)


def storage_rank_tolerance(dtype) -> float:
    """Relative singular-value tolerance appropriate for snapshots stored in ``dtype``."""
    return 1e-6 if np.dtype(dtype).itemsize <= 4 else 1e-12
