"""Marker → q estimators (geometry Jacobian LS and learned ridge)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..rendering.markers import SurfaceMarkers
from ..rom.pod import PODBasis
from .dataset import VisionDataset
from .features import (
    image_jacobian,
    marker_delta_features,
    mode_marker_displacements,
    solve_q_from_jacobian,
    undeformed_marker_pixels,
)


@dataclass
class GeometricMarkerModel:
    """Per-sample Jacobian LS: Δuv ≈ J(c) q, no training required."""

    basis: PODBasis
    markers: SurfaceMarkers
    mesh: Any
    ridge: float = 1e-4
    eps: float = 1e-4
    use_noisy: bool = True
    n_iter: int = 1
    huber: float | None = None
    _p0: np.ndarray = field(init=False, repr=False)
    _dp: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._p0 = self.markers.positions(self.mesh, None)
        self._dp = mode_marker_displacements(self.markers, self.mesh, self.basis)

    @property
    def r(self) -> int:
        return self.basis.r

    def predict_one(self, camera, marker_px: np.ndarray, visible: np.ndarray) -> np.ndarray:
        q = np.zeros(self.r)
        marker_px = np.asarray(marker_px, float)
        for _ in range(max(1, int(self.n_iter))):
            p = self._p0 + np.einsum("mkr,r->mk", self._dp, q)
            J = image_jacobian(camera, p, self._dp, eps=self.eps)
            uv_pred, _ = camera.project(p)
            dq = solve_q_from_jacobian(
                J, marker_px - uv_pred, visible, ridge=self.ridge, huber=self.huber,
            )
            q = q + dq
            if float(np.linalg.norm(dq)) < 1e-8:
                break
        return q

    def predict(self, vds: VisionDataset, indices: np.ndarray | None = None) -> np.ndarray:
        idx = np.arange(len(vds)) if indices is None else np.asarray(indices, int)
        px = vds.marker_px if self.use_noisy else vds.marker_px_clean
        out = np.zeros((len(idx), self.r))
        for i, s in enumerate(idx):
            out[i] = self.predict_one(vds.camera(int(s)), px[s], vds.marker_visible[s])
        return out


def fuse_q_by_fem_index(q: np.ndarray, fem_index: np.ndarray) -> np.ndarray:
    """Average ``q`` over rendered views that share a FEM sample."""
    q = np.asarray(q, float).copy()
    fem_index = np.asarray(fem_index).reshape(-1)
    if len(q) != len(fem_index):
        raise ValueError("q and fem_index length mismatch")
    for fid in np.unique(fem_index):
        m = fem_index == fid
        if int(m.sum()) < 2:
            continue
        q[m] = q[m].mean(axis=0)
    return q


@dataclass
class RidgeMarkerModel:
    """Learned linear map Δuv → q (ridge), camera-pose variation absorbed into W."""

    W: np.ndarray  # (2M, r)
    bias: np.ndarray  # (r,)
    ridge_lambda: float
    use_noisy: bool = True
    markers: SurfaceMarkers | None = None
    mesh: Any = None
    uv0_cache: np.ndarray | None = None  # (S, M, 2) optional precomputed
    meta: dict = field(default_factory=dict)

    @property
    def r(self) -> int:
        return int(self.W.shape[1])

    @property
    def n_markers(self) -> int:
        return int(self.W.shape[0] // 2)

    @classmethod
    def fit(
        cls,
        vds: VisionDataset,
        train_mask: np.ndarray,
        markers: SurfaceMarkers,
        mesh,
        ridge_lambda: float = 1e-2,
        use_noisy: bool = True,
        q: np.ndarray | None = None,
    ) -> "RidgeMarkerModel":
        uv0 = undeformed_marker_pixels(vds, markers, mesh)
        px = vds.marker_px if use_noisy else vds.marker_px_clean
        X = marker_delta_features(px, uv0, vds.marker_visible)
        Y = vds.q if q is None else np.asarray(q, float)
        Xt, Yt = X[train_mask], Y[train_mask]
        # centre features for a bias term
        x_mean = Xt.mean(axis=0)
        y_mean = Yt.mean(axis=0)
        Xt_c, Yt_c = Xt - x_mean, Yt - y_mean
        A = Xt_c.T @ Xt_c + float(ridge_lambda) * np.eye(Xt_c.shape[1])
        W = np.linalg.solve(A, Xt_c.T @ Yt_c)
        bias = y_mean - x_mean @ W
        return cls(
            W=W, bias=bias, ridge_lambda=float(ridge_lambda), use_noisy=use_noisy,
            markers=markers, mesh=mesh, uv0_cache=uv0,
            meta={"x_mean": x_mean, "y_mean": y_mean, "n_train": int(train_mask.sum())},
        )

    def predict(self, vds: VisionDataset, indices: np.ndarray | None = None) -> np.ndarray:
        idx = np.arange(len(vds)) if indices is None else np.asarray(indices, int)
        uv0 = self.uv0_cache if self.uv0_cache is not None else undeformed_marker_pixels(vds, self.markers, self.mesh)
        px = vds.marker_px if self.use_noisy else vds.marker_px_clean
        X = marker_delta_features(px[idx], uv0[idx], vds.marker_visible[idx])
        return X @ self.W + self.bias

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, W=self.W, bias=self.bias, ridge_lambda=np.array(self.ridge_lambda),
            use_noisy=np.array(self.use_noisy), meta_n_train=np.array(self.meta.get("n_train", -1)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path, markers: SurfaceMarkers | None = None, mesh=None) -> "RidgeMarkerModel":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                W=z["W"], bias=z["bias"], ridge_lambda=float(z["ridge_lambda"]),
                use_noisy=bool(z["use_noisy"]), markers=markers, mesh=mesh,
                meta={"n_train": int(z["meta_n_train"])},
            )
