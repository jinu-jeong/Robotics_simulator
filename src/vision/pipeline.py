"""Vision → q → ROM force pipeline and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..contact.contact_mapping import PointContactMapping, SurfaceContact
from ..rom.pod import PODBasis
from ..rom.reduced_mechanics import ReducedModel
from .dataset import VisionDataset


@dataclass
class SamplePrediction:
    q_hat: np.ndarray
    u_hat: np.ndarray  # (N, 3)
    force_vector: np.ndarray
    force_magnitude: float
    contact: SurfaceContact
    extra: dict = field(default_factory=dict)


@dataclass
class VisionMechanicsPipeline:
    """q_estimator(sample_index) → q̂; reconstruct û = Φ q̂; estimate force at known contact."""

    basis: PODBasis
    rom: ReducedModel
    q_predict: Callable[[np.ndarray], np.ndarray]  # indices -> (len, r)
    force_method: str = "displacement"
    force_mode: str = "normal"
    nonneg: bool = True
    name: str = "vision"

    def predict_q(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.q_predict(np.asarray(indices, int)), float)

    def contact_of(self, vds: VisionDataset, s: int) -> SurfaceContact:
        return self.rom.mapping.locate(vds.contact_position[s])

    def predict_one(self, vds: VisionDataset, s: int) -> SamplePrediction:
        q = self.predict_q(np.array([s]))[0]
        c = self.contact_of(vds, s)
        est = self.rom.estimate_force(q, c, method=self.force_method, mode=self.force_mode, nonneg=self.nonneg)
        return SamplePrediction(q, self.basis.reconstruct(q), est.force_vector, est.magnitude, c, extra={"residual": est.residual_rel})

    def evaluate(self, vds: VisionDataset, indices: np.ndarray, fem_U: np.ndarray | None = None) -> dict[str, Any]:
        """Compute q / displacement / force errors on ``indices``.

        ``fem_U`` optional (len(vds unique fem) or full contact-sweep U) – if given as
        the FEM dataset's U with rows indexed by ``vds.fem_index``, displacement RMSE
        is reported against the true field.
        """
        idx = np.asarray(indices, int)
        q_hat = self.predict_q(idx)
        q_true = vds.q[idx]
        q_rel = np.linalg.norm(q_hat - q_true, axis=1) / np.maximum(np.linalg.norm(q_true, axis=1), 1e-30)
        q_rmse = float(np.sqrt(np.mean((q_hat - q_true) ** 2)))

        F_hat = np.zeros((len(idx), 3))
        for i, s in enumerate(idx):
            c = self.contact_of(vds, int(s))
            est = self.rom.estimate_force(q_hat[i], c, method=self.force_method, mode=self.force_mode, nonneg=self.nonneg)
            F_hat[i] = est.force_vector
        F_true = vds.force_vector[idx]
        mag_hat = np.linalg.norm(F_hat, axis=1)
        mag_true = vds.force_magnitude[idx]
        force_rel = np.abs(mag_hat - mag_true) / np.maximum(mag_true, 1e-30)
        force_mae = float(np.mean(np.abs(mag_hat - mag_true)))
        force_rmse = float(np.sqrt(np.mean((mag_hat - mag_true) ** 2)))
        dir_cos = np.sum(F_hat * F_true, axis=1) / np.maximum(mag_hat * np.linalg.norm(F_true, axis=1), 1e-30)

        out: dict[str, Any] = {
            "n": int(len(idx)),
            "q_rel_mean": float(q_rel.mean()),
            "q_rel_median": float(np.median(q_rel)),
            "q_rmse": q_rmse,
            "force_rel_mean": float(force_rel.mean()),
            "force_rel_median": float(np.median(force_rel)),
            "force_rel_p90": float(np.percentile(force_rel, 90)),
            "force_mae_N": force_mae,
            "force_rmse_N": force_rmse,
            "force_rel_by_sample": force_rel,
            "q_rel_by_sample": q_rel,
            "mag_hat": mag_hat,
            "mag_true": mag_true,
            "direction_cos_mean": float(dir_cos.mean()),
            "q_hat": q_hat,
            "F_hat": F_hat,
        }
        if fem_U is not None:
            # fem_U: (n_fem, 3N) or (n_fem, N, 3)
            U = np.asarray(fem_U)
            if U.ndim == 3:
                U = U.reshape(U.shape[0], -1)
            u_err = []
            for i, s in enumerate(idx):
                u_t = U[int(vds.fem_index[s])]
                u_h = self.basis.reconstruct(q_hat[i]).reshape(-1)
                u_err.append(np.linalg.norm(u_h - u_t) / max(np.linalg.norm(u_t), 1e-30))
            out["u_rel_mean"] = float(np.mean(u_err))
            out["u_rel_median"] = float(np.median(u_err))
        return out


def build_rom(model, basis: PODBasis, contact_face: str = "top") -> ReducedModel:
    from ..rendering.markers import finger_face_ids

    mapping = PointContactMapping(model.mesh, finger_face_ids(model.mesh, model.geometry, contact_face))
    return ReducedModel.from_fem(model.fem, basis, mapping)
