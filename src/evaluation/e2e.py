"""End-to-end pipeline: observation → q̂ → Φ q̂ → contact force (± unknown contact)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from ..contact.contact_mapping import SurfaceContact
from ..fem.contact_dataset import ContactDataset
from ..fem.finger_model import FingerFEMModel
from ..rendering.markers import SurfaceMarkers
from ..rom.pod import PODBasis
from ..rom.reduced_mechanics import ReducedModel
from ..vision.dataset import VisionDataset
from ..vision.marker_model import GeometricMarkerModel, RidgeMarkerModel, fuse_q_by_fem_index
from ..vision.pipeline import build_rom
from ..vision.splits import VisionSplit, make_vision_split
from .metrics import force_metrics, relative_errors, summarize, timing_stats
from .unknown_contact import UnknownContactLocalizer

EstimatorName = Literal["geometric", "iterative", "robust", "ridge", "cnn", "oracle"]


@dataclass
class E2EPrediction:
    q_hat: np.ndarray
    u_hat: np.ndarray
    force_vector: np.ndarray
    force_magnitude: float
    contact: SurfaceContact
    contact_known: bool
    residual_rel: float
    time_s: float
    extra: dict = field(default_factory=dict)


@dataclass
class EndToEndPipeline:
    """Packaged research pipeline used by the M7 eval / inspector."""

    basis: PODBasis
    rom: ReducedModel
    model: FingerFEMModel
    markers: SurfaceMarkers
    vds: VisionDataset
    fem: ContactDataset
    q_predict: Callable[[np.ndarray], np.ndarray]
    estimator_name: str
    localizer: UnknownContactLocalizer | None = None
    force_method: str = "displacement"
    force_mode: str = "normal"
    nonneg: bool = True

    # ------------------------------------------------------------ predict
    def predict_q(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.q_predict(np.asarray(indices, int)), float)

    def predict_one(self, s: int, *, known_contact: bool = True, q: np.ndarray | None = None) -> E2EPrediction:
        t0 = time.perf_counter()
        q_hat = self.predict_q(np.array([s]))[0] if q is None else np.asarray(q, float).reshape(-1)
        if known_contact:
            c = self.rom.mapping.locate(self.vds.contact_position[s])
            est = self.rom.estimate_force(q_hat, c, method=self.force_method, mode=self.force_mode, nonneg=self.nonneg)
            F, mag, rrel = est.force_vector, est.magnitude, est.residual_rel
        else:
            if self.localizer is None:
                raise RuntimeError("unknown-contact localizer not configured")
            loc = self.localizer.localize(q_hat)
            c, F, mag, rrel = loc.contact, loc.force_vector, loc.magnitude, loc.residual_rel
        dt = time.perf_counter() - t0
        return E2EPrediction(q_hat, self.basis.reconstruct(q_hat), F, mag, c, known_contact, rrel, dt)

    def evaluate(
        self,
        indices: np.ndarray,
        *,
        known_contact: bool = True,
        q_override: np.ndarray | None = None,
    ) -> dict[str, Any]:
        idx = np.asarray(indices, int)
        if q_override is None:
            q_hat = self.predict_q(idx)
        else:
            q_hat = np.asarray(q_override, float)
            if q_hat.shape != (len(idx), self.basis.r):
                raise ValueError("q_override shape mismatch")
        q_true = self.vds.q[idx]
        q_rel = relative_errors(q_hat, q_true)

        F_hat = np.zeros((len(idx), 3))
        mag_hat = np.zeros(len(idx))
        pos_err = np.zeros(len(idx))
        times = np.zeros(len(idx))
        for i, s in enumerate(idx):
            pred = self.predict_one(int(s), known_contact=known_contact, q=q_hat[i])
            F_hat[i] = pred.force_vector
            mag_hat[i] = pred.force_magnitude
            pos_err[i] = np.linalg.norm(pred.contact.position - self.vds.contact_position[s])
            times[i] = pred.time_s

        u_rel = []
        for i, s in enumerate(idx):
            u_t = self.fem.displacement(int(self.vds.fem_index[s])).reshape(-1)
            u_h = self.basis.reconstruct(q_hat[i]).reshape(-1)
            u_rel.append(np.linalg.norm(u_h - u_t) / max(np.linalg.norm(u_t), 1e-30))
        u_rel = np.asarray(u_rel)

        out: dict[str, Any] = {
            "n": int(len(idx)),
            "known_contact": bool(known_contact),
            "estimator": self.estimator_name,
            **summarize(q_rel, "q_rel_"),
            **summarize(u_rel, "u_rel_"),
            **force_metrics(mag_hat, self.vds.force_magnitude[idx], F_hat, self.vds.force_vector[idx]),
            **timing_stats(times),
            "q_rel_by_sample": q_rel,
            "u_rel_by_sample": u_rel,
            "contact_pos_err_m": pos_err,
            **summarize(1e3 * pos_err, "contact_err_mm_"),
            "q_hat": q_hat,
            "F_hat": F_hat,
            "indices": idx,
        }
        return out


def build_q_estimator(
    name: EstimatorName,
    vds: VisionDataset,
    train_mask: np.ndarray,
    markers: SurfaceMarkers,
    mesh,
    basis: PODBasis,
    *,
    use_noisy: bool = True,
    ridge_lambda: float = 1e-2,
    geometric_ridge: float = 1e-4,
    n_iter: int | None = None,
    huber: float | None = None,
    fuse_multiview: bool | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], Any]:
    """Return ``(indices → q)`` callable and the underlying model object.

    ``iterative`` = Gauss–Newton (2 iters) + Huber IRLS.
    ``robust``    = iterative + average ``q`` across views of the same FEM sample.
    """
    name = name.lower()  # type: ignore[assignment]
    if name == "oracle":
        return (lambda idx: vds.q[np.asarray(idx, int)]), None
    if name in ("geometric", "iterative", "robust"):
        defaults = {
            "geometric": (1, None, False),
            "iterative": (2, 1.5, False),
            "robust": (2, 1.5, True),
        }
        d_iter, d_huber, d_fuse = defaults[name]
        geo = GeometricMarkerModel(
            basis, markers, mesh, ridge=geometric_ridge, use_noisy=use_noisy,
            n_iter=int(d_iter if n_iter is None else n_iter),
            huber=(d_huber if huber is None else huber),
        )
        do_fuse = d_fuse if fuse_multiview is None else bool(fuse_multiview)

        def _pred(idx, g=geo, fuse=do_fuse):
            q = g.predict(vds, idx)
            if fuse:
                q = fuse_q_by_fem_index(q, vds.fem_index[np.asarray(idx, int)])
            return q

        return _pred, geo
    if name == "ridge":
        ridge = RidgeMarkerModel.fit(vds, train_mask, markers, mesh, ridge_lambda=ridge_lambda, use_noisy=use_noisy, q=vds.q)
        if checkpoint_dir is not None:
            ridge.save(Path(checkpoint_dir) / "ridge_marker.npz")
        return (lambda idx, m=ridge: m.predict(vds, idx)), ridge
    if name == "cnn":
        from ..vision.image_model import ImageCNNModel

        ckpt = Path(checkpoint_dir or "results/etc/checkpoints/vision_model") / "image_cnn.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"CNN checkpoint not found: {ckpt} (run train_vision_model.py --models cnn)")
        cnn = ImageCNNModel.load(ckpt)
        return (lambda idx, m=cnn: m.predict_dataset(vds, idx)), cnn
    raise ValueError(f"unknown estimator {name!r}")


def build_e2e_pipeline(cfg: dict[str, Any], estimator: EstimatorName | None = None) -> tuple[EndToEndPipeline, VisionSplit]:
    """Construct the packaged pipeline from ``configs/e2e.yaml`` (or compatible dict)."""
    vds = VisionDataset.load(cfg["vision_dataset"])
    fem = ContactDataset.load(cfg.get("fem_dataset") or vds.meta["fem_dataset"])
    model = FingerFEMModel.from_config(fem.meta["fem_config"])
    face = str(cfg.get("unknown_contact", {}).get("face") or vds.meta.get("config", {}).get("markers", {}).get("face", "top"))
    model.restrict_contact_surface(face)
    markers = SurfaceMarkers.from_dict(vds.meta["markers"])
    r = int(cfg.get("q_modes", vds.q.shape[1]))
    basis = PODBasis.load(cfg.get("rom_basis") or vds.meta["rom_basis"]).truncate(r)
    if vds.q.shape[1] != r:
        vds.q = np.stack([basis.project(fem.displacement(int(i))) for i in vds.fem_index])
    rom = build_rom(model, basis, face)
    split_cfg = cfg.get("split", {})
    split = make_vision_split(
        vds, fem.contact_position,
        hold_every=int(split_cfg.get("hold_every", 3)),
        hold_force_above=split_cfg.get("hold_force_above", None),
    )
    vis = cfg.get("vision", {})
    name: EstimatorName = (estimator or vis.get("estimator", "ridge"))  # type: ignore[assignment]
    q_fn, _model = build_q_estimator(
        name, vds, split.train, markers, model.mesh, basis,
        use_noisy=bool(vis.get("use_noisy_markers", True)),
        ridge_lambda=float(vis.get("ridge_lambda", 1e-2)),
        geometric_ridge=float(vis.get("geometric_ridge", 1e-4)),
        n_iter=vis.get("n_iter"),
        huber=vis.get("huber_px"),
        fuse_multiview=vis.get("fuse_multiview"),
        checkpoint_dir=cfg.get("checkpoint_dir"),
    )
    force = cfg.get("force", {})
    localizer = None
    uc = cfg.get("unknown_contact", {})
    if uc.get("enabled", True):
        localizer = UnknownContactLocalizer.from_rom(
            rom, model.surface_face_ids(face),
            barycentric=uc.get("barycentric"),
            face_stride=int(uc.get("face_stride", 2)),
            min_influence=float(uc.get("min_influence", 1e-8)),
            nonneg=bool(force.get("nonneg", True)),
            r_loc=uc.get("r_loc", 4),
            refine=bool(uc.get("refine", True)),
            refine_n=int(uc.get("refine_n", 7)),
            weight=str(uc.get("weight", "leading")),
        )
    pipe = EndToEndPipeline(
        basis=basis, rom=rom, model=model, markers=markers, vds=vds, fem=fem,
        q_predict=q_fn, estimator_name=str(name), localizer=localizer,
        force_method=str(force.get("method", "displacement")),
        force_mode=str(force.get("mode", "normal")),
        nonneg=bool(force.get("nonneg", True)),
    )
    return pipe, split
