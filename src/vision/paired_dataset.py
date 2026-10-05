"""Paired marker-on / marker-free vision dataset (markers as training-time privileged information).

For every rendered sample the *same* deformed state, camera, lighting and
image-noise realisation is rendered twice:

    images_marker  (S, H, W, C) uint8   with marker discs   (teacher modality)
    images_raw     (S, H, W, C) uint8   without markers     (deployment modality)

Two reduced-coordinate labels are stored and kept distinct:

    q_sim     (S, r)  exact  Φ_rᵀ u  of the FEM field           ("simulation ground truth")
    q_marker  (S, r)  q recovered by the existing marker pipeline from the
                      (noisy) marker pixels of ``images_marker``  ("marker teacher")

Everything else mirrors :class:`~src.vision.dataset.VisionDataset` so the
existing splits / ROM / unknown-contact tooling can be reused through
:meth:`PairedVisionDataset.as_vision_dataset`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .dataset import VisionDataset

ImageKind = Literal["raw", "marker"]
QKind = Literal["sim", "marker"]


@dataclass
class PairedVisionDataset:
    images_marker: np.ndarray
    images_raw: np.ndarray
    marker_px: np.ndarray
    marker_px_clean: np.ndarray
    marker_visible: np.ndarray
    q_sim: np.ndarray
    q_marker: np.ndarray
    force_magnitude: np.ndarray
    force_vector: np.ndarray
    contact_position: np.ndarray
    contact_normal: np.ndarray
    camera_K: np.ndarray
    camera_T_cw: np.ndarray
    camera_position: np.ndarray
    camera_fov_y_deg: np.ndarray
    fem_index: np.ndarray
    appearance: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.images_marker.shape != self.images_raw.shape:
            raise ValueError("images_marker and images_raw must have identical shapes (paired samples)")
        if self.q_sim.shape != self.q_marker.shape:
            raise ValueError("q_sim and q_marker must have identical shapes")

    def __len__(self) -> int:
        return int(self.images_raw.shape[0])

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.images_raw.shape[1:])

    @property
    def n_markers(self) -> int:
        return int(self.marker_px.shape[1])

    @property
    def r(self) -> int:
        return int(self.q_sim.shape[1])

    # ------------------------------------------------------------ selectors
    def images(self, kind: ImageKind) -> np.ndarray:
        if kind == "raw":
            return self.images_raw
        if kind == "marker":
            return self.images_marker
        raise ValueError(f"image kind must be 'raw' or 'marker', got {kind!r}")

    def q(self, kind: QKind) -> np.ndarray:
        if kind == "sim":
            return self.q_sim
        if kind == "marker":
            return self.q_marker
        raise ValueError(f"q kind must be 'sim' or 'marker', got {kind!r}")

    def as_vision_dataset(self, images: ImageKind = "raw", q: QKind = "sim") -> VisionDataset:
        """Shallow :class:`VisionDataset` view (arrays shared, not copied).

        ``q`` selects which label populates ``VisionDataset.q`` so existing
        evaluation code (which reads ``vds.q`` as truth) can be pointed at
        either the simulation ground truth or the marker teacher explicitly.
        """
        return VisionDataset(
            images=self.images(images), marker_px=self.marker_px, marker_px_clean=self.marker_px_clean,
            marker_visible=self.marker_visible, q=self.q(q),
            force_magnitude=self.force_magnitude, force_vector=self.force_vector,
            contact_position=self.contact_position, contact_normal=self.contact_normal,
            camera_K=self.camera_K, camera_T_cw=self.camera_T_cw, camera_position=self.camera_position,
            camera_fov_y_deg=self.camera_fov_y_deg, fem_index=self.fem_index,
            appearance=self.appearance, meta={**self.meta, "paired_view": {"images": images, "q": q}},
        )

    def subset(self, indices: np.ndarray) -> "PairedVisionDataset":
        """Row subset (copies the per-sample arrays, shares ``meta``)."""
        idx = np.asarray(indices, int)
        kw = {}
        for name in self.__dataclass_fields__:
            v = getattr(self, name)
            if name == "appearance":
                kw[name] = [v[int(i)] for i in idx] if len(v) else []
            elif isinstance(v, np.ndarray):
                kw[name] = v[idx]
            else:
                kw[name] = v
        return PairedVisionDataset(**kw)

    def summary(self) -> dict[str, Any]:
        q_rel = np.linalg.norm(self.q_marker - self.q_sim, axis=1) / np.maximum(np.linalg.norm(self.q_sim, axis=1), 1e-30)
        return {
            "n_samples": len(self), "image_shape": self.image_shape, "n_markers": self.n_markers, "q_modes": self.r,
            "force_range_N": [float(self.force_magnitude.min()), float(self.force_magnitude.max())],
            "visible_marker_fraction": float(self.marker_visible.mean()),
            "n_fem_samples": int(len(np.unique(self.fem_index))),
            "q_marker_vs_sim_rel_median": float(np.median(q_rel)),
            "size_MB": (self.images_raw.nbytes + self.images_marker.nbytes) / 1e6,
        }

    # ------------------------------------------------------------ io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, images_marker=self.images_marker, images_raw=self.images_raw,
            marker_px=self.marker_px, marker_px_clean=self.marker_px_clean, marker_visible=self.marker_visible,
            q_sim=self.q_sim, q_marker=self.q_marker,
            force_magnitude=self.force_magnitude, force_vector=self.force_vector,
            contact_position=self.contact_position, contact_normal=self.contact_normal,
            camera_K=self.camera_K, camera_T_cw=self.camera_T_cw, camera_position=self.camera_position,
            camera_fov_y_deg=self.camera_fov_y_deg, fem_index=self.fem_index,
            appearance=np.array(json.dumps(self.appearance)), meta=np.array(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PairedVisionDataset":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                images_marker=z["images_marker"], images_raw=z["images_raw"],
                marker_px=z["marker_px"], marker_px_clean=z["marker_px_clean"], marker_visible=z["marker_visible"],
                q_sim=z["q_sim"], q_marker=z["q_marker"],
                force_magnitude=z["force_magnitude"], force_vector=z["force_vector"],
                contact_position=z["contact_position"], contact_normal=z["contact_normal"],
                camera_K=z["camera_K"], camera_T_cw=z["camera_T_cw"], camera_position=z["camera_position"],
                camera_fov_y_deg=z["camera_fov_y_deg"], fem_index=z["fem_index"],
                appearance=json.loads(str(z["appearance"])), meta=json.loads(str(z["meta"])),
            )

    def camera(self, i: int):
        return self.as_vision_dataset().camera(i)
