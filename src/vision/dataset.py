"""Synthetic vision dataset: rendered observation-camera images with mechanics labels.

Arrays (S rendered samples, M markers, r reduced coordinates)
------------------------------------------------------------
images          (S, H, W, C) uint8         C = 3 (RGB) or 1 (grayscale)
marker_px       (S, M, 2) float32          pixel coordinates (u right, v down), noisy if configured
marker_px_clean (S, M, 2) float32          noise-free projections
marker_visible  (S, M) bool
q               (S, r) float64             reduced coordinates Φ_rᵀ u of the *exact* field
force_magnitude (S,), force_vector (S, 3)  [N]
contact_position (S, 3), contact_normal (S, 3)
camera_K (S, 3, 3), camera_T_cw (S, 4, 4), camera_position (S, 3), camera_fov_y_deg (S,)
fem_index       (S,) int64                 index of the underlying sample in the FEM dataset (-> u)
appearance      json list                  per-sample nuisance parameters
meta            json                       config snapshot, marker definition, basis path, ...

The full displacement field is *not* duplicated here; ``fem_index`` refers
back to the contact-sweep dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VisionDataset:
    images: np.ndarray
    marker_px: np.ndarray
    marker_px_clean: np.ndarray
    marker_visible: np.ndarray
    q: np.ndarray
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

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.images.shape[1:])

    @property
    def n_markers(self) -> int:
        return int(self.marker_px.shape[1])

    def image_float(self, i: int) -> np.ndarray:
        return self.images[i].astype(np.float32) / 255.0

    def summary(self) -> dict[str, Any]:
        return {
            "n_samples": len(self), "image_shape": self.image_shape, "n_markers": self.n_markers,
            "q_modes": int(self.q.shape[1]), "force_range_N": [float(self.force_magnitude.min()), float(self.force_magnitude.max())],
            "visible_marker_fraction": float(self.marker_visible.mean()),
            "n_fem_samples": int(len(np.unique(self.fem_index))), "size_MB": self.images.nbytes / 1e6,
        }

    # ------------------------------------------------------------ io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, images=self.images, marker_px=self.marker_px, marker_px_clean=self.marker_px_clean,
            marker_visible=self.marker_visible, q=self.q, force_magnitude=self.force_magnitude, force_vector=self.force_vector,
            contact_position=self.contact_position, contact_normal=self.contact_normal, camera_K=self.camera_K,
            camera_T_cw=self.camera_T_cw, camera_position=self.camera_position, camera_fov_y_deg=self.camera_fov_y_deg,
            fem_index=self.fem_index, appearance=np.array(json.dumps(self.appearance)), meta=np.array(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VisionDataset":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                images=z["images"], marker_px=z["marker_px"], marker_px_clean=z["marker_px_clean"], marker_visible=z["marker_visible"],
                q=z["q"], force_magnitude=z["force_magnitude"], force_vector=z["force_vector"],
                contact_position=z["contact_position"], contact_normal=z["contact_normal"], camera_K=z["camera_K"],
                camera_T_cw=z["camera_T_cw"], camera_position=z["camera_position"], camera_fov_y_deg=z["camera_fov_y_deg"],
                fem_index=z["fem_index"], appearance=json.loads(str(z["appearance"])), meta=json.loads(str(z["meta"])),
            )

    def camera(self, i: int):
        """Rebuild the :class:`ObservationCamera` of sample ``i``."""
        from ..visualization.camera import ObservationCamera

        c = self.meta["cameras"][i] if "cameras" in self.meta and i < len(self.meta["cameras"]) else None
        if c is not None:
            return ObservationCamera.from_config(c)
        T = self.camera_T_cw[i]
        R_wc = T[:3, :3].T
        pos = -R_wc @ T[:3, 3]
        fwd = R_wc[:, 2]
        H, W = self.image_shape[:2]
        return ObservationCamera(position=pos, target=pos + fwd * 0.1, up=-R_wc[:, 1], fov_y_deg=float(self.camera_fov_y_deg[i]),
                                 image_width=W, image_height=H)
