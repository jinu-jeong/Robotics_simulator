#!/usr/bin/env python
"""Milestone 6 – inspect vision → q → force predictions in the Taichi viewer.

Shows, for each sample:
* GT deformation vs vision-reconstructed Φ q̂ (toggle ``A``)
* GT force (green) vs ROM force from q̂ (magenta)
* observation-camera image as picture-in-picture inset
* q / force error in the info panel

Keys (plus viewer defaults):
    Left / Right   previous / next (±10 with Shift)     Space   random sample
    D              cycle estimator (geometric / ridge / cnn / oracle)
    Y              toggle noisy / clean markers for geometric & ridge
    Home / End     first / last

Run:
    python scripts/inspect_vision_model.py
    python scripts/inspect_vision_model.py --model ridge --sample 100 --headless
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fem.contact_dataset import ContactDataset  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.rendering.markers import SurfaceMarkers  # noqa: E402
from src.rom.pod import PODBasis  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402
from src.vision.dataset import VisionDataset  # noqa: E402
from src.vision.marker_model import GeometricMarkerModel, RidgeMarkerModel  # noqa: E402
from src.vision.pipeline import VisionMechanicsPipeline, build_rom  # noqa: E402
from src.vision.splits import make_vision_split  # noqa: E402


def paint_markers(img: np.ndarray, px: np.ndarray, visible: np.ndarray, size: int = 1) -> np.ndarray:
    out = np.array(img, dtype=np.float32, copy=True)
    if out.ndim == 2 or out.shape[-1] == 1:
        out = np.repeat(out.reshape(out.shape[0], out.shape[1], 1), 3, axis=2)
    H, W = out.shape[:2]
    for (u, v), vis in zip(px, visible):
        i, j = int(np.floor(u)), int(np.floor(v))
        if not (0 <= i < W and 0 <= j < H):
            continue
        c = (0.0, 1.0, 1.0) if vis else (0.55, 0.55, 0.55)
        out[j, max(i - size, 0) : i + size + 1] = c
        out[max(j - size, 0) : j + size + 1, i] = c
    return out


class VisionModelInspector:
    def __init__(self, vds, fem, markers, mesh, basis, rom, estimators: dict, viewer, start: int, force_cfg: dict):
        self.vds, self.fem, self.markers, self.mesh = vds, fem, markers, mesh
        self.basis, self.rom, self.estimators, self.v = basis, rom, estimators, viewer
        self.names = list(estimators.keys())
        self.mi = 0
        self.i = start % len(vds)
        self.use_noisy = True
        self.force_cfg = force_cfg
        self.rng = np.random.default_rng(0)
        self.p0 = markers.positions(mesh, None)
        self._install_keys()
        self.apply(first=True)

    def _install_keys(self) -> None:
        import taichi as ti

        v = self.v
        shift = lambda: v.window.is_pressed(ti.ui.SHIFT)  # noqa: E731
        v.register_key(ti.ui.RIGHT, lambda _: self.step(10 if shift() else 1), "next sample")
        v.register_key(ti.ui.LEFT, lambda _: self.step(-10 if shift() else -1), "previous sample")
        v.register_key("Home", lambda _: self.goto(0), "first")
        v.register_key("End", lambda _: self.goto(len(self.vds) - 1), "last")
        v.register_key(ti.ui.SPACE, lambda _: self.goto(int(self.rng.integers(len(self.vds)))), "random")
        v.register_key("d", lambda _: self.cycle_model(), "cycle vision estimator")
        v.register_key("y", lambda _: self.toggle_noise(), "toggle noisy/clean markers")

    def cycle_model(self) -> None:
        self.mi = (self.mi + 1) % len(self.names)
        self.apply()

    def toggle_noise(self) -> None:
        self.use_noisy = not self.use_noisy
        for est in self.estimators.values():
            if hasattr(est, "use_noisy"):
                est.use_noisy = self.use_noisy
        self.apply()

    def step(self, d: int) -> None:
        self.goto(self.i + d)

    def goto(self, i: int) -> None:
        self.i = int(i) % len(self.vds)
        self.apply()

    def _q_hat(self, s: int) -> np.ndarray:
        name = self.names[self.mi]
        est = self.estimators[name]
        if name == "oracle":
            return self.vds.q[s]
        if name == "cnn":
            return est.predict(self.vds.images[s : s + 1])[0]
        if name == "ridge":
            return est.predict(self.vds, np.array([s]))[0]
        # geometric
        px = self.vds.marker_px[s] if self.use_noisy else self.vds.marker_px_clean[s]
        return est.predict_one(self.vds.camera(s), px, self.vds.marker_visible[s])

    def apply(self, first: bool = False) -> None:
        s = self.i
        name = self.names[self.mi]
        q_hat = self._q_hat(s)
        q_true = self.vds.q[s]
        u_hat = self.basis.reconstruct(q_hat)
        c = self.rom.mapping.locate(self.vds.contact_position[s])
        est = self.rom.estimate_force(
            q_hat, c, method=self.force_cfg.get("method", "displacement"),
            mode=self.force_cfg.get("mode", "normal"), nonneg=bool(self.force_cfg.get("nonneg", True)),
        )
        q_rel = float(np.linalg.norm(q_hat - q_true) / max(np.linalg.norm(q_true), 1e-30))
        f_rel = abs(est.magnitude - self.vds.force_magnitude[s]) / max(self.vds.force_magnitude[s], 1e-30)
        u_true = self.fem.displacement(int(self.vds.fem_index[s]))
        u_rel = float(np.linalg.norm(u_hat - u_true) / max(np.linalg.norm(u_true), 1e-30))

        info = {
            "vision sample": f"{s} / {len(self.vds) - 1}  (FEM {int(self.vds.fem_index[s])})",
            "estimator (D)": f"{name}   markers={'noisy' if self.use_noisy else 'clean'} (Y)",
            "q error": f"{100 * q_rel:.2f} %   ‖q̂‖={np.linalg.norm(q_hat):.3e}",
            "u error (Φq̂ vs FEM)": f"{100 * u_rel:.2f} %",
            "force": f"true {self.vds.force_magnitude[s]:.3f} N, est {est.magnitude:.3f} N, rel {100 * f_rel:.1f} %",
            "q̂ (first 4)": "  ".join(f"{x:+.2e}" for x in q_hat[:4]),
        }
        state = self.fem.to_visualization_state(int(self.vds.fem_index[s]), estimated_force=est.force_vector, extra_info=info)
        state.displacement_alt = u_hat
        state.displacement_alt_name = f"vision/{name}"
        state.observation_camera = self.vds.camera(s)
        vis = self.vds.marker_visible[s]
        state.keypoints = self.p0
        state.keypoints_visible = vis
        self.v.options.use_alt_displacement = False
        self.v.set_state(state, refit_camera=first)
        img = self.vds.image_float(s)
        px = self.vds.marker_px[s] if self.use_noisy else self.vds.marker_px_clean[s]
        self.v.set_overlay_image(paint_markers(img, px, vis))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="vision_model")
    p.add_argument("--model", default="ridge", help="initial estimator")
    p.add_argument("--sample", type=int, default=-1)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--out", default="results/etc/vision_model_inspect")
    args = p.parse_args()

    cfg = load_config(args.config)
    vds = VisionDataset.load(cfg["vision_dataset"])
    fem = ContactDataset.load(cfg.get("fem_dataset") or vds.meta["fem_dataset"])
    model = FingerFEMModel.from_config(fem.meta["fem_config"])
    markers = SurfaceMarkers.from_dict(vds.meta["markers"])
    r = int(cfg.get("q_modes", vds.q.shape[1]))
    basis = PODBasis.load(cfg.get("rom_basis") or vds.meta["rom_basis"]).truncate(r)
    if vds.q.shape[1] != r:
        vds.q = np.stack([basis.project(fem.displacement(int(i))) for i in vds.fem_index])
    rom = build_rom(model, basis, contact_face=str(vds.meta.get("config", {}).get("markers", {}).get("face", "top")))
    mk_cfg = cfg.get("markers", {})
    split = make_vision_split(vds, fem.contact_position, hold_every=int(cfg.get("split", {}).get("hold_every", 3)))

    estimators = {
        "geometric": GeometricMarkerModel(
            basis, markers, model.mesh,
            ridge=float(mk_cfg.get("geometric_ridge", 1e-4)),
            eps=float(mk_cfg.get("eps", 1e-4)),
            use_noisy=bool(mk_cfg.get("use_noisy", True)),
        ),
        "ridge": RidgeMarkerModel.fit(
            vds, split.train, markers, model.mesh,
            ridge_lambda=float(mk_cfg.get("ridge_lambda", 1e-2)),
            use_noisy=bool(mk_cfg.get("use_noisy", True)),
        ),
        "oracle": object(),
    }
    ckpt = Path(cfg.get("checkpoint_dir", "results/etc/checkpoints/vision_model")) / "image_cnn.pt"
    if ckpt.exists():
        try:
            from src.vision.image_model import ImageCNNModel
            estimators["cnn"] = ImageCNNModel.load(ckpt)
            print(f"loaded CNN from {ckpt}")
        except Exception as e:
            print(f"CNN checkpoint not loaded: {e}")
    else:
        # still offer cnn if we can train quickly? skip – train script owns that
        print(f"no CNN checkpoint at {ckpt} (run train_vision_model.py --models cnn)")

    # order: put requested model first
    names = [args.model] + [n for n in estimators if n != args.model]
    estimators = {n: estimators[n] for n in names if n in estimators}

    from src.visualization.taichi_viewer import TaichiViewer

    run_dir = make_run_dir(args.out)
    viewer = TaichiViewer(show_window=not args.headless, title="vision model – GT vs Φq̂ (A), force GT/est")
    viewer.options.amplification = 1.0
    insp = VisionModelInspector(vds, fem, markers, model.mesh, basis, rom, estimators, viewer, args.sample, cfg.get("force", {}))
    # quick metrics snapshot for the current sample
    pred = insp._q_hat(insp.i)
    save_json({"sample": insp.i, "model": insp.names[insp.mi], "q_hat": pred.tolist(), "q_true": vds.q[insp.i].tolist()}, run_dir / "sample.json")
    viewer.save_screenshot(run_dir / "debug_view.png")
    # also dump a second screenshot with alt displacement on
    viewer.options.use_alt_displacement = True
    insp.apply()
    viewer.save_screenshot(run_dir / "debug_view_pred.png")
    viewer.options.use_alt_displacement = False
    print(f"figures -> {run_dir}")
    if not args.headless:
        viewer.run()
    else:
        # step through a few held-out samples
        test_idx = np.nonzero(split.test)[0]
        for s in test_idx[:: max(1, len(test_idx) // 3)][:3]:
            insp.goto(int(s))
            viewer.save_screenshot(run_dir / f"sample_{s}.png")
        viewer.destroy()


if __name__ == "__main__":
    main()
