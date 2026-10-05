#!/usr/bin/env python
"""Milestone 7 – inspect the end-to-end pipeline in the Taichi viewer.

Shows the full chain on each sample:
  observation image (inset) → q̂ → Φ q̂ (toggle A) → estimated force
  with optional unknown-contact localization (toggle U).

Keys (plus viewer defaults):
    Left / Right   previous / next (±10 with Shift)     Space   random
    D              cycle estimator (ridge / geometric / iterative / robust / cnn / oracle)
    U              toggle known ↔ unknown contact
    Y              noisy ↔ clean markers (marker estimators)
    Home / End     first / last

Run:
    python scripts/inspect_e2e.py
    python scripts/inspect_e2e.py --estimator ridge --sample 200 --headless
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

from src.evaluation.e2e import build_e2e_pipeline, build_q_estimator  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402


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


class E2EInspector:
    def __init__(self, cfg, pipe, split, viewer, start: int, estimators: list[str]):
        self.cfg, self.pipe, self.split, self.v = cfg, pipe, split, viewer
        self.names = estimators
        self.mi = 0
        self.i = start % len(pipe.vds)
        self.known_contact = True
        self.use_noisy = bool(cfg.get("vision", {}).get("use_noisy_markers", True))
        self.rng = np.random.default_rng(0)
        self.p0 = pipe.markers.positions(pipe.model.mesh, None)
        self._install_keys()
        self._rebuild_estimator(self.names[0])
        self.apply(first=True)

    def _install_keys(self) -> None:
        import taichi as ti

        v = self.v
        shift = lambda: v.window.is_pressed(ti.ui.SHIFT)  # noqa: E731
        v.register_key(ti.ui.RIGHT, lambda _: self.step(10 if shift() else 1), "next")
        v.register_key(ti.ui.LEFT, lambda _: self.step(-10 if shift() else -1), "previous")
        v.register_key("Home", lambda _: self.goto(0), "first")
        v.register_key("End", lambda _: self.goto(len(self.pipe.vds) - 1), "last")
        v.register_key(ti.ui.SPACE, lambda _: self.goto(int(self.rng.integers(len(self.pipe.vds)))), "random")
        v.register_key("d", lambda _: self.cycle_estimator(), "cycle estimator")
        v.register_key("u", lambda _: self.toggle_contact_mode(), "toggle known/unknown contact")
        v.register_key("y", lambda _: self.toggle_noise(), "noisy/clean markers")

    def _rebuild_estimator(self, name: str) -> None:
        q_fn, model = build_q_estimator(
            name, self.pipe.vds, self.split.train, self.pipe.markers, self.pipe.model.mesh, self.pipe.basis,  # type: ignore[arg-type]
            use_noisy=self.use_noisy,
            ridge_lambda=float(self.cfg.get("vision", {}).get("ridge_lambda", 1e-2)),
            geometric_ridge=float(self.cfg.get("vision", {}).get("geometric_ridge", 1e-4)),
            checkpoint_dir=self.cfg.get("checkpoint_dir"),
        )
        self.pipe.q_predict = q_fn
        self.pipe.estimator_name = name
        self._est_model = model

    def cycle_estimator(self) -> None:
        self.mi = (self.mi + 1) % len(self.names)
        try:
            self._rebuild_estimator(self.names[self.mi])
        except FileNotFoundError as e:
            print(e)
            self.mi = (self.mi + 1) % len(self.names)
            self._rebuild_estimator(self.names[self.mi])
        self.apply()

    def toggle_contact_mode(self) -> None:
        if self.pipe.localizer is None:
            print("unknown-contact localizer disabled")
            return
        self.known_contact = not self.known_contact
        self.apply()

    def toggle_noise(self) -> None:
        self.use_noisy = not self.use_noisy
        self._rebuild_estimator(self.names[self.mi])
        self.apply()

    def step(self, d: int) -> None:
        self.goto(self.i + d)

    def goto(self, i: int) -> None:
        self.i = int(i) % len(self.pipe.vds)
        self.apply()

    def apply(self, first: bool = False) -> None:
        s = self.i
        pred = self.pipe.predict_one(s, known_contact=self.known_contact)
        q_true = self.pipe.vds.q[s]
        u_true = self.pipe.fem.displacement(int(self.pipe.vds.fem_index[s]))
        q_rel = float(np.linalg.norm(pred.q_hat - q_true) / max(np.linalg.norm(q_true), 1e-30))
        u_rel = float(np.linalg.norm(pred.u_hat - u_true) / max(np.linalg.norm(u_true), 1e-30))
        f_true = self.pipe.vds.force_magnitude[s]
        f_rel = abs(pred.force_magnitude - f_true) / max(f_true, 1e-30)
        pos_err = float(np.linalg.norm(pred.contact.position - self.pipe.vds.contact_position[s]))
        hold = "TEST" if self.split.test[s] else "train"
        info = {
            "e2e sample": f"{s} / {len(self.pipe.vds) - 1}  [{hold}]",
            "estimator (D)": f"{self.pipe.estimator_name}   markers={'noisy' if self.use_noisy else 'clean'} (Y)",
            "contact mode (U)": "known GT location" if self.known_contact else f"unknown search ({len(self.pipe.localizer)} cands)",
            "q / u error": f"{100 * q_rel:.2f} % / {100 * u_rel:.2f} %",
            "force": f"true {f_true:.3f} N, est {pred.force_magnitude:.3f} N, rel {100 * f_rel:.1f} %",
            "contact err": f"{1e3 * pos_err:.2f} mm   residual {pred.residual_rel:.3f}   {1e3 * pred.time_s:.1f} ms",
        }
        state = self.pipe.fem.to_visualization_state(
            int(self.pipe.vds.fem_index[s]), estimated_force=pred.force_vector, extra_info=info,
        )
        # show localized contact when unknown
        if not self.known_contact:
            from src.visualization.state import ContactPoint, ForceVector

            state.contact_points = [ContactPoint(pred.contact.position, normal=pred.contact.normal)]
            state.estimated_force = ForceVector(pred.contact.position, pred.force_vector, label="F_est")
        state.displacement_alt = pred.u_hat
        state.displacement_alt_name = f"e2e/{self.pipe.estimator_name}"
        state.observation_camera = self.pipe.vds.camera(s)
        state.keypoints = self.p0
        state.keypoints_visible = self.pipe.vds.marker_visible[s]
        self.v.options.use_alt_displacement = False
        self.v.set_state(state, refit_camera=first)
        img = self.pipe.vds.image_float(s)
        px = self.pipe.vds.marker_px[s] if self.use_noisy else self.pipe.vds.marker_px_clean[s]
        self.v.set_overlay_image(paint_markers(img, px, self.pipe.vds.marker_visible[s]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="e2e")
    p.add_argument("--estimator", default="ridge")
    p.add_argument("--sample", type=int, default=-1)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--out", default="results/etc/e2e_inspect")
    args = p.parse_args()

    cfg = load_config(args.config)
    pipe, split = build_e2e_pipeline(cfg, estimator=args.estimator)
    estimators = [args.estimator] + [n for n in ("ridge", "geometric", "iterative", "robust", "cnn", "oracle") if n != args.estimator]

    from src.visualization.taichi_viewer import TaichiViewer

    run_dir = make_run_dir(args.out)
    viewer = TaichiViewer(show_window=not args.headless, title="E2E – observation → q → Φq → force")
    viewer.options.amplification = 1.0
    insp = E2EInspector(cfg, pipe, split, viewer, args.sample, estimators)
    pred = pipe.predict_one(insp.i, known_contact=True)
    save_json({
        "sample": insp.i, "estimator": pipe.estimator_name,
        "force_true": float(pipe.vds.force_magnitude[insp.i]), "force_est": pred.force_magnitude,
        "q_hat": pred.q_hat.tolist(),
    }, run_dir / "sample.json")
    viewer.save_screenshot(run_dir / "debug_view.png")
    viewer.options.use_alt_displacement = True
    insp.apply()
    viewer.save_screenshot(run_dir / "debug_view_pred.png")
    if pipe.localizer is not None:
        insp.known_contact = False
        insp.apply()
        viewer.save_screenshot(run_dir / "debug_view_unknown.png")
        insp.known_contact = True
    print(f"figures -> {run_dir}")
    if not args.headless:
        viewer.run()
    else:
        test_idx = np.nonzero(split.test)[0]
        for s in test_idx[:: max(1, len(test_idx) // 3)][:3]:
            insp.goto(int(s))
            viewer.save_screenshot(run_dir / f"sample_{s}.png")
        viewer.destroy()


if __name__ == "__main__":
    main()
