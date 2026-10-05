#!/usr/bin/env python
"""Stages A / B – fixed parallel-jaw grasp with compliant fingers.

A (``--mode gt``): the controller measures the *true* contact force.
B (``--mode markers``): the controller measures force from virtual markers
projected through the observation camera (pixel noise) + ROM inverse.

The jaws close, regulate to ``force_target``, then lift. The object is held
only if ``2 μ λ ≥ m g``.

Keys (plus viewer defaults):
    Space   pause / resume
    4 / 5   Stage A (GT force) / Stage B (marker estimate)
    9       lift now
    0       reset
    [ / ]   force target − / + 0.2 N

Run:
    python scripts/run_grasp_sim.py                  # interactive, Stage A
    python scripts/run_grasp_sim.py --mode markers   # Stage B
    python scripts/run_grasp_eval.py                 # A vs B batch + figures
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grasp.factory import build_grasp_sim  # noqa: E402
from src.grasp.scene import build_grasp_state  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import make_run_dir  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="grasp")
    p.add_argument("--mode", default=None, choices=["gt", "markers", "inverse"])
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    sim = build_grasp_sim(cfg, estimator_mode=args.mode)
    if args.target is not None:
        sim.force_target = float(args.target)

    from src.visualization.taichi_viewer import TaichiViewer

    run_dir = make_run_dir(args.out or cfg.get("output_dir", "results/etc/grasp"))
    viewer = TaichiViewer(show_window=not args.headless, title=f"grasp {sim.estimator.mode} – A=GT, B=markers")
    viewer.options.amplification = 1.0
    paused = {"v": False}

    def apply(first: bool = False) -> None:
        viewer.set_state(build_grasp_state(sim), refit_camera=first)

    def on_frame(v) -> None:
        if not paused["v"]:
            # a few physics steps per drawn frame
            for _ in range(2):
                sim.step()
            apply()

    def toggle_pause(_v) -> None:
        paused["v"] = not paused["v"]

    def set_mode(mode: str):
        def _fn(_v):
            sim.set_mode(mode)
            print(f"[grasp] estimator → {mode}")
        return _fn

    def bump_target(d: float):
        def _fn(_v):
            sim.force_target = max(0.2, sim.force_target + d)
            print(f"[grasp] target → {sim.force_target:.2f} N  (hold ≥ {sim.mech.hold_force:.2f} N)")
        return _fn

    viewer.register_key(" ", toggle_pause, "pause / resume")
    viewer.register_key("4", set_mode("gt"), "Stage A: GT force")
    viewer.register_key("5", set_mode("markers"), "Stage B: marker force")
    viewer.register_key("9", lambda _v: setattr(sim, "force_lift", True), "lift now")
    viewer.register_key("0", lambda _v: (sim.reset(), apply(True)), "reset grasp")
    viewer.register_key("[", bump_target(-0.2), "target −0.2 N")
    viewer.register_key("]", bump_target(+0.2), "target +0.2 N")

    apply(first=True)
    viewer.save_screenshot(run_dir / f"start_{sim.estimator.mode}.png")
    print(f"hold threshold {sim.mech.hold_force:.2f} N, target {sim.force_target:.2f} N, "
          f"k = {sim.mech.stiffness:.0f} N/m, estimator={sim.estimator.mode}")
    if args.headless:
        log = sim.run(t_max=8.0)
        apply()
        viewer.save_screenshot(run_dir / f"end_{sim.estimator.mode}.png")
        print(f"phase={sim.state.phase} held={sim.state.held} success={sim.success()}  → {run_dir}")
        viewer.destroy()
    else:
        viewer.run(on_frame=on_frame)


if __name__ == "__main__":
    main()
