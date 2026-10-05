#!/usr/bin/env python
"""Batch A vs B grasp: force tracking, lift success, and viewer screenshots."""

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

from src.grasp.factory import build_grasp_sim  # noqa: E402
from src.grasp.scene import build_grasp_state  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402


def _plot(run_dir: Path, logs: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)
    for name, log in logs.items():
        t = np.asarray(log["t"])
        ax[0].plot(t, log["lam_true"], "-", label=f"{name} true")
        if name != "gt":
            ax[0].plot(t, log["lam_meas"], "--", label=f"{name} meas")
        ax[1].plot(t, 1e3 * np.asarray(log["opening"]), label=name)
        ax[2].plot(t, 1e3 * np.asarray(log["obj_z"]), label=name)
    ax[0].axhline(logs["gt"]["hold_force"][0], color="r", ls=":", lw=1, label="hold threshold")
    ax[0].axhline(logs["gt"]["force_target"][0], color="k", ls="--", lw=1, label="target")
    ax[0].set(ylabel="normal force [N]", title="Stages A / B: force tracking then lift")
    ax[1].set(ylabel="opening [mm]")
    ax[2].set(ylabel="object z [mm]", xlabel="t [s]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "grasp_ab.png", dpi=120)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="grasp")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    run_dir = make_run_dir(args.out or cfg.get("output_dir", "results/etc/grasp"))
    save_yaml(dict(cfg), run_dir / "config.yaml")

    logs, summary = {}, {}
    from src.visualization.taichi_viewer import TaichiViewer

    viewer = None
    for mode in ("gt", "markers"):
        sim = build_grasp_sim(cfg, estimator_mode=mode)
        log = sim.run(t_max=12.0)
        logs[mode] = log
        summary[mode] = {
            "success": bool(sim.success()),
            "phase": sim.state.phase,
            "held": bool(sim.state.held),
            "lam_final": float(sim.state.lam_true),
            "lam_meas_final": float(sim.state.lam_meas),
            "obj_z_final_mm": float(1e3 * sim.state.obj_z),
            "force_target": float(sim.force_target),
            "hold_force": float(sim.mech.hold_force),
            "tracking_mae_N": float(np.mean(np.abs(np.asarray(log["lam_meas"]) - sim.force_target))),
        }
        print(f"[{mode}] success={sim.success()} phase={sim.state.phase} "
              f"λ={sim.state.lam_true:.2f} N  z={1e3 * sim.state.obj_z:.1f} mm")
        if viewer is None:
            viewer = TaichiViewer(show_window=False, title="grasp eval")
            viewer.options.amplification = 1.0
        viewer.set_state(build_grasp_state(sim), refit_camera=True)
        viewer.save_screenshot(run_dir / f"{mode}_final.png")

    _plot(run_dir, logs)
    save_json({"summary": summary, "hold_force_N": summary["gt"]["hold_force"]}, run_dir / "metrics.json")
    print(f"results → {run_dir}")
    if viewer is not None:
        viewer.destroy()


if __name__ == "__main__":
    main()
