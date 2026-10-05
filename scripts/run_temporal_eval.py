#!/usr/bin/env python
"""Level 4: consecutive frames → Kalman / EMA / batch mean on q.

Same cameras and 0.5 px marker noise as the current dataset. Each held-out
FEM sample is observed for T frames with independent pixel noise (a hold),
then again with a linear squeeze ``u(t) = lerp(start, 1) u``.

Run:
    python scripts/run_temporal_eval.py
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

from src.evaluation.e2e import build_e2e_pipeline  # noqa: E402
from src.evaluation.metrics import relative_errors, summarize  # noqa: E402
from src.evaluation.snr_sweep import estimate_q_multiview, pick_heldout_fem  # noqa: E402
from src.rendering.synthetic_camera import camera_ring  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402
from src.vision.marker_model import GeometricMarkerModel  # noqa: E402
from src.vision.temporal import filter_sequence  # noqa: E402


def _collect_frames(geo, cams, mesh, markers, samples, noise, rng, T, scale_of):
    """Return ``q_meas (N, T, r)``."""
    n, r = len(samples), geo.r
    out = np.zeros((n, T, r))
    for i, s in enumerate(samples):
        for t in range(T):
            u = scale_of(t) * s["u"]
            out[i, t], _ = estimate_q_multiview(geo, cams, mesh, markers, u, noise, rng)
    return out


def _score(q_hat, samples, localizer, scale: float = 1.0) -> dict:
    q_true = np.stack([s["q"] for s in samples]) * float(scale)
    F_true = np.array([s["force"] for s in samples], float) * float(scale)
    pos = np.zeros(len(samples))
    mag = np.zeros(len(samples))
    for i, s in enumerate(samples):
        loc = localizer.localize(q_hat[i])
        pos[i] = np.linalg.norm(loc.contact.position - s["contact"])
        mag[i] = loc.magnitude
    q_rel = relative_errors(q_hat, q_true)
    f_rel = np.abs(mag - F_true) / np.maximum(F_true, 1e-30)
    return {
        **summarize(q_rel, "q_rel_"),
        **summarize(1e3 * pos, "contact_err_mm_"),
        **summarize(f_rel, "force_rel_"),
        "n": int(len(samples)),
    }


def _plot(run_dir: Path, hold: dict, ramp: dict, counts: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "raw": ("#9D755D", "last frame"),
        "ema": ("#F58518", "EMA"),
        "kalman": ("#4C78A8", "Kalman"),
        "batch": ("#54A24B", "running mean"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for name, (color, lab) in styles.items():
        if name not in hold:
            continue
        xs = [c for c in counts if str(c) in hold[name]]
        ys = [hold[name][str(c)]["contact_err_mm_median"] for c in xs]
        fs = [100 * hold[name][str(c)]["force_rel_median"] for c in xs]
        axes[0].plot(xs, ys, "o-", color=color, label=lab)
        axes[1].plot(xs, fs, "o-", color=color, label=lab)
    axes[0].axhline(2.0, color="0.4", ls="--", lw=1, label="2 mm")
    axes[1].axhline(10.0, color="0.4", ls="--", lw=1, label="10 %")
    axes[0].set(xlabel="frames (static hold)", ylabel="contact error [mm]",
                title="Static hold · 2 views · 0.5 px · 192×144")
    axes[1].set(xlabel="frames (static hold)", ylabel="force rel. median [%]",
                title="Static hold · unknown force")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(counts)
        ax.set_xticklabels([str(c) for c in counts])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "temporal_hold.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, (color, lab) in styles.items():
        if name not in ramp:
            continue
        xs = [c for c in counts if str(c) in ramp[name]]
        ys = [ramp[name][str(c)]["contact_err_mm_median"] for c in xs]
        ax.plot(xs, ys, "o-", color=color, label=lab)
    ax.axhline(2.0, color="0.4", ls="--", lw=1, label="2 mm")
    ax.set(xlabel="frame in a 32-frame squeeze", ylabel="contact error [mm]",
           title="Slow squeeze · unknown contact at that frame")
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "temporal_ramp.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="temporal")
    p.add_argument("--out", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    e2e_cfg = load_config(cfg.get("e2e_config", "e2e"))
    vis_cfg = load_config(cfg.get("vision_config", "vision"))
    pipe, split = build_e2e_pipeline(e2e_cfg, estimator="oracle")
    if pipe.localizer is None:
        raise RuntimeError("unknown-contact localizer required")

    test_idx = np.nonzero(split.test)[0]
    max_s = args.max_samples if args.max_samples is not None else cfg.get("max_samples")
    fids = pick_heldout_fem(pipe.vds, test_idx, max_s)
    first = {int(fid): int(np.nonzero(pipe.vds.fem_index == fid)[0][0]) for fid in fids}
    samples = []
    for fid in fids:
        s = first[int(fid)]
        samples.append({
            "u": pipe.fem.displacement(int(fid)),
            "q": pipe.vds.q[s],
            "contact": pipe.vds.contact_position[s],
            "force": float(pipe.vds.force_magnitude[s]),
        })

    geo = GeometricMarkerModel(
        pipe.basis, pipe.markers, pipe.model.mesh,
        ridge=float(cfg.get("geometric_ridge", 1e-4)),
        use_noisy=False, n_iter=int(cfg.get("n_iter", 2)), huber=cfg.get("huber_px", 1.5),
    )
    cams = camera_ring(
        pipe.model.geometry, vis_cfg.get("camera", {}),
        int(cfg.get("n_views", 2)),
        int(cfg.get("width", 192)), int(cfg.get("height", 144)),
    )
    T = int(cfg.get("n_frames", 32))
    counts = [int(c) for c in cfg.get("frame_counts", [1, 2, 4, 8, 16, 32])]
    noise = float(cfg.get("noise_px", 0.5))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    print(f"samples={len(samples)}  frames={T}  views={len(cams)}  noise={noise:g} px")

    print("hold frames …")
    hold_z = _collect_frames(
        geo, cams, pipe.model.mesh, pipe.markers, samples, noise, rng, T, lambda t: 1.0,
    )
    start = float(cfg.get("ramp_start", 0.3))
    print("ramp frames …")
    ramp_z = _collect_frames(
        geo, cams, pipe.model.mesh, pipe.markers, samples, noise, rng, T,
        lambda t: start + (1.0 - start) * (t / max(T - 1, 1)),
    )

    kinds = {
        "raw": {},
        "ema": {"ema_alpha": float(cfg.get("ema_alpha", 0.25))},
        "kalman": {
            "process": float(cfg.get("kalman_process", 1e-8)),
            "measure": float(cfg.get("kalman_measure", 1e-4)),
        },
        "batch": {},
    }
    hold, ramp = {}, {}
    for name, kw in kinds.items():
        hold[name], ramp[name] = {}, {}
        kw_ramp = dict(kw)
        if name == "kalman":
            kw_ramp["process"] = float(cfg.get("kalman_process_ramp", 3e-6))
        filt_h = np.stack([filter_sequence(hold_z[i], name, **kw) for i in range(len(samples))])
        filt_r = np.stack([filter_sequence(ramp_z[i], name, **kw_ramp) for i in range(len(samples))])
        for n in counts:
            t = n - 1
            hold[name][str(n)] = _score(filt_h[:, t], samples, pipe.localizer, 1.0)
            scale = start + (1.0 - start) * (t / max(T - 1, 1))
            ramp[name][str(n)] = _score(filt_r[:, t], samples, pipe.localizer, scale)
            h, r = hold[name][str(n)], ramp[name][str(n)]
            print(
                f"[{name:6s} n={n:2d}]  hold {h['contact_err_mm_median']:.2f} mm / "
                f"{100 * h['force_rel_median']:.1f} %   "
                f"ramp {r['contact_err_mm_median']:.2f} mm / {100 * r['force_rel_median']:.1f} %"
            )

    run_dir = make_run_dir(args.out or cfg.get("output_dir", "results/etc/e2e"))
    save_yaml({**cfg, "n_samples": len(samples)}, run_dir / "config.yaml")
    save_json({"n": len(samples), "hold": hold, "ramp": ramp}, run_dir / "temporal.json")
    _plot(run_dir, hold, ramp, counts)
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
