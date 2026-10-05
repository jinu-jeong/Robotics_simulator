#!/usr/bin/env python
"""Level-4 SNR sweep: n_views × marker noise × resolution (marker re-projection).

Does not regenerate images. For each held-out FEM sample the surface markers
are projected through a calibrated camera ring, pixel noise is added, q is
estimated (iterative geometric, averaged across views), then the existing
unknown-contact localizer runs.

Run:
    python scripts/run_snr_sweep.py
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
from src.evaluation.snr_sweep import (  # noqa: E402
    SNRCondition,
    default_conditions,
    evaluate_condition,
    pick_heldout_fem,
)
from src.rendering.synthetic_camera import camera_ring  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402
from src.vision.marker_model import GeometricMarkerModel  # noqa: E402


def _plot(run_dir: Path, blocks: dict[str, dict], conds: list[SNRCondition]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _xy(noise: float, scale: int):
        xs, c, f = [], [], []
        for cond in conds:
            if cond.noise_px != noise or cond.res_scale != scale:
                continue
            b = blocks[cond.key]["all"]
            xs.append(cond.n_views)
            c.append(b["contact_err_mm_median"])
            f.append(100 * b["force_rel_median"])
        order = np.argsort(xs)
        return np.asarray(xs)[order], np.asarray(c)[order], np.asarray(f)[order]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for noise, color in ((0.2, "#54A24B"), (0.5, "#4C78A8"), (1.0, "#E45756")):
        x, c, f = _xy(noise, 1)
        if len(x) == 0:
            continue
        axes[0].plot(x, c, "o-", color=color, label=f"{noise:g} px")
        axes[1].plot(x, f, "o-", color=color, label=f"{noise:g} px")
    axes[0].axhline(2.0, color="0.4", ls="--", lw=1, label="2 mm")
    axes[1].axhline(10.0, color="0.4", ls="--", lw=1, label="10 %")
    axes[0].set(xlabel="n views", ylabel="contact error [mm]", title="192×144 · unknown contact")
    axes[1].set(xlabel="n views", ylabel="force rel. median [%]", title="192×144 · unknown force")
    for ax in axes:
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "snr_vs_views.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for scale, color, lab in ((1, "#4C78A8", "192×144"), (2, "#F58518", "384×288"), (4, "#B279A2", "768×576")):
        x, c, f = _xy(0.5, scale)
        if len(x) == 0:
            continue
        axes[0].plot(x, c, "o-", color=color, label=lab)
        axes[1].plot(x, f, "o-", color=color, label=lab)
    axes[0].axhline(2.0, color="0.4", ls="--", lw=1, label="2 mm")
    axes[1].axhline(10.0, color="0.4", ls="--", lw=1, label="10 %")
    axes[0].set(xlabel="n views", ylabel="contact error [mm]", title="0.5 px noise · unknown contact")
    axes[1].set(xlabel="n views", ylabel="force rel. median [%]", title="0.5 px noise · unknown force")
    for ax in axes:
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "snr_vs_res.png", dpi=140)
    plt.close(fig)

    views = sorted({c.n_views for c in conds if c.res_scale == 1})
    noises = sorted({c.noise_px for c in conds if c.res_scale == 1})
    heat = np.full((len(noises), len(views)), np.nan)
    for i, n in enumerate(noises):
        for j, v in enumerate(views):
            key = SNRCondition(v, n, 1).key
            if key in blocks:
                heat[i, j] = blocks[key]["all"]["contact_err_mm_median"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    im = ax.imshow(heat, origin="lower", cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(views)), [str(v) for v in views])
    ax.set_yticks(range(len(noises)), [f"{n:g} px" for n in noises])
    ax.set(xlabel="n views", ylabel="marker noise", title="Median contact error [mm] at 192×144")
    for i in range(len(noises)):
        for j in range(len(views)):
            if np.isfinite(heat[i, j]):
                ax.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="mm")
    fig.tight_layout()
    fig.savefig(run_dir / "snr_heatmap.png", dpi=140)
    plt.close(fig)


def _plot_cameras(run_dir: Path, cameras, geometry) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    L, W = geometry.length, geometry.width
    ax.add_patch(plt.Rectangle((0, 0), W, L, fill=False, color="0.3", lw=1.2))
    for cam in cameras:
        ax.plot(cam.position[1], cam.position[0], "o", color="#4C78A8")
        ax.annotate("", xy=(cam.target[1], cam.target[0]),
                    xytext=(cam.position[1], cam.position[0]),
                    arrowprops=dict(arrowstyle="->", color="#4C78A8", lw=1))
    ax.set(xlabel="y [m]", ylabel="x [m]", title=f"camera ring · {len(cameras)} views · top-down")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "snr_cameras.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="snr_sweep")
    p.add_argument("--out", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    sweep = load_config(args.config)
    e2e_cfg = load_config(sweep.get("e2e_config", "e2e"))
    vis_cfg = load_config(sweep.get("vision_config", "vision"))
    pipe, split = build_e2e_pipeline(e2e_cfg, estimator="oracle")
    if pipe.localizer is None:
        raise RuntimeError("unknown-contact localizer required")

    test_idx = np.nonzero(split.test)[0]
    max_s = args.max_samples if args.max_samples is not None else sweep.get("max_samples")
    fids = pick_heldout_fem(pipe.vds, test_idx, max_s)
    # one vision row per FEM sample (same q / contact / force)
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
        ridge=float(sweep.get("geometric_ridge", e2e_cfg.get("vision", {}).get("geometric_ridge", 1e-4))),
        use_noisy=False,
        n_iter=int(sweep.get("n_iter", 2)),
        huber=sweep.get("huber_px", 1.5),
    )
    cam_cfg = vis_cfg.get("camera", {})
    W0 = int(sweep.get("base_width", vis_cfg.get("image", {}).get("width", 192)))
    H0 = int(sweep.get("base_height", vis_cfg.get("image", {}).get("height", 144)))
    conds = default_conditions(
        n_views=sweep.get("n_views", [1, 2, 4, 8]),
        noise_px=sweep.get("noise_px", [0.2, 0.5, 1.0]),
        res_scales=sweep.get("res_scales", [1, 2, 4]),
        base_noise=float(sweep.get("base_noise_px", 0.5)),
    )
    rng = np.random.default_rng(int(sweep.get("seed", 0)))

    blocks: dict[str, dict] = {}
    print(f"samples={len(samples)}  conditions={len(conds)}  localizer={len(pipe.localizer)}")
    for cond in conds:
        W, H = W0 * cond.res_scale, H0 * cond.res_scale
        cams = camera_ring(pipe.model.geometry, cam_cfg, cond.n_views, W, H)
        raw = evaluate_condition(
            geo=geo, localizer=pipe.localizer, cameras=cams,
            mesh=pipe.model.mesh, markers=pipe.markers, samples=samples,
            noise_px=cond.noise_px, rng=rng,
        )
        blocks[cond.key] = {
            "n_views": cond.n_views, "noise_px": cond.noise_px, "res_scale": cond.res_scale,
            "width": W, "height": H,
            "all": {k: v for k, v in raw.items() if k not in ("q_rel", "contact_err_mm", "force_rel", "mag_true")},
        }
        a = blocks[cond.key]["all"]
        print(
            f"[{cond.key}]  contact {a['contact_err_mm_median']:.2f} mm  "
            f"F_unk {100 * a['force_rel_median']:.1f} %  "
            f"q {100 * a['q_rel_median']:.1f} %  vis {a['visible_frac']:.2f}"
        )

    # oracle ceiling on the same subset
    pos = np.zeros(len(samples))
    mag = np.zeros(len(samples))
    F_true = np.array([s["force"] for s in samples])
    for i, s in enumerate(samples):
        loc = pipe.localizer.localize(s["q"])
        pos[i] = np.linalg.norm(loc.contact.position - s["contact"])
        mag[i] = loc.magnitude
    f_rel = np.abs(mag - F_true) / np.maximum(F_true, 1e-30)
    blocks["oracle"] = {
        "all": {
            "n": len(samples),
            "contact_err_mm_median": float(np.median(1e3 * pos)),
            "force_rel_median": float(np.median(f_rel)),
        }
    }
    print(f"[oracle]  contact {1e3 * np.median(pos):.2f} mm  F_unk {100 * np.median(f_rel):.1f} %")

    run_dir = make_run_dir(args.out or sweep.get("output_dir", "results/etc/e2e"))
    save_yaml({**sweep, "n_samples": len(samples), "n_conditions": len(conds)}, run_dir / "config.yaml")
    save_json({"n": len(samples), "methods": {k: v for k, v in blocks.items()}}, run_dir / "snr_sweep.json")
    _plot(run_dir, blocks, conds)
    _plot_cameras(run_dir, camera_ring(pipe.model.geometry, cam_cfg, max(c.n_views for c in conds), W0, H0),
                  pipe.model.geometry)
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
