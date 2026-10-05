#!/usr/bin/env python
"""Milestone 7 – end-to-end evaluation: observation → q → deformation → force.

Runs the packaged pipeline on generalization Tests A–D (brief §16) and, when
enabled, the Level-4 unknown-contact localizer on the same held-out set.

  Test A  held-out force magnitudes (``hold_force_above``)
  Test B  held-out contact locations (``hold_every``)
  Test C  extra marker pixel noise
  Test D  extra camera-pose jitter (geometric estimator)

Also reports inference time and contact-localization error.

Run:
    python scripts/run_e2e_eval.py
    python scripts/run_e2e_eval.py --estimator ridge
    python scripts/run_e2e_eval.py --estimator geometric --no-unknown
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
from src.rendering.synthetic_camera import jitter_camera  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402
from src.vision.marker_model import GeometricMarkerModel  # noqa: E402
from src.vision.splits import make_vision_split  # noqa: E402


def _slim(metrics: dict) -> dict:
    drop = {"q_hat", "F_hat", "q_rel_by_sample", "u_rel_by_sample", "force_rel_by_sample",
            "mag_hat", "mag_true", "contact_pos_err_m", "indices"}
    return {k: v for k, v in metrics.items() if k not in drop}


def _plot(run_dir: Path, known: dict, unknown: dict | None, noise: dict, pose: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, block, title in (
        (ax[0], known, "known contact"),
        (ax[1], unknown if unknown else known, "unknown contact" if unknown else "known contact"),
    ):
        a.scatter(block["mag_true"], block["mag_hat"], s=10, alpha=0.45)
        # axis from true range so a few localization outliers do not hide the bulk
        lim = [0.0, float(block["mag_true"].max() * 1.15)]
        a.plot(lim, lim, "k--", lw=1)
        a.set(xlabel="F_true [N]", ylabel="F_hat [N]", aspect="equal", xlim=lim, ylim=lim,
              title=f"{title}: median rel {100 * block['force_rel_median']:.1f} %")
        a.grid(alpha=0.3)
        n_out = int(np.sum(block["mag_hat"] > lim[1]))
        if n_out:
            a.text(0.02, 0.98, f"{n_out} outliers above axis", transform=a.transAxes, va="top", fontsize=8, color="0.3")
    fig.suptitle(f"end-to-end ({known['estimator']}) on held-out contacts/forces", fontsize=11)
    fig.tight_layout()
    fig.savefig(run_dir / "force_true_vs_est.png", dpi=120)

    # error vs load
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    levels = np.unique(np.round(known["mag_true"], 6))
    series = [("known", known, "-")]
    if unknown is not None:
        series.append(("unknown", unknown, "--"))
    for label, block, ls in series:
        med = [np.median(block["force_rel_by_sample"][np.isclose(block["mag_true"], lv)]) for lv in levels]
        ax.plot(levels, med, "o" + ls, label=label, ms=4)
    ax.axhline(0.10, color="r", ls=":", lw=1, label="10 % soft target")
    ax.set(xlabel="F_true [N]", ylabel="median relative force error", yscale="log",
           title="E2E force error vs load (Tests A+B pool)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(run_dir / "force_error_vs_load.png", dpi=120)

    # Test C noise sweep
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    sigmas = sorted(noise.keys())
    ax.plot(sigmas, [noise[s]["force_rel_median"] for s in sigmas], "o-", label="force median")
    ax.plot(sigmas, [noise[s]["q_rel_median"] for s in sigmas], "s-", label="q median")
    ax.set(xlabel="extra marker noise σ [px]", ylabel="median relative error",
           title="Test C – marker noise sensitivity (held-out pool)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "test_c_noise.png", dpi=120)

    if unknown is not None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.hist(1e3 * unknown["contact_pos_err_m"], bins=40, color="C0", alpha=0.85)
        ax.axvline(1e3 * np.median(unknown["contact_pos_err_m"]), color="r", ls="--",
                   label=f"median {1e3 * np.median(unknown['contact_pos_err_m']):.1f} mm")
        ax.set(xlabel="contact localization error [mm]", ylabel="count",
               title="Level 4 – unknown-contact localization error")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "contact_localization.png", dpi=120)

    if pose is not None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.bar(["nominal", "jittered pose"], [pose["nominal_force_rel_median"], pose["jitter_force_rel_median"]], color=["C0", "C1"])
        ax.set(ylabel="median relative force error", title="Test D – camera pose jitter (geometric)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "test_d_pose.png", dpi=120)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="e2e")
    p.add_argument("--estimator", default=None, choices=["geometric", "iterative", "robust", "ridge", "cnn", "oracle"])
    p.add_argument("--no-unknown", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.no_unknown:
        cfg.setdefault("unknown_contact", {})["enabled"] = False
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    pipe, split = build_e2e_pipeline(cfg, estimator=args.estimator)
    run_dir = make_run_dir(args.out or cfg.get("output_dir", "results/etc/e2e"))
    save_yaml(cfg, run_dir / "config.yaml")
    test_idx = np.nonzero(split.test)[0]
    print(f"estimator={pipe.estimator_name}  test={len(test_idx)}  "
          f"split={split.meta}  unknown_cands={len(pipe.localizer) if pipe.localizer else 0}")

    # ---- Tests A+B pool (whatever hold-out the split defines)
    known = pipe.evaluate(test_idx, known_contact=True)
    print(f"[known]   F_rel mean/med {known['force_rel_mean']:.3f}/{known['force_rel_median']:.3f}  "
          f"q_rel {known['q_rel_mean']:.3f}/{known['q_rel_median']:.3f}  "
          f"u_rel {known['u_rel_mean']:.3f}  time {known['time_median_ms']:.2f} ms")

    unknown = None
    if pipe.localizer is not None:
        unknown = pipe.evaluate(test_idx, known_contact=False)
        print(f"[unknown] F_rel mean/med {unknown['force_rel_mean']:.3f}/{unknown['force_rel_median']:.3f}  "
              f"contact_err med {unknown['contact_err_mm_median']:.2f} mm  "
              f"time {unknown['time_median_ms']:.2f} ms")

    # ---- Test C: extra marker noise (re-fit / re-predict with corrupted pixels)
    gen = cfg.get("generalization", {})
    noise_results = {}
    # work on a copy of marker_px
    px_orig = pipe.vds.marker_px.copy()
    for sigma in gen.get("extra_marker_noise_px", [0.0, 0.5, 1.0, 2.0]):
        pipe.vds.marker_px = px_orig + (0.0 if sigma == 0 else rng.normal(0.0, float(sigma), px_orig.shape))
        # rebuild ridge/geometric with corrupted markers if they depend on marker_px
        if pipe.estimator_name in ("ridge", "geometric", "iterative", "robust"):
            q_fn, _ = build_q_estimator(
                pipe.estimator_name, pipe.vds, split.train, pipe.markers, pipe.model.mesh, pipe.basis,  # type: ignore[arg-type]
                use_noisy=True,
                ridge_lambda=float(cfg.get("vision", {}).get("ridge_lambda", 1e-2)),
                geometric_ridge=float(cfg.get("vision", {}).get("geometric_ridge", 1e-4)),
            )
            pipe.q_predict = q_fn
        m = pipe.evaluate(test_idx, known_contact=True)
        noise_results[float(sigma)] = _slim(m)
        print(f"[Test C] σ={sigma:g} px  F_rel med {m['force_rel_median']:.3f}  q_rel med {m['q_rel_median']:.3f}")
    pipe.vds.marker_px = px_orig
    # restore estimator after noise sweep
    if pipe.estimator_name in ("ridge", "geometric", "iterative", "robust"):
        q_fn, _ = build_q_estimator(
            pipe.estimator_name, pipe.vds, split.train, pipe.markers, pipe.model.mesh, pipe.basis,  # type: ignore[arg-type]
            use_noisy=bool(cfg.get("vision", {}).get("use_noisy_markers", True)),
            ridge_lambda=float(cfg.get("vision", {}).get("ridge_lambda", 1e-2)),
            geometric_ridge=float(cfg.get("vision", {}).get("geometric_ridge", 1e-4)),
        )
        pipe.q_predict = q_fn

    # ---- Test D: camera pose jitter (geometric only – needs explicit camera)
    pose_results = None
    if pipe.estimator_name in ("geometric", "iterative", "robust", "oracle", "ridge"):
        # evaluate geometric under jittered cameras on a subset
        geo = GeometricMarkerModel(
            pipe.basis, pipe.markers, pipe.model.mesh,
            ridge=float(cfg.get("vision", {}).get("geometric_ridge", 1e-4)), use_noisy=True,
        )
        subset = test_idx[:: max(1, len(test_idx) // 80)]
        # nominal
        q_nom = geo.predict(pipe.vds, subset)
        nom = pipe.evaluate(subset, known_contact=True, q_override=q_nom)
        # jittered: replace each sample's projection using a jittered camera, keep stored marker noise model by re-projecting clean positions + noise
        from src.rendering.markers import SurfaceMarkers  # noqa: F401
        g = pipe.model.geometry
        jit = {
            "position_rel": float(gen.get("pose_jitter_position_rel", 0.05)),
            "target_rel": 0.02,
            "fov_deg": float(gen.get("pose_jitter_fov_deg", 2.0)),
        }
        n_trials = int(gen.get("pose_trials", 3))
        q_jit = np.zeros((len(subset), pipe.basis.r))
        p0 = pipe.markers.positions(pipe.model.mesh, None)
        for i, s in enumerate(subset):
            qs = []
            for t in range(n_trials):
                cam0 = pipe.vds.camera(int(s))
                cam = jitter_camera(cam0, g, jit, np.random.default_rng(int(cfg.get("seed", 0)) + 1000 * int(s) + t))
                # observation = true deformed markers through jittered camera + dataset noise level
                u = pipe.fem.displacement(int(pipe.vds.fem_index[s]))
                uv_true, _ = cam.project(pipe.markers.positions(pipe.model.mesh, u))
                noise = float(pipe.vds.meta.get("config", {}).get("markers", {}).get("pixel_noise_std", 0.5))
                uv_obs = uv_true + rng.normal(0.0, noise, uv_true.shape)
                qs.append(geo.predict_one(cam, uv_obs, pipe.vds.marker_visible[s]))
            q_jit[i] = np.mean(qs, axis=0)
        jit_m = pipe.evaluate(subset, known_contact=True, q_override=q_jit)
        pose_results = {
            "nominal_force_rel_median": nom["force_rel_median"],
            "jitter_force_rel_median": jit_m["force_rel_median"],
            "nominal_q_rel_median": nom["q_rel_median"],
            "jitter_q_rel_median": jit_m["q_rel_median"],
            "n": int(len(subset)),
            "trials": n_trials,
        }
        print(f"[Test D] geometric pose jitter: F_rel med {nom['force_rel_median']:.3f} → {jit_m['force_rel_median']:.3f}")

    # ---- Test A vs B breakdown: contacts-only vs forces-only if hold_force_above set
    breakdown = {}
    hold_F = cfg.get("split", {}).get("hold_force_above", None)
    if hold_F is not None:
        # pure contact holdout (ignore force hold) for comparison
        split_B = make_vision_split(pipe.vds, pipe.fem.contact_position, hold_every=int(cfg.get("split", {}).get("hold_every", 3)), hold_force_above=None)
        # samples that are test only because of force (train contact, high force)
        force_only = split.test & split_B.train
        contact_only = split_B.test
        if force_only.any():
            mA = pipe.evaluate(np.nonzero(force_only)[0], known_contact=True)
            breakdown["test_A_unseen_force"] = _slim(mA)
            print(f"[Test A] unseen force (seen contact): n={mA['n']}  F_rel med {mA['force_rel_median']:.3f}")
        if contact_only.any():
            mB = pipe.evaluate(np.nonzero(contact_only)[0], known_contact=True)
            breakdown["test_B_unseen_contact"] = _slim(mB)
            print(f"[Test B] unseen contact: n={mB['n']}  F_rel med {mB['force_rel_median']:.3f}")

    payload = {
        "split": split.meta,
        "known_contact": _slim(known),
        "unknown_contact": None if unknown is None else _slim(unknown),
        "test_C_noise": {str(k): v for k, v in noise_results.items()},
        "test_D_pose": pose_results,
        "breakdown": breakdown,
        "estimator": pipe.estimator_name,
        "n_unknown_candidates": 0 if pipe.localizer is None else len(pipe.localizer),
    }
    # keep arrays needed for plots in memory; write slim JSON
    save_json(payload, run_dir / "metrics.json")
    _plot(run_dir, known, unknown, noise_results, pose_results)
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
