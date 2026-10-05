#!/usr/bin/env python
"""Compare vision→q estimators on the held-out pool, then Level-4 search.

The L4 residual search is already accurate given oracle q. This script measures
whether a better marker→q map closes the remaining contact/force gap.

Estimators:
  ridge       learned Δuv → q (current default)
  geometric   one-shot image Jacobian LS
  iterative   Gauss–Newton (2 iters) + Huber IRLS
  robust      iterative + average q across views of the same FEM sample
  oracle      ground-truth q (ceiling)

Also reports unknown-contact metrics after rejecting low-signal samples
(small ‖q̂‖). Coverage is printed so the gate is not a silent cherry-pick.

Run:
    python scripts/run_q_improve_eval.py
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
from src.evaluation.metrics import relative_errors, summarize  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402


def _block(pipe, idx, q, localizer, accept: np.ndarray | None = None) -> dict:
    q = np.asarray(q, float)
    q_true = pipe.vds.q[idx]
    F_true = pipe.vds.force_magnitude[idx]
    c_true = pipe.vds.contact_position[idx]
    q_rel = relative_errors(q, q_true)

    known = pipe.evaluate(idx, known_contact=True, q_override=q)
    pos = np.zeros(len(idx))
    mag = np.zeros(len(idx))
    prev = pipe.localizer
    pipe.localizer = localizer
    try:
        for i, s in enumerate(idx):
            pred = pipe.predict_one(int(s), known_contact=False, q=q[i])
            pos[i] = np.linalg.norm(pred.contact.position - c_true[i])
            mag[i] = pred.force_magnitude
    finally:
        pipe.localizer = prev
    unk_rel = np.abs(mag - F_true) / np.maximum(F_true, 1e-30)
    q_norm = np.linalg.norm(q, axis=1)
    mask = np.ones(len(idx), bool) if accept is None else np.asarray(accept, bool)
    high = F_true > 1.0

    def _pack(m: np.ndarray) -> dict:
        m = np.asarray(m, bool)
        n = int(m.sum())
        if n == 0:
            return {"n": 0, "coverage": 0.0}
        return {
            "n": n,
            "coverage": float(n / len(idx)),
            **summarize(q_rel[m], "q_rel_"),
            "force_rel_median_known": float(np.median(known["force_rel_by_sample"][m])),
            "contact_err_mm_median": float(np.median(1e3 * pos[m])),
            "contact_err_mm_mean": float(np.mean(1e3 * pos[m])),
            "contact_err_mm_p90": float(np.percentile(1e3 * pos[m], 90)),
            "force_rel_median_unknown": float(np.median(unk_rel[m])),
            "force_rel_mean_unknown": float(np.mean(unk_rel[m])),
        }

    out = {
        "all": _pack(mask),
        "Fgt1": _pack(mask & high),
        "q_norm": q_norm,
        "contact_err_mm": 1e3 * pos,
        "force_rel_unknown": unk_rel,
        "force_rel_known": known["force_rel_by_sample"],
        "q_rel": q_rel,
        "mag_true": F_true,
        "n": int(len(idx)),
    }
    return out


def _plot(run_dir: Path, blocks: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [k for k in ("ridge", "geometric", "iterative", "robust", "robust_reject", "oracle") if k in blocks]
    labels = {
        "ridge": "ridge",
        "geometric": "geometric",
        "iterative": "iterative",
        "robust": "robust",
        "robust_reject": "robust + reject",
        "oracle": "oracle q",
    }

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    xs = np.arange(len(names))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#9D755D"]
    series = (
        (axes[0, 0], [100 * blocks[k]["all"]["q_rel_median"] for k in names], "q̂ error (median)", "%"),
        (axes[0, 1], [100 * blocks[k]["all"]["force_rel_median_known"] for k in names], "known-contact force (median)", "%"),
        (axes[1, 0], [blocks[k]["all"]["contact_err_mm_median"] for k in names], "unknown-contact location (median)", "mm"),
        (axes[1, 1], [100 * blocks[k]["all"]["force_rel_median_unknown"] for k in names], "unknown-contact force (median)", "%"),
    )
    for ax, vals, title, ylab in series:
        ax.bar(xs, vals, color=colors[: len(names)])
        ax.set_xticks(xs, [labels[k] for k in names], rotation=25, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "q_improve_bars.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for k, c in (("ridge", "#4C78A8"), ("robust", "#E45756"), ("oracle", "#9D755D")):
        if k not in blocks:
            continue
        b = blocks[k]
        ax.scatter(b["mag_true"], b["contact_err_mm"], s=8, alpha=0.35, c=c, label=labels[k])
    ax.set(xlabel="true force [N]", ylabel="contact error [mm]",
           title="Unknown-contact error vs load", ylim=(0, None))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "q_improve_contact_vs_load.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for k, c in (("ridge", "#4C78A8"), ("robust", "#E45756"), ("oracle", "#9D755D")):
        if k not in blocks:
            continue
        ax.hist(blocks[k]["contact_err_mm"], bins=40, range=(0, 40), histtype="step",
                density=True, color=c, label=labels[k], linewidth=1.6)
    ax.set(xlabel="contact error [mm]", ylabel="density", title="Unknown-contact error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "q_improve_contact_hist.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="e2e")
    p.add_argument("--out", default=None)
    p.add_argument("--min-q-norm", type=float, default=None,
                   help="reject samples with ‖q̂‖ below this (default: 2× median ‖q̂‖ of F<0.5 N)")
    args = p.parse_args()

    cfg = load_config(args.config)
    pipe, split = build_e2e_pipeline(cfg, estimator="ridge")
    idx = np.nonzero(split.test)[0]
    vis = cfg.get("vision", {})

    estimators = {}
    for name in ("ridge", "geometric", "iterative", "robust", "oracle"):
        fn, _ = build_q_estimator(
            name, pipe.vds, split.train, pipe.markers, pipe.model.mesh, pipe.basis,  # type: ignore[arg-type]
            use_noisy=bool(vis.get("use_noisy_markers", True)),
            ridge_lambda=float(vis.get("ridge_lambda", 1e-2)),
            geometric_ridge=float(vis.get("geometric_ridge", 1e-4)),
        )
        estimators[name] = fn(idx)
        print(f"predicted {name}: {estimators[name].shape}")

    localizer = pipe.localizer
    if localizer is None:
        raise RuntimeError("unknown-contact localizer required (set unknown_contact.enabled)")

    blocks = {}
    for name, q in estimators.items():
        print(f"evaluating {name} …")
        blocks[name] = _block(pipe, idx, q, localizer)

    q_rob = estimators["robust"]
    qn = np.linalg.norm(q_rob, axis=1)
    F_true = pipe.vds.force_magnitude[idx]
    low = F_true < 0.5
    q_floor = float(np.median(qn[low])) if low.any() else float(np.median(qn))
    q_min = float(args.min_q_norm) if args.min_q_norm is not None else 2.0 * q_floor
    accept = qn >= q_min
    blocks["robust_reject"] = _block(pipe, idx, q_rob, localizer, accept=accept)
    blocks["robust_reject"]["q_min"] = q_min
    blocks["robust_reject"]["q_floor"] = q_floor

    run_dir = make_run_dir(args.out or "results/etc/e2e")
    save_yaml(cfg, run_dir / "config.yaml")
    slim = {}
    for name, b in blocks.items():
        slim[name] = {k: v for k, v in b.items() if k not in (
            "q_norm", "contact_err_mm", "force_rel_unknown", "force_rel_known", "q_rel", "mag_true",
        )}
        a = b["all"]
        print(
            f"[{name}] n={a.get('n', 0)} cov={a.get('coverage', 1):.2f}  "
            f"q_rel med {100 * a.get('q_rel_median', float('nan')):.1f} %  "
            f"F_known {100 * a.get('force_rel_median_known', float('nan')):.1f} %  "
            f"contact {a.get('contact_err_mm_median', float('nan')):.2f} mm  "
            f"F_unk {100 * a.get('force_rel_median_unknown', float('nan')):.1f} %"
        )
    save_json({
        "n_test": int(len(idx)),
        "q_min": q_min,
        "q_floor": q_floor,
        "methods": slim,
        "note": "robust = iterative Gauss–Newton + Huber + multi-view mean; "
                "robust_reject gates on ‖q̂‖ ≥ q_min",
    }, run_dir / "q_improve.json")
    _plot(run_dir, blocks)
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
