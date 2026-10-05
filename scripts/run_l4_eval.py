#!/usr/bin/env python
"""Level 4 comparison: known contact vs unknown, old search vs improved search.

Runs the packaged vision→q pipeline on the held-out pool, then localizes
contact two ways:

  old   unweighted residual, discrete grid only (the original L4 start)
  new   leading-mode residual + barycentric refine + full-mode force refit

Also reports the oracle-q ceiling (search given perfect reduced coordinates).

Run:
    python scripts/run_l4_eval.py
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
from src.evaluation.unknown_contact import UnknownContactLocalizer  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402


def _run(pipe, idx, q, localizer) -> dict:
    pos = np.zeros(len(idx))
    mag = np.zeros(len(idx))
    F_true = pipe.vds.force_magnitude[idx]
    c_true = pipe.vds.contact_position[idx]
    prev = pipe.localizer
    pipe.localizer = localizer
    try:
        for i, s in enumerate(idx):
            pred = pipe.predict_one(int(s), known_contact=False, q=q[i])
            pos[i] = np.linalg.norm(pred.contact.position - c_true[i])
            mag[i] = pred.force_magnitude
    finally:
        pipe.localizer = prev
    rel = np.abs(mag - F_true) / np.maximum(F_true, 1e-30)
    high = F_true > 1.0
    return {
        "contact_err_mm": 1e3 * pos,
        "force_rel": rel,
        "mag_hat": mag,
        "mag_true": F_true,
        "n": int(len(idx)),
        "contact_err_mm_median": float(np.median(1e3 * pos)),
        "contact_err_mm_mean": float(np.mean(1e3 * pos)),
        "contact_err_mm_p90": float(np.percentile(1e3 * pos, 90)),
        "force_rel_median": float(np.median(rel)),
        "force_rel_mean": float(np.mean(rel)),
        "force_rel_p90": float(np.percentile(rel, 90)),
        "n_high": int(high.sum()),
        "contact_err_mm_median_Fgt1": float(np.median(1e3 * pos[high])) if high.any() else float("nan"),
        "force_rel_median_Fgt1": float(np.median(rel[high])) if high.any() else float("nan"),
    }


def _plot(run_dir: Path, blocks: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [k for k in ("old_ridge", "new_ridge", "old_oracle", "new_oracle") if k in blocks]
    labels = {
        "old_ridge": "old + ridge q",
        "new_ridge": "new + ridge q",
        "old_oracle": "old + oracle q",
        "new_oracle": "new + oracle q",
    }

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
    names = [labels[k] for k in order]
    ax[0].bar(names, [blocks[k]["contact_err_mm_median"] for k in order], color=["0.65", "C0", "0.65", "C2"])
    ax[0].set(ylabel="median contact error [mm]", title="Contact localization")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(names, [100 * blocks[k]["force_rel_median"] for k in order], color=["0.65", "C0", "0.65", "C2"])
    ax[1].set(ylabel="median force rel. error [%]", title="Force (unknown contact)")
    ax[1].tick_params(axis="x", rotation=20)
    for a in ax:
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "l4_compare_bars.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for k, ls in (("old_ridge", "--"), ("new_ridge", "-")):
        if k not in blocks:
            continue
        b = blocks[k]
        levels = np.unique(np.round(b["mag_true"], 6))
        med = [np.median(b["contact_err_mm"][np.isclose(b["mag_true"], lv)]) for lv in levels]
        ax.plot(levels, med, "o" + ls, label=labels[k], ms=4)
    ax.set(xlabel="F_true [N]", ylabel="median contact error [mm]",
           title="Level 4: contact error vs load (ridge q̂)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "l4_contact_vs_load.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for k, col, lab in (
        ("old_ridge", "0.55", "old + ridge"),
        ("new_ridge", "C0", "new + ridge"),
        ("new_oracle", "C2", "new + oracle"),
    ):
        if k not in blocks:
            continue
        ax.hist(blocks[k]["contact_err_mm"], bins=40, range=(0, 60), histtype="step",
                lw=1.6, color=col, label=lab)
    ax.set(xlabel="contact localization error [mm]", ylabel="count",
           title="Level 4: localization error histogram (held-out pool)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "l4_contact_hist.png", dpi=120)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="e2e")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    pipe, split = build_e2e_pipeline(cfg, estimator="ridge")
    idx = np.nonzero(split.test)[0]
    q_ridge = pipe.predict_q(idx)
    q_oracle = pipe.vds.q[idx]

    uc = cfg.get("unknown_contact", {})
    face = str(uc.get("face") or "top")
    face_ids = pipe.model.surface_face_ids(face)
    common = dict(
        barycentric=uc.get("barycentric"),
        face_stride=int(uc.get("face_stride", 2)),
        min_influence=float(uc.get("min_influence", 1e-8)),
        nonneg=True,
    )
    loc_old = UnknownContactLocalizer.from_rom(
        pipe.rom, face_ids, **common, r_loc=None, refine=False, weight="none",
    )
    loc_new = UnknownContactLocalizer.from_rom(
        pipe.rom, face_ids, **common,
        r_loc=int(uc.get("r_loc", 4)),
        refine=bool(uc.get("refine", True)),
        refine_n=int(uc.get("refine_n", 7)),
        weight=str(uc.get("weight", "leading")),
    )

    known = pipe.evaluate(idx, known_contact=True, q_override=q_ridge)
    blocks = {
        "known_ridge": {
            "contact_err_mm_median": 0.0,
            "force_rel_median": known["force_rel_median"],
            "force_rel_median_Fgt1": float(np.median(
                known["force_rel_by_sample"][known["mag_true"] > 1.0]
            )),
            "n": known["n"],
        },
        "old_ridge": _run(pipe, idx, q_ridge, loc_old),
        "new_ridge": _run(pipe, idx, q_ridge, loc_new),
        "old_oracle": _run(pipe, idx, q_oracle, loc_old),
        "new_oracle": _run(pipe, idx, q_oracle, loc_new),
    }

    run_dir = make_run_dir(args.out or "results/etc/e2e")
    save_yaml(cfg, run_dir / "config.yaml")
    slim = {}
    for name, b in blocks.items():
        slim[name] = {k: v for k, v in b.items() if k not in ("contact_err_mm", "force_rel", "mag_hat", "mag_true")}
        print(
            f"[{name}] n={b.get('n', '?')}  "
            f"contact med {b.get('contact_err_mm_median', 0):.2f} mm  "
            f"F_rel med {100 * b.get('force_rel_median', 0):.1f} %  "
            f"(F>1 N: {b.get('contact_err_mm_median_Fgt1', float('nan')):.2f} mm / "
            f"{100 * b.get('force_rel_median_Fgt1', 0):.1f} %)"
        )
    save_json({
        "n_test": int(len(idx)),
        "n_candidates": len(loc_new),
        "methods": slim,
        "baseline_note": "old = unweighted discrete grid; new = leading-4 modes + refine",
    }, run_dir / "l4_compare.json")
    _plot(run_dir, blocks)
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
