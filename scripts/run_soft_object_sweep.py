#!/usr/bin/env python
"""Deformable object sweep: vision force vs jaw-opening force as the object softens.

The object is a lumped spring ``k_obj`` in series with the two fingers. For each
``k_obj`` the Stage A (GT) and Stage B (markers → q → ROM; known and unknown contact) grasps are run and,
over the lift / hold window, we report

* vision force error ``|λ̂ − λ| / λ`` (should not depend on ``k_obj``: the finger
  FEM does not see the object);
* the "proprioceptive" baseline that infers force from the jaw opening assuming
  a rigid object, ``λ_open = k (w − g)/2`` (wrong by the object's share of the closure);
* the object stiffness recovered from opening + λ̂.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grasp.factory import build_grasp_sim  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402

WINDOW_PHASES = ("lift", "hold")


def _run_case(cfg: dict, k_obj: float, mode: str, t_max: float, seed: int) -> dict:
    c = copy.deepcopy(cfg)
    c["object"]["stiffness"] = None if np.isinf(k_obj) else float(k_obj)
    c["seed"] = seed
    est_mode = mode.removesuffix("-known")
    if mode.endswith("-known"):
        c["estimator"]["unknown_contact"] = False
    sim = build_grasp_sim(c, estimator_mode=est_mode)
    mech = sim.mech
    rigid = dataclasses.replace(mech, object_stiffness=np.inf)
    log = sim.run(t_max=t_max)

    phase = np.asarray(log["phase"])
    lam = np.asarray(log["lam_true"])
    lam_hat = np.asarray(log["lam_meas"])
    opening = np.asarray(log["opening"])
    win = np.isin(phase, WINDOW_PHASES) & (lam > 0.3)
    if not win.any():
        win = lam > 0.3
    lam_open = np.array([rigid.force_from_opening(g) for g in opening])
    c_hat = np.array([mech.estimate_object_compliance(g, l) for g, l in zip(opening[win], lam_hat[win])])
    c_med = float(np.nanmedian(c_hat)) if c_hat.size else float("nan")
    ce = sim.estimator.last_contact_err_m
    return {
        "k_obj_N_per_m": None if np.isinf(k_obj) else float(k_obj),
        "mode": mode,
        "seed": seed,
        "success": bool(sim.success()),
        "contact_err_mm": None if ce is None else float(1e3 * ce),
        "phase": sim.state.phase,
        "lam_final_N": float(sim.state.lam_true),
        "compression_final_mm": float(1e3 * mech.object_compression(sim.state.lam_true)),
        "opening_final_mm": float(1e3 * sim.state.opening),
        "vision_force_rel_median": float(np.median(np.abs(lam_hat[win] - lam[win]) / lam[win])),
        "opening_force_rel_median": float(np.median(np.abs(lam_open[win] - lam[win]) / lam[win])),
        "k_obj_hat_N_per_m": float(1.0 / c_med) if c_med > 1e-9 else None,
        "compliance_hat_mm_per_N": float(1e3 * c_med),
        "compliance_true_mm_per_N": float(1e3 / k_obj),
        "_log": {"t": np.asarray(log["t"]), "lam": lam, "lam_hat": lam_hat, "lam_open": lam_open, "opening": opening},
    }


MODE_STYLE = {"gt": ("o", "C0"), "markers-known": ("s", "C1"), "markers": ("^", "C2")}


def _summarize(rows: list[dict], ks: list[float], modes: list[str]) -> list[dict]:
    out = []
    for k in ks:
        for mode in modes:
            rs = [r for r in rows if r["_k"] == k and r["mode"] == mode]
            ce = [r["contact_err_mm"] for r in rs if r["contact_err_mm"] is not None]
            out.append({
                "k_obj_N_per_m": rs[0]["k_obj_N_per_m"],
                "mode": mode,
                "n_seeds": len(rs),
                "success_rate": float(np.mean([r["success"] for r in rs])),
                "vision_force_rel_median": float(np.median([r["vision_force_rel_median"] for r in rs])),
                "opening_force_rel_median": float(np.median([r["opening_force_rel_median"] for r in rs])),
                "compliance_hat_mm_per_N": float(np.median([r["compliance_hat_mm_per_N"] for r in rs])),
                "compliance_true_mm_per_N": rs[0]["compliance_true_mm_per_N"],
                "contact_err_mm_median": float(np.median(ce)) if ce else None,
                "compression_final_mm_median": float(np.median([r["compression_final_mm"] for r in rs])),
            })
    return out


def _plot(run_dir: Path, summary: list[dict], rows: list[dict], ks: list[float], modes: list[str], target: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["rigid" if np.isinf(k) else f"{k:g}" for k in ks]
    x = np.arange(len(ks))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for mode in modes:
        mk, col = MODE_STYLE.get(mode, ("x", "C3"))
        s = [r for r in summary if r["mode"] == mode]
        if mode != "gt":
            ax[0].plot(x, [100 * r["vision_force_rel_median"] for r in s], mk + "-", color=col, label=f"vision λ̂ ({mode})")
        ax[1].plot(x, [r["compliance_hat_mm_per_N"] for r in s], mk + "-", color=col, label=f"opening + λ̂ ({mode})")
    s = [r for r in summary if r["mode"] == modes[0]]
    ax[0].plot(x, [100 * r["opening_force_rel_median"] for r in s], "k--", label="jaw opening, rigid-object assumption")
    ax[1].plot(x, [r["compliance_true_mm_per_N"] for r in s], "k:", lw=2, label="true 1/k_obj")
    ax[0].set(xticks=x, xticklabels=labels, xlabel="object stiffness k_obj [N/m]",
              ylabel="force error, median [%]", title="force during lift / hold")
    ax[1].set(xticks=x, xticklabels=labels, xlabel="object stiffness k_obj [N/m]",
              ylabel="object compliance [mm/N]", title="object compliance from vision force + opening")
    for r in rows:
        if r["mode"] == "markers-known" and r["seed"] == 0:
            lg = r["_log"]
            ax[2].plot(lg["t"], lg["lam_hat"], "-", lw=1, label=f"k={labels[ks.index(r['_k'])]}")
    ax[2].axhline(target, color="k", ls="--", lw=1)
    ax[2].set(xlabel="t [s]", ylabel="λ̂ [N]", title="markers, known contact: force tracking")
    for a in ax:
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "soft_object_sweep.png", dpi=130)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="grasp")
    p.add_argument("--k-obj", type=float, nargs="+", default=[np.inf, 5000.0, 2000.0, 1000.0, 600.0])
    p.add_argument("--modes", nargs="+", default=["gt", "markers-known", "markers"])
    p.add_argument("--seeds", type=int, default=3, help="noise seeds per vision mode (gt runs once)")
    p.add_argument("--t-max", type=float, default=16.0)
    p.add_argument("--out", default="results/etc/soft_object")
    args = p.parse_args()
    cfg = dict(load_config(args.config))
    run_dir = make_run_dir(args.out)
    save_yaml(cfg, run_dir / "config.yaml")
    ks = list(args.k_obj)

    rows = []
    for k in ks:
        for mode in args.modes:
            for seed in range(1 if mode == "gt" else args.seeds):
                r = _run_case(cfg, k, mode, args.t_max, seed)
                r["_k"] = k
                rows.append(r)
    summary = _summarize(rows, ks, args.modes)
    for s in summary:
        k = s["k_obj_N_per_m"]
        ce = s["contact_err_mm_median"]
        print(f"k_obj={'rigid' if k is None else f'{k:>5.0f}'} {s['mode']:>13}: success {100 * s['success_rate']:3.0f} %  "
              f"squash {s['compression_final_mm_median']:.2f} mm  "
              f"vision err {100 * s['vision_force_rel_median']:5.1f} %  "
              f"opening(rigid) err {100 * s['opening_force_rel_median']:5.1f} %  "
              f"1/k̂ {s['compliance_hat_mm_per_N']:.3f} (true {s['compliance_true_mm_per_N']:.3f}) mm/N  "
              f"contact {'-' if ce is None else f'{ce:.1f} mm'}")

    _plot(run_dir, summary, rows, ks, args.modes, float(cfg["control"]["force_target"]))
    save_json({
        "summary": summary,
        "cases": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }, run_dir / "metrics.json")
    print(f"results → {run_dir}")


if __name__ == "__main__":
    main()
