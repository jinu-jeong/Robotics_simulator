#!/usr/bin/env python
"""Marker-free experiment – evaluate q sources through the SAME physics back-end.

Every q source is pushed through the identical, unchanged ROM stages so the
comparison isolates the state estimator:

    Level A  q          : q̂ vs q_sim (relative error, per-mode normalised RMSE), q̂ vs q_marker
    Level B  force      : q̂ → ReducedModel.estimate_force at the KNOWN contact
    Level C  contact    : q̂ → UnknownContactLocalizer (residual search) → contact position + force

q sources
    sim       q_sim (exact Φ_rᵀu; ROM upper bound)
    marker    q_marker (existing marker pipeline on the marker-on image / noisy pixels)
    nn        ResNet-18 checkpoints (``resnet18_q_<target>_<input>.pt``); the image
              modality (raw / marker) is read from the checkpoint meta.

Run:
    python scripts/eval_image_to_q.py                                  # all checkpoints in checkpoint_dir
    python scripts/eval_image_to_q.py --checkpoints results/etc/checkpoints/markerfree/resnet18_q_sim_raw.pt
    python scripts/eval_image_to_q.py --no-unknown --max-test 200       # quick
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.e2e import EndToEndPipeline  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402
from src.evaluation.unknown_contact import UnknownContactLocalizer  # noqa: E402
from src.fem.contact_dataset import ContactDataset  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.rendering.markers import SurfaceMarkers  # noqa: E402
from src.rom.pod import PODBasis  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.utils.figure_res import MPL_DPI  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402
from src.vision.paired_dataset import PairedVisionDataset  # noqa: E402
from src.vision.pipeline import build_rom  # noqa: E402
from src.vision.splits import make_vision_split  # noqa: E402

KEEP_KEYS = (
    "n", "q_rel_mean", "q_rel_median", "q_rel_p90", "u_rel_mean", "u_rel_median",
    "force_rel_mean", "force_rel_median", "force_rel_p90", "force_mae_N", "force_rmse_N", "direction_cos_mean",
    "contact_err_mm_mean", "contact_err_mm_median", "contact_err_mm_p90", "time_mean_ms",
)


def level_a(q_hat: np.ndarray, q_sim: np.ndarray, q_marker: np.ndarray, q_std: np.ndarray) -> dict:
    """q-space errors against both labels."""
    rel_sim = np.linalg.norm(q_hat - q_sim, axis=1) / np.maximum(np.linalg.norm(q_sim, axis=1), 1e-30)
    rel_mk = np.linalg.norm(q_hat - q_marker, axis=1) / np.maximum(np.linalg.norm(q_marker, axis=1), 1e-30)
    nrmse_mode = np.sqrt(np.mean(((q_hat - q_sim) / q_std) ** 2, axis=0))
    return {
        **summarize(rel_sim, "q_rel_sim_"), **summarize(rel_mk, "q_rel_marker_"),
        "q_nrmse_per_mode_vs_sim": nrmse_mode.tolist(), "q_nrmse_vs_sim": float(np.sqrt(np.mean(nrmse_mode ** 2))),
        "q_rel_sim_by_sample": rel_sim,
    }


def by_force_level(values: np.ndarray, mag_true: np.ndarray) -> dict[str, list[float]]:
    levels = np.unique(np.round(mag_true, 6))
    med = [float(np.median(values[np.isclose(mag_true, lv)])) for lv in levels]
    return {"force_levels_N": levels.tolist(), "median": med}


def _plot(run_dir: Path, results: dict, unknown: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    x = np.arange(len(names))
    panels = [("q_rel_sim_median", "‖q̂−q_sim‖/‖q_sim‖ (median)"), ("known.force_rel_median", "force rel. err, known contact (median)")]
    if unknown:
        panels += [("unknown.force_rel_median", "force rel. err, unknown contact (median)"),
                   ("unknown.contact_err_mm_median", "contact position err [mm] (median)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8))
    for ax, (key, title) in zip(np.atleast_1d(axes), panels):
        vals = []
        for n in names:
            d = results[n]
            for k in key.split("."):
                d = d.get(k, {}) if isinstance(d, dict) else {}
            vals.append(float(d) if isinstance(d, (int, float)) else np.nan)
        ax.bar(x, vals, color=["#777" if n == "sim" else "#d95f02" if n == "marker" else "#1b9e77" for n in names])
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("held-out contacts: sim q / marker q / image-network q through the same ROM", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "summary_bars.png", dpi=MPL_DPI)

    # force error vs load
    fig, axes = plt.subplots(1, 2 if unknown else 1, figsize=(10 if unknown else 5.5, 3.8), squeeze=False)
    for n in names:
        b = results[n]["known"]["by_force_level"]
        axes[0, 0].plot(b["force_levels_N"], b["median"], "o-", ms=3, label=n)
        if unknown:
            b = results[n]["unknown"]["contact_err_by_force_level"]
            axes[0, 1].plot(b["force_levels_N"], b["median"], "o-", ms=3, label=n)
    axes[0, 0].set(xlabel="F_true [N]", ylabel="median rel. force error (known contact)", yscale="log")
    axes[0, 0].axhline(0.10, color="r", ls="--", lw=1)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3, which="both")
    if unknown:
        axes[0, 1].set(xlabel="F_true [N]", ylabel="median contact error [mm] (unknown contact)", yscale="log")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(run_dir / "error_vs_load.png", dpi=MPL_DPI)

    # true vs estimated force (known contact)
    fig, axes = plt.subplots(1, len(names), figsize=(3.8 * len(names), 3.6), squeeze=False)
    for ax, n in zip(axes[0], names):
        mt, mh = np.asarray(results[n]["known"]["mag_true"]), np.asarray(results[n]["known"]["mag_hat"])
        lim = [0.0, float(max(mt.max(), mh.max()) * 1.05)]
        ax.scatter(mt, mh, s=6, alpha=0.5)
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(xlabel="F_true [N]", ylabel="F̂ [N]", title=f"{n}: median rel {100 * results[n]['known']['force_rel_median']:.1f} %",
               xlim=lim, ylim=lim, aspect="equal")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "force_true_vs_est_known.png", dpi=MPL_DPI)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="markerfree")
    p.add_argument("--dataset", default=None)
    p.add_argument("--checkpoints", nargs="*", default=None, help="ResNet checkpoints (default: all in checkpoint_dir)")
    p.add_argument("--sources", default="sim,marker,nn", help="subset of sim,marker,nn")
    p.add_argument("--no-unknown", action="store_true", help="skip Level C (unknown-contact localisation)")
    p.add_argument("--max-test", type=int, default=None, help="evaluate on a random subset of the test split")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    wanted = {s.strip() for s in args.sources.split(",") if s.strip()}

    ds_path = Path(args.dataset or cfg["output"])
    pds = PairedVisionDataset.load(ds_path)
    fem = ContactDataset.load(cfg.get("fem_dataset") or pds.meta["fem_dataset"])
    model = FingerFEMModel.from_config(fem.meta["fem_config"])
    face = str(cfg.get("unknown_contact", {}).get("face") or pds.meta.get("config", {}).get("markers", {}).get("face", "top"))
    model.restrict_contact_surface(face)
    basis = PODBasis.load(cfg.get("rom_basis") or pds.meta["rom_basis"]).truncate(pds.r)
    rom = build_rom(model, basis, face)
    markers = SurfaceMarkers.from_dict(pds.meta["markers"])
    vds = pds.as_vision_dataset(images="raw", q="sim")  # truth for the physics stages = q_sim
    split_cfg = cfg.get("split", {})
    split = make_vision_split(vds, fem.contact_position, hold_every=int(split_cfg.get("hold_every", 3)),
                              hold_force_above=split_cfg.get("hold_force_above", None))
    force_cfg = cfg.get("force", {})
    uc = cfg.get("unknown_contact", {})
    unknown = bool(uc.get("enabled", True)) and not args.no_unknown
    localizer = None
    if unknown:
        localizer = UnknownContactLocalizer.from_rom(
            rom, model.surface_face_ids(face), barycentric=uc.get("barycentric"),
            face_stride=int(uc.get("face_stride", 2)), min_influence=float(uc.get("min_influence", 1e-8)),
            nonneg=bool(force_cfg.get("nonneg", True)), r_loc=uc.get("r_loc", 4),
            refine=bool(uc.get("refine", True)), refine_n=int(uc.get("refine_n", 7)), weight=str(uc.get("weight", "leading")),
        )
    pipe = EndToEndPipeline(
        basis=basis, rom=rom, model=model, markers=markers, vds=vds, fem=fem,
        q_predict=lambda idx: vds.q[np.asarray(idx, int)], estimator_name="q_override", localizer=localizer,
        force_method=str(force_cfg.get("method", "displacement")), force_mode=str(force_cfg.get("mode", "normal")),
        nonneg=bool(force_cfg.get("nonneg", True)),
    )

    test_idx = np.nonzero(split.test)[0]
    if args.max_test is not None and args.max_test < len(test_idx):
        test_idx = np.sort(np.random.default_rng(seed).choice(test_idx, int(args.max_test), replace=False))
    q_std = np.maximum(pds.q_sim[split.train].std(axis=0), 1e-12)

    # ---- q sources
    sources: dict[str, tuple[np.ndarray, dict]] = {}
    if "sim" in wanted:
        sources["sim"] = (pds.q_sim[test_idx], {"kind": "sim", "note": "exact Φ_rᵀu (ROM upper bound)"})
    if "marker" in wanted:
        sources["marker"] = (pds.q_marker[test_idx], {"kind": "marker", **pds.meta.get("q_marker_teacher", {})})
    if "nn" in wanted:
        from src.vision.resnet_q import ResNetQModel

        ckpt_dir = Path(cfg.get("checkpoint_dir", "results/etc/checkpoints/markerfree"))
        ckpts = [Path(c) for c in args.checkpoints] if args.checkpoints else sorted(ckpt_dir.glob("resnet18_q_*.pt"))
        if not ckpts:
            print(f"[nn] no checkpoints found in {ckpt_dir} (run train_image_to_q.py)")
        for c in ckpts:
            net = ResNetQModel.load(c)
            inp = str(net.meta.get("input", "raw"))
            tgt = str(net.meta.get("target", "?"))
            mw = str(net.meta.get("mode_weight", "none"))
            name = f"nn[{inp}→q_{tgt}" + (f",{mw}]" if mw != "none" else "]")
            t0 = time.time()
            q_hat = net.predict(pds.images(inp)[test_idx])
            sources[name] = (q_hat, {"kind": "nn", "checkpoint": str(c), "input": inp, "target": tgt,
                                     "n_params": net.n_params, "infer_ms_per_image": 1e3 * (time.time() - t0) / len(test_idx)})

    print(f"dataset {ds_path.name}: {len(pds)} pairs; evaluating {len(test_idx)} held-out samples "
          f"({split.meta['n_held_contacts']}/{split.meta['n_contacts']} contacts); unknown-contact={unknown}")
    results: dict[str, dict] = {}
    for name, (q_hat, info) in sources.items():
        t0 = time.time()
        res: dict = {"info": info, **level_a(q_hat, pds.q_sim[test_idx], pds.q_marker[test_idx], q_std)}
        known = pipe.evaluate(test_idx, known_contact=True, q_override=q_hat)
        res["known"] = {k: known[k] for k in KEEP_KEYS if k in known}
        res["known"]["by_force_level"] = by_force_level(known["force_rel_by_sample"], known["mag_true"])
        res["known"]["mag_true"], res["known"]["mag_hat"] = known["mag_true"], known["mag_hat"]
        if unknown:
            unk = pipe.evaluate(test_idx, known_contact=False, q_override=q_hat)
            res["unknown"] = {k: unk[k] for k in KEEP_KEYS if k in unk}
            res["unknown"]["by_force_level"] = by_force_level(unk["force_rel_by_sample"], unk["mag_true"])
            res["unknown"]["contact_err_by_force_level"] = by_force_level(1e3 * unk["contact_pos_err_m"], unk["mag_true"])
            res["unknown"]["contact_err_mm_by_sample"] = 1e3 * unk["contact_pos_err_m"]
        res["seconds"] = time.time() - t0
        results[name] = res
        line = (f"[{name:22s}] A q_rel(sim) med {res['q_rel_sim_median']:.3f}  "
                f"B force rel med {res['known']['force_rel_median']:.3f} (MAE {res['known']['force_mae_N']:.3f} N)")
        if unknown:
            line += (f"  C contact err med {res['unknown']['contact_err_mm_median']:.2f} mm, "
                     f"force rel med {res['unknown']['force_rel_median']:.3f}")
        print(line)

    run_dir = make_run_dir(args.out or cfg.get("output_dir", "results/etc/markerfree"), name="eval")
    save_yaml(cfg, run_dir / "config.yaml")
    save_json({"split": split.meta, "n_test": int(len(test_idx)), "unknown_contact": unknown, "results": results}, run_dir / "metrics.json")
    _plot(run_dir, results, unknown)
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
