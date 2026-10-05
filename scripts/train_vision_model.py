#!/usr/bin/env python
"""Milestone 6 – train / evaluate vision → q → force.

Estimators
----------
* ``geometric`` – per-sample marker image Jacobian LS (no training)
* ``ridge``     – learned Δuv → q ridge regression
* ``cnn``       – small CNN on the observation image (needs PyTorch)
* ``oracle``    – ground-truth q (upper bound of the ROM force path)

Splits follow brief §16: held-out contact locations (Test B). Force errors are
broken down by magnitude so the M5 noise-floor finding is visible.

Run:
    python scripts/train_vision_model.py
    python scripts/train_vision_model.py --models geometric,ridge,cnn --epochs 30
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

from src.fem.contact_dataset import ContactDataset  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.rendering.markers import SurfaceMarkers  # noqa: E402
from src.rom.pod import PODBasis  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402
from src.vision.dataset import VisionDataset  # noqa: E402
from src.vision.marker_model import GeometricMarkerModel, RidgeMarkerModel  # noqa: E402
from src.vision.pipeline import VisionMechanicsPipeline, build_rom  # noqa: E402
from src.vision.splits import make_vision_split  # noqa: E402


def _plot_results(run_dir: Path, results: dict, vds: VisionDataset, split) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(results.keys())
    # 1) force rel error summary bars
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(models))
    means = [results[m]["test"]["force_rel_mean"] for m in models]
    meds = [results[m]["test"]["force_rel_median"] for m in models]
    ax.bar(x - 0.18, means, 0.35, label="mean")
    ax.bar(x + 0.18, meds, 0.35, label="median")
    ax.axhline(0.10, color="r", ls="--", lw=1, label="10 % (Level-3 soft target)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set(ylabel="relative force error", title="held-out contacts: force error after vision → q → ROM", ylim=(0, max(0.5, max(means) * 1.2)))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "force_error_summary.png", dpi=120)

    # 2) true vs estimated force scatter for each model
    ncol = len(models)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
    idx = np.nonzero(split.test)[0]
    sc = None
    for ax, m in zip(axes[0], models):
        r = results[m]["test"]
        sc = ax.scatter(r["mag_true"], r["mag_hat"], s=8, alpha=0.5, c=1e3 * vds.contact_position[idx, 0], cmap="viridis")
        lim = [0.0, float(max(r["mag_true"].max(), r["mag_hat"].max()) * 1.05)]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(xlabel="F_true [N]", ylabel="F_hat [N]", title=f"{m}: median rel {100 * r['force_rel_median']:.1f} %",
               aspect="equal", xlim=lim, ylim=lim)
        ax.grid(alpha=0.3)
    if sc is not None:
        fig.colorbar(sc, ax=axes[0].tolist(), label="contact x [mm]", fraction=0.02)
    fig.tight_layout()
    fig.savefig(run_dir / "force_true_vs_est.png", dpi=120)

    # 3) error vs force magnitude (noise floor)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m in models:
        r = results[m]["test"]
        idx = np.nonzero(split.test)[0]
        # bin by true force
        levels = np.unique(np.round(r["mag_true"], 6))
        med = [np.median(r["force_rel_by_sample"][np.isclose(r["mag_true"], lv)]) for lv in levels]
        ax.plot(levels, med, "o-", label=m, ms=4)
    ax.axhline(0.10, color="r", ls="--", lw=1)
    ax.set(xlabel="F_true [N]", ylabel="median relative force error", yscale="log",
           title="force error vs load (held-out contacts) – low-F dominated by marker/image noise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(run_dir / "force_error_vs_load.png", dpi=120)

    # 4) q error bars
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x - 0.18, [results[m]["test"]["q_rel_mean"] for m in models], 0.35, label="mean")
    ax.bar(x + 0.18, [results[m]["test"]["q_rel_median"] for m in models], 0.35, label="median")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set(ylabel="‖q̂ − q‖ / ‖q‖", title="held-out contacts: reduced-coordinate error")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "q_error_summary.png", dpi=120)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="vision_model")
    p.add_argument("--models", default="geometric,ridge,cnn,oracle", help="comma-separated subset")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    np.random.seed(seed)
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]

    vds = VisionDataset.load(cfg["vision_dataset"])
    fem = ContactDataset.load(cfg.get("fem_dataset") or vds.meta["fem_dataset"])
    model = FingerFEMModel.from_config(fem.meta["fem_config"])
    markers = SurfaceMarkers.from_dict(vds.meta["markers"])
    r = int(cfg.get("q_modes", vds.q.shape[1]))
    basis = PODBasis.load(cfg.get("rom_basis") or vds.meta["rom_basis"]).truncate(r)
    # dataset q may already be r-dim; truncate if needed
    if vds.q.shape[1] != r:
        # recompute q from FEM if ranks differ
        Q = np.stack([basis.project(fem.displacement(int(i))) for i in vds.fem_index])
    else:
        Q = vds.q
    # temporarily expose matching q on a shallow copy view
    vds.q = Q

    split_cfg = cfg.get("split", {})
    split = make_vision_split(vds, fem.contact_position, hold_every=int(split_cfg.get("hold_every", 3)),
                              hold_force_above=split_cfg.get("hold_force_above", None))
    rom = build_rom(model, basis, contact_face=str(vds.meta.get("config", {}).get("markers", {}).get("face", "top")))
    force_cfg = cfg.get("force", {})
    mk_cfg = cfg.get("markers", {})
    run_dir = make_run_dir(args.out or cfg.get("output_dir", "results/etc/vision_model"))
    ckpt_dir = Path(cfg.get("checkpoint_dir", "results/etc/checkpoints/vision_model"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "config.yaml")
    print(f"dataset {len(vds)} images, q in R^{r}, split train={split.meta['n_train']} test={split.meta['n_test']} "
          f"(held {split.meta['n_held_contacts']}/{split.meta['n_contacts']} contacts)")

    results: dict = {}
    pipelines: dict[str, VisionMechanicsPipeline] = {}

    def attach(name: str, q_fn) -> VisionMechanicsPipeline:
        pipe = VisionMechanicsPipeline(
            basis, rom, q_fn, force_method=force_cfg.get("method", "displacement"),
            force_mode=force_cfg.get("mode", "normal"), nonneg=bool(force_cfg.get("nonneg", True)), name=name,
        )
        pipelines[name] = pipe
        return pipe

    # ---- geometric
    if "geometric" in wanted:
        t0 = time.time()
        geo = GeometricMarkerModel(
            basis, markers, model.mesh,
            ridge=float(mk_cfg.get("geometric_ridge", 1e-4)),
            eps=float(mk_cfg.get("eps", 1e-4)),
            use_noisy=bool(mk_cfg.get("use_noisy", True)),
        )
        pipe = attach("geometric", lambda idx, g=geo: g.predict(vds, idx))
        results["geometric"] = {
            "train": pipe.evaluate(vds, np.nonzero(split.train)[0], fem_U=fem.U),
            "test": pipe.evaluate(vds, np.nonzero(split.test)[0], fem_U=fem.U),
            "seconds": time.time() - t0,
        }
        print(f"[geometric] test q_rel {results['geometric']['test']['q_rel_mean']:.3f}  "
              f"force_rel mean {results['geometric']['test']['force_rel_mean']:.3f}  "
              f"median {results['geometric']['test']['force_rel_median']:.3f}  ({results['geometric']['seconds']:.1f}s)")

    # ---- ridge
    if "ridge" in wanted:
        t0 = time.time()
        ridge = RidgeMarkerModel.fit(
            vds, split.train, markers, model.mesh,
            ridge_lambda=float(mk_cfg.get("ridge_lambda", 1e-2)),
            use_noisy=bool(mk_cfg.get("use_noisy", True)),
            q=Q,
        )
        ridge.save(ckpt_dir / "ridge_marker.npz")
        pipe = attach("ridge", lambda idx, m=ridge: m.predict(vds, idx))
        results["ridge"] = {
            "train": pipe.evaluate(vds, np.nonzero(split.train)[0], fem_U=fem.U),
            "test": pipe.evaluate(vds, np.nonzero(split.test)[0], fem_U=fem.U),
            "seconds": time.time() - t0,
        }
        print(f"[ridge]     test q_rel {results['ridge']['test']['q_rel_mean']:.3f}  "
              f"force_rel mean {results['ridge']['test']['force_rel_mean']:.3f}  "
              f"median {results['ridge']['test']['force_rel_median']:.3f}  ({results['ridge']['seconds']:.1f}s)")

    # ---- cnn
    if "cnn" in wanted:
        try:
            from src.vision.image_model import train_image_cnn
            t0 = time.time()
            cnn_cfg = cfg.get("image_cnn", {})
            epochs = int(args.epochs if args.epochs is not None else cnn_cfg.get("epochs", 40))
            cnn, hist = train_image_cnn(
                vds, split.train, split.test,
                epochs=epochs,
                batch_size=int(cnn_cfg.get("batch_size", 64)),
                lr=float(cnn_cfg.get("lr", 1e-3)),
                weight_decay=float(cnn_cfg.get("weight_decay", 1e-4)),
                hidden=int(cnn_cfg.get("hidden", 64)),
                dropout=float(cnn_cfg.get("dropout", 0.1)),
                patience=int(cnn_cfg.get("patience", 8)),
                seed=seed,
            )
            cnn.save(ckpt_dir / "image_cnn.pt")
            pipe = attach("cnn", lambda idx, m=cnn: m.predict_dataset(vds, idx))
            results["cnn"] = {
                "train": pipe.evaluate(vds, np.nonzero(split.train)[0], fem_U=fem.U),
                "test": pipe.evaluate(vds, np.nonzero(split.test)[0], fem_U=fem.U),
                "seconds": time.time() - t0,
                "history": hist,
            }
            print(f"[cnn]       test q_rel {results['cnn']['test']['q_rel_mean']:.3f}  "
                  f"force_rel mean {results['cnn']['test']['force_rel_mean']:.3f}  "
                  f"median {results['cnn']['test']['force_rel_median']:.3f}  "
                  f"({results['cnn']['seconds']:.1f}s, best epoch {hist['best_epoch']})")
            # learning curve
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(hist["train_rmse"], label="train")
            ax.plot(hist["val_rmse"], label="val (held-out contacts)")
            ax.set(xlabel="epoch", ylabel="normalised q RMSE", title="image CNN training")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(run_dir / "cnn_training.png", dpi=120)
        except ImportError as e:
            print(f"[cnn] skipped: {e}")

    # ---- oracle q
    if "oracle" in wanted:
        t0 = time.time()
        pipe = attach("oracle", lambda idx: vds.q[idx])
        results["oracle"] = {
            "train": pipe.evaluate(vds, np.nonzero(split.train)[0], fem_U=fem.U),
            "test": pipe.evaluate(vds, np.nonzero(split.test)[0], fem_U=fem.U),
            "seconds": time.time() - t0,
        }
        print(f"[oracle]    test q_rel {results['oracle']['test']['q_rel_mean']:.3e}  "
              f"force_rel mean {results['oracle']['test']['force_rel_mean']:.3e}  "
              f"median {results['oracle']['test']['force_rel_median']:.3e}  ({results['oracle']['seconds']:.1f}s)")

    # strip large arrays for JSON (keep summaries + per-sample vectors as lists)
    json_results = {}
    for name, block in results.items():
        json_results[name] = {}
        for split_name, metrics in block.items():
            if split_name in ("seconds", "history"):
                json_results[name][split_name] = metrics
                continue
            keep = {k: v for k, v in metrics.items() if k not in ("q_hat", "F_hat", "force_rel_by_sample", "q_rel_by_sample", "mag_hat", "mag_true")}
            # store compact per-sample for plots reload
            keep["force_rel_by_sample"] = metrics["force_rel_by_sample"]
            keep["mag_true"] = metrics["mag_true"]
            keep["mag_hat"] = metrics["mag_hat"]
            json_results[name][split_name] = keep

    save_json({"split": split.meta, "models": json_results, "q_modes": r, "seed": seed}, run_dir / "metrics.json")
    # rebuild mag arrays as numpy for plotting
    for name in results:
        for sp in ("train", "test"):
            if sp in results[name]:
                for k in ("mag_true", "mag_hat", "force_rel_by_sample", "q_rel_by_sample"):
                    results[name][sp][k] = np.asarray(results[name][sp][k])
    _plot_results(run_dir, results, vds, split)
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
