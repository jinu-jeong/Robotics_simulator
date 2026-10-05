#!/usr/bin/env python
"""Marker-free experiment – train ResNet-18: RGB image → reduced coordinates q.

Markers are treated as training-time privileged information:

    marker-on image  → existing marker pipeline → q_marker   (teacher, stored in the dataset)
    FEM state        → Φ_rᵀ u                   → q_sim      (exact simulation state)
    marker-free image → ResNet-18               → q_pred     (student)

    L_q = MSE( (q_pred − μ) / σ , (q_target − μ) / σ ),   q_target ∈ {q_sim, q_marker}

The network predicts only q (same POD basis / dimension as q_sim); force and
contact follow from the existing ROM at evaluation / deployment time
(``scripts/eval_image_to_q.py``). Per-mode normalisation statistics (μ, σ) are
saved with the checkpoint (``*.pt`` and ``*.norm.json``).

Splits: contact-wise hold-out via ``make_vision_split`` (held-out contacts are
the validation set for early stopping).

Run:
    python scripts/train_image_to_q.py                          # --target sim, --input raw
    python scripts/train_image_to_q.py --target marker          # supervise with the marker teacher
    python scripts/train_image_to_q.py --target marker --mode-weight snr  # teacher target, noisy modes down-weighted
    python scripts/train_image_to_q.py --input marker           # sanity: network sees the marker-on images
    python scripts/train_image_to_q.py --epochs 5 --limit 200   # quick check
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
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402
from src.vision.paired_dataset import PairedVisionDataset  # noqa: E402
from src.vision.splits import make_vision_split  # noqa: E402


def _plot_history(run_dir: Path, hist: dict, target: str, inp: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(hist["train_loss"], label="train loss (normalised MSE)")
    axes[0].plot(np.asarray(hist["val_rmse_norm"]) ** 2, label="val (normalised MSE)")
    axes[0].axvline(hist["best_epoch"], color="k", ls=":", lw=1)
    axes[0].set(xlabel="epoch", yscale="log", title=f"ResNet-18 {inp} image → q_{target}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].plot(hist["val_q_rel_target"], label=f"‖q̂−q_{target}‖/‖q_{target}‖")
    if np.isfinite(hist["val_q_rel_reference"]).any():
        axes[1].plot(hist["val_q_rel_reference"], label="‖q̂−q_sim‖/‖q_sim‖")
    axes[1].set(xlabel="epoch", ylabel="relative q error (held-out contacts)", ylim=(0, None))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=120)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="markerfree")
    p.add_argument("--dataset", default=None, help="paired dataset npz (default: config output)")
    p.add_argument("--target", choices=["sim", "marker"], default=None, help="supervision: q_sim or q_marker")
    p.add_argument("--input", choices=["raw", "marker"], default=None, help="image modality fed to the network")
    p.add_argument("--mode-weight", choices=["none", "snr"], default="none",
                   help="snr: weight each mode's loss by var(q_sim)/var(q_target) clipped to [0,1] "
                        "(down-weights noise-dominated modes of a marker teacher; no effect for --target sim)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="use only the first N paired samples (debug)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    m_cfg = cfg.get("model", {})
    target = args.target or str(m_cfg.get("target", "sim"))
    inp = args.input or str(m_cfg.get("input", "raw"))
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    epochs = int(args.epochs if args.epochs is not None else m_cfg.get("epochs", 40))

    from src.vision.resnet_q import teacher_snr_weights, train_resnet_q  # torch import deferred to give a clear error

    ds_path = Path(args.dataset or cfg["output"])
    pds = PairedVisionDataset.load(ds_path)
    if args.limit is not None and args.limit < len(pds):
        pds = pds.subset(np.arange(int(args.limit)))
    if pds.meta.get("self_contained"):
        # e.g. grasp-scene dataset: contacts stored per sample, fem_index = arange(S)
        contact_positions = pds.contact_position
    else:
        contact_positions = ContactDataset.load(cfg.get("fem_dataset") or pds.meta["fem_dataset"]).contact_position
    split_cfg = cfg.get("split", {})
    split = make_vision_split(pds.as_vision_dataset(), contact_positions,
                              hold_every=int(split_cfg.get("hold_every", 3)),
                              hold_force_above=split_cfg.get("hold_force_above", None))

    images = pds.images(inp)
    q_target = pds.q(target)
    q_ref = pds.q_sim if target == "marker" else None
    mode_w = None
    if args.mode_weight == "snr" and target != "sim":
        mode_w = teacher_snr_weights(q_target[split.train], pds.q_sim[split.train])
        print("SNR mode weights:", np.round(mode_w, 3).tolist())
    tag = f"{target}_{inp}" + ("_snr" if mode_w is not None else "")

    run_dir = make_run_dir(args.out or cfg.get("output_dir", "results/etc/markerfree"), name=f"train_{tag}")
    ckpt_dir = Path(args.checkpoint_dir or cfg.get("checkpoint_dir", "results/etc/checkpoints/markerfree"))
    save_yaml(cfg, run_dir / "config.yaml")
    print(f"paired dataset {ds_path}: {len(pds)} samples, images={inp} {pds.image_shape}, target=q_{target} (r={pds.r})")
    print(f"split: train {split.meta['n_train']}  val/test {split.meta['n_test']}  "
          f"(held {split.meta['n_held_contacts']}/{split.meta['n_contacts']} contacts)")

    t0 = time.time()
    model, hist = train_resnet_q(
        images, q_target, split.train, split.test,
        q_reference=q_ref,
        mode_weights=mode_w,
        epochs=epochs,
        batch_size=int(args.batch_size or m_cfg.get("batch_size", 32)),
        lr=float(args.lr or m_cfg.get("lr", 5e-4)),
        weight_decay=float(m_cfg.get("weight_decay", 1e-4)),
        patience=int(m_cfg.get("patience", 12)),
        augment=not args.no_augment and bool(m_cfg.get("augment", True)),
        width=int(args.width or m_cfg.get("width", 64)),
        seed=seed,
    )
    secs = time.time() - t0
    model.meta.update({
        "target": target, "input": inp, "dataset": str(ds_path), "rom_basis": pds.meta.get("rom_basis"),
        "q_modes": pds.r, "q_convention": "q = Phi_r^T u (same basis as q_sim)", "mode_weight": args.mode_weight if mode_w is not None else "none",
        "basis_fingerprint": pds.meta.get("basis_fingerprint"), "scene_appearance": pds.meta.get("scene_appearance"),
        "split": split.meta, "seconds": secs,
    })
    ckpt = model.save(ckpt_dir / f"resnet18_q_{tag}.pt")
    model.q_norm.save_json(run_dir / "q_norm.json")
    save_json({"history": hist, "meta": model.meta, "checkpoint": str(ckpt)}, run_dir / "training.json")
    _plot_history(run_dir, hist, target, inp)
    b = hist["best_epoch"]
    print(f"done in {secs:.0f}s  best epoch {b + 1}  val nRMSE {hist['val_rmse_norm'][b]:.4f}  "
          f"q_rel(target) {hist['val_q_rel_target'][b]:.3f}"
          + (f"  q_rel(q_sim) {hist['val_q_rel_reference'][b]:.3f}" if q_ref is not None else ""))
    print(f"checkpoint -> {ckpt}\nresults -> {run_dir}")


if __name__ == "__main__":
    main()
