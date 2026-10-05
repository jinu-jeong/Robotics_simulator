#!/usr/bin/env python
"""Milestone 4 – POD reduced-order model of the finger deformation.

1. Split the contact-sweep dataset by *contact location* (every k-th grid point is
   held out) – with a linear model all force levels of one contact are collinear,
   so only unseen contacts test generalisation.
2. POD basis Φ from the training snapshots (SVD, no mean-centring).
3. For r = 1 … r_max:
      displacement error   ‖u − Φ_r Φ_rᵀ u‖ / ‖u‖              (train / test)
      force error          u -> q̂ = Φ_rᵀ u -> λ̂ (reduced displacement-/force-space LS)
      noise                the same with u + ε, compared with the full-space estimators
4. Save the basis (data/processed/rom_basis_<name>.npz) and plots/metrics
   (results/etc/rom/<timestamp>/).

Run:
    python scripts/run_rom_analysis.py
    python scripts/run_rom_analysis.py data/processed/contact_sweep_top.npz --holdout 3 --r 1 2 4 8 16 32 --noise 1e-5 1e-4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contact.inverse_force import InverseForceSolver, add_displacement_noise  # noqa: E402
from src.fem.contact_dataset import ContactDataset  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.rom.pod import compute_pod, storage_rank_tolerance  # noqa: E402
from src.rom.reduced_mechanics import ReducedModel  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402

R_DEFAULT = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def contact_split(ds: ContactDataset, holdout: int):
    """Indices of train / test samples; every ``holdout``-th distinct contact is test."""
    keys = np.round(ds.contact_position, 9)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    test_contacts = np.arange(len(uniq))[1::holdout] if holdout > 0 else np.array([], dtype=int)
    is_test = np.isin(inv, test_contacts)
    return np.nonzero(~is_test)[0], np.nonzero(is_test)[0], len(uniq), len(test_contacts)


def rel_err(a, b):
    return np.abs(np.asarray(a) - b) / np.maximum(np.abs(b), 1e-300)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default="data/processed/contact_sweep_top.npz")
    p.add_argument("--holdout", type=int, default=3, help="hold out every k-th contact location (0 = none)")
    p.add_argument("--r", type=int, nargs="*", default=None)
    p.add_argument("--r-max", type=int, default=64, help="largest basis kept / saved (capped at the numerical rank)")
    p.add_argument("--noise", type=float, nargs="*", default=[1e-5, 1e-4], help="noise std σ [m] for the noisy force test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--basis-out", default=None, help="where to save the basis (default data/processed/rom_basis_<dataset>.npz)")
    p.add_argument("--out", default="results/etc/rom")
    args = p.parse_args()

    ds = ContactDataset.load(args.dataset)
    model = FingerFEMModel.from_config(ds.meta["fem_config"])
    if not np.allclose(model.mesh.nodes, ds.nodes):
        raise RuntimeError("rebuilt FEM mesh does not match the dataset mesh")
    run_dir = make_run_dir(args.out)
    rng = np.random.default_rng(args.seed)
    train, test, n_contacts, n_test_contacts = contact_split(ds, args.holdout)
    print(f"dataset {len(ds)} samples, {n_contacts} contacts -> train {len(train)} samples ({n_contacts - n_test_contacts} contacts), "
          f"test {len(test)} samples ({n_test_contacts} contacts)")

    # ------------------------------------------------------------------ POD
    t0 = time.time()
    basis = compute_pod(ds.snapshot_matrix(train), r_max=args.r_max, rel_tol=storage_rank_tolerance(ds.U.dtype),
                        meta={"dataset": str(args.dataset), "holdout": args.holdout,
                              "train_indices": train.tolist(), "test_indices": test.tolist()})
    s = basis.singular_values
    rank = basis.meta["numerical_rank"]
    print(f"POD of {len(train)} snapshots ({ds.n_dofs} dofs, {ds.U.dtype}): numerical rank {rank} "
          f"(σ_{rank}/σ_1 = {s[rank - 1] / s[0]:.1e}, σ_{rank + 1}/σ_1 = {s[rank] / s[0]:.1e} = storage noise floor), {time.time() - t0:.1f}s")
    for frac in (0.99, 0.999, 0.9999, 0.99999):
        print(f"  {100 * frac:.3f} % energy: r = {basis.modes_for_energy(frac)}")
    r_list = sorted({r for r in (args.r or R_DEFAULT) if r <= basis.r} | {basis.r})
    rom_full = ReducedModel.from_fem(model.fem, basis, model.contact)
    basis_path = Path(args.basis_out or f"data/processed/rom_basis_{Path(args.dataset).stem.replace('contact_sweep_', '')}.npz")
    basis.save(basis_path)
    print(f"basis saved -> {basis_path}  (r = {basis.r})")

    contacts = [model.contact.contact_from_face(ds.contact_face[i], ds.contact_bary[i]) for i in range(len(ds))]
    F_true = ds.force_magnitude
    U = np.asarray(ds.U, dtype=np.float64).reshape(len(ds), -1)  # (S, 3N)
    Q = U @ basis.Phi  # (S, r_max) exact projections
    norm_u = np.linalg.norm(U, axis=1)

    # noisy copies (same draw for every r) + full-space reference estimators
    full_inv = InverseForceSolver(model.fem, model.contact)
    noisy = {}
    for sigma in args.noise:
        Un = np.stack([add_displacement_noise(U[i], sigma, rng, model.partition.free_dofs) for i in range(len(ds))])
        ref = {m: np.array([full_inv.estimate(Un[i], contacts[i], method=m).magnitude for i in range(len(ds))]) for m in ("displacement", "force")}
        noisy[sigma] = {"Q": Un @ basis.Phi, "ref": ref}
        print(f"  σ={sigma:.0e}: full-space reference mean rel error  displacement {100 * rel_err(ref['displacement'], F_true).mean():.3f} %   "
              f"force {100 * rel_err(ref['force'], F_true).mean():.1f} %")

    # ------------------------------------------------------------------ r sweep
    rows = []
    print(f"\n{'r':>3} {'energy':>9} | {'disp rel err train':>18} {'test':>9} | {'force rel err (exact) disp-LS train/test':>40} {'force-LS train/test':>21} | "
          + " ".join(f"σ={sg:.0e} disp/force" for sg in args.noise))
    for r in r_list:
        rom = rom_full.truncate(r)
        # displacement reconstruction error from the projections (Φ orthonormal)
        res_norm = np.sqrt(np.maximum(norm_u**2 - np.sum(Q[:, :r] ** 2, axis=1), 0.0))
        d_rel = res_norm / norm_u
        d_rmse = res_norm / np.sqrt(ds.n_dofs)
        row = {"r": r, "energy": basis.energy_fraction(r),
               "disp_rel_train": float(d_rel[train].mean()), "disp_rel_test": float(d_rel[test].mean()) if len(test) else None,
               "disp_rmse_train": float(d_rmse[train].mean()), "disp_rmse_test": float(d_rmse[test].mean()) if len(test) else None,
               "disp_rel_test_max": float(d_rel[test].max()) if len(test) else None}
        for method in ("displacement", "force"):
            est = np.array([rom.estimate_force(Q[i, :r], contacts[i], method=method).magnitude for i in range(len(ds))])
            e = rel_err(est, F_true)
            row[f"force_{method}_train"] = float(e[train].mean())
            row[f"force_{method}_test"] = float(e[test].mean()) if len(test) else None
            row[f"force_{method}_test_max"] = float(e[test].max()) if len(test) else None
            for sigma in args.noise:
                Qn = noisy[sigma]["Q"]
                estn = np.array([rom.estimate_force(Qn[i, :r], contacts[i], method=method).magnitude for i in range(len(ds))])
                row[f"force_{method}_noise_{sigma:.0e}"] = float(rel_err(estn, F_true).mean())
        rows.append(row)
        te = lambda k: (f"{100 * row[k]:8.3f} %" if row[k] is not None else "     n/a")  # noqa: E731
        print(f"{r:3d} {100 * row['energy']:8.4f}% | {100 * row['disp_rel_train']:17.3f} % {te('disp_rel_test'):>9} | "
              f"{100 * row['force_displacement_train']:19.3f} % / {te('force_displacement_test')} {100 * row['force_force_train']:9.3f} % / {te('force_force_test')} | "
              + " ".join(f"{100 * row[f'force_displacement_noise_{sg:.0e}']:7.3f}/{100 * row[f'force_force_noise_{sg:.0e}']:7.2f} %" for sg in args.noise))

    # ------------------------------------------------------------------ plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = np.array([row["r"] for row in rows])
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    k = np.arange(1, len(s) + 1)
    ax[0, 0].semilogy(k, s / s[0], "o-", ms=3, label="σ_i / σ_1")
    ax[0, 0].semilogy(k, 1 - np.cumsum(s**2) / np.sum(s**2) + 1e-17, "s-", ms=3, label="1 − captured energy")
    ax[0, 0].axvline(basis.meta["numerical_rank"], color="grey", ls=":", label=f"numerical rank {basis.meta['numerical_rank']}")
    ax[0, 0].set(xlabel="mode index", ylabel="", title=f"POD spectrum ({len(train)} train snapshots, {n_contacts - n_test_contacts} contacts)",
                 xlim=(0, min(len(s), 100)))
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].loglog(rs, 100 * np.array([row["disp_rel_train"] for row in rows]), "o-", label="train contacts")
    if len(test):
        ax[0, 1].loglog(rs, 100 * np.array([row["disp_rel_test"] for row in rows]), "s-", label="held-out contacts (mean)")
        ax[0, 1].loglog(rs, 100 * np.array([row["disp_rel_test_max"] for row in rows]), "s--", color="C1", alpha=0.6, label="held-out (max)")
    ax[0, 1].set(xlabel="number of modes r", ylabel="relative displacement error ‖u − Φ_rΦ_rᵀu‖/‖u‖ [%]", title="deformation reconstruction")
    ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3, which="both")

    for method, mk in (("displacement", "o"), ("force", "s")):
        ax[1, 0].loglog(rs, 100 * np.maximum([row[f"force_{method}_train"] for row in rows], 1e-12), mk + "-", label=f"rom {method}-space, train")
        if len(test):
            ax[1, 0].loglog(rs, 100 * np.maximum([row[f"force_{method}_test"] for row in rows], 1e-12), mk + "--", label=f"rom {method}-space, held-out")
    ax[1, 0].set(xlabel="number of modes r", ylabel="mean relative force error [%]", title="force from q̂ = Φ_rᵀu (exact u)")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3, which="both")

    for j, sigma in enumerate(args.noise):
        for method, mk in (("displacement", "o"), ("force", "s")):
            ax[1, 1].loglog(rs, 100 * np.array([row[f"force_{method}_noise_{sigma:.0e}"] for row in rows]), mk + "-", color=f"C{j}",
                            alpha=1.0 if method == "displacement" else 0.6, label=f"rom {method}-space, σ={sigma:.0e}")
        ref = noisy[sigma]["ref"]
        ax[1, 1].axhline(100 * rel_err(ref["displacement"], F_true).mean(), color=f"C{j}", ls=":", label=f"full displacement-space, σ={sigma:.0e}")
    ax[1, 1].set(xlabel="number of modes r", ylabel="mean relative force error [%]", title="force from noisy u + ε (all samples)")
    ax[1, 1].legend(fontsize=7); ax[1, 1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(run_dir / "rom_analysis.png", dpi=130)

    # first modes as a picture strip (top-surface u_z)
    n_show = min(6, basis.r)
    fig, ax = plt.subplots(1, n_show, figsize=(2.6 * n_show, 3.2))
    top = np.nonzero(np.abs(ds.nodes[:, 2] - model.geometry.height) < 1e-9)[0]
    for kk in range(n_show):
        m = basis.mode(kk)
        sc = ax[kk].scatter(1e3 * ds.nodes[top, 0], 1e3 * ds.nodes[top, 1], c=m[top, 2], cmap="coolwarm", s=12, marker="s",
                            vmin=-np.abs(m[top, 2]).max(), vmax=np.abs(m[top, 2]).max())
        ax[kk].set(title=f"mode {kk + 1}, σ={s[kk] / s[0]:.2e}", xlabel="x [mm]", aspect="equal")
        ax[kk].set_yticks([])
    ax[0].set_ylabel("top-surface φ_z")
    fig.tight_layout()
    fig.savefig(run_dir / "pod_modes.png", dpi=130)

    save_json({"dataset": str(args.dataset), "holdout": args.holdout, "n_train": len(train), "n_test": len(test),
               "n_contacts": n_contacts, "n_test_contacts": n_test_contacts, "numerical_rank": basis.meta["numerical_rank"],
               "singular_values": s.tolist(), "basis_path": str(basis_path), "noise_sigmas": args.noise,
               "full_space_reference_noisy": {f"{sg:.0e}": {m: float(rel_err(v, F_true).mean()) for m, v in noisy[sg]["ref"].items()} for sg in args.noise},
               "sweep": rows}, run_dir / "metrics.json")
    print(f"\nresults -> {run_dir}")


if __name__ == "__main__":
    main()
