#!/usr/bin/env python
"""Milestone 3 – recover the contact force from the FEM displacement field (known contact).

For every sample of a contact-sweep dataset the normal force is estimated from
``u`` (and the known contact ``c``) with the two least-squares formulations of
``src/contact/inverse_force.py`` and compared with the ground truth:

* metrics: MAE, RMSE, relative / max error, y = x fit, error vs force level and
  vs contact location, force *direction* error of the vector estimate
* noise sensitivity: ``u + ε``, ε ~ N(0, σ²) for a sweep of σ, for both methods
  and several observation operators (full field, surface nodes, top-face z only)

Outputs in results/etc/inverse_force/<timestamp>/: metrics.json, per_sample.npz,
true_vs_estimated.png, error_maps.png, noise_sensitivity.png.

Run:
    python scripts/run_inverse_force_eval.py
    python scripts/run_inverse_force_eval.py data/processed/contact_sweep_top.npz --noise 0 1e-6 1e-5 1e-4 --repeats 3
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

from src.contact.inverse_force import (  # noqa: E402
    InverseForceSolver,
    add_displacement_noise,
    direction_error_deg,
    force_error_metrics,
)
from src.contact.observation import observation_from_name  # noqa: E402
from src.fem.contact_dataset import ContactDataset  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402

DEFAULT_NOISE = [0.0, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]


def contacts_for(ds: ContactDataset, model: FingerFEMModel):
    return [model.contact.contact_from_face(ds.contact_face[i], ds.contact_bary[i]) for i in range(len(ds))]


def evaluate(solver: InverseForceSolver, ds: ContactDataset, contacts, method: str, mode: str = "normal",
             sigma: float = 0.0, rng: np.random.Generator | None = None, indices=None) -> dict[str, np.ndarray]:
    idx = np.arange(len(ds)) if indices is None else np.asarray(indices)
    est_mag = np.zeros(len(idx))
    est_vec = np.zeros((len(idx), 3))
    clipped = np.zeros(len(idx), dtype=bool)
    for k, i in enumerate(idx):
        u = ds.displacement(i)
        if sigma > 0.0:
            u = add_displacement_noise(u, sigma, rng, solver.partition.free_dofs)
        e = solver.estimate(u, contacts[i], method=method, mode=mode)
        est_mag[k], est_vec[k], clipped[k] = e.magnitude, e.force_vector, e.clipped
    return {"indices": idx, "est_mag": est_mag, "est_vec": est_vec, "clipped": clipped}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default="data/processed/contact_sweep_top.npz")
    p.add_argument("--noise", type=float, nargs="*", default=None, help="noise std σ [m] sweep")
    p.add_argument("--repeats", type=int, default=1, help="random noise draws per sample and σ")
    p.add_argument("--observations", nargs="*", default=["full", "surface", "top:z"],
                   help="observation operators for the displacement-space method")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/etc/inverse_force")
    args = p.parse_args()
    sigmas = DEFAULT_NOISE if args.noise is None else list(args.noise)

    ds = ContactDataset.load(args.dataset)
    model = FingerFEMModel.from_config(ds.meta["fem_config"])
    if not np.allclose(model.mesh.nodes, ds.nodes):
        raise RuntimeError("rebuilt FEM mesh does not match the dataset mesh")
    contacts = contacts_for(ds, model)
    run_dir = make_run_dir(args.out)
    rng = np.random.default_rng(args.seed)
    F_true = ds.force_magnitude
    print(f"dataset {args.dataset}: {len(ds)} samples, {ds.n_dofs} dofs, F ∈ [{F_true.min():.2f}, {F_true.max():.2f}] N, "
          f"stored as {ds.U.dtype}")

    solvers = {name: InverseForceSolver(model.fem, model.contact, observation_from_name(name, model.mesh, model.partition, model.geometry))
               for name in args.observations}
    full = solvers.get("full") or InverseForceSolver(model.fem, model.contact)

    # ------------------------------------------------------------------ 1) exact u
    print("\n1) exact displacement field")
    exact = {}
    for method in ("force", "displacement"):
        t0 = time.time()
        r = evaluate(full, ds, contacts, method)
        m = force_error_metrics(F_true, r["est_mag"])
        m["time_per_sample_ms"] = 1e3 * (time.time() - t0) / len(ds)
        m["n_clipped"] = int(r["clipped"].sum())
        exact[method] = {"metrics": m, **r}
        print(f"  {method:>12s}: MAE {m['mae']:.2e} N  RMSE {m['rmse']:.2e} N  mean rel {100 * m['mean_rel_error']:.4f} %  "
              f"max rel {100 * m['max_rel_error']:.4f} %  slope {m['fit_slope']:.6f}  R² {m['r2']:.8f}  "
              f"({m['time_per_sample_ms']:.2f} ms/sample)")
    rv = evaluate(full, ds, contacts, "displacement", mode="vector")
    ang = direction_error_deg(ds.force_vector, rv["est_vec"])
    mag_vec = force_error_metrics(F_true, rv["est_mag"])
    exact["displacement_vector"] = {"metrics": mag_vec, "direction_error_deg_mean": float(ang.mean()),
                                    "direction_error_deg_max": float(ang.max()), **rv}
    print(f"  vector (3 unknowns): |F| mean rel {100 * mag_vec['mean_rel_error']:.4f} %, direction error mean {ang.mean():.4f}°, max {ang.max():.4f}°")

    for name, s in solvers.items():
        if name == "full":
            continue
        r = evaluate(s, ds, contacts, "displacement")
        m = force_error_metrics(F_true, r["est_mag"])
        exact[f"displacement_{name}"] = {"metrics": m, **r}
        print(f"  displacement, observation {name:>8s} ({s.observation.n_obs:5d} dofs): mean rel {100 * m['mean_rel_error']:.4f} %  max rel {100 * m['max_rel_error']:.4f} %")

    # error vs force level / vs contact location (force-space method shows the float32 storage effect)
    per_force, per_contact = {}, {}
    for method in ("force", "displacement"):
        err = np.abs(exact[method]["est_mag"] - F_true) / F_true
        per_force[method] = {f"{f:.3f}": float(err[F_true == f].mean()) for f in np.unique(F_true)}
        keys = np.round(ds.contact_position[:, :2], 9)
        uniq, inv_idx = np.unique(keys, axis=0, return_inverse=True)
        per_contact[method] = {"xy": uniq, "mean_rel_error": np.array([err[inv_idx == k].mean() for k in range(len(uniq))])}

    # ------------------------------------------------------------------ 2) noise sweep
    print(f"\n2) noise sensitivity: σ ∈ {sigmas} m, {args.repeats} draw(s) per sample"
          f"  (max|u| in dataset: {1e3 * ds.max_displacement.min():.3f} … {1e3 * ds.max_displacement.max():.3f} mm)")
    configs = [("force", "full")] + [("displacement", n) for n in solvers]
    noise = {f"{m}|{o}": {"sigma": [], "mean_rel_error": [], "rmse": [], "median_rel_error": [], "p90_rel_error": [], "frac_clipped": []}
             for m, o in configs}
    hdr = "  σ [m]     " + "".join(f"{m[:4]}|{o:>8s}   " for m, o in configs)
    print(hdr)
    for s in sigmas:
        line = f"  {s:8.1e}  "
        for m, o in configs:
            solver = full if o == "full" else solvers[o]
            rels, errs, clip = [], [], []
            for _ in range(args.repeats):
                r = evaluate(solver, ds, contacts, m, sigma=s, rng=rng)
                e = r["est_mag"] - F_true
                rels.append(np.abs(e) / F_true)
                errs.append(e)
                clip.append(r["clipped"])
            rels, errs, clip = np.concatenate(rels), np.concatenate(errs), np.concatenate(clip)
            d = noise[f"{m}|{o}"]
            d["sigma"].append(s)
            d["mean_rel_error"].append(float(rels.mean()))
            d["median_rel_error"].append(float(np.median(rels)))
            d["p90_rel_error"].append(float(np.percentile(rels, 90)))
            d["rmse"].append(float(np.sqrt(np.mean(errs**2))))
            d["frac_clipped"].append(float(clip.mean()))
            line += f"{100 * rels.mean():9.3f} %    "
        print(line)

    # ------------------------------------------------------------------ plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, method in zip(ax, ("force", "displacement")):
        est = exact[method]["est_mag"]
        m = exact[method]["metrics"]
        a.plot([0, F_true.max()], [0, F_true.max()], "k--", lw=1, label="y = x")
        a.scatter(F_true, est, s=10, alpha=0.6, c=ds.contact_position[:, 0] / model.geometry.length, cmap="viridis")
        a.set(xlabel="true normal force [N]", ylabel="estimated normal force [N]",
              title=f"{method}-space LS, exact u\nMAE {m['mae']:.1e} N, mean rel {100 * m['mean_rel_error']:.4f} %, slope {m['fit_slope']:.6f}")
        a.grid(alpha=0.3)
        a.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(run_dir / "true_vs_estimated.png", dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for method, mk in (("force", "o"), ("displacement", "s")):
        fs = np.array([float(k) for k in per_force[method]])
        ax[0].semilogy(fs, np.maximum(list(per_force[method].values()), 1e-16), mk + "-", label=method)
    ax[0].set(xlabel="true force [N]", ylabel="mean relative error", title="error vs force level (exact u)")
    ax[0].grid(alpha=0.3)
    ax[0].legend()
    for a, method in zip(ax[1:], ("force", "displacement")):
        pc = per_contact[method]
        sc = a.scatter(1e3 * pc["xy"][:, 0], 1e3 * pc["xy"][:, 1], c=np.maximum(pc["mean_rel_error"], 1e-16), s=80, marker="s",
                       norm=matplotlib.colors.LogNorm(), cmap="magma")
        a.set(xlabel="contact x [mm]", ylabel="contact y [mm]", title=f"{method}-space: mean rel error vs contact")
        a.set_xlim(0, 1e3 * model.geometry.length)
        a.set_ylim(0, 1e3 * model.geometry.width)
        a.set_aspect("equal")
        fig.colorbar(sc, ax=a, shrink=0.8)
    fig.tight_layout()
    fig.savefig(run_dir / "error_maps.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for key, d in noise.items():
        s = np.array(d["sigma"])
        pos = s > 0
        ax[0].loglog(s[pos], np.maximum(np.array(d["mean_rel_error"])[pos], 1e-9), "o-", label=key)
        ax[1].semilogx(s[pos], np.array(d["frac_clipped"])[pos], "o-", label=key)
    umax = ds.max_displacement
    for a in ax:
        a.axvspan(umax.min() * 1e-3, umax.max() * 1e-3, color="grey", alpha=0.15, label="0.1 % of max|u|")
        a.axvspan(umax.min() * 1e-2, umax.max() * 1e-2, color="grey", alpha=0.3, label="1 % of max|u|")
        a.grid(alpha=0.3, which="both")
    ax[0].set(xlabel="noise std σ [m]", ylabel="mean relative force error", title="noise sensitivity  (u + ε, ε ~ N(0, σ²) per DOF)")
    ax[0].axhline(0.05, color="r", lw=0.8, ls=":")
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel="noise std σ [m]", ylabel="fraction of estimates clipped to 0", title="non-negativity constraint active")
    fig.tight_layout()
    fig.savefig(run_dir / "noise_sensitivity.png", dpi=130)

    # ------------------------------------------------------------------ save
    np.savez_compressed(
        run_dir / "per_sample.npz", F_true=F_true, contact_position=ds.contact_position,
        **{f"est_{k}": v["est_mag"] for k, v in exact.items()}, est_vec_displacement=exact["displacement_vector"]["est_vec"],
        direction_error_deg=ang,
    )
    save_json({
        "dataset": str(args.dataset), "n_samples": len(ds), "n_dofs": ds.n_dofs, "storage_dtype": str(ds.U.dtype),
        "exact": {k: {kk: vv for kk, vv in v.items() if kk in ("metrics", "direction_error_deg_mean", "direction_error_deg_max")} for k, v in exact.items()},
        "error_vs_force": per_force,
        "error_vs_contact": {m: {"xy": v["xy"].tolist(), "mean_rel_error": v["mean_rel_error"].tolist()} for m, v in per_contact.items()},
        "noise": noise, "noise_repeats": args.repeats, "seed": args.seed,
        "observations": {n: s.observation.n_obs for n, s in solvers.items()},
    }, run_dir / "metrics.json")
    print(f"\nresults -> {run_dir}")


if __name__ == "__main__":
    main()
