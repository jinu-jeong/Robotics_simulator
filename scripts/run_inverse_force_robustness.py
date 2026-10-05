#!/usr/bin/env python
"""Milestone 3b – robustness of the inverse force solver to *systematic* model errors.

The 900-sample evaluation (run_inverse_force_eval.py) uses the same FEM to
generate and to invert the displacement field, so exact recovery is guaranteed
by construction. Real measurements differ from the model in systematic ways
that least squares cannot average out. This script quantifies three of them:

(a) mesh / discretisation mismatch
    "truth" from a nested, ``--refine``× finer mesh (softer, closer to the
    converged solution) sampled at the nodes of the default model, inverted
    with the default (over-stiff T4) model -> constant over-estimation.
(b) contact-location error
    same mesh, but the contact handed to the solver is shifted by (dx, dy)
    along the finger / across its width.
(c) Young's-modulus error
    the model uses E_model = (1 + δ) E_true; the estimate scales exactly with
    E_model / E_true (verified numerically).

Both LS formulations are evaluated in (a) and (b). Outputs land in
results/etc/inverse_force_robustness/<timestamp>/ (robustness.png, metrics.json).

Run:
    python scripts/run_inverse_force_robustness.py                 # ~1 min (fine mesh 100x12x20)
    python scripts/run_inverse_force_robustness.py --refine 1      # skip the fine solve (mesh mismatch = 0)
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contact.inverse_force import InverseForceSolver  # noqa: E402
from src.contact.observation import observation_from_name  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.geometry.mesh_utils import coincident_node_map  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="fem")
    p.add_argument("--dataset-config", default="dataset", help="contact grid of the dataset (for the mismatch sweep)")
    p.add_argument("--refine", type=int, default=2, help="refinement factor of the 'truth' mesh (1 = off)")
    p.add_argument("--force", type=float, default=1.0)
    p.add_argument("--observations", nargs="*", default=["full", "top:z"])
    p.add_argument("--out", default="results/etc/inverse_force_robustness")
    args = p.parse_args()

    cfg = load_config(args.config)
    ds_cfg = load_config(args.dataset_config)["contact"]
    F = args.force
    run_dir = make_run_dir(args.out)

    model = FingerFEMModel.from_config(cfg)
    model.restrict_contact_surface("top")
    g, m = model.geometry, cfg["mesh"]
    solvers = {n: InverseForceSolver(model.fem, model.contact, observation_from_name(n, model.mesh, model.partition, g)) for n in args.observations}
    full = solvers.get("full") or InverseForceSolver(model.fem, model.contact)
    print(f"model mesh {m['nx']}x{m['ny']}x{m['nz']} ({model.mesh.n_dofs} dofs)")

    xr, yr = ds_cfg["x_rel"], ds_cfg["y_rel"]
    x_rels = np.linspace(xr["min"], xr["max"], int(xr["n"]))
    y_rels = np.linspace(yr["min"], yr["max"], int(yr["n"]))

    # ------------------------------------------------------------------ (a) mesh mismatch
    mismatch = None
    if args.refine > 1:
        r = args.refine
        t0 = time.time()
        fine = FingerFEMModel.from_config(cfg, r * m["nx"], r * m["ny"], r * m["nz"])
        fine.restrict_contact_surface("top")
        node_map = coincident_node_map(model.mesh.nodes, fine.mesh.nodes)
        print(f"\n(a) truth mesh {r * m['nx']}x{r * m['ny']}x{r * m['nz']} ({fine.mesh.n_dofs} dofs, factorised in {time.time() - t0:.0f}s)")
        lam = {n: np.zeros((len(x_rels), len(y_rels))) for n in list(solvers) + ["force-space"]}
        tip_ratio = np.zeros((len(x_rels), len(y_rels)))
        for i, xr_ in enumerate(x_rels):
            for j, yr_ in enumerate(y_rels):
                pnt = g.point_from_relative([xr_, yr_, 1.0])
                res_f, _ = fine.solve_normal_contact(pnt, F)
                u_truth = res_f.u[node_map]
                c = model.contact.locate(pnt)
                for n, s in solvers.items():
                    lam[n][i, j] = s.estimate(u_truth, c).magnitude
                lam["force-space"][i, j] = full.estimate(u_truth, c, method="force").magnitude
                res_c, _ = model.solve_normal_contact(pnt, F)
                tip_ratio[i, j] = res_f.max_displacement() / res_c.max_displacement()
        mismatch = {"x_rel": x_rels, "y_rel": y_rels, "lambda_est": lam, "max_u_ratio_fine_over_coarse": tip_ratio}
        for n, arr in lam.items():
            e = arr / F - 1.0
            print(f"  {n:>12s}: λ̂/λ − 1 = {100 * e.mean():+6.2f} % (min {100 * e.min():+6.2f}, max {100 * e.max():+6.2f})")
        print(f"  max|u| fine / coarse = {tip_ratio.mean():.4f} (coarse T4 mesh is stiffer)")

    # ------------------------------------------------------------------ (b) contact-location error
    print("\n(b) contact location error (same mesh)")
    dx = np.array([-10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10]) * 1e-3
    dy = np.array([0, 0.5, 1, 2, 3, 4, 5, 6]) * 1e-3
    bases = [(0.5, 0.5), (0.9, 0.5), (0.7, 0.3)]
    loc = {"dx": dx, "dy": dy, "bases": bases, "lambda_dx": {}, "lambda_dy": {}, "residual_dx": {}, "residual_dy": {}}
    for b in bases:
        res, c_true = model.solve_normal_contact(g.point_from_relative([*b, 1.0]), F)
        for key, shifts, axis in (("dx", dx, 0), ("dy", dy, 1)):
            lam_full, lam_z, lam_force, resid = [], [], [], []
            for s in shifts:
                q = c_true.position.copy()
                q[axis] += s
                c_wrong = model.contact.locate(q)
                e = full.estimate(res.u, c_wrong)
                lam_full.append(e.magnitude)
                resid.append(e.residual_rel)
                lam_z.append(solvers["top:z"].estimate(res.u, c_wrong).magnitude if "top:z" in solvers else np.nan)
                lam_force.append(full.estimate(res.u, c_wrong, method="force").magnitude)
            loc[f"lambda_{key}"][str(b)] = {"full": lam_full, "top:z": lam_z, "force-space": lam_force}
            loc[f"residual_{key}"][str(b)] = resid
        lx = np.array(loc["lambda_dx"][str(b)]["full"]) / F
        sens = np.polyfit(dx * 1e3, lx, 1)[0]
        print(f"  base {b}: dλ/dx ≈ {100 * sens:+.2f} %/mm along the finger;  "
              f"dy = 3 mm -> {100 * (loc['lambda_dy'][str(b)]['full'][4] / F - 1):+.2f} %;  "
              f"force-space at dx=+1 mm -> {loc['lambda_dx'][str(b)]['force-space'][8] / F:.3f}")

    # ------------------------------------------------------------------ (c) Young's modulus error
    print("\n(c) Young's modulus error (model E = (1+δ) E_true)")
    deltas = np.array([-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2])
    res, c_true = model.solve_normal_contact(g.point_from_relative([0.9, 0.5, 1.0]), F)
    lam_E = []
    for d in deltas:
        cfg_d = copy.deepcopy(cfg)
        cfg_d["material"]["youngs_modulus"] = float(cfg["material"]["youngs_modulus"]) * (1.0 + d)
        model_d = FingerFEMModel.from_config(cfg_d)
        model_d.restrict_contact_surface("top")
        lam_E.append(InverseForceSolver(model_d.fem, model_d.contact).estimate(res.u, c_true).magnitude)
    lam_E = np.array(lam_E)
    print("  δE      : " + " ".join(f"{100 * d:+6.0f}%" for d in deltas))
    print("  λ̂/λ − 1 : " + " ".join(f"{100 * (l / F - 1):+6.1f}%" for l in lam_E) + "   (expected: identical to δE)")
    assert np.allclose(lam_E / F, 1.0 + deltas, rtol=1e-6)

    # ------------------------------------------------------------------ plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    if mismatch is not None:
        for n, arr in mismatch["lambda_est"].items():
            if n == "force-space":
                continue
            ax[0].plot(x_rels, 100 * (arr.mean(axis=1) / F - 1), "o-", label=f"displacement-space, obs={n}")
        fs = mismatch["lambda_est"]["force-space"].mean(axis=1) / F - 1
        ax[0].plot(x_rels, 100 * fs, "x--", color="C3", label=f"force-space (mean {100 * fs.mean():+.0f} %)")
        ax[0].plot(x_rels, 100 * (mismatch["max_u_ratio_fine_over_coarse"].mean(axis=1) - 1), "k:", label="max|u| fine/coarse − 1")
        ax[0].set_ylim(-5, 30)
        ax[0].set(xlabel="contact x / L", ylabel="λ̂ / λ_true − 1  [%]",
                  title=f"(a) truth on {args.refine}× finer mesh, inverted with the model mesh")
        ax[0].legend(fontsize=8)
    else:
        ax[0].set_title("(a) skipped (--refine 1)")
    ax[0].grid(alpha=0.3)
    for k, b in enumerate(bases):
        ax[1].plot(1e3 * dx, 100 * (np.array(loc["lambda_dx"][str(b)]["full"]) / F - 1), "o-", color=f"C{k}", label=f"along x, contact {b}")
        ax[1].plot(1e3 * dy, 100 * (np.array(loc["lambda_dy"][str(b)]["full"]) / F - 1), "s--", color=f"C{k}", label=f"across y, contact {b}")
    ax[1].set(xlabel="contact position error [mm]", ylabel="λ̂ / λ_true − 1  [%]", title="(b) contact-location error (displacement-space)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    ax[2].plot(100 * deltas, 100 * (lam_E / F - 1), "o-", label="numerical")
    ax[2].plot(100 * deltas, 100 * deltas, "k--", lw=1, label="λ̂/λ = E_model/E_true")
    ax[2].set(xlabel="Young's modulus error δE [%]", ylabel="λ̂ / λ_true − 1  [%]", title="(c) material error")
    ax[2].grid(alpha=0.3)
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "robustness.png", dpi=130)

    def _j(o):
        if isinstance(o, dict):
            return {str(k): _j(v) for k, v in o.items()}
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (list, tuple)):
            return [_j(v) for v in o]
        return o

    save_json({
        "model_mesh": m, "refine": args.refine, "force": F,
        "mesh_mismatch": _j(mismatch), "contact_location": _j(loc),
        "youngs_modulus": {"delta": deltas.tolist(), "lambda_est": lam_E.tolist()},
    }, run_dir / "metrics.json")
    print(f"\nresults -> {run_dir}")


if __name__ == "__main__":
    main()
