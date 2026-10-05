#!/usr/bin/env python
"""Milestone 1/2 – Does the contact-sweep FEM agree with strength-of-materials beam theory?

The cantilever benchmark only checked a centred tip load. Dataset samples have
(a) the load at an interior station ``x = a`` and (b) a lateral eccentricity
``e = y - W/2`` that twists the finger. This script compares the FEM against:

* bending  – Euler–Bernoulli / Timoshenko point-load deflection curve ``w(x)``
* torsion  – Saint-Venant twist ``θ(x) = T min(x, a) / (G J)`` of a rectangle

FEM section quantities come from a per-station least-squares fit
``u_z = c0 + c1 (y - W/2) + c2 (z - H/2)`` (c0: mean deflection, c1: twist),
which separates bending from torsion and filters the local dent under the load.

Outputs (results/etc/beam_theory/<timestamp>/): comparison.png, metrics.json, and
an interactive viewer where ``A`` toggles FEM <-> beam-theory displacement.

Run:
    python scripts/run_beam_theory_comparison.py                 # default mesh + one refinement
    python scripts/run_beam_theory_comparison.py --fine 100 10 20 # add a finer mesh for the curves
    python scripts/run_beam_theory_comparison.py --headless
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

from src.fem.beam_theory import PointLoadCantilever, section_fit  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402


def theory_for(model: FingerFEMModel, contact, F: float) -> PointLoadCantilever:
    g = model.geometry
    return PointLoadCantilever(g.length, g.width, g.height, model.material.E, model.material.nu,
                              force_z=-F, a=float(contact.position[0]), e=float(contact.position[1] - 0.5 * g.width))


def fem_sections(model: FingerFEMModel, u: np.ndarray):
    g = model.geometry
    return section_fit(model.mesh.nodes, u, g.width, g.height)


def rel(a: float, b: float) -> float:
    return (a - b) / b if abs(b) > 0 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="fem")
    p.add_argument("--force", type=float, default=1.0)
    p.add_argument("--curve-rel", type=float, nargs=2, default=(0.98, 0.85), metavar=("AX", "AY"),
                   help="contact (x/L, y/W) for the deflection/twist curves (default = dataset sample 899)")
    p.add_argument("--fine", type=int, nargs=3, action="append", default=None, metavar=("NX", "NY", "NZ"),
                   help="extra mesh resolutions for the curve comparison (default: 80 8 16)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--out", default="results/etc/beam_theory")
    args = p.parse_args()
    fine_meshes = args.fine if args.fine else [(80, 8, 16)]

    cfg = load_config(args.config)
    ds_cfg = load_config("dataset")["contact"]
    F = args.force
    run_dir = make_run_dir(args.out)

    t0 = time.time()
    model = FingerFEMModel.from_config(cfg)
    model.restrict_contact_surface("top")
    g = model.geometry
    print(f"default mesh {cfg['mesh']} -> {model.mesh.n_dofs} dofs (build {time.time() - t0:.1f}s)")
    print(f"beam: L/H = {g.length / g.height:.0f}, W/H = {g.width / g.height:.1f}, "
          f"J = {PointLoadCantilever(g.length, g.width, g.height, 1, 0.4, 1, 1).J:.3e} m^4\n")

    # ------------------------------------------------------------------ A) bending vs load position (centre line)
    xr = ds_cfg["x_rel"]
    a_rels = np.linspace(xr["min"], xr["max"], int(xr["n"]))
    rows_a = []
    print(f"A) centred load F={F} N at x=a : tip deflection (section mean at x=L)  [mm]")
    print(f"{'a/L':>5} {'FEM':>8} {'Timosh.':>8} {'E-B':>8} {'err vs T':>9} {'err vs EB':>9}")
    for ar in a_rels:
        res, c = model.solve_normal_contact(g.point_from_relative([ar, 0.5, 1.0]), F)
        th = theory_for(model, c, F)
        xs, w_fem, _ = fem_sections(model, res.u)
        d_fem, d_t, d_eb = w_fem[-1], th.tip_deflection(True), th.tip_deflection(False)
        rows_a.append({"a_rel": float(ar), "tip_fem": float(d_fem), "tip_timoshenko": d_t, "tip_euler": d_eb,
                       "rel_err_timoshenko": rel(d_fem, d_t), "rel_err_euler": rel(d_fem, d_eb)})
        print(f"{ar:5.2f} {1e3 * d_fem:8.4f} {1e3 * d_t:8.4f} {1e3 * d_eb:8.4f} {100 * rel(d_fem, d_t):8.1f}% {100 * rel(d_fem, d_eb):8.1f}%")

    # ------------------------------------------------------------------ B) torsion vs eccentricity (a = 0.9 L)
    yr = ds_cfg["y_rel"]
    y_rels = np.linspace(yr["min"], yr["max"], int(yr["n"]))
    rows_b = []
    # Mesh-asymmetry bias: the Kuhn tet split is not mirror-symmetric in y, so a
    # centred load produces a small spurious twist. Report it and subtract it.
    res0, _ = model.solve_normal_contact(g.point_from_relative([0.9, 0.5, 1.0]), F)
    bias = float(fem_sections(model, res0.u)[2][-1])
    print(f"\nB) F={F} N at a=0.9L, eccentric y : tip twist angle (section fit c1 at x=L)  [mrad]")
    print(f"   spurious twist of the centred load (mesh asymmetry bias): {1e3 * bias:+.4f} mrad")
    print(f"{'y/W':>5} {'e [mm]':>7} {'FEM':>8} {'FEM-bias':>8} {'St-Venant':>9} {'err':>7} {'err(-bias)':>10} {'tip w FEM':>10} {'tip w T':>8}")
    for yr_ in y_rels:
        res, c = model.solve_normal_contact(g.point_from_relative([0.9, yr_, 1.0]), F)
        th = theory_for(model, c, F)
        xs, w_fem, t_fem = fem_sections(model, res.u)
        t_th, t_f, t_fc = th.tip_twist(), float(t_fem[-1]), float(t_fem[-1] - bias)
        ecc = abs(th.e) > 1e-9
        rows_b.append({"y_rel": float(yr_), "e": th.e, "twist_fem": t_f, "twist_fem_debiased": t_fc, "twist_saint_venant": t_th,
                       "rel_err": rel(t_f, t_th) if ecc else None, "rel_err_debiased": rel(t_fc, t_th) if ecc else None,
                       "tip_fem": float(w_fem[-1]), "tip_timoshenko": th.tip_deflection()})
        err = f"{100 * rel(t_f, t_th):6.1f}%" if ecc else "   n/a"
        errc = f"{100 * rel(t_fc, t_th):9.1f}%" if ecc else "       n/a"
        print(f"{yr_:5.2f} {1e3 * th.e:7.2f} {1e3 * t_f:8.4f} {1e3 * t_fc:8.4f} {1e3 * t_th:9.4f} {err} {errc} {1e3 * w_fem[-1]:10.4f} {1e3 * th.tip_deflection():8.4f}")

    # ------------------------------------------------------------------ C) full curves w(x), θ(x) for one eccentric contact, mesh refinement
    ax_rel, ay_rel = args.curve_rel
    curves = []
    print(f"\nC) curves for contact ({ax_rel:.2f} L, {ay_rel:.2f} W), F={F} N")
    meshes = [(cfg["mesh"]["nx"], cfg["mesh"]["ny"], cfg["mesh"]["nz"])] + [tuple(m) for m in fine_meshes]
    viewer_payload = None
    for (nx, ny, nz) in meshes:
        t0 = time.time()
        m = model if (nx, ny, nz) == meshes[0] else FingerFEMModel.from_config(cfg, nx, ny, nz)
        m.restrict_contact_surface("top")
        res, c = m.solve_normal_contact(g.point_from_relative([ax_rel, ay_rel, 1.0]), F)
        res0, _ = m.solve_normal_contact(g.point_from_relative([ax_rel, 0.5, 1.0]), F)
        th = theory_for(m, c, F)
        xs, w_fem, t_fem = fem_sections(m, res.u)
        t_bias = fem_sections(m, res0.u)[2]  # spurious twist of the centred load (mesh asymmetry)
        t_fem = t_fem - t_bias
        w_th, t_th = th.bending_deflection(xs), th.twist(xs)
        curves.append({"mesh": [nx, ny, nz], "n_dofs": m.mesh.n_dofs, "x": xs, "w_fem": w_fem, "w_theory": w_th,
                       "twist_fem_debiased": t_fem, "twist_bias": t_bias, "twist_theory": t_th,
                       "tip_deflection_err": rel(w_fem[-1], w_th[-1]), "tip_twist_err_debiased": rel(t_fem[-1], t_th[-1]),
                       "load_point_uz_fem": float(res.u[c.nearest_node, 2]),
                       "load_point_uz_theory": float(th.displacement_field(m.mesh.nodes)[c.nearest_node, 2])})
        print(f"  mesh {nx}x{ny}x{nz} ({m.mesh.n_dofs} dofs, {time.time() - t0:.1f}s): "
              f"tip w FEM {1e3 * w_fem[-1]:.4f} / theory {1e3 * w_th[-1]:.4f} mm ({100 * rel(w_fem[-1], w_th[-1]):+.1f}%),  "
              f"tip θ FEM(-bias) {1e3 * t_fem[-1]:.4f} / theory {1e3 * t_th[-1]:.4f} mrad ({100 * rel(t_fem[-1], t_th[-1]):+.1f}%, bias {1e3 * t_bias[-1]:+.3f}),  "
              f"u_z under load: FEM {1e3 * res.u[c.nearest_node, 2]:.4f} / beam {1e3 * th.displacement_field(m.mesh.nodes)[c.nearest_node, 2]:.4f} mm")
        if viewer_payload is None:
            viewer_payload = (m, res, c, th)

    # ------------------------------------------------------------------ plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    aL = np.array([r["a_rel"] for r in rows_a])
    ax[0, 0].plot(aL, 1e3 * np.array([r["tip_euler"] for r in rows_a]), "k--", label="Euler–Bernoulli")
    ax[0, 0].plot(aL, 1e3 * np.array([r["tip_timoshenko"] for r in rows_a]), "k-", label="Timoshenko")
    ax[0, 0].plot(aL, 1e3 * np.array([r["tip_fem"] for r in rows_a]), "o", color="C0", label=f"FEM {meshes[0]}")
    ax[0, 0].set(xlabel="load position a / L", ylabel="tip deflection [mm]", title=f"Bending: centred load F={F} N at x=a")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    e_mm = 1e3 * np.array([r["e"] for r in rows_b])
    ax[0, 1].plot(e_mm, 1e3 * np.array([r["twist_saint_venant"] for r in rows_b]), "k-", label="Saint-Venant  T·a/(GJ)")
    ax[0, 1].plot(e_mm, 1e3 * np.array([r["twist_fem"] for r in rows_b]), "x", color="C1", label=f"FEM {meshes[0]} raw")
    ax[0, 1].plot(e_mm, 1e3 * np.array([r["twist_fem_debiased"] for r in rows_b]), "o", color="C1", label="FEM − mesh bias")
    ax[0, 1].set(xlabel="eccentricity e = y − W/2 [mm]", ylabel="tip twist angle [mrad]", title="Torsion: F=%.1f N at a=0.9 L" % F)
    ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

    for k, cv in enumerate(curves):
        lab = f"FEM {cv['mesh'][0]}x{cv['mesh'][1]}x{cv['mesh'][2]}"
        ax[1, 0].plot(1e3 * cv["x"], 1e3 * cv["w_fem"], "-", color=f"C{k}", label=lab)
        ax[1, 1].plot(1e3 * cv["x"], 1e3 * cv["twist_fem_debiased"], "-", color=f"C{k}", label=lab)
    ax[1, 0].plot(1e3 * curves[-1]["x"], 1e3 * curves[-1]["w_theory"], "k--", label="Timoshenko w(x)")
    ax[1, 1].plot(1e3 * curves[-1]["x"], 1e3 * curves[-1]["twist_theory"], "k--", label="Saint-Venant θ(x)")
    ax[1, 0].axvline(1e3 * th.a, color="grey", lw=0.8, ls=":")
    ax[1, 1].axvline(1e3 * th.a, color="grey", lw=0.8, ls=":")
    ax[1, 0].set(xlabel="x [mm]", ylabel="section-mean u_z [mm]", title=f"Deflection curve, contact ({ax_rel:.2f}L, {ay_rel:.2f}W)")
    ax[1, 1].set(xlabel="x [mm]", ylabel="section twist [mrad]", title=f"Twist curve, e = {1e3 * th.e:+.1f} mm")
    for a_ in ax[1]:
        a_.legend(); a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "comparison.png", dpi=130)
    print(f"\nplot -> {run_dir / 'comparison.png'}")

    save_json({
        "force": F, "geometry": g.__dict__, "material": {"E": model.material.E, "nu": model.material.nu},
        "bending_vs_position": rows_a, "torsion_vs_eccentricity": rows_b,
        "curves": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cv.items()} for cv in curves],
    }, run_dir / "metrics.json")

    # ------------------------------------------------------------------ viewer: FEM vs beam-theory kinematics
    m, res, c, th = viewer_payload
    state = m.visualization_state(res, c, -F * c.normal, with_stress=False,
                                  info={"tip w FEM/theory": f"{1e3 * curves[0]['w_fem'][-1]:.3f} / {1e3 * curves[0]['w_theory'][-1]:.3f} mm",
                                        "tip θ FEM/theory": f"{1e3 * curves[0]['twist_fem_debiased'][-1]:.3f} / {1e3 * curves[0]['twist_theory'][-1]:.3f} mrad",
                                        "A": "toggle FEM <-> beam theory"})
    state.displacement_alt = th.displacement_field(m.mesh.nodes)
    state.displacement_alt_name = "beam theory (Timoshenko + Saint-Venant)"

    from src.visualization.taichi_viewer import TaichiViewer

    viewer = TaichiViewer(show_window=not args.headless)
    viewer.set_state(state)
    viewer.save_screenshot(run_dir / "fem.png")
    viewer.options.use_alt_displacement = True
    viewer.save_screenshot(run_dir / "beam_theory.png")
    viewer.options.use_alt_displacement = False
    if not args.headless:
        viewer.run()


if __name__ == "__main__":
    main()
