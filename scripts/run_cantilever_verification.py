#!/usr/bin/env python
"""Milestone 1 – cantilever verification of the 3D quadratic hex-dominant (H20 + T10) FEM.

Numerical verification
    FEM tip deflection vs Euler–Bernoulli  δ = F L³ / (3 E I)  and Timoshenko
    (Cowper shear correction) on a sequence of refined meshes. Produces a table,
    a convergence plot, ``metrics.json`` and a config snapshot in
    ``results/etc/cantilever/<timestamp>/``.

Visual verification
    The finest mesh result is shown in the Taichi viewer (or rendered head-less
    with ``--screenshot``): the beam must bend along the load (−z top-face
    benchmark, −y lateral / inner-face press), the root must stay put,
    displacement must grow monotonically from root to tip, and the reaction
    force must balance the applied load.

Run:
    python scripts/run_cantilever_verification.py            # study + viewer
    python scripts/run_cantilever_verification.py --no-view  # study only
    python scripts/run_cantilever_verification.py --fine     # add the 80x16x16 level
    python scripts/run_cantilever_verification.py --config cantilever_lateral --no-view
        # lateral (−y) press as in the grasp → Fig. 1 panel 03
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

from src.fem.beam_theory import CantileverReference  # noqa: E402
from src.fem.boundary import clamp_nodes  # noqa: E402
from src.fem.loads import total_force_on_plane  # noqa: E402
from src.fem.material import LinearElasticMaterial  # noqa: E402
from src.fem.solver import LinearFEM  # noqa: E402
from src.geometry.finger import FingerGeometry, make_rectangular_finger_mesh, root_fixed_nodes  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.figure_res import MPL_DPI  # noqa: E402
from src.utils.io import make_run_dir, save_json, snapshot_config  # noqa: E402
from src.visualization.state import ForceVector  # noqa: E402


def bending_axis(direction: np.ndarray) -> str:
    """'y' for a lateral (inner-face) load, 'z' for a top-face load."""
    d = np.abs(np.asarray(direction, float))
    if d[0] > 1e-9 or (d[1] > 1e-9 and d[2] > 1e-9):
        raise ValueError("load direction must be purely along y or z")
    return "y" if d[1] > d[2] else "z"


def solve_level(geom, mat, res, force_vec):
    nx, ny, nz = res
    t0 = time.time()
    mesh = make_rectangular_finger_mesh(geom, nx, ny, nz)
    fem = LinearFEM(mesh, mat, clamp_nodes(mesh.n_nodes, root_fixed_nodes(mesh)))
    f = total_force_on_plane(mesh, 0, geom.length, force_vec)  # consistent traction on the end face
    result = fem.solve(f, meta={"resolution": list(res)})
    tip_nodes = mesh.nodes_on_plane(0, geom.length)
    d = np.asarray(force_vec, float) / np.linalg.norm(force_vec)
    tip_w = float((result.u[tip_nodes] @ d).mean())  # deflection magnitude along the load direction
    axis = bending_axis(d)
    hx, hy, hz = geom.length / nx, geom.width / ny, geom.height / nz
    return {
        "resolution": list(res),
        "n_nodes": mesh.n_nodes,
        "n_tets": mesh.n_tets,
        "n_hexes": mesh.n_hexes,
        "n_ties": mesh.n_ties,
        "n_dofs": mesh.n_dofs,
        "bending_axis": axis,
        # Bending-thickness size alone collapses 2×1×1 / 3×1×1 / 5×1×1 onto one
        # abscissa (ny=1) and looks like a vertical spike; use a cell measure too.
        "h_z": hz if axis == "z" else hy,
        "h_char": float((hx * hy * hz) ** (1.0 / 3.0)),
        "tip_deflection": tip_w,
        "max_displacement": result.max_displacement(),
        "equilibrium_residual": result.equilibrium_residual(),
        "reaction_resultant": result.reaction_resultant().tolist(),
        "solve_time_s": time.time() - t0,
    }, fem, result


def convergence_order(hs, errs):
    """Least-squares slope of log|err| vs log h."""
    hs, errs = np.asarray(hs), np.abs(np.asarray(errs))
    if len(hs) < 2 or np.any(errs <= 0):
        return float("nan")
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


def _richardson_limit(hs, tips, p: float = 2.0) -> float:
    """Extrapolate δ(h→0) from the two finest levels assuming O(h^p)."""
    h1, h2 = float(hs[-1]), float(hs[-2])
    d1, d2 = float(tips[-1]), float(tips[-2])
    r = (h2 / h1) ** p
    if abs(r - 1.0) < 1e-12:
        return d1
    return (r * d1 - d2) / (r - 1.0)


def make_plot(levels, ref, out_png, load_label: str = "tip −z"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dofs = [lv["n_dofs"] for lv in levels]
    tips_m = [lv["tip_deflection"] for lv in levels]
    tips = [1e3 * d for d in tips_m]
    hs = [lv.get("h_char", lv["h_z"]) for lv in levels]
    errs_t = [abs(lv["rel_error_timoshenko"]) for lv in levels]
    # Discretisation error vs a 3D continuum estimate (Richardson). Vs Timoshenko
    # alone floors at ~1–2 % (clamped-root gap) and hides the mesh trend.
    delta_3d = _richardson_limit(hs, tips_m, p=2.0)
    errs_3d = [abs(d / delta_3d - 1.0) for d in tips_m]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogx(dofs, tips, "o-", label="FEM (H20 + T10 tip)")
    ax.axhline(1e3 * ref.tip_deflection_euler_bernoulli, color="k", ls="--", label="Euler–Bernoulli")
    ax.axhline(1e3 * ref.tip_deflection_timoshenko, color="r", ls=":", label="Timoshenko (Cowper κ)")
    # Zoomed y-axes (0.96–1.03) make a ~1 % 3D-vs-1D gap look dramatic; start
    # near the coarsest tip (~0.86 mm at 2×1×1) so refinement is readable.
    y_lo = min(tips) * 0.95
    y_hi = max(tips + [1e3 * ref.tip_deflection_timoshenko]) * 1.02
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("tip deflection [mm]")
    ax.set_title(f"Cantilever tip deflection vs mesh refinement ({load_label})", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    hs_mm = [1e3 * h for h in hs]
    ax.loglog(hs_mm, errs_3d, "s-", color="C0", label="|δ_FEM/δ_3D − 1| (Richardson)")
    ax.loglog(hs_mm, errs_t, "o--", color="C3", alpha=0.75, label="|δ_FEM/δ_T − 1| (1D floor)")
    floor_t = abs(errs_t[-1])
    ax.axhline(floor_t, color="C3", ls=":", lw=0.8, alpha=0.5)
    ax.annotate(
        f"Timoshenko floor ≈ {100 * floor_t:.1f} %",
        xy=(max(hs_mm), floor_t),
        xytext=(-8, 8),
        textcoords="offset points",
        fontsize=7,
        color="C3",
        ha="right",
    )
    # Fit mesh error on pre-plateau levels only.
    fit_hs, fit_e = hs[:-2], errs_3d[:-2]
    p = convergence_order(fit_hs, fit_e) if len(fit_hs) >= 2 else float("nan")
    if np.isfinite(p) and min(fit_e) > 0:
        h0, e0 = fit_hs[-1], fit_e[-1]
        hh = np.array([min(hs), max(hs)])
        ax.loglog(1e3 * hh, e0 * (hh / h0) ** p, "k--", alpha=0.6, label=f"fit slope = {p:.2f}")
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, ScalarFormatter

    ax.set_xscale("log")
    ax.set_yscale("log")
    xt = [v for v in (1, 2, 3, 5, 10, 20, 30, 40) if min(hs_mm) * 0.85 <= v <= max(hs_mm) * 1.15]
    ax.set_xticks(xt)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=12))
    ax.set_xlabel(r"characteristic element size $(h_x h_y h_z)^{1/3}$ [mm]")
    ax.set_ylabel("relative error")
    ax.set_title("Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=MPL_DPI)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="cantilever_benchmark")
    p.add_argument("--fine", action="store_true", help="include the extra fine level(s)")
    p.add_argument("--no-view", action="store_true", help="skip the Taichi viewer")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--screenshot", type=str, default=None)
    p.add_argument("--view-level", type=int, default=-1, help="which refinement level to visualise (default: finest)")
    args = p.parse_args()

    cfg = load_config(args.config)
    np.random.seed(int(cfg.get("seed", 0)))
    geom = FingerGeometry.from_config(cfg)
    mat = LinearElasticMaterial.from_config(cfg)
    F = float(cfg["load"]["force_magnitude"])
    direction = np.asarray(cfg["load"]["direction"], float)
    direction /= np.linalg.norm(direction)
    force_vec = F * direction
    axis = bending_axis(direction)
    ref = CantileverReference(geom.length, geom.width, geom.height, mat.E, mat.nu, F, axis=axis)
    load_label = f"tip {'−' if direction[1 if axis == 'y' else 2] < 0 else '+'}{axis}" + (" lateral, inner-face press" if axis == "y" else "")

    levels_cfg = [tuple(r) for r in cfg["refinement"]]
    if args.fine:
        levels_cfg += [tuple(r) for r in cfg.get("refinement_fine", [])]

    run_dir = make_run_dir(cfg.get("output_dir", "results/etc/cantilever"))
    snapshot_config(cfg, run_dir)

    print("=" * 96)
    print(f"Cantilever benchmark  L={geom.length} W={geom.width} H={geom.height} m  E={mat.E:.3g} Pa  nu={mat.nu}  F={F} N  along {direction}")
    s = ref.summary()
    print(f"Euler–Bernoulli δ = {1e3 * ref.tip_deflection_euler_bernoulli:.4f} mm   Timoshenko δ = {1e3 * ref.tip_deflection_timoshenko:.4f} mm"
          f"   (shear part {100 * s['shear_fraction']:.2f} %,  L/thickness = {s['slenderness_L_over_H']:.0f}, bending axis {axis})")
    print("=" * 96)
    hdr = f"{'mesh':>12} {'nodes':>7} {'dofs':>7} {'δ_FEM [mm]':>11} {'err vs EB':>10} {'err vs T':>10} {'|ΣR+ΣF|/|ΣF|':>13} {'time [s]':>9}"
    print(hdr)
    print("-" * len(hdr))

    levels = []
    kept = None
    view_idx = args.view_level % len(levels_cfg)
    for i, res in enumerate(levels_cfg):
        lv, fem, result = solve_level(geom, mat, res, force_vec)
        lv["rel_error_euler_bernoulli"] = lv["tip_deflection"] / ref.tip_deflection_euler_bernoulli - 1.0
        lv["rel_error_timoshenko"] = lv["tip_deflection"] / ref.tip_deflection_timoshenko - 1.0
        levels.append(lv)
        print(f"{'x'.join(map(str, res)):>12} {lv['n_nodes']:>7} {lv['n_dofs']:>7} {1e3 * lv['tip_deflection']:>11.4f} "
              f"{100 * lv['rel_error_euler_bernoulli']:>9.2f}% {100 * lv['rel_error_timoshenko']:>9.2f}% "
              f"{lv['equilibrium_residual']:>13.1e} {lv['solve_time_s']:>9.2f}")
        if i == view_idx:
            kept = (fem, result, res)

    order = convergence_order(
        [lv.get("h_char", lv["h_z"]) for lv in levels[:-1]],
        [abs(lv["tip_deflection"] / levels[-1]["tip_deflection"] - 1.0) for lv in levels[:-1]],
    )
    finest = levels[-1]
    target = float(cfg.get("target_relative_error", 0.1))
    print("-" * len(hdr))
    print(f"observed convergence order (vs Richardson 3D, excl. finest): {order:.2f}")
    print(f"finest-mesh relative error: {100 * finest['rel_error_timoshenko']:.2f} %  (reported target ±{100 * target:.0f} %)")
    print("quadratic elements: displacement-based FEM is stiffer than the exact solution -> convergence from below.")

    metrics = {
        "reference": s | {"F": F, "E": mat.E, "nu": mat.nu, "L": geom.length, "W": geom.width, "H": geom.height},
        "levels": levels,
        "convergence_order": order,
        "finest_rel_error_timoshenko": finest["rel_error_timoshenko"],
        "target_relative_error": target,
        "passed_target": abs(finest["rel_error_timoshenko"]) <= target,
    }
    save_json(metrics, run_dir / "metrics.json")
    make_plot(levels, ref, run_dir / "convergence.png", load_label)
    print(f"results written to {run_dir}")
    if cfg.get("plot_copy"):  # fixed-path copy for the paper figure export
        import shutil

        dst = ROOT / cfg["plot_copy"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / "convergence.png", dst)
        save_json(metrics, dst.with_suffix(".json"))
        print(f"plot copied to {dst}")

    if args.no_view and not args.screenshot:
        return

    # ---------------------------------------------------------------- viewer
    from src.visualization.taichi_viewer import TaichiViewer

    fem, result, res = kept
    mesh = fem.mesh
    stresses = fem.stresses(result)
    tip_center = np.array([geom.length, 0.5 * geom.width, 0.5 * geom.height])
    if axis == "y":  # lateral press: anchor the arrow on the loaded (inner) face
        tip_center[1] = geom.width if direction[1] < 0 else 0.0
    lv = levels[view_idx]
    state = result.to_visualization_state(
        ground_truth_force=ForceVector(tip_center, force_vec, label="F_tip (distributed on end face)"),
        node_scalar=stresses["von_mises_nodal"],
        node_scalar_name="von Mises [Pa]",
        info={
            "benchmark": f"cantilever {'x'.join(map(str, res))}",
            "δ_FEM tip": f"{1e3 * lv['tip_deflection']:.4f} mm",
            "δ Euler-Bernoulli": f"{1e3 * ref.tip_deflection_euler_bernoulli:.4f} mm",
            "δ Timoshenko": f"{1e3 * ref.tip_deflection_timoshenko:.4f} mm",
            "rel. error vs Timoshenko": f"{100 * lv['rel_error_timoshenko']:.2f} %",
            "max von Mises": f"{stresses['von_mises'].max() / 1e6:.3f} MPa",
        },
    )
    viewer = TaichiViewer(show_window=not args.headless)
    viewer.options.amplification = 1.0 if result.max_displacement() > 0.02 * geom.length else 5.0
    viewer.set_state(state)
    print("\n".join(viewer.renderer.summary_lines(viewer.options)))
    if args.screenshot:
        viewer.save_screenshot(args.screenshot)
        viewer.options.color_mode = "scalar"
        viewer.options.mode = "deformed"
        shot = Path(args.screenshot)
        viewer.save_screenshot(shot.with_name(shot.stem + "_vonmises" + shot.suffix))
    if not args.headless:
        viewer.run()


if __name__ == "__main__":
    main()
