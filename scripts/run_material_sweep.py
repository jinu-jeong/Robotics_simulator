#!/usr/bin/env python
"""Fig. 5 – same finger, different material: only ``K`` changes after ``q``.

Frozen: POD basis Φ (reference E₀, ν₀), marker geometry / image Jacobian, the
marker-free ResNet. Swept: (E, ν) → FEM ``K`` → ROM ``K_r = ΦᵀKΦ`` and the
contact-localiser influence fields. Nothing is retrained.

Offline (held-out inner-face contacts × forces, per material)
  q sources : oracle (Φᵀu), marker (Stage-D pixels → q), nn (marker-free RGB → q)
  physics   : updated K_r(E, ν)  vs  stale K_r(E₀, ν₀)  → known-contact force,
              unknown-contact force + location
Closed loop (optional)
  ``run_arm_grasp`` mechanics rebuilt with the new E, Φ pinned to the reference
  material (``estimator.pod_reference_material``), modes world / nn.

Outputs → results/figure5/ : material_sweep.png, closed_loop.png, metrics.json

    python scripts/run_material_sweep.py                 # full (≈ 3 min incl. closed loop)
    python scripts/run_material_sweep.py --quick         # smoke test
    python scripts/run_material_sweep.py --no-nn --no-closed-loop
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.material_sweep import (  # noqa: E402
    MaterialCase,
    evaluate_case,
    held_out_inner_face_points,
    material_variants,
)
from src.grasp.factory import build_arm_grasp_sim  # noqa: E402
from src.grasp.mechanics import GraspMechanics  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.utils.figure_res import MPL_DPI  # noqa: E402
from src.utils.io import save_json  # noqa: E402


# ---------------------------------------------------------------- plotting
def _plot_offline(out: Path, results: list[dict], sources: list[str]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E_rows = [r for r in results if r["material"]["sweep"] == "E"]
    nu_rows = [r for r in results if r["material"]["sweep"] == "nu" or (r["material"]["sweep"] == "E" and r["material"]["E_factor"] == 1.0)]
    nu_rows = sorted(nu_rows, key=lambda r: r["material"]["poisson_ratio"])
    E_rows = sorted(E_rows, key=lambda r: r["material"]["E_factor"])
    Ef = np.array([r["material"]["E_factor"] for r in E_rows])
    nus = np.array([r["material"]["poisson_ratio"] for r in nu_rows])
    colors = {"oracle": "k", "marker": "tab:blue", "nn": "tab:orange"}
    labels = {"oracle": "oracle q = Φᵀu", "marker": "markers → q", "nn": "marker-free NN → q"}

    def pick(rows, src, key):
        return np.array([100.0 * r["sources"].get(src, {}).get(key, np.nan) if "rel" in key else r["sources"].get(src, {}).get(key, np.nan) for r in rows])

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0))
    a = ax[0, 0]
    for s in sources:
        a.plot(Ef, pick(E_rows, s, "force_known_rel_median"), "o-", color=colors[s], label=f"{labels[s]}, K updated")
    a.plot(Ef, pick(E_rows, "oracle", "force_known_rel_median_staleK"), "x--", color="tab:red", label="oracle q, K stale (E₀)")
    a.plot(Ef, 100.0 * np.abs(1.0 / Ef - 1.0), ":", color="gray", lw=1, label="|E₀/E − 1| (theory, stale)")
    a.set(xscale="log", xlabel="E / E₀", ylabel="known-contact force error [%] (median)", title="(a) force from q: only K changes")
    a.set_xticks(Ef); a.set_xticklabels([f"{f:g}" for f in Ef])
    a.grid(alpha=0.3); a.legend(fontsize=7.5)

    a = ax[0, 1]
    for s in sources:
        a.plot(Ef, pick(E_rows, s, "contact_err_mm_median"), "o-", color=colors[s], label=f"{labels[s]}, K updated")
        a.plot(Ef, pick(E_rows, s, "contact_err_mm_median_staleK"), "x--", color=colors[s], alpha=0.45, label=f"{labels[s]}, K stale")
    a.set(xscale="log", xlabel="E / E₀", ylabel="unknown-contact location error [mm] (median)", title="(b) contact localisation")
    a.set_xticks(Ef); a.set_xticklabels([f"{f:g}" for f in Ef])
    a.grid(alpha=0.3); a.legend(fontsize=7.5)

    a = ax[1, 0]
    for s in sources:
        a.plot(Ef, pick(E_rows, s, "force_unknown_rel_median"), "o-", color=colors[s], label=f"{labels[s]}, K updated")
        a.plot(Ef, pick(E_rows, s, "force_unknown_rel_median_staleK"), "x--", color=colors[s], alpha=0.45, label=f"{labels[s]}, K stale")
    a.set(xscale="log", xlabel="E / E₀", ylabel="unknown-contact force error [%] (median)", title="(c) force with contact found from q")
    a.set_xticks(Ef); a.set_xticklabels([f"{f:g}" for f in Ef])
    a.grid(alpha=0.3); a.legend(fontsize=7.5)

    a = ax[1, 1]
    for s in sources:
        a.plot(nus, pick(nu_rows, s, "force_known_rel_median"), "o-", color=colors[s], label=f"{labels[s]} force err [%]")
    a.plot(nus, 100.0 * np.array([r["proj_residual_median"] for r in nu_rows]), "s--", color="tab:green", label="Φ projection residual [%]")
    a.plot(nus, pick(nu_rows, "oracle", "contact_err_mm_median"), "d-.", color="tab:purple", label="oracle contact err [mm]")
    a.set(xlabel="Poisson ratio ν  (E = E₀)", ylabel="[%] or [mm]", title="(d) ν sweep with the reference Φ")
    a.grid(alpha=0.3); a.legend(fontsize=7.5)
    fig.suptitle("Material generalisation: vision → q frozen (Φ, markers, ResNet); only K → K_r = ΦᵀKΦ rebuilt", fontsize=11)
    fig.tight_layout()
    p = out / "material_sweep.png"
    fig.savefig(p, dpi=MPL_DPI)
    plt.close(fig)
    return p


def _plot_closed_loop(out: Path, rows: list[dict]) -> Path | None:
    if not rows:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted({r["mode"] for r in rows}, key=lambda m: ["world", "nn"].index(m) if m in ("world", "nn") else 9)
    Ef = sorted({r["E_factor"] for r in rows})
    x = np.arange(len(Ef))
    w = 0.8 / max(1, len(modes))
    colors = {"world": "tab:blue", "nn": "tab:orange"}
    labels = {"world": "markers (D)", "nn": "marker-free NN (E)"}
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for k, m in enumerate(modes):
        rr = {r["E_factor"]: r for r in rows if r["mode"] == m}
        vals = lambda key: np.array([rr[f][key] if f in rr and rr[f][key] is not None else np.nan for f in Ef], float)  # noqa: E731
        off = (k - 0.5 * (len(modes) - 1)) * w
        ax[0].bar(x + off, vals("success").astype(float), w, color=colors.get(m), label=labels.get(m, m))
        ax[1].bar(x + off, vals("contact_err_mm"), w, color=colors.get(m), label=labels.get(m, m))
        ax[2].bar(x + off, 100.0 * vals("force_rel_err_median"), w, color=colors.get(m), label=labels.get(m, m))
    for a, t, yl in zip(ax, ("lift success", "final contact error", "force error while loaded (median)"), ("1 = held & lifted", "[mm]", "[%]")):
        a.set_xticks(x); a.set_xticklabels([f"E×{f:g}" for f in Ef]); a.set(title=t, ylabel=yl); a.grid(alpha=0.3, axis="y")
    ax[0].set_ylim(0, 1.15); ax[0].legend(fontsize=8)
    fig.suptitle("Closed-loop arm grasp with the material changed at run time (Φ pinned to E₀; K rebuilt; no retraining)", fontsize=10.5)
    fig.tight_layout()
    p = out / "closed_loop.png"
    fig.savefig(p, dpi=MPL_DPI)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- closed loop
def _closed_loop_run(gcfg: dict, e_factor: float, mode: str, *, t_max: float, scale_gates: bool) -> dict:
    cfg = copy.deepcopy(dict(gcfg))
    mat0 = dict(cfg["fem"]["material"])
    cfg["fem"]["material"]["youngs_modulus"] = float(mat0["youngs_modulus"]) * float(e_factor)
    est = cfg.setdefault("estimator", {})
    est["pod_reference_material"] = {"youngs_modulus": float(mat0["youngs_modulus"]), "poisson_ratio": float(mat0["poisson_ratio"])}
    if scale_gates:
        for k, d in (("q_min_norm", 8.0e-3), ("q_lock_norm", 1.2e-2)):
            est[k] = float(est.get(k, d)) / float(e_factor)
    sim = build_arm_grasp_sim(cfg, estimator_mode=mode)
    t0 = time.time()
    log = sim.run(t_max=float(t_max))
    wall = time.time() - t0
    yt, ym = np.asarray(log["lam_true"], float), np.asarray(log["lam_meas"], float)
    m = yt > 0.3
    rel = np.abs(ym[m] - yt[m]) / yt[m] if np.any(m) else np.array([])
    lift_t = next((t for t, ph in zip(log["t"], log["phase"]) if ph == "lift"), None)
    row = {
        "E_factor": float(e_factor), "mode": mode, "success": bool(sim.success()), "phase": sim.phase, "held": bool(sim.held()),
        "stiffness_N_per_m": float(sim.mech.stiffness), "hold_force_N": float(sim.mech.hold_force),
        "lam_final": float(sim.grasp.state.lam_true), "lam_meas_final": float(sim.grasp.state.lam_meas),
        "force_rel_err_median": float(np.median(rel)) if rel.size else None,
        "force_rel_err_p90": float(np.percentile(rel, 90)) if rel.size else None,
        "contact_err_mm": None if sim.estimator.last_contact_err_m is None else float(1e3 * sim.estimator.last_contact_err_m),
        "lift_started_s": lift_t, "sim_t": float(log["t"][-1]), "wall_s": wall,
        "basis_fingerprint": sim.estimator.basis_fingerprint,
    }
    sim.estimator.close()
    return row


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="material_sweep")
    p.add_argument("--out", default=None)
    p.add_argument("--no-nn", action="store_true")
    p.add_argument("--no-marker", action="store_true")
    p.add_argument("--no-closed-loop", action="store_true")
    p.add_argument("--quick", action="store_true", help="2 contacts x 2 forces x {0.5,1} E; skips closed loop")
    args = p.parse_args()

    cfg = load_config(args.config)
    gcfg = load_config(cfg.get("grasp_config", "grasp"))
    out = Path(args.out or cfg.get("output_dir", "results/figure5"))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    mats = cfg.get("materials", {})
    e_factors = list(mats.get("e_factors", [0.5, 1.0, 2.0]))
    nu_values = list(mats.get("nu_values", []))
    ho = cfg.get("held_out", {})
    x_rel, z_rel = list(ho.get("x_rel", [0.6, 0.8])), list(ho.get("z_rel", [0.5]))
    forces = np.asarray(cfg.get("forces_N", [0.5, 1.5, 2.5]), float)
    src_cfg = dict(cfg.get("sources", {}))
    use_marker = bool(src_cfg.get("marker", True)) and not args.no_marker
    use_nn = bool(src_cfg.get("nn", True)) and not args.no_nn
    marker_frames = max(1, int(src_cfg.get("marker_frames", 30)))
    if args.quick:
        e_factors, nu_values = [0.5, 1.0], []
        x_rel, z_rel, forces = x_rel[:2], z_rel[:1], forces[:2]

    # ---- reference machinery (Φ, markers, world camera, arm pose) -------------------------
    t0 = time.time()
    sim_ref = build_arm_grasp_sim(gcfg, estimator_mode="world")
    est_ref, mech_ref = sim_ref.estimator, sim_ref.mech
    basis, rom_ref = est_ref.geo.basis, est_ref.geo._rom
    base_mat = dict(gcfg["fem"]["material"])
    fem_cfg = dict(gcfg["fem"])
    T_ee = sim_ref.T_grasp
    obj = gcfg["object"]
    contact_rel = gcfg.get("contact", {}).get("position_rel", [0.85, 1.0, 0.5])
    pts = held_out_inner_face_points(mech_ref.model.geometry, x_rel, z_rel)
    print(f"reference built in {time.time() - t0:.1f} s: Φ r={basis.r} ({est_ref.basis_fingerprint}), "
          f"{len(pts)} held-out contacts, {len(forces)} forces, E×{e_factors}, ν {nu_values}")

    nn_est = None
    q_train_max = None
    if use_nn:
        from src.grasp.nn_vision import NNStateEstimator

        nn_cfg = dict(gcfg.get("estimator", {}).get("nn") or {})
        ckpt = nn_cfg.get("checkpoint", "results/etc/checkpoints/grasp_nn/resnet18_q_sim_raw.pt")
        if not Path(ckpt).exists():
            print(f"[warn] NN checkpoint {ckpt} not found → skipping nn source")
            use_nn = False
        else:
            nn_est = NNStateEstimator.from_checkpoint(
                ckpt, mech_ref, est_ref.world_camera,
                width=int(nn_cfg.get("image_width", 240)), height=int(nn_cfg.get("image_height", 180)),
                remote=True, expected_basis_fingerprint=est_ref.basis_fingerprint,
            )
            ds = Path(load_config("grasp_nn").get("output", "data/processed/grasp_nn_top.npz"))
            if ds.exists():
                q_train_max = float(np.percentile(np.linalg.norm(np.load(ds)["q_sim"], axis=1), 99.5))
            print(f"nn source: {ckpt}  (training |q| p99.5 = {q_train_max})")

    # ---- offline sweep ---------------------------------------------------------------------
    variants = material_variants(base_mat, e_factors, nu_values)
    results = []
    for mat in variants:
        t1 = time.time()
        case = MaterialCase.build(fem_cfg, mat, basis, rom_ref, pts,
                                  localizer_cfg=gcfg.get("estimator", {}).get("unknown_contact_cfg"))
        mech_case = GraspMechanics.build(case.model, contact_rel, obj["size"], float(obj["mass"]), float(obj["friction"]))

        # `u` handed to the sources is the contact field; the observed finger also sags under
        # its own weight (fem.gravity) – the vision sources see u + u_g, the oracle q is contact-only.
        u_g_case = mech_case.gravity_displacement(T_ee[:3, :3])

        def q_oracle(u, c, F):
            return basis.project(u)

        def q_marker(u, c, F, _mech=mech_case):
            # Stage-D pixels → q, averaged over `marker_frames` noisy frames (≈ the Kalman hold in the demo)
            g = _mech.opening_from_force(F)
            qs = []
            mech_saved = est_ref.mech
            est_ref.mech = _mech  # sag compensation uses the *case* material's u_g (same mesh)
            try:
                for _ in range(marker_frames):
                    est_ref.rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                    qs.append(est_ref._q_from_world(u + u_g_case, T_ee, g))
            finally:
                est_ref.mech = mech_saved
            return np.mean(qs, axis=0)

        def q_nn(u, c, F, _mech=mech_case):
            obj_c = T_ee[:3, :3] @ np.array([c.position[0], 0.0, c.position[2]]) + T_ee[:3, 3]
            return nn_est.predict_q(_mech, u + u_g_case, _mech.opening_from_force(F), T_ee, obj_c, T_ee[:3, :3])

        sources = {"oracle": q_oracle}
        if use_marker:
            sources["marker"] = q_marker
        if use_nn:
            sources["nn"] = q_nn
        summ = evaluate_case(case, basis, forces, sources)
        summ["stiffness_N_per_m"] = float(mech_case.stiffness)
        if use_nn and q_train_max is not None:
            qn = np.asarray(summ["sources"]["oracle"]["samples"]["q_norm"], float)
            summ["nn_in_distribution_fraction"] = float(np.mean(qn <= q_train_max))
        results.append(summ)
        s = summ["sources"]
        line = "  ".join(
            f"{k}: F_known {100 * v['force_known_rel_median']:.1f}% (stale {100 * v['force_known_rel_median_staleK']:.1f}%), "
            f"contact {v['contact_err_mm_median']:.1f} mm, F_unk {100 * v['force_unknown_rel_median']:.1f}%"
            for k, v in s.items()
        )
        print(f"[{mat['name']:>8}] k={mech_case.stiffness:.0f} N/m  projΦ {100 * summ['proj_residual_median']:.2f}%  |  {line}  ({time.time() - t1:.1f} s)")

    if nn_est is not None:
        nn_est.close()
    plot_path = _plot_offline(out, results, list(results[0]["sources"].keys()))
    print(f"offline figure → {plot_path}")

    # ---- closed loop -------------------------------------------------------------------------
    cl_cfg = dict(cfg.get("closed_loop", {}))
    cl_rows: list[dict] = []
    if bool(cl_cfg.get("enabled", True)) and not args.no_closed_loop and not args.quick:
        modes = [m for m in cl_cfg.get("modes", ["world", "nn"]) if not (m == "nn" and not use_nn)]
        for f in cl_cfg.get("e_factors", [0.5, 1.0, 2.0]):
            for mode in modes:
                row = _closed_loop_run(gcfg, float(f), mode, t_max=float(cl_cfg.get("t_max_s", 10.0)),
                                       scale_gates=bool(cl_cfg.get("scale_q_gates", True)))
                cl_rows.append(row)
                ce = "—" if row["contact_err_mm"] is None else f"{row['contact_err_mm']:.1f} mm"
                fe = "—" if row["force_rel_err_median"] is None else f"{100 * row['force_rel_err_median']:.1f}%"
                print(f"[closed loop E×{f:g} {mode:5s}] success={row['success']} k={row['stiffness_N_per_m']:.0f} N/m "
                      f"contact {ce} force err {fe} lift@{row['lift_started_s']} ({row['wall_s']:.1f} s)")
        cl_path = _plot_closed_loop(out, cl_rows)
        if cl_path:
            print(f"closed-loop figure → {cl_path}")

    # ---- save ----------------------------------------------------------------------------------
    slim = []
    for r in results:
        rr = copy.deepcopy(r)
        for v in rr["sources"].values():
            v.pop("samples", None)
        slim.append(rr)
    save_json({
        "reference_material": base_mat, "basis_fingerprint": est_ref.basis_fingerprint, "q_modes": int(basis.r),
        "held_out_contacts": pts.tolist(), "forces_N": forces.tolist(), "q_train_max_p99_5": q_train_max,
        "marker_frames_averaged": marker_frames, "pixel_noise_std": float(gcfg.get("estimator", {}).get("pixel_noise_std", 0.0)),
        "offline": slim, "closed_loop": cl_rows,
    }, out / "metrics.json")
    save_yaml(cfg, out / "config.yaml")
    (out / "README.txt").write_text(
        "Paper Figure 5 – material generalisation (same finger, different E / ν)\n\n"
        "Frozen: POD basis Φ (reference material), marker geometry / image Jacobian, marker-free ResNet.\n"
        "Rebuilt: FEM K → ROM K_r = ΦᵀKΦ and contact-localiser influence fields. No retraining.\n\n"
        "material_sweep.png  offline: oracle / marker / nn q → force & contact, updated vs stale K\n"
        "closed_loop.png     arm grasp with the material changed at run time (world vs nn)\n"
        "metrics.json        all numbers (per-material medians, closed-loop rows)\n\n"
        "Regenerate: python scripts/run_material_sweep.py\n"
    )
    print(f"done → {out}")


if __name__ == "__main__":
    main()
