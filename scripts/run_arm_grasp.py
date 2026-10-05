#!/usr/bin/env python
"""Stages C / D / E – 7-DoF arm + IK, world-camera force feedback.

C: the arm reaches a pregrasp, approaches, then A/B force control squeezes
and the arm lifts via IK.
D (``--mode world``): markers through a world-fixed camera → ``q`` → force.
E (``--mode nn``): same camera, **no markers** – ResNet-18 RGB → ``q`` → force.

Paper Figure 2 outputs (``--mode world`` / ``nn``) go to::

    results/figure2/{world|nn}/stage_<phase>.png
    results/figure2/{world|nn}/force_plot.png

Other timestamped clutter under ``results/etc/grasp/`` is temporarily disabled.

Keys (plus viewer defaults):
    Space   pause / resume
    4 / 5 / 6 / 7   Stage A (GT) / B (finger-cam) / D (world-cam markers) / E (NN)
    9       lift now (once squeezing)
    0       reset
    [ / ]   force target − / + 0.2 N

Run:
    python scripts/run_arm_grasp.py --mode world
    python scripts/run_arm_grasp.py --mode nn
    python scripts/run_arm_grasp.py --mode world --headless
    python scripts/run_arm_grasp.py --mode nn --video   # → results/videos/nn/{viewer,nn_input}.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grasp.estimator import observation_marker_image  # noqa: E402
from src.grasp.factory import build_arm_grasp_sim  # noqa: E402
from src.grasp.live_force_plot import (  # noqa: E402
    display_force_estimate,
    render_force_strip,
    stack_camera_and_force,
)
from src.grasp.scene import build_arm_grasp_state  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.figure_res import MPL_DPI, viewer_window_config  # noqa: E402
from src.utils.video_io import RGBVideoWriter  # noqa: E402
# Temporarily disabled (was writing timestamped junk into results/etc/):
# from src.utils.config import save_yaml
# from src.utils.io import make_run_dir, save_json

FIGURE2_ROOT = ROOT / "results" / "figure2"
# Cap sim time for the Figure-2 demos, then quit automatically (interactive + headless).
T_MAX_S = 10.0
VIDEO_SIZE = (1280, 720)
VIDEO_EVERY = 3  # sim steps per video frame (dt = 0.01 → 33.3 fps, real time)
NN_INPUT_SCALE = 4  # 240×180 network input → 960×720


def _figure2_dir(mode: str) -> Path:
    """Fixed paper Figure-2 folder: results/figure2/{mode}/."""
    d = FIGURE2_ROOT / str(mode)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plot(run_dir: Path, log: dict, *, ema: float = 0.18) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.asarray(log["t"])
    yt = np.asarray(log["lam_true"], float)
    ym = display_force_estimate(np.asarray(log["lam_meas"], float), ema=ema)
    fig, ax = plt.subplots(3, 1, figsize=(8.2, 7.4), sharex=True)
    ax[0].plot(t, yt, label="true")
    ax[0].plot(t, ym, "--", label="estimate (EMA)" if ema < 1.0 else "estimate")
    ax[0].axhline(log["hold_force"][0], color="r", ls=":", lw=1, label="hold threshold")
    ax[0].axhline(log["force_target"][0], color="k", ls="--", lw=1, label="target")
    ax[0].set(ylabel="normal force [N]", title="Arm grasp: IK approach, squeeze, lift")
    ax[1].plot(t, 1e3 * np.asarray(log["opening"]), label="opening")
    ax[1].plot(t, 1e3 * np.asarray(log["ik_err"]), label="IK ‖e‖")
    ax[1].set(ylabel="mm")
    ax[2].plot(t, 1e3 * np.asarray(log["ee_z"]), label="EE z")
    ax[2].plot(t, 1e3 * np.asarray(log["obj_z"]), label="object z")
    ax[2].set(ylabel="z [mm]", xlabel="t [s]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = run_dir / "force_plot.png"
    fig.savefig(out, dpi=MPL_DPI)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="grasp")
    p.add_argument("--mode", default=None, choices=["gt", "markers", "inverse", "world", "nn"])
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--out", default=None, help="override output dir (default: results/figure2/{mode})")
    p.add_argument("--video", action="store_true",
                   help="headless run that records viewer.mp4 (+ nn_input.mp4 in nn mode) to results/videos/{mode}")
    p.add_argument("--raw-force", action="store_true",
                   help="plot / inset the raw estimate (no display EMA)")
    p.add_argument("--object-stiffness", type=float, default=None,
                   help="deformable object: width stiffness k_obj [N/m] (default: config, rigid)")
    args = p.parse_args()
    if args.video:
        args.headless = True

    cfg = load_config(args.config)
    if args.object_stiffness is not None:
        cfg["object"]["stiffness"] = float(args.object_stiffness)
    sim = build_arm_grasp_sim(cfg, estimator_mode=args.mode)
    if args.target is not None:
        sim.grasp.force_target = float(args.target)
    disp = dict(cfg.get("display") or {})
    beautify = bool(disp.get("force_beautify", True)) and not args.raw_force
    force_ema = float(disp.get("force_ema", 0.18))
    mode = sim.estimator.mode

    from src.visualization.taichi_viewer import TaichiViewer

    # Figure 2: world / nn → results/figure2/{mode}/
    # Older timestamped results/etc/grasp/<ts>/ dumps are temporarily disabled.
    if args.out is not None:
        run_dir = Path(args.out)
        run_dir.mkdir(parents=True, exist_ok=True)
    elif args.video:  # video-size stills must not replace the paper's Figure-2 stills
        name = mode
        if args.object_stiffness is not None:
            name += f"_soft{args.object_stiffness:g}"
        if args.raw_force:
            name += "_raw"
        run_dir = ROOT / "results" / "videos" / name
        run_dir.mkdir(parents=True, exist_ok=True)
    elif mode in ("world", "nn"):
        run_dir = _figure2_dir(mode)
    else:
        # Temporarily do not write into results/etc/ for other modes.
        run_dir = Path("/tmp") / f"arm_grasp_{mode}"
        run_dir.mkdir(parents=True, exist_ok=True)
    # save_yaml(dict(cfg), run_dir / "config.yaml")  # temporarily disabled

    viewer = TaichiViewer(
        show_window=not args.headless,
        title=f"arm grasp {mode} – D=world E=nn",
        config=viewer_window_config(*VIDEO_SIZE) if args.video else viewer_window_config(),
    )
    viewer.options.amplification = 1.0
    paused = {"v": False}
    live = {"strip_every": 0, "force_img": None}
    shot = {"last_phase": None}

    def _overlay(st) -> None:
        est = sim.estimator
        if est.mode not in ("world", "nn") or est.world_camera is None:
            viewer.set_overlay_image(None)
            return
        cam = None
        if est.mode == "nn":
            cam = None if est.last_image is None else est.last_image.astype("float32") / 255.0
        else:
            uv, vis = est.last_uv, est.last_visible
            if uv is None and st.keypoints is not None:
                uv, z = est.world_camera.project(st.keypoints)
                vis = z > 0 if st.keypoints_visible is None else st.keypoints_visible
            if uv is not None:
                cam = observation_marker_image(est.world_camera, uv, vis)
        live["strip_every"] += 1
        if live["force_img"] is None or live["strip_every"] % 2 == 0:
            live["force_img"] = render_force_strip(
                sim.log, width=420, height=200,
                beautify=beautify, ema=force_ema,
            )
        force = live["force_img"]
        if cam is None:
            viewer.set_overlay_image(force, max_width_frac=0.42)
        else:
            viewer.set_overlay_image(stack_camera_and_force(cam, force), max_width_frac=0.42)

    def apply(first: bool = False) -> None:
        st = build_arm_grasp_state(sim)
        viewer.set_state(st, refit_camera=first)
        _overlay(st)

    def shot_if_phase_changed(*, force: bool = False) -> None:
        """Save a viewer screenshot once per arm phase (paper Figure 2)."""
        ph = str(sim.phase)
        if not force and ph == shot["last_phase"]:
            return
        shot["last_phase"] = ph
        apply()
        path = run_dir / f"stage_{ph}.png"
        viewer.save_screenshot(path)
        print(f"[arm] screenshot → {path}")

    def on_frame(v) -> None:
        if not paused["v"]:
            for _ in range(2):
                if sim.t >= T_MAX_S:
                    break
                sim.step()
            shot_if_phase_changed()
            apply()
        if sim.t >= T_MAX_S:
            print(f"[arm] t={sim.t:.2f} s ≥ {T_MAX_S:.0f} s — quitting")
            viewer.close()

    def toggle_pause(_v) -> None:
        paused["v"] = not paused["v"]

    def set_mode(m: str):
        def _fn(_v):
            sim.set_mode(m)
            live["force_img"] = None
            print(f"[arm] estimator → {m}")
        return _fn

    def bump_target(d: float):
        def _fn(_v):
            sim.grasp.force_target = max(0.2, sim.grasp.force_target + d)
            print(f"[arm] target → {sim.grasp.force_target:.2f} N  (hold ≥ {sim.mech.hold_force:.2f} N)")
        return _fn

    def on_reset(_v) -> None:
        sim.reset()
        live["force_img"] = None
        live["strip_every"] = 0
        shot["last_phase"] = None
        apply(True)
        shot_if_phase_changed(force=True)

    viewer.register_key(" ", toggle_pause, "pause / resume")
    viewer.register_key("4", set_mode("gt"), "Stage A: GT force")
    viewer.register_key("5", set_mode("markers"), "Stage B: finger-cam force")
    viewer.register_key("6", set_mode("world"), "Stage D: world-cam force")
    viewer.register_key("7", set_mode("nn"), "Stage E: marker-free NN force")
    viewer.register_key("9", lambda _v: setattr(sim.grasp, "force_lift", True), "lift now")
    viewer.register_key("0", on_reset, "reset arm grasp")
    viewer.register_key("[", bump_target(-0.2), "target −0.2 N")
    viewer.register_key("]", bump_target(+0.2), "target +0.2 N")

    if sim.estimator.mode == "nn":
        sim.estimator.ensure_nn()
        nn = sim.estimator.nn
        print(f"[arm] marker-free NN: {nn.meta['checkpoint']}  ({nn.camera.image_width}x{nn.camera.image_height}, "
              f"every {sim.estimator.nn_every} steps, q-basis {nn.meta.get('basis_fingerprint')})")

    apply(first=True)
    shot_if_phase_changed(force=True)
    # viewer.save_screenshot(run_dir / f"start_{mode}.png")  # temporarily replaced by stage_*.png
    uc = "unknown contact" if sim.estimator.unknown_contact else "known contact"
    print(f"hold threshold {sim.mech.hold_force:.2f} N, target {sim.grasp.force_target:.2f} N, "
          f"k = {sim.mech.stiffness:.0f} N/m, estimator={mode} ({uc})")
    print(f"[arm] figure2 output → {run_dir}")
    print(f"[arm] auto-quit at t = {T_MAX_S:.0f} s")
    if mode in ("world", "nn") and not args.headless:
        print("[arm] viewer inset: camera view + GT vs estimate (display-smoothed)")

    if args.headless:
        # Step ourselves so we can capture each phase (sim.run has no hooks).
        video = cam_video = None
        if args.video:
            fps = 1.0 / (VIDEO_EVERY * sim.grasp.dt)
            video = RGBVideoWriter(run_dir / "viewer.mp4", fps=fps)
            if mode == "nn":
                cam_video = RGBVideoWriter(run_dir / "nn_input.mp4", fps=fps, scale=NN_INPUT_SCALE)
                nn_cam = sim.estimator.nn.camera
                blank = np.zeros((nn_cam.image_height, nn_cam.image_width, 3), np.uint8)
        n_step = 0
        while sim.t < T_MAX_S:
            sim.step()
            shot_if_phase_changed()
            if video is not None and n_step % VIDEO_EVERY == 0:
                apply()
                video.write(viewer.render_offscreen())
                if cam_video is not None:
                    img = sim.estimator.last_image
                    cam_video.write(blank if img is None else img)
            n_step += 1
        for w in (video, cam_video):
            if w is not None:
                print(f"[arm] video → {w.close()}  ({w.n_frames} frames)")
        apply()
        shot_if_phase_changed(force=True)
        # viewer.save_screenshot(run_dir / f"end_{mode}.png")  # temporarily replaced by stage_*.png
        plot_path = _plot(
            run_dir, sim.log,
            ema=force_ema if beautify else 1.0,
        )
        # Temporarily disabled clutter:
        # if sim.estimator.last_image is not None:
        #     import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        #     plt.imsave(run_dir / "nn_camera_final.png", sim.estimator.last_image)
        # save_json({...}, run_dir / "metrics.json")
        print(f"phase={sim.phase} held={sim.held()} success={sim.success()}  → {run_dir}")
        lam = np.asarray(sim.log["lam_true"], float)
        hat = np.asarray(sim.log["lam_meas"], float)
        win = np.isin(np.asarray(sim.log["phase"]), ("lift", "hold")) & (lam > 0.3)
        ce = sim.estimator.last_contact_err_m
        if win.any():
            err = np.abs(hat[win] - lam[win])
            print(f"raw |λ̂−λ| median over lift/hold: {np.median(err):.3f} N "
                  f"({100 * np.median(err / lam[win]):.1f} %), contact error "
                  f"{'-' if ce is None else f'{1e3 * ce:.1f} mm'}")
        print(f"force plot → {plot_path}")
        viewer.destroy()
    else:
        viewer.run(on_frame=on_frame)
        plot_path = _plot(
            run_dir, sim.log,
            ema=force_ema if beautify else 1.0,
        )
        print(f"force plot → {plot_path}")
    sim.estimator.close()


if __name__ == "__main__":
    main()
