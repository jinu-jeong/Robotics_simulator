#!/usr/bin/env python
"""Paper Fig.1 panel 01 – clean early grasp scene (photo only).

Steps the arm to ``approach_ee`` (fingers around the object, before squeeze),
frames the full arm + table + gripper, and writes a screenshot with no HUD
(no force plot / camera inset / frustum).

Zoom / framing (larger = more zoomed out):
    python scripts/render_figure1_grasp_scene.py --zoom 1.55
    python scripts/render_figure1_grasp_scene.py --zoom 1.2   # closer
    python scripts/render_figure1_grasp_scene.py --zoom 2.0   # farther

Resolution (screenshot = viewer window size; default from ``src.utils.figure_res``):
    python scripts/render_figure1_grasp_scene.py --width 12800 --height 7200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.grasp.factory import build_arm_grasp_sim
from src.grasp.scene import build_arm_grasp_state
from src.utils.config import load_config
from src.utils.figure_res import VIEWER_HEIGHT, VIEWER_WIDTH, viewer_window_config
from src.visualization.taichi_viewer import TaichiViewer

OUT = ROOT / "results" / "figure1" / "02_grasp_scene.png"
T_MAX_S = 8.0
# Leave a bit of IK residual so we do not tip into squeeze.
APPROACH_STOP_ERR_M = 0.012


def _step_to_early_approach(sim, *, t_max: float = T_MAX_S) -> None:
    """Stop in ``approach_ee`` with fingers near the object, before squeeze."""
    while sim.t < t_max and sim.phase != "approach_ee":
        sim.step()
    if sim.phase != "approach_ee":
        raise RuntimeError(f"did not reach approach_ee (stuck at {sim.phase!r}, t={sim.t:.2f}s)")
    while sim.t < t_max and sim.phase == "approach_ee" and float(sim.ik_err) > APPROACH_STOP_ERR_M:
        sim.step()
    if sim.phase != "approach_ee":
        raise RuntimeError(f"overshot into {sim.phase!r}; lower APPROACH_STOP_ERR_M or re-tune")


def _scene_bounds(st) -> tuple[np.ndarray, np.ndarray]:
    """Full workspace: soft fingers + arm links + table + object (no camera)."""
    alpha = 1.0
    display = st.displaced(alpha)
    bmin = np.minimum(st.nodes_original.min(axis=0), display.min(axis=0))
    bmax = np.maximum(st.nodes_original.max(axis=0), display.max(axis=0))
    for obj in st.objects or []:
        v = np.asarray(obj.vertices, float)
        if v.size == 0:
            continue
        bmin = np.minimum(bmin, v.min(axis=0))
        bmax = np.maximum(bmax, v.max(axis=0))
    pad = 0.02 * np.ones(3)
    return bmin - pad, bmax + pad


def render(
    out: Path = OUT,
    *,
    zoom: float = 1.55,
    width: int = VIEWER_WIDTH,
    height: int = VIEWER_HEIGHT,
    yaw: float = -55.0,
    pitch: float = 28.0,
    fov: float = 38.0,
) -> Path:
    cfg = load_config("grasp")
    sim = build_arm_grasp_sim(cfg, estimator_mode="world")
    _step_to_early_approach(sim)

    st = build_arm_grasp_state(sim, show_markers=False)

    viewer = TaichiViewer(
        show_window=False,
        title="fig1 grasp scene",
        config=viewer_window_config(width, height),
    )
    viewer.options.mode = "deformed"  # no undeformed ghost boxes
    viewer.options.amplification = 1.0
    viewer.options.show_gui_panel = False
    viewer.options.show_wireframe = False
    viewer.options.show_tet_edges = False
    viewer.options.show_nodes = False
    viewer.options.show_fixed_nodes = False
    viewer.options.show_contact = False
    viewer.options.show_gt_force = False
    viewer.options.show_estimated_force = False
    viewer.options.show_extra_forces = False
    viewer.options.show_objects = True
    viewer.options.show_observation_camera = False  # frustum dominated the frame; photo-only
    viewer.options.show_keypoints = False
    viewer.set_overlay_image(None)

    viewer.set_state(st, refit_camera=False)
    bmin, bmax = _scene_bounds(st)
    # distance_rel: larger → farther (zoom out), smaller → closer (zoom in).
    viewer.orbit.fit(bmin, bmax, float(zoom))
    viewer.orbit.yaw_deg = float(yaw)
    viewer.orbit.pitch_deg = float(pitch)
    viewer.orbit.fov_deg = float(fov)

    out.parent.mkdir(parents=True, exist_ok=True)
    viewer.save_screenshot(out)
    print(
        f"  → {out.relative_to(ROOT)}  "
        f"(phase={sim.phase}, t={sim.t:.2f}s, zoom={zoom:g}, {width}x{height})"
    )
    try:
        viewer.destroy()
    except Exception:  # noqa: BLE001 — Vulkan teardown often aborts after a good shot
        pass
    sim.estimator.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zoom", type=float, default=1.55,
                   help="orbit distance_rel (default 1.55; smaller=zoom in, larger=zoom out)")
    p.add_argument("--width", type=int, default=VIEWER_WIDTH, help="screenshot width [px]")
    p.add_argument("--height", type=int, default=VIEWER_HEIGHT, help="screenshot height [px]")
    p.add_argument("--yaw", type=float, default=-55.0)
    p.add_argument("--pitch", type=float, default=28.0)
    p.add_argument("--fov", type=float, default=38.0, help="developer camera FOV [deg]")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    render(
        args.out,
        zoom=args.zoom,
        width=args.width,
        height=args.height,
        yaw=args.yaw,
        pitch=args.pitch,
        fov=args.fov,
    )


if __name__ == "__main__":
    main()
