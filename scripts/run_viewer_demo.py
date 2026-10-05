#!/usr/bin/env python
"""Milestone 0A – Taichi 3D viewer demo with an *artificial* deformation.

No FEM is involved yet. A rectangular finger is meshed, clamped at x = 0 and
given an Euler–Bernoulli-shaped cantilever deflection so every viewer feature
(overlay, amplification, fixed nodes, contact marker, GT/estimated force
arrows, object, observation camera) can be checked visually.

Artificial displacement field (tip load at x = L, deflection in -z):

    w(x)   = -delta_tip * x^2 (3L - x) / (2 L^3)      (transverse, u_z)
    u_x    = -(z - H/2) * dw/dx                       (beam kinematics)
    u_y    = 0

Run:
    python scripts/run_viewer_demo.py
    python scripts/run_viewer_demo.py --nx 40 --ny 8 --nz 4 --tip-deflection-mm 3
    python scripts/run_viewer_demo.py --headless --screenshot /tmp/demo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.finger import FingerGeometry, make_rectangular_finger_mesh, root_fixed_nodes  # noqa: E402
from src.geometry.mesh_utils import nearest_node, vertex_normals  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.visualization.camera import ObservationCamera  # noqa: E402
from src.visualization.state import ContactPoint, ForceVector, ObjectGeometry, VisualizationState  # noqa: E402


def cantilever_artificial_displacement(nodes: np.ndarray, length: float, height: float, tip_deflection: float) -> np.ndarray:
    """Euler–Bernoulli tip-load deflection shape, used only to exercise the viewer."""
    x, z = nodes[:, 0], nodes[:, 2]
    L = length
    w = -tip_deflection * x**2 * (3.0 * L - x) / (2.0 * L**3)
    dwdx = -tip_deflection * (6.0 * L * x - 3.0 * x**2) / (2.0 * L**3)
    u = np.zeros_like(nodes)
    u[:, 0] = -(z - 0.5 * height) * dwdx
    u[:, 2] = w
    return u


def build_demo_state(args, fem_cfg) -> VisualizationState:
    geom = FingerGeometry.from_config(fem_cfg)
    mesh = make_rectangular_finger_mesh(geom, args.nx, args.ny, args.nz)

    u = cantilever_artificial_displacement(mesh.nodes, geom.length, geom.height, args.tip_deflection_mm * 1e-3)
    fixed = root_fixed_nodes(mesh)
    u[fixed] = 0.0  # a clamped root must not move (visual check of the BC marker)

    # Contact: nearest top-surface node to the configured relative position.
    load_cfg = fem_cfg["loading"]
    target = geom.point_from_relative(load_cfg["contact_position_rel"])
    contact_node = nearest_node(mesh.nodes, target, candidates=mesh.surface_nodes)
    contact_pos = mesh.nodes[contact_node]
    normals = vertex_normals(mesh.nodes, mesh.surface_faces)
    contact_normal = normals[contact_node]

    f_dir = np.asarray(load_cfg["force_direction"], float)
    f_dir /= np.linalg.norm(f_dir)
    f_mag = float(load_cfg["force_magnitude"])
    gt_force = ForceVector(contact_pos, f_mag * f_dir, label="F_true")
    # Deliberately imperfect "estimate" so both arrows are distinguishable.
    est_vec = 0.88 * f_mag * f_dir + 0.06 * f_mag * np.array([1.0, 0.3, 0.0])
    est_force = ForceVector(contact_pos, est_vec, label="F_est")

    # Object: a sphere touching the contact point along the outward normal. It
    # is attached to the contact so it rides on the (amplified) deformed surface.
    obj_radius = 0.4 * geom.height
    sphere = ObjectGeometry.sphere(
        contact_pos + contact_normal * obj_radius, obj_radius, name="probe sphere", attach_to=contact_pos
    )

    cam_target = np.array([0.5 * geom.length, 0.5 * geom.width, 0.5 * geom.height])
    cam_pos = cam_target + np.array([0.05 * geom.length, -1.1 * geom.length, 0.55 * geom.length])
    obs_cam = ObservationCamera(
        position=cam_pos,
        target=cam_target,
        fov_y_deg=40.0,
        image_width=640,
        image_height=480,
        near=0.15 * geom.length,
        far=float(np.linalg.norm(cam_pos - cam_target)),  # frustum drawn up to the look-at point
        name="obs_cam_0",
    )

    return VisualizationState(
        nodes_original=mesh.nodes,
        surface_faces=mesh.surface_faces,
        element_edges=mesh.element_edges,
        surface_edges=mesh.surface_edges,
        displacement=u,
        fixed_nodes=fixed,
        contact_points=[ContactPoint(contact_pos, node_index=contact_node, normal=contact_normal)],
        ground_truth_force=gt_force,
        estimated_force=est_force,
        objects=[sphere],
        observation_camera=obs_cam,
        info={
            "demo": "artificial cantilever deflection (no FEM yet)",
            "tip deflection (input)": f"{args.tip_deflection_mm:.2f} mm",
            "mesh": f"{args.nx}x{args.ny}x{args.nz} cells: {mesh.n_hexes} hex20 + {mesh.n_tets} tet10",
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fem-config", default="fem", help="configs/<name>.yaml or path")
    p.add_argument("--viewer-config", default="viewer")
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--nz", type=int, default=None)
    p.add_argument("--tip-deflection-mm", type=float, default=2.0, help="artificial tip deflection [mm]")
    p.add_argument("--amplification", type=float, default=None, help="initial visual amplification")
    p.add_argument("--mode", choices=["original", "deformed", "overlay"], default=None)
    p.add_argument("--backend", default=None, help="auto|cpu|cuda|vulkan|metal")
    p.add_argument("--headless", action="store_true", help="do not open a window")
    p.add_argument("--screenshot", type=str, default=None, help="save a PNG (implies one rendered frame)")
    args = p.parse_args()

    fem_cfg = load_config(args.fem_config)
    m = fem_cfg.get("mesh", {})
    args.nx = args.nx or int(m.get("nx", 20))
    args.ny = args.ny or int(m.get("ny", 4))
    args.nz = args.nz or int(m.get("nz", 2))

    overrides = {}
    if args.backend:
        overrides["backend"] = args.backend

    # Import here so `--help` works without a Taichi runtime.
    from src.visualization.taichi_viewer import TaichiViewer

    viewer = TaichiViewer(config=overrides, config_name=args.viewer_config, show_window=not args.headless)
    if args.amplification is not None:
        viewer.options.amplification = args.amplification
    if args.mode is not None:
        viewer.options.mode = args.mode

    state = build_demo_state(args, fem_cfg)
    viewer.set_state(state)

    print("\n".join(viewer.renderer.summary_lines(viewer.options)))
    if args.screenshot:
        viewer.save_screenshot(args.screenshot)
    if not args.headless:
        viewer.run()


if __name__ == "__main__":
    main()
