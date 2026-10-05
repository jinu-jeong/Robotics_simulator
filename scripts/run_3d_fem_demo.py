#!/usr/bin/env python
"""Milestone 1 – 3D FEM of the compliant finger with a single point contact.

Builds the finger from ``configs/fem.yaml`` (geometry, mesh, material,
clamped root, default contact), applies the contact force through the
operator ``B(c)`` and shows the result in the Taichi viewer: deformed
surface coloured by |u| (press ``M`` for von Mises stress), fixed root nodes,
contact marker, force arrow, probe sphere.

Run:
    python scripts/run_3d_fem_demo.py
    python scripts/run_3d_fem_demo.py --force 2.0 --contact-rel 0.6 0.5 1.0
    python scripts/run_3d_fem_demo.py --nx 40 --ny 8 --nz 6 --headless --screenshot /tmp/fem.png
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

from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.visualization.camera import ObservationCamera  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="fem")
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--nz", type=int, default=None)
    p.add_argument("--force", type=float, default=None, help="contact force magnitude [N] (default from config)")
    p.add_argument("--contact-rel", type=float, nargs=3, default=None, metavar=("FX", "FY", "FZ"),
                   help="contact location as fractions of (L, W, H)")
    p.add_argument("--direction", type=float, nargs=3, default=None, help="force direction (default from config)")
    p.add_argument("--amplification", type=float, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--screenshot", type=str, default=None)
    p.add_argument("--no-gravity", action="store_true", help="ignore fem.gravity (contact load only)")
    args = p.parse_args()

    cfg = load_config(args.config)
    t0 = time.time()
    model = FingerFEMModel.from_config(cfg, args.nx, args.ny, args.nz)
    t_build = time.time() - t0

    load = cfg["loading"]
    F = float(args.force if args.force is not None else load["force_magnitude"])
    d = np.asarray(args.direction if args.direction is not None else load["force_direction"], float)
    d /= np.linalg.norm(d)
    point = model.geometry.point_from_relative(args.contact_rel) if args.contact_rel else model.default_contact_point()

    t0 = time.time()
    result, contact = model.solve_point_force(point, F * d)
    t_solve = time.time() - t0

    g = model.geometry
    print(f"finger {g.length}x{g.width}x{g.height} m, E={model.material.E:.3g} Pa, nu={model.material.nu}")
    print(f"mesh {model.mesh.summary()} / {model.mesh.n_dofs} dofs  (build+factorize {t_build:.2f}s, solve {t_solve * 1e3:.1f} ms)")
    print(f"contact at {contact.position} (face normal {contact.normal}), force {F * d} N")
    print(f"max |u| = {1e3 * result.max_displacement():.4f} mm, equilibrium residual {result.equilibrium_residual():.1e}")
    tip = result.u[model.mesh.nodes_on_plane(0, g.length), 2].mean()
    print(f"mean tip u_z = {1e3 * tip:.4f} mm")

    # Self-weight (config fem.gravity) by superposition: u = u_contact + u_g.
    use_gravity = model.has_gravity and not args.no_gravity
    if use_gravity:
        u_g = model.gravity_displacement()
        tip_g = u_g[model.mesh.nodes_on_plane(0, g.length), 2].mean()
        print(f"self-weight: ρV = {1e3 * model.material.density * g.length * g.width * g.height:.1f} g, "
              f"tip sag {1e3 * tip_g:.4f} mm (contact-only tip {1e3 * tip:.4f} mm)")
        result.u = result.u + u_g
        result.meta["self_weight"] = True

    cam_target = 0.5 * g.size
    obs_cam = ObservationCamera(
        position=cam_target + np.array([0.05 * g.length, -1.1 * g.length, 0.55 * g.length]),
        target=cam_target, fov_y_deg=40.0, near=0.15 * g.length, far=1.2 * g.length, name="obs_cam_0",
    )
    state = model.visualization_state(
        result, contact, F * d, observation_camera=obs_cam,
        info={"solve time": f"{t_solve * 1e3:.1f} ms", "mean tip u_z": f"{1e3 * tip:.4f} mm",
              "self-weight": "on" if use_gravity else "off"},
    )

    from src.visualization.taichi_viewer import TaichiViewer

    viewer = TaichiViewer(show_window=not args.headless)
    if args.amplification is not None:
        viewer.options.amplification = args.amplification
    viewer.set_state(state)
    if args.screenshot:
        viewer.save_screenshot(args.screenshot)
    if not args.headless:
        viewer.run()


if __name__ == "__main__":
    main()
