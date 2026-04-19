"""MPM plasticity demo — elastic / metal / sand side-by-side.

Drops an initially-elevated cube with a downward kick onto the floor and
runs the MPM solver under one of three materials:

* ``--material elastic``  — Neo-Hookean, bounces back and wobbles.
* ``--material metal``    — Von Mises J2, keeps a permanent flatten.
* ``--material sand``     — Drucker-Prager, collapses into a granular heap.

Uses :class:`SimViewer` (same Taichi GGUI style as grasp_demo).

Usage
-----
    python examples/mpm_plasticity_demo.py --material metal
    python examples/mpm_plasticity_demo.py --material sand --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.physics.mpm.grid import Grid                             # noqa: E402
from robosim.physics.mpm.materials import (                           # noqa: E402
    DruckerPragerPlastic, NeoHookean, VonMisesPlastic,
)
from robosim.physics.mpm.solver import BoxBC, MPMSolver, sample_box_particles  # noqa: E402


DOMAIN_LOWER = np.array([0.0, 0.0, 0.0])
DOMAIN_UPPER = np.array([1.0, 1.0, 1.0])
CUBE_LOWER   = np.array([0.45, 0.45, 0.35])
CUBE_UPPER   = np.array([0.55, 0.55, 0.45])
DX           = 0.02
N_PER_AXIS   = 8
DENSITY      = 1200.0
GRAVITY      = np.array([0.0, 0.0, -9.81])
DT           = 3e-4
N_STEPS      = 1500
SUBSTEPS_PER_FRAME = 4
INITIAL_KICK = np.array([0.0, 0.0, -2.0])


def make_material(kind: str):
    if kind == "elastic":
        return NeoHookean(young=1e5, poisson=0.3)
    if kind == "metal":
        return VonMisesPlastic(young=3e5, poisson=0.3, yield_stress=3e3)
    if kind == "sand":
        return DruckerPragerPlastic(young=8e4, poisson=0.3,
                                    friction_angle=np.deg2rad(35.0))
    raise ValueError(kind)


def build_solver(kind: str) -> MPMSolver:
    pts = sample_box_particles(CUBE_LOWER, CUBE_UPPER, N_PER_AXIS, DENSITY)
    pts.v[:] = INITIAL_KICK
    grid = Grid.from_bounds(DOMAIN_LOWER, DOMAIN_UPPER, DX, pad=3)
    return MPMSolver(
        particles=pts, grid=grid,
        material=make_material(kind),
        gravity=GRAVITY,
        bcs=[BoxBC(lower=DOMAIN_LOWER,
                   upper=np.array([DOMAIN_UPPER[0], DOMAIN_UPPER[1], 10.0]),
                   mode="slip")],
    )


def _ground_mesh(size: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    s = size / 2.0
    verts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]],
                     dtype=np.float64) + np.array([0.5, 0.5, 0.0])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, faces


def _height_colors(z: np.ndarray) -> np.ndarray:
    t = np.clip(z / 0.5, 0, 1)
    return np.column_stack([0.3 + 0.5*t, 0.5 + 0.3*t, 1.0 - 0.4*t]).astype(np.float32)


def run_headless(kind: str) -> None:
    solver = build_solver(kind)
    pts = solver.particles
    print(f"[init] material={kind}, {pts.n} particles")
    for step in range(N_STEPS + 1):
        if step % 100 == 0:
            com = pts.x.mean(axis=0)
            xy_ext = np.ptp(pts.x[:, :2], axis=0).mean()
            vcom_z = float((pts.m * pts.v[:, 2]).sum() / pts.m.sum())
            print(f"t={step*DT:5.3f}s  com_z={com[2]:.3f}  xy_ext={xy_ext:.3f}"
                  f"  v_com_z={vcom_z:+.3f}")
        if step < N_STEPS:
            solver.step(DT)


def run_gui(kind: str) -> None:
    from robosim.viz.viewer import SimViewer

    solver = build_solver(kind)
    pts = solver.particles

    viewer = SimViewer(
        title=f"RoboSim — MPM Plasticity [{kind.upper()}]",
        window_size=(1280, 800),
        background=(0.08, 0.08, 0.10),
    )
    viewer.initialize()

    gv, gf = _ground_mesh(size=1.2)
    viewer.add_mesh("ground", gv, gf, color=np.array([0.22, 0.22, 0.25]))
    viewer.add_particles("mpm", pts.x, radius=0.008,
                         per_vertex_color=_height_colors(pts.x[:, 2]))

    step_count = [0]

    def step(frame: int) -> None:
        if step_count[0] >= N_STEPS:
            return
        for _ in range(SUBSTEPS_PER_FRAME):
            solver.step(DT)
            step_count[0] += 1
            if step_count[0] >= N_STEPS:
                break
        viewer.update_particles("mpm", pts.x,
                                per_vertex_color=_height_colors(pts.x[:, 2]))
        com = pts.x.mean(axis=0)
        xy_ext = float(np.ptp(pts.x[:, :2], axis=0).mean())
        z_lo, z_hi = float(pts.x[:, 2].min()), float(pts.x[:, 2].max())
        t_sim = step_count[0] * DT
        viewer.add_text(
            f"t = {t_sim:6.3f} s   step {step_count[0]}/{N_STEPS}   [{kind.upper()}]\n"
            f"CoM   : ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f})\n"
            f"z range: [{z_lo:+.3f}, {z_hi:+.3f}]   xy extent: {xy_ext:.3f}\n"
            f"[LDrag=orbit  Scroll=zoom  ESC=quit]"
        )

    viewer.add_callback(step)
    viewer.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", choices=["elastic", "metal", "sand"],
                    default="metal")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    if args.headless:
        run_headless(args.material)
    else:
        import taichi as ti
        ti.init(arch=ti.metal)
        run_gui(args.material)


if __name__ == "__main__":
    main()
