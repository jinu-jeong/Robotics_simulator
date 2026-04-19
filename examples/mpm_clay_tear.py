"""Milestone 3 demo — clay bar torn apart via CD-MPM damage.

A horizontal clay bar is pulled at both ends with a uniform tensile
strain rate. Interior particles stretch past the critical principal
stretch, accumulate damage, and eventually stop transmitting stress —
so the bar separates into two fragments.

Colour encodes per-particle damage ``d ∈ [0, 1]`` (cool → hot).
Rendering via :class:`SimViewer` (same Taichi GGUI style as grasp_demo).

Usage
-----
    python examples/mpm_clay_tear.py
    python examples/mpm_clay_tear.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.physics.mpm.grid import Grid                             # noqa: E402
from robosim.physics.mpm.materials import DamagedNeoHookean           # noqa: E402
from robosim.physics.mpm.particles import Particles                   # noqa: E402
from robosim.physics.mpm.solver import MPMSolver                      # noqa: E402


DOMAIN_LOWER = np.array([0.0, 0.0, 0.0])
DOMAIN_UPPER = np.array([1.0, 1.0, 1.0])
BAR_LOWER    = np.array([0.30, 0.48, 0.48])
BAR_UPPER    = np.array([0.70, 0.52, 0.52])
DX           = 0.02
N_X, N_YZ    = 41, 5
DENSITY      = 1200.0
STRAIN_RATE  = 10.0
DT           = 2e-4
N_STEPS      = 900
SUBSTEPS_PER_FRAME = 4


def build_solver() -> MPMSolver:
    xs = np.linspace(BAR_LOWER[0], BAR_UPPER[0], N_X)
    ys = np.linspace(BAR_LOWER[1], BAR_UPPER[1], N_YZ)
    zs = np.linspace(BAR_LOWER[2], BAR_UPPER[2], N_YZ)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    P = x.shape[0]
    V = float(np.prod(BAR_UPPER - BAR_LOWER))
    pts = Particles(
        x=x, v=np.zeros_like(x),
        m=np.full(P, DENSITY * V / P),
        V0=np.full(P, V / P),
    )
    pts.v[:, 0] = STRAIN_RATE * (pts.x[:, 0] - 0.5)
    grid = Grid.from_bounds(DOMAIN_LOWER, DOMAIN_UPPER, DX, pad=3)
    return MPMSolver(
        particles=pts, grid=grid,
        material=DamagedNeoHookean(young=1.5e4, poisson=0.3,
                                   stretch_c=1.05, softening=3.0),
        gravity=np.zeros(3), bcs=[],
    )


def _ground_mesh(size: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    s = size / 2.0
    verts = np.array([[-s, -s, 0.47], [s, -s, 0.47], [s, s, 0.47], [-s, s, 0.47]],
                     dtype=np.float64) + np.array([0.5, 0.5, 0.0])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, faces


def _damage_colors(d: np.ndarray) -> np.ndarray:
    """Inferno-ish: cool blue (pristine) → orange/yellow (fully broken)."""
    d = np.clip(d, 0.0, 1.0).astype(np.float32)
    r = np.clip(1.6 * d, 0, 1)
    g = np.clip(0.2 + 0.8 * d * d, 0, 1)
    b = np.clip(0.8 * (1.0 - d), 0, 1)
    return np.column_stack([r, g, b])


def run_headless() -> None:
    solver = build_solver()
    pts = solver.particles
    print(f"[init] {pts.n} particles, grid {solver.grid.shape}")
    for step in range(N_STEPS + 1):
        if step % 100 == 0:
            ext = float(np.ptp(pts.x[:, 0]))
            print(f"t={step*DT:5.3f}s  x_ext={ext:.3f}  "
                  f"d_max={pts.d.max():.3f}  d_mean={pts.d.mean():.3f}  "
                  f"broken={float((pts.d > 0.99).mean()):.2%}")
        if step < N_STEPS:
            solver.step(DT)


def run_gui() -> None:
    from robosim.viz.viewer import SimViewer

    solver = build_solver()
    pts = solver.particles

    viewer = SimViewer(
        title="RoboSim — CD-MPM Clay Tear",
        window_size=(1280, 800),
        background=(0.08, 0.08, 0.10),
    )
    viewer.initialize()

    gv, gf = _ground_mesh(size=1.4)
    viewer.add_mesh("ground", gv, gf, color=np.array([0.22, 0.22, 0.25]))
    viewer.add_particles("mpm", pts.x, radius=0.007,
                         per_vertex_color=_damage_colors(pts.d))

    step_count = [0]

    def step(frame: int) -> None:
        for _ in range(SUBSTEPS_PER_FRAME):
            try:
                solver.step(DT)
            except IndexError:
                return  # fragments flew off the grid — freeze, keep drawing
            step_count[0] += 1
        viewer.update_particles("mpm", pts.x,
                                per_vertex_color=_damage_colors(pts.d))
        ext = float(np.ptp(pts.x[:, 0]))
        t_sim = step_count[0] * DT
        viewer.add_text(
            f"t = {t_sim:6.3f} s   step {step_count[0]}\n"
            f"x extent : {ext:.3f} m\n"
            f"d_max = {pts.d.max():.3f}   d_mean = {pts.d.mean():.3f}\n"
            f"broken   : {float((pts.d > 0.99).mean()):.2%}\n"
            f"[LDrag=orbit  Scroll=zoom  ESC=quit]"
        )

    viewer.add_callback(step)
    viewer.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    if args.headless:
        run_headless()
    else:
        import taichi as ti
        ti.init(arch=ti.metal)
        run_gui()


if __name__ == "__main__":
    main()
