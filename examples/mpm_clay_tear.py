"""Milestone 3 demo — clay bar torn apart via CD-MPM damage.

A horizontal clay bar is pulled at both ends with a uniform tensile
strain rate. Interior particles stretch past the critical principal
stretch, accumulate damage, and eventually stop transmitting stress —
so the bar separates into two fragments. Colour encodes per-particle
damage ``d ∈ [0, 1]``.

Usage
-----
    python examples/mpm_clay_tear.py
    python examples/mpm_clay_tear.py --headless
    python examples/mpm_clay_tear.py --save tear.mp4
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
STRAIN_RATE  = 10.0                       # 1/s — tensile pull rate
DT           = 2e-4
N_STEPS      = 900
SNAPSHOT_EVERY = 10


def build_solver() -> MPMSolver:
    xs = np.linspace(BAR_LOWER[0], BAR_UPPER[0], N_X)
    ys = np.linspace(BAR_LOWER[1], BAR_UPPER[1], N_YZ)
    zs = np.linspace(BAR_LOWER[2], BAR_UPPER[2], N_YZ)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    P = x.shape[0]
    V = float(np.prod(BAR_UPPER - BAR_LOWER))
    pts = Particles(
        x=x,
        v=np.zeros_like(x),
        m=np.full(P, DENSITY * V / P),
        V0=np.full(P, V / P),
    )
    # Uniform extension in x.
    pts.v[:, 0] = STRAIN_RATE * (pts.x[:, 0] - 0.5)

    grid = Grid.from_bounds(lower=DOMAIN_LOWER, upper=DOMAIN_UPPER, dx=DX, pad=3)
    return MPMSolver(
        particles=pts,
        grid=grid,
        material=DamagedNeoHookean(
            young=1.5e4, poisson=0.3, stretch_c=1.05, softening=3.0
        ),
        gravity=np.zeros(3),
        bcs=[],
    )


def run_headless() -> None:
    solver = build_solver()
    pts = solver.particles
    print(f"[init] {pts.n} particles, grid {solver.grid.shape}")
    for step in range(N_STEPS + 1):
        if step % 100 == 0:
            ext = float(np.ptp(pts.x[:, 0]))
            d_max = float(pts.d.max())
            d_mean = float(pts.d.mean())
            frac_broken = float((pts.d > 0.99).mean())
            print(
                f"t={step*DT:5.3f}s  x_ext={ext:.3f}  d_max={d_max:.3f}  "
                f"d_mean={d_mean:.3f}  broken={frac_broken:.2%}"
            )
        if step < N_STEPS:
            solver.step(DT)


def run_animated(save_path: str | None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    solver = build_solver()
    pts = solver.particles

    frames = [(pts.x.copy(), pts.d.copy())]
    for step in range(N_STEPS):
        solver.step(DT)
        if step % SNAPSHOT_EVERY == 0:
            frames.append((pts.x.copy(), pts.d.copy()))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.3, 0.7); ax.set_zlim(0.3, 0.7)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    x0, d0 = frames[0]
    scat = ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], s=10, c=d0,
                      cmap="inferno", vmin=0.0, vmax=1.0)
    fig.colorbar(scat, ax=ax, label="damage d")

    def update(i: int):
        x, d = frames[i]
        scat._offsets3d = (x[:, 0], x[:, 1], x[:, 2])
        scat.set_array(d)
        ax.set_title(f"CD-MPM clay tear — t={i*SNAPSHOT_EVERY*DT:.2f}s")
        return (scat,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False)

    if save_path is not None:
        anim.save(save_path, dpi=150, fps=25)
        print(f"saved animation to {save_path}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.headless and args.save is None:
        run_headless()
    else:
        run_animated(save_path=args.save)


if __name__ == "__main__":
    main()
