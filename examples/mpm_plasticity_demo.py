"""Milestone 2 demo — plastic MPM materials compared side by side.

Drops an initially-elevated cube with a downward kick onto the floor and
runs the MPM solver under one of three materials:

* ``--material elastic``  — pure Neo-Hookean, bounces back and wobbles.
* ``--material metal``    — Von Mises J2, keeps a permanent flatten.
* ``--material sand``     — Drucker-Prager, collapses into a granular heap.

Usage
-----
    python examples/mpm_plasticity_demo.py --material metal
    python examples/mpm_plasticity_demo.py --material sand --headless
    python examples/mpm_plasticity_demo.py --material elastic --save elastic.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.physics.mpm.grid import Grid                             # noqa: E402
from robosim.physics.mpm.materials import (                           # noqa: E402
    DruckerPragerPlastic,
    NeoHookean,
    VonMisesPlastic,
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
SNAPSHOT_EVERY = 20
INITIAL_KICK = np.array([0.0, 0.0, -2.0])     # extra downward velocity


def make_material(kind: str):
    if kind == "elastic":
        return NeoHookean(young=1e5, poisson=0.3)
    if kind == "metal":
        # Metal-like: stiff bulk, moderate yield so dents stay without
        # behaving like a fluid.
        return VonMisesPlastic(young=3e5, poisson=0.3, yield_stress=3e3)
    if kind == "sand":
        return DruckerPragerPlastic(
            young=8e4, poisson=0.3, friction_angle=np.deg2rad(35.0)
        )
    raise ValueError(f"unknown material: {kind!r}")


def build_solver(kind: str) -> MPMSolver:
    pts = sample_box_particles(
        lower=CUBE_LOWER,
        upper=CUBE_UPPER,
        n_per_axis=N_PER_AXIS,
        density=DENSITY,
    )
    pts.v[:] = INITIAL_KICK
    grid = Grid.from_bounds(lower=DOMAIN_LOWER, upper=DOMAIN_UPPER, dx=DX, pad=3)
    return MPMSolver(
        particles=pts,
        grid=grid,
        material=make_material(kind),
        gravity=GRAVITY,
        bcs=[BoxBC(
            lower=DOMAIN_LOWER,
            upper=np.array([DOMAIN_UPPER[0], DOMAIN_UPPER[1], 10.0]),
            mode="slip",
        )],
    )


def run_headless(kind: str) -> None:
    solver = build_solver(kind)
    pts = solver.particles
    print(f"[init] material={kind}, {pts.n} particles, grid {solver.grid.shape}")
    for step in range(N_STEPS + 1):
        if step % 100 == 0:
            com = pts.x.mean(axis=0)
            z_lo, z_hi = pts.x[:, 2].min(), pts.x[:, 2].max()
            xy_ext = np.ptp(pts.x[:, :2], axis=0).mean()
            vcom_z = float((pts.m * pts.v[:, 2]).sum() / pts.m.sum())
            print(
                f"t={step*DT:5.3f}s  com_z={com[2]:.3f}  z=[{z_lo:.3f},{z_hi:.3f}]  "
                f"xy_extent={xy_ext:.3f}  v_com_z={vcom_z:+.3f}"
            )
        if step < N_STEPS:
            solver.step(DT)


def run_animated(kind: str, save_path: str | None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    solver = build_solver(kind)
    pts = solver.particles

    frames = [pts.x.copy()]
    for step in range(N_STEPS):
        solver.step(DT)
        if step % SNAPSHOT_EVERY == 0:
            frames.append(pts.x.copy())

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(DOMAIN_LOWER[0], DOMAIN_UPPER[0])
    ax.set_ylim(DOMAIN_LOWER[1], DOMAIN_UPPER[1])
    ax.set_zlim(DOMAIN_LOWER[2], DOMAIN_UPPER[2])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    scat = ax.scatter(
        frames[0][:, 0], frames[0][:, 1], frames[0][:, 2],
        s=8, c=frames[0][:, 2], cmap="plasma",
    )

    def update(i: int):
        x = frames[i]
        scat._offsets3d = (x[:, 0], x[:, 1], x[:, 2])
        scat.set_array(x[:, 2])
        ax.set_title(f"{kind} — t={i * SNAPSHOT_EVERY * DT:.2f}s")
        return (scat,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False)

    if save_path is not None:
        anim.save(save_path, dpi=150, fps=25)
        print(f"saved animation to {save_path}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", choices=["elastic", "metal", "sand"], default="metal")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    if args.headless and args.save is None:
        run_headless(args.material)
    else:
        run_animated(args.material, save_path=args.save)


if __name__ == "__main__":
    main()
