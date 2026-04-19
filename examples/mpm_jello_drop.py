"""Standalone MPM demo — elastic jello cube dropped onto a floor.

Milestone 1 deliverable for the MPM solver. Runs the MLS-MPM solver
directly (Scene API / rigid-body coupling arrives in Milestone 4).

Usage
-----
    python examples/mpm_jello_drop.py                  # live matplotlib animation
    python examples/mpm_jello_drop.py --headless       # print diagnostics, no GUI
    python examples/mpm_jello_drop.py --save out.mp4   # record an MP4 (needs ffmpeg)

The cube starts at rest above the floor, falls under gravity, compresses
on impact, rebounds, and wobbles. Centre-of-mass trajectory and total
kinetic + elastic potential energy are printed along the way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.physics.mpm.grid import Grid                             # noqa: E402
from robosim.physics.mpm.materials import NeoHookean                  # noqa: E402
from robosim.physics.mpm.solver import BoxBC, MPMSolver, sample_box_particles  # noqa: E402


# ── Physical setup ───────────────────────────────────────────────────────────

DOMAIN_LOWER = np.array([0.0, 0.0, 0.0])
DOMAIN_UPPER = np.array([0.6, 0.6, 0.6])
CUBE_LOWER   = np.array([0.25, 0.25, 0.35])
CUBE_UPPER   = np.array([0.35, 0.35, 0.45])
DX           = 0.015
N_PER_AXIS   = 8        # 512 particles inside the cube
DENSITY      = 1000.0   # kg/m³  (water-like jello)
YOUNG        = 4e4      # Pa
POISSON      = 0.3
GRAVITY      = np.array([0.0, 0.0, -9.81])
DT           = 3e-4
N_STEPS      = 1200     # ~0.36 s of sim time
SNAPSHOT_EVERY = 20     # frames captured per dt_snapshot for animation


def build_solver() -> MPMSolver:
    pts = sample_box_particles(
        lower=CUBE_LOWER,
        upper=CUBE_UPPER,
        n_per_axis=N_PER_AXIS,
        density=DENSITY,
    )
    grid = Grid.from_bounds(lower=DOMAIN_LOWER, upper=DOMAIN_UPPER, dx=DX, pad=3)
    return MPMSolver(
        particles=pts,
        grid=grid,
        material=NeoHookean(young=YOUNG, poisson=POISSON),
        gravity=GRAVITY,
        bcs=[
            BoxBC(
                lower=DOMAIN_LOWER,
                upper=np.array([DOMAIN_UPPER[0], DOMAIN_UPPER[1], 10.0]),
                mode="slip",
            ),
        ],
    )


def kinetic_energy(solver: MPMSolver) -> float:
    p = solver.particles
    return 0.5 * float((p.m * np.einsum("pa,pa->p", p.v, p.v)).sum())


def elastic_energy(solver: MPMSolver) -> float:
    """Neo-Hookean strain energy density, integrated over the particles."""
    p = solver.particles
    mat = solver.material
    J = np.linalg.det(p.F)
    IC = np.einsum("pab,pab->p", p.F, p.F)       # tr(F F^T)
    psi = 0.5 * mat.mu * (IC - 3.0) - mat.mu * np.log(J) + 0.5 * mat.lam * np.log(J) ** 2
    return float((psi * p.V0).sum())


# ── Drivers ──────────────────────────────────────────────────────────────────

def run_headless() -> None:
    solver = build_solver()
    pts = solver.particles
    print(f"[init] {pts.n} particles, grid {solver.grid.shape}, dx={solver.grid.dx}")

    for step in range(N_STEPS + 1):
        if step % 60 == 0:
            com = pts.x.mean(axis=0)
            vcom = (pts.m[:, None] * pts.v).sum(axis=0) / pts.m.sum()
            KE = kinetic_energy(solver)
            PE = elastic_energy(solver)
            t = step * DT
            print(
                f"t={t:5.3f}s  com=({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f})  "
                f"v_com_z={vcom[2]:+.3f}  KE={KE:.3e}  PE={PE:.3e}  total={KE+PE:.3e}"
            )
        if step < N_STEPS:
            solver.step(DT)


def run_animated(save_path: str | None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    solver = build_solver()
    pts = solver.particles

    # Pre-roll one step so the scatter has something non-degenerate to show.
    frames_x = [pts.x.copy()]
    for step in range(N_STEPS):
        solver.step(DT)
        if step % SNAPSHOT_EVERY == 0:
            frames_x.append(pts.x.copy())

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(DOMAIN_LOWER[0], DOMAIN_UPPER[0])
    ax.set_ylim(DOMAIN_LOWER[1], DOMAIN_UPPER[1])
    ax.set_zlim(DOMAIN_LOWER[2], DOMAIN_UPPER[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("MPM jello drop")

    scat = ax.scatter(
        frames_x[0][:, 0], frames_x[0][:, 1], frames_x[0][:, 2],
        s=8, c=frames_x[0][:, 2], cmap="viridis",
    )

    def update(i: int):
        x = frames_x[i]
        scat._offsets3d = (x[:, 0], x[:, 1], x[:, 2])
        scat.set_array(x[:, 2])
        ax.set_title(f"MPM jello drop — t={i * SNAPSHOT_EVERY * DT:.2f}s")
        return (scat,)

    anim = FuncAnimation(fig, update, frames=len(frames_x), interval=40, blit=False)

    if save_path is not None:
        anim.save(save_path, dpi=150, fps=25)
        print(f"saved animation to {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="MPM jello drop demo")
    parser.add_argument("--headless", action="store_true", help="print diagnostics, no GUI")
    parser.add_argument("--save", default=None, help="save animation to this path (mp4/gif)")
    args = parser.parse_args()

    if args.headless and args.save is None:
        run_headless()
    else:
        run_animated(save_path=args.save)


if __name__ == "__main__":
    main()
