"""Milestone 4 demo — scripted rigid box dropped onto an MPM clay pile.

Stands in for the full Scene-API integration (deferred). A kinematic
rigid box is scripted through four phases — approach, lift, translate,
release — and then falls under gravity onto a Drucker-Prager sand pile
sitting on the opposite side. The MPM material reacts through a
:class:`KinematicBoxCollider` hook, so the pile deforms, piles up, and
scatters when the block lands.

The "grasp" is stylised (box is teleported to the gripper pose during
lift), but the physical contact between the flying box and the MPM
pile is fully solved by the MLS-MPM step + collider projection.

Usage
-----
    python examples/mpm_grasp_drop.py
    python examples/mpm_grasp_drop.py --headless
    python examples/mpm_grasp_drop.py --material metal
    python examples/mpm_grasp_drop.py --save drop.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.physics.mpm.boundary import KinematicBoxCollider                # noqa: E402
from robosim.physics.mpm.grid import Grid                                     # noqa: E402
from robosim.physics.mpm.materials import (                                   # noqa: E402
    DruckerPragerPlastic,
    NeoHookean,
    VonMisesPlastic,
)
from robosim.physics.mpm.solver import BoxBC, MPMSolver, sample_box_particles  # noqa: E402


DOMAIN_LOWER = np.array([0.0, 0.0, 0.0])
DOMAIN_UPPER = np.array([1.0, 1.0, 0.8])
DX           = 0.025

# MPM clay pile — sits on the floor at +y side.
PILE_LOWER = np.array([0.58, 0.58, 0.0])
PILE_UPPER = np.array([0.82, 0.82, 0.15])
PILE_N     = 10
PILE_DENSITY = 1400.0

# Rigid box — initially resting on the floor at the −x/−y corner.
BOX_HALF   = np.array([0.05, 0.05, 0.05])
BOX_START  = np.array([0.15, 0.15, BOX_HALF[2]])       # resting
BOX_LIFT   = np.array([0.15, 0.15, 0.45])              # picked up
BOX_OVER   = np.array([0.70, 0.70, 0.45])              # over the pile
# Release point is the same as BOX_OVER — gravity then pulls it onto the pile.

# Scripted phases (tuples of (duration_s, target_center)).
PHASES = [
    (0.30, BOX_START),   # settle
    (0.40, BOX_LIFT),    # lift straight up
    (0.60, BOX_OVER),    # translate over the pile
]
RELEASE_AT = sum(d for d, _ in PHASES)
TOTAL_T    = RELEASE_AT + 0.60         # 0.6 s of free-fall drop
DT         = 4e-4
SNAPSHOT_EVERY = 20
GRAVITY = np.array([0.0, 0.0, -9.81])
BOX_MASS = 2.0                                         # for future two-way


def make_material(kind: str):
    if kind == "clay":
        # Cohesive clay — ductile, retains a dent after impact.
        return VonMisesPlastic(young=1.5e5, poisson=0.3, yield_stress=8e2)
    if kind == "sand":
        return DruckerPragerPlastic(
            young=1e5, poisson=0.3, friction_angle=np.deg2rad(40.0)
        )
    if kind == "metal":
        return VonMisesPlastic(young=2e5, poisson=0.3, yield_stress=3e3)
    if kind == "elastic":
        return NeoHookean(young=8e4, poisson=0.3)
    raise ValueError(f"unknown material: {kind!r}")


def scripted_box_pose(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, velocity) at time ``t`` following the phase plan."""
    # Before release: linear interpolation through phase waypoints.
    if t < RELEASE_AT:
        t0 = 0.0
        prev = BOX_START
        for duration, target in PHASES:
            if t < t0 + duration:
                alpha = (t - t0) / duration
                center = (1.0 - alpha) * prev + alpha * target
                velocity = (target - prev) / duration
                return center, velocity
            t0 += duration
            prev = target
        return prev, np.zeros(3)
    # After release: ballistic. Initial state = last phase end, velocity 0.
    dt_post = t - RELEASE_AT
    center = BOX_OVER + 0.5 * GRAVITY * dt_post ** 2
    velocity = GRAVITY * dt_post
    return center, velocity


def build_solver(kind: str) -> tuple[MPMSolver, KinematicBoxCollider]:
    pts = sample_box_particles(
        lower=PILE_LOWER, upper=PILE_UPPER, n_per_axis=PILE_N, density=PILE_DENSITY
    )
    grid = Grid.from_bounds(lower=DOMAIN_LOWER, upper=DOMAIN_UPPER, dx=DX, pad=3)

    collider = KinematicBoxCollider(
        center=BOX_START.copy(),
        half_extent=BOX_HALF.copy(),
        velocity=np.zeros(3),
    )
    solver = MPMSolver(
        particles=pts,
        grid=grid,
        material=make_material(kind),
        gravity=GRAVITY,
        bcs=[BoxBC(
            lower=DOMAIN_LOWER,
            upper=np.array([DOMAIN_UPPER[0], DOMAIN_UPPER[1], 10.0]),
            mode="slip",
        )],
        colliders=[collider],
    )
    return solver, collider


def simulate(kind: str, record: bool):
    solver, collider = build_solver(kind)
    pts = solver.particles
    n_steps = int(round(TOTAL_T / DT))
    frames = [] if record else None

    for step in range(n_steps + 1):
        t = step * DT
        center, vel = scripted_box_pose(t)
        collider.center = center
        collider.velocity = vel

        if record and step % SNAPSHOT_EVERY == 0:
            frames.append((t, pts.x.copy(), center.copy()))

        if step % 100 == 0:
            z_top = float(pts.x[:, 2].max())
            xy_ext = float(np.ptp(pts.x[:, :2], axis=0).mean())
            print(
                f"t={t:5.3f}s  box=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})  "
                f"pile z_top={z_top:.3f}  xy_ext={xy_ext:.3f}"
            )

        if step < n_steps:
            solver.step(DT)

    return frames


def run_headless(kind: str) -> None:
    simulate(kind, record=False)


def run_animated(kind: str, save_path: str | None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    frames = simulate(kind, record=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(DOMAIN_LOWER[0], DOMAIN_UPPER[0])
    ax.set_ylim(DOMAIN_LOWER[1], DOMAIN_UPPER[1])
    ax.set_zlim(DOMAIN_LOWER[2], DOMAIN_UPPER[2])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    t0, x0, c0 = frames[0]
    scat = ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], s=8, c=x0[:, 2],
                      cmap="viridis", vmin=0.0, vmax=0.4)

    # Wireframe box: represent the collider as a line cube.
    def cube_lines(center, half):
        c, h = center, half
        corners = np.array([
            [c[0]-h[0], c[1]-h[1], c[2]-h[2]], [c[0]+h[0], c[1]-h[1], c[2]-h[2]],
            [c[0]+h[0], c[1]+h[1], c[2]-h[2]], [c[0]-h[0], c[1]+h[1], c[2]-h[2]],
            [c[0]-h[0], c[1]-h[1], c[2]+h[2]], [c[0]+h[0], c[1]-h[1], c[2]+h[2]],
            [c[0]+h[0], c[1]+h[1], c[2]+h[2]], [c[0]-h[0], c[1]+h[1], c[2]+h[2]],
        ])
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        segs = np.stack([corners[np.array(e)] for e in edges])
        return segs

    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    box_lines = Line3DCollection(cube_lines(c0, BOX_HALF), colors="crimson", lw=1.5)
    ax.add_collection(box_lines)

    def update(i: int):
        t, x, c = frames[i]
        scat._offsets3d = (x[:, 0], x[:, 1], x[:, 2])
        scat.set_array(x[:, 2])
        box_lines.set_segments(cube_lines(c, BOX_HALF))
        ax.set_title(f"{kind} pile — t={t:.2f}s")
        return (scat, box_lines)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False)

    if save_path is not None:
        anim.save(save_path, dpi=150, fps=25)
        print(f"saved animation to {save_path}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", choices=["clay", "sand", "metal", "elastic"],
                    default="clay")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    if args.headless and args.save is None:
        run_headless(args.material)
    else:
        run_animated(args.material, save_path=args.save)


if __name__ == "__main__":
    main()
