"""Milestone 4 demo — scripted rigid box dropped onto an MPM clay pile.

Standalone demo (full Scene-API integration of MPM is a later milestone).
A kinematic rigid box is scripted through four phases — settle, lift,
translate, release — and then falls ballistically onto a VM-clay
(default) / DP-sand / metal / elastic pile sitting on the opposite
corner. Coupling is one-way via :class:`KinematicBoxCollider`.

Rendering via :class:`SimViewer` (same Taichi GGUI style as grasp_demo)
with the kinematic box drawn as a wireframe cube and the pile as a
particle cloud coloured by height.

Usage
-----
    python examples/mpm_grasp_drop.py                  # clay
    python examples/mpm_grasp_drop.py --material sand
    python examples/mpm_grasp_drop.py --material metal --headless
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
    DruckerPragerPlastic, NeoHookean, VonMisesPlastic,
)
from robosim.physics.mpm.solver import BoxBC, MPMSolver, sample_box_particles  # noqa: E402


DOMAIN_LOWER = np.array([0.0, 0.0, 0.0])
DOMAIN_UPPER = np.array([1.0, 1.0, 0.8])
DX           = 0.025

PILE_LOWER = np.array([0.58, 0.58, 0.0])
PILE_UPPER = np.array([0.82, 0.82, 0.15])
PILE_N     = 10
PILE_DENSITY = 1400.0

BOX_HALF   = np.array([0.05, 0.05, 0.05])
BOX_START  = np.array([0.15, 0.15, BOX_HALF[2]])
BOX_LIFT   = np.array([0.15, 0.15, 0.45])
BOX_OVER   = np.array([0.70, 0.70, 0.45])

PHASES = [(0.30, BOX_START), (0.40, BOX_LIFT), (0.60, BOX_OVER)]
RELEASE_AT = sum(d for d, _ in PHASES)
TOTAL_T    = RELEASE_AT + 0.60
DT         = 4e-4
SUBSTEPS_PER_FRAME = 4
GRAVITY = np.array([0.0, 0.0, -9.81])


def make_material(kind: str):
    if kind == "clay":
        return VonMisesPlastic(young=1.5e5, poisson=0.3, yield_stress=8e2)
    if kind == "sand":
        return DruckerPragerPlastic(young=1e5, poisson=0.3,
                                    friction_angle=np.deg2rad(40.0))
    if kind == "metal":
        return VonMisesPlastic(young=2e5, poisson=0.3, yield_stress=3e3)
    if kind == "elastic":
        return NeoHookean(young=8e4, poisson=0.3)
    raise ValueError(kind)


def scripted_box_pose(t: float) -> tuple[np.ndarray, np.ndarray]:
    if t < RELEASE_AT:
        t0 = 0.0
        prev = BOX_START
        for dur, tgt in PHASES:
            if t < t0 + dur:
                a = (t - t0) / dur
                return (1 - a) * prev + a * tgt, (tgt - prev) / dur
            t0 += dur
            prev = tgt
        return prev, np.zeros(3)
    dt_post = t - RELEASE_AT
    return BOX_OVER + 0.5 * GRAVITY * dt_post ** 2, GRAVITY * dt_post


def build_solver(kind: str):
    pts = sample_box_particles(PILE_LOWER, PILE_UPPER, PILE_N, PILE_DENSITY)
    grid = Grid.from_bounds(DOMAIN_LOWER, DOMAIN_UPPER, DX, pad=3)
    collider = KinematicBoxCollider(
        center=BOX_START.copy(), half_extent=BOX_HALF.copy(),
        velocity=np.zeros(3),
    )
    solver = MPMSolver(
        particles=pts, grid=grid,
        material=make_material(kind),
        gravity=GRAVITY,
        bcs=[BoxBC(lower=DOMAIN_LOWER,
                   upper=np.array([DOMAIN_UPPER[0], DOMAIN_UPPER[1], 10.0]),
                   mode="slip")],
        colliders=[collider],
    )
    return solver, collider


# ── SimViewer helpers: box wireframe + ground plane ─────────────────────────

def _ground_mesh(size: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    s = size / 2.0
    verts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]],
                     dtype=np.float64) + np.array([0.5, 0.5, 0.0])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, faces


def _box_wireframe_mesh(center: np.ndarray, half: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Corner verts + triangle indices for a thin-face box. Rendered as
    wireframe via SimViewer.set_meshes_wireframe_by_prefix."""
    c, h = center, half
    verts = np.array([
        [c[0]-h[0], c[1]-h[1], c[2]-h[2]], [c[0]+h[0], c[1]-h[1], c[2]-h[2]],
        [c[0]+h[0], c[1]+h[1], c[2]-h[2]], [c[0]-h[0], c[1]+h[1], c[2]-h[2]],
        [c[0]-h[0], c[1]-h[1], c[2]+h[2]], [c[0]+h[0], c[1]-h[1], c[2]+h[2]],
        [c[0]+h[0], c[1]+h[1], c[2]+h[2]], [c[0]-h[0], c[1]+h[1], c[2]+h[2]],
    ], dtype=np.float64)
    faces = np.array([
        [0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
        [2,3,7],[2,7,6],[0,4,7],[0,7,3],[1,2,6],[1,6,5],
    ], dtype=np.int32)
    return verts, faces


def _height_colors(z: np.ndarray) -> np.ndarray:
    t = np.clip(z / 0.4, 0, 1).astype(np.float32)
    return np.column_stack([0.3 + 0.5*t, 0.5 + 0.3*t, 1.0 - 0.4*t])


def run_headless(kind: str) -> None:
    solver, collider = build_solver(kind)
    pts = solver.particles
    n_steps = int(round(TOTAL_T / DT))
    for step in range(n_steps + 1):
        t = step * DT
        c, v = scripted_box_pose(t)
        collider.center = c
        collider.velocity = v
        if step % 100 == 0:
            z_top = float(pts.x[:, 2].max())
            xy_ext = float(np.ptp(pts.x[:, :2], axis=0).mean())
            print(f"t={t:5.3f}s  box=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})  "
                  f"z_top={z_top:.3f}  xy_ext={xy_ext:.3f}")
        if step < n_steps:
            solver.step(DT)


def run_gui(kind: str) -> None:
    from robosim.viz.viewer import SimViewer

    solver, collider = build_solver(kind)
    pts = solver.particles
    n_steps = int(round(TOTAL_T / DT))

    viewer = SimViewer(
        title=f"RoboSim — MPM Grasp-Drop [{kind.upper()}]",
        window_size=(1280, 800),
        background=(0.08, 0.08, 0.10),
    )
    viewer.initialize()

    gv, gf = _ground_mesh(size=1.2)
    viewer.add_mesh("ground", gv, gf, color=np.array([0.22, 0.22, 0.25]))

    bv, bf = _box_wireframe_mesh(collider.center, collider.half_extent)
    viewer.add_mesh("box/collider", bv, bf, color=np.array([0.95, 0.25, 0.25]))
    viewer.set_meshes_wireframe_by_prefix("box/", True)

    viewer.add_particles("mpm", pts.x, radius=0.009,
                         per_vertex_color=_height_colors(pts.x[:, 2]))

    step_count = [0]

    def step(frame: int) -> None:
        if step_count[0] >= n_steps:
            return
        for _ in range(SUBSTEPS_PER_FRAME):
            t = step_count[0] * DT
            c, v = scripted_box_pose(t)
            collider.center = c
            collider.velocity = v
            solver.step(DT)
            step_count[0] += 1
            if step_count[0] >= n_steps:
                break
        # Refresh box wireframe corners for the new pose.
        bv2, _ = _box_wireframe_mesh(collider.center, collider.half_extent)
        viewer.update_mesh_vertices("box/collider", bv2)
        viewer.update_particles("mpm", pts.x,
                                per_vertex_color=_height_colors(pts.x[:, 2]))
        t_sim = step_count[0] * DT
        z_top = float(pts.x[:, 2].max())
        xy_ext = float(np.ptp(pts.x[:, :2], axis=0).mean())
        phase = ("SETTLE" if t_sim < 0.30
                 else "LIFT" if t_sim < 0.70
                 else "TRANSLATE" if t_sim < RELEASE_AT
                 else "DROP")
        viewer.add_text(
            f"t = {t_sim:6.3f} s   step {step_count[0]}/{n_steps}   "
            f"[{kind.upper()}]   phase: {phase}\n"
            f"Box  : ({collider.center[0]:+.3f}, "
            f"{collider.center[1]:+.3f}, {collider.center[2]:+.3f})\n"
            f"Pile z_top: {z_top:.3f}   xy extent: {xy_ext:.3f}\n"
            f"[LDrag=orbit  Scroll=zoom  ESC=quit]"
        )

    viewer.add_callback(step)
    viewer.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", choices=["clay", "sand", "metal", "elastic"],
                    default="clay")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    if args.headless:
        run_headless(args.material)
    else:
        run_gui(args.material)


if __name__ == "__main__":
    main()
