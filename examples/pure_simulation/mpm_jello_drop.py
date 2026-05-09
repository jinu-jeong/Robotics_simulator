"""Standalone MPM demo — elastic jello cube dropped onto a floor.

Taichi GGUI live view via :class:`SimViewer`.
Headless mode prints diagnostics.

Usage
-----
    python examples/mpm_jello_drop.py                # GUI
    python examples/mpm_jello_drop.py --headless     # diagnostics only
"""

from __future__ import annotations

import argparse
import sys
import time
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
N_PER_AXIS   = 8
DENSITY      = 1000.0
YOUNG        = 4e4
POISSON      = 0.3
GRAVITY      = np.array([0.0, 0.0, -9.81])
DT           = 3e-4
N_STEPS      = 1200
SUBSTEPS_PER_FRAME = 4


def build_solver(backend: str = "numpy"):
    pts  = sample_box_particles(CUBE_LOWER, CUBE_UPPER, N_PER_AXIS, DENSITY)
    grid = Grid.from_bounds(DOMAIN_LOWER, DOMAIN_UPPER, DX, pad=3)
    if backend == "taichi":
        from robosim.physics.mpm.taichi_solver import TaichiMPMSolver
        bc_upper = np.array([DOMAIN_UPPER[0], DOMAIN_UPPER[1], 10.0])
        return TaichiMPMSolver(
            particles_x=pts.x.copy(), particles_v=pts.v.copy(),
            particles_m=pts.m.copy(), particles_V0=pts.V0.copy(),
            grid_origin=grid.origin, grid_dx=grid.dx, grid_shape=grid.shape,
            young=YOUNG, poisson=POISSON, gravity=GRAVITY,
            bc_lower=DOMAIN_LOWER, bc_upper=bc_upper,
        )
    return MPMSolver(
        particles=pts, grid=grid,
        material=NeoHookean(young=YOUNG, poisson=POISSON),
        gravity=GRAVITY,
        bcs=[BoxBC(lower=DOMAIN_LOWER,
                   upper=np.array([DOMAIN_UPPER[0], DOMAIN_UPPER[1], 10.0]),
                   mode="slip")],
    )


def _solver_x(solver) -> np.ndarray:
    """Return particle positions as numpy from either backend."""
    return solver.particles_x() if hasattr(solver, "particles_x") else solver.particles.x


def _solver_v(solver) -> np.ndarray:
    return solver.particles_v() if hasattr(solver, "particles_v") else solver.particles.v


def _solver_m(solver) -> np.ndarray:
    if hasattr(solver, "m_p"):
        return solver.m_p.to_numpy()
    return solver.particles.m


# ── Ground-plane mesh + grid lines ────────────────────────────────────────────

def _ground_mesh(size: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    s = size / 2.0
    verts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]],
                     dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts + np.array([DOMAIN_UPPER[0]/2, DOMAIN_UPPER[1]/2, 0]), faces


def kinetic_energy(solver: MPMSolver) -> float:
    p = solver.particles
    return 0.5 * float((p.m * np.einsum("pa,pa->p", p.v, p.v)).sum())


def elastic_energy(solver: MPMSolver) -> float:
    p, mat = solver.particles, solver.material
    J  = np.linalg.det(p.F)
    IC = np.einsum("pab,pab->p", p.F, p.F)
    psi = 0.5*mat.mu*(IC-3.0) - mat.mu*np.log(J) + 0.5*mat.lam*np.log(J)**2
    return float((psi * p.V0).sum())


def run_headless(backend: str = "numpy") -> None:
    import time
    from robosim.util.fps import FPSCounter
    solver = build_solver(backend)
    n_p = (_solver_x(solver)).shape[0]
    print(f"[init] backend={backend}  {n_p} particles")
    fps = FPSCounter()
    t_wall = time.perf_counter()
    for step in range(N_STEPS + 1):
        if step < N_STEPS:
            solver.step(DT)
        fps.tick()
        if step % 60 == 0:
            x = _solver_x(solver); v = _solver_v(solver); m = _solver_m(solver)
            com = x.mean(axis=0)
            vcom_z = float((m * v[:, 2]).sum() / m.sum())
            print(f"t={step*DT:5.3f}s  com_z={com[2]:.3f}  v_com_z={vcom_z:+.3f}  "
                  f"{fps.format()}")
    elapsed = time.perf_counter() - t_wall
    print(f"\nWall time: {elapsed:.2f} s  ({N_STEPS*DT/elapsed:.2f}× real-time)")
    print(f"Frames    : {fps.n_frames}  (avg {fps.average:.1f} FPS, "
          f"last-window {fps.current:.1f} FPS)")


def run_gui(backend: str = "numpy") -> None:
    from robosim.viz.viewer import SimViewer
    from robosim.util.fps import FPSCounter

    solver = build_solver(backend)
    pts = solver.particles

    viewer = SimViewer(
        title="RoboSim — MPM Jello Drop",
        window_size=(1280, 800),
        background=(0.08, 0.08, 0.10),
    )
    viewer.initialize()

    gv, gf = _ground_mesh(size=0.8)
    viewer.add_mesh("ground", gv, gf, color=np.array([0.22, 0.22, 0.25]))
    viewer.add_particles(
        "mpm", pts.x, radius=0.006,
        per_vertex_color=_height_colors(pts.x[:, 2]),
    )

    step_count = [0]
    fps = FPSCounter()

    def step(frame: int) -> None:
        for _ in range(SUBSTEPS_PER_FRAME):
            try:
                solver.step(DT)
            except IndexError:
                return  # particle left the grid — freeze physics, keep drawing
            step_count[0] += 1
        viewer.update_particles("mpm", pts.x,
                                per_vertex_color=_height_colors(pts.x[:, 2]))
        fps.tick()
        com = pts.x.mean(axis=0)
        vcom_z = float((pts.m * pts.v[:, 2]).sum() / pts.m.sum())
        KE, PE = kinetic_energy(solver), elastic_energy(solver)
        t_sim = step_count[0] * DT
        viewer.add_text(
            f"t = {t_sim:6.3f} s   step {step_count[0]}\n"
            f"{fps.format()}\n"
            f"CoM z  : {com[2]:+.3f} m   v_com_z: {vcom_z:+.3f} m/s\n"
            f"KE = {KE:8.2e}  PE = {PE:8.2e}  total = {KE+PE:8.2e}\n"
            f"[LDrag=orbit  Scroll=zoom  ESC=quit]"
        )

    viewer.add_callback(step)
    viewer.show()


def _height_colors(z: np.ndarray) -> np.ndarray:
    t = np.clip((z - DOMAIN_LOWER[2]) / (DOMAIN_UPPER[2] - DOMAIN_LOWER[2]), 0, 1)
    return np.column_stack([0.3 + 0.5*t, 0.5 + 0.3*t, 1.0 - 0.4*t]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--backend", choices=["numpy", "taichi"], default="numpy",
                    help="numpy (default, single-thread CPU) or taichi "
                         "(arch=metal on M-series Macs, 20-40× faster).")
    args = ap.parse_args()
    if args.backend == "taichi" or not args.headless:
        import taichi as ti
        ti.init(arch=ti.metal)
    if args.headless:
        run_headless(backend=args.backend)
    else:
        run_gui(backend=args.backend)


if __name__ == "__main__":
    main()
