"""CB-reduced elastic vs CB-hybrid plastic — side-by-side cantilever.

Mirror of ``fem_plastic_demo.py`` (clamped bar, tip pull, release,
settle) but both bars run through the **Craig-Bampton reduced-order
basis** instead of full FEM:

  • Left  bar — :class:`CraigBamptonBody` with ``CorotationalElastic``.
                Reduced step every substep; bar fully recovers when
                released (no plastic mechanism in this body).
  • Right bar — :class:`HybridCBPlasticBody` with linear isotropic
                hardening. CB step while every region is ELASTIC; falls
                to full FEM J2 only on regions whose σ_eq crosses σ_Y;
                rebuilds the basis at the deformed shape once plastic
                flow has settled. Keeps a clear permanent set after
                release — same physics signature as the FEM-side
                ``fem_plastic_demo.py``, but driven through the
                CB / hybrid pipeline.

GUI mode colours the hybrid bar's surface elements by region state:

  blue   — ELASTIC          (CB fast path)
  red    — PLASTIC_ACTIVE   (full FEM J2 return mapping running here)
  yellow — REBUILD_PENDING  (no plastic flow lately, queued for rebuild)

Usage:
  python examples/pure_simulation/cb_plastic_demo.py --headless
  python examples/pure_simulation/cb_plastic_demo.py --gui
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.physics.fem.materials import CorotationalElastic         # noqa: E402
from robosim.physics.fem.mesh import TetMesh                          # noqa: E402
from robosim.physics.fem.reduced import (                             # noqa: E402
    CraigBamptonBody, CraigBamptonSolver,
)
from robosim.physics.fem.hybrid import (                              # noqa: E402
    HybridCBPlasticBody, RegionState,
)


# ── Geometry & material (matches fem_plastic_demo apart from mesh size) ─────
L, W, H = 0.40, 0.05, 0.05
DIVISIONS = (16, 3, 3)         # same mesh resolution as fem_plastic_demo
DENSITY = 1000.0
YOUNG = 5e7
POISSON = 0.3
YIELD_STRESS = 2.5e5
HARDENING = 8e6

DT = 5e-4
DAMPING = 5.0
N_LOAD_STEPS = 400
N_RELEASE_STEPS = 800
TIP_FORCE = 35.0

# Hybrid-only knobs.
N_REGIONS = 4
N_MODES = 10
REBUILD_AFTER_STEADY_STEPS = 30


# ── Mesh helper (same 5-tet decomposition as fem_plastic_demo) ──────────────

def _hex_to_5tet(nx, ny, nz, lx, ly, lz, x_offset=0.0) -> TetMesh:
    xs = np.linspace(0.0, lx, nx + 1) + x_offset
    ys = np.linspace(0.0, ly, ny + 1)
    zs = np.linspace(0.0, lz, nz + 1)
    nodes = np.array([[x, y, z] for x in xs for y in ys for z in zs],
                     dtype=np.float64)

    def nid(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = [
                    nid(i,     j,     k),     nid(i + 1, j,     k),
                    nid(i + 1, j + 1, k),     nid(i,     j + 1, k),
                    nid(i,     j,     k + 1), nid(i + 1, j,     k + 1),
                    nid(i + 1, j + 1, k + 1), nid(i,     j + 1, k + 1),
                ]
                if (i + j + k) % 2 == 0:
                    tets += [
                        [v[0], v[1], v[3], v[4]],
                        [v[1], v[2], v[3], v[6]],
                        [v[1], v[5], v[4], v[6]],
                        [v[3], v[4], v[7], v[6]],
                        [v[1], v[3], v[4], v[6]],
                    ]
                else:
                    tets += [
                        [v[0], v[1], v[2], v[5]],
                        [v[0], v[2], v[3], v[7]],
                        [v[0], v[2], v[7], v[5]],
                        [v[0], v[5], v[7], v[4]],
                        [v[2], v[5], v[6], v[7]],
                    ]
    return TetMesh(nodes=nodes, elements=np.asarray(tets, dtype=np.int64))


# ── Body builders ───────────────────────────────────────────────────────────

def _make_elastic_cb_bar() -> tuple[CraigBamptonBody, CraigBamptonSolver]:
    mesh = _hex_to_5tet(*DIVISIONS, lx=L, ly=W, lz=H)
    fixed = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    body = CraigBamptonBody(
        mesh=mesh,
        material=CorotationalElastic(young=YOUNG, poisson=POISSON),
        density=DENSITY,
        n_modes=N_MODES,
        fixed_nodes=fixed,
        gravity=np.zeros(3),
        damping=DAMPING,
        name="cb_elastic",
    )
    solver = CraigBamptonSolver(bodies=[body])
    solver.initialize(dt=DT)
    return body, solver


def _make_hybrid_plastic_bar() -> HybridCBPlasticBody:
    mesh = _hex_to_5tet(*DIVISIONS, lx=L, ly=W, lz=H)
    fixed = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    body = HybridCBPlasticBody(
        mesh=mesh,
        material=CorotationalElastic(young=YOUNG, poisson=POISSON),
        density=DENSITY, fixed_nodes=fixed,
        gravity=np.zeros(3), damping=DAMPING,
        yield_stress=YIELD_STRESS, hardening=HARDENING,
        n_modes=N_MODES, n_regions=N_REGIONS,
        rebuild_after_steady_steps=REBUILD_AFTER_STEADY_STEPS,
        name="cb_hybrid_plastic",
    )
    body.initialize(dt=DT)
    return body


# ── Common diagnostics ──────────────────────────────────────────────────────

def _tip_force_vector(mesh: TetMesh, total_force_z: float) -> np.ndarray:
    n_dof = mesh.n_nodes * 3
    f = np.zeros(n_dof)
    tip = np.where(mesh.nodes[:, 0] > L - 1e-9)[0]
    per_node = total_force_z / len(tip)
    for n in tip:
        f[n * 3 + 2] = per_node
    return f


def _tip_z_displacement(x: np.ndarray, ref_nodes: np.ndarray) -> float:
    tip = np.where(ref_nodes[:, 0] > L - 1e-9)[0]
    return float(np.mean(x[tip, 2] - ref_nodes[tip, 2]))


# ── Headless ────────────────────────────────────────────────────────────────

def run_headless() -> None:
    bar_e, solver_e = _make_elastic_cb_bar()
    bar_p           = _make_hybrid_plastic_bar()

    # The hybrid body mutates its own mesh.nodes at each rebuild, so
    # snapshot the original geometry now for tip-deflection reporting.
    ref_e = bar_e.mesh.nodes.copy()
    ref_p = bar_p.mesh.nodes.copy()

    f_pull_e = _tip_force_vector(bar_e.mesh, -TIP_FORCE)
    f_pull_p = _tip_force_vector(bar_p.mesh, -TIP_FORCE)

    print(f"[init] mesh: {bar_e.mesh.n_nodes} nodes / {bar_e.mesh.n_elements} tets")
    print(f"[init] dt={DT}s  load={N_LOAD_STEPS} steps  release={N_RELEASE_STEPS} steps")
    print(f"[init] young={YOUNG:.1e}  yield={YIELD_STRESS:.1e}  H={HARDENING:.1e}")
    print(f"[init] hybrid: K={N_REGIONS} regions, {N_MODES} CB modes")
    print()
    print(f"{'phase':<10}{'t (s)':>8}"
          f"{'tip_z elastic (mm)':>22}{'tip_z plastic (mm)':>22}"
          f"{'states (HYB)':>16}{'rebuilds':>10}")

    def _log(phase: str, step: int) -> None:
        de = _tip_z_displacement(bar_e.x, ref_e) * 1000
        dp = _tip_z_displacement(bar_p.x, ref_p) * 1000
        states = ''.join(s.value[0].upper() for s in bar_p.region_state)
        print(f"{phase:<10}{step*DT:>8.3f}{de:>22.3f}{dp:>22.3f}"
              f"{states:>16s}{bar_p._rebuild_count:>10d}")

    # Loading phase
    for k in range(N_LOAD_STEPS):
        solver_e.step(dt=DT, extra_forces={0: f_pull_e})
        bar_p.step(DT, extra_forces={0: f_pull_p})
        if k % 100 == 0:
            _log("load", k)
    _log("load_end", N_LOAD_STEPS)

    # Release phase
    for k in range(N_RELEASE_STEPS):
        solver_e.step(dt=DT)
        bar_p.step(DT)
        if k % 200 == 0:
            _log("release", N_LOAD_STEPS + k)
    _log("settle", N_LOAD_STEPS + N_RELEASE_STEPS)

    # Summary
    final_e = _tip_z_displacement(bar_e.x, ref_e) * 1000
    final_p = _tip_z_displacement(bar_p.x, ref_p) * 1000
    peak_p = float(np.linalg.norm(bar_p.eps_p, axis=(1, 2)).max())
    print()
    print(f"Final tip-z displacement after release:")
    print(f"  elastic CB     : {final_e:+8.3f} mm")
    print(f"  hybrid plastic : {final_p:+8.3f} mm  (permanent set; "
          f"rebuilds={bar_p._rebuild_count}, ‖eps_p‖_peak={peak_p:.4f})")


# ── GUI ─────────────────────────────────────────────────────────────────────

def _state_color(state: RegionState) -> tuple[float, float, float]:
    if state == RegionState.ELASTIC:         return (0.30, 0.55, 0.95)   # cool blue
    if state == RegionState.PLASTIC_ACTIVE:  return (0.95, 0.30, 0.20)   # hot red
    return (0.95, 0.85, 0.20)                                            # yellow


def run_gui() -> None:
    import taichi as ti
    ti.init(arch=ti.metal)

    bar_e, solver_e = _make_elastic_cb_bar()
    bar_p           = _make_hybrid_plastic_bar()
    ref_e = bar_e.mesh.nodes.copy()
    ref_p = bar_p.mesh.nodes.copy()

    f_pull_e = _tip_force_vector(bar_e.mesh, -TIP_FORCE)
    f_pull_p = _tip_force_vector(bar_p.mesh, -TIP_FORCE)

    surf_e = bar_e.mesh.extract_surface()
    surf_p = bar_p.mesh.extract_surface()
    y_shift = W * 1.6

    n_v_e = bar_e.mesh.n_nodes
    n_v_p = bar_p.mesh.n_nodes
    v_e = ti.Vector.field(3, dtype=ti.f32, shape=n_v_e)
    c_e = ti.Vector.field(3, dtype=ti.f32, shape=n_v_e)
    i_e = ti.field(dtype=ti.i32, shape=surf_e.shape[0] * 3)
    v_p = ti.Vector.field(3, dtype=ti.f32, shape=n_v_p)
    c_p = ti.Vector.field(3, dtype=ti.f32, shape=n_v_p)
    i_p = ti.field(dtype=ti.i32, shape=surf_p.shape[0] * 3)
    i_e.from_numpy(surf_e.ravel().astype(np.int32))
    i_p.from_numpy(surf_p.ravel().astype(np.int32))
    # Elastic bar: solid cool blue.
    c_e.from_numpy(np.tile([0.30, 0.55, 0.95],
                           (n_v_e, 1)).astype(np.float32))

    def _hybrid_node_colors() -> np.ndarray:
        """Per-node colour from the region containing each element it
        belongs to. Cheap, good-enough visual."""
        n_nodes = bar_p.mesh.n_nodes
        cols = np.tile([0.30, 0.55, 0.95], (n_nodes, 1)).astype(np.float32)
        for r, elem_idx in enumerate(bar_p.partition.region_elements):
            col = _state_color(bar_p.region_state[r])
            for e in elem_idx:
                for n in bar_p.mesh.elements[e]:
                    cols[n] = col
        return cols

    window = ti.ui.Window("CB elastic vs CB-hybrid plastic",
                          (1400, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.08, 0.08, 0.10))
    scene = window.get_scene()
    cam = ti.ui.Camera()
    cam.position(0.5, -0.6, 0.5)
    cam.lookat(0.20, y_shift / 2, 0.0)
    cam.up(0, 0, 1)

    sub = 4
    step_count = [0]
    last_wall = [time.perf_counter()]
    fps_window_steps = [0]
    fps_value = [0.0]

    while window.running:
        for _ in range(sub):
            k = step_count[0]
            ext_e = {0: f_pull_e} if k < N_LOAD_STEPS else None
            ext_p = {0: f_pull_p} if k < N_LOAD_STEPS else None
            solver_e.step(dt=DT, extra_forces=ext_e)
            bar_p.step(DT, extra_forces=ext_p)
            step_count[0] += 1
            fps_window_steps[0] += 1

        v_e.from_numpy(bar_e.x.astype(np.float32))
        v_p_arr = bar_p.x.astype(np.float32).copy()
        v_p_arr[:, 1] += y_shift
        v_p.from_numpy(v_p_arr)
        c_p.from_numpy(_hybrid_node_colors())

        now = time.perf_counter()
        if now - last_wall[0] > 0.25:
            fps_value[0] = fps_window_steps[0] / (now - last_wall[0])
            last_wall[0] = now
            fps_window_steps[0] = 0

        cam.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
        scene.set_camera(cam)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(1.0, 1.0, 1.0), color=(0.9, 0.9, 0.9))
        scene.mesh(v_e, indices=i_e, per_vertex_color=c_e, two_sided=True)
        scene.mesh(v_p, indices=i_p, per_vertex_color=c_p, two_sided=True)
        canvas.scene(scene)

        de = _tip_z_displacement(bar_e.x, ref_e) * 1000
        dp = _tip_z_displacement(bar_p.x, ref_p) * 1000
        phase = "LOADING" if step_count[0] < N_LOAD_STEPS else "RELEASE"
        n_active = sum(1 for s in bar_p.region_state
                       if s == RegionState.PLASTIC_ACTIVE)
        n_pending = sum(1 for s in bar_p.region_state
                        if s == RegionState.REBUILD_PENDING)
        window.GUI.begin("status", 0.02, 0.02, 0.34, 0.28)
        window.GUI.text(f"phase    : {phase}")
        window.GUI.text(f"step     : {step_count[0]}  t = {step_count[0]*DT:.3f} s")
        window.GUI.text(f"sim FPS  : {fps_value[0]:.1f}")
        window.GUI.text("")
        window.GUI.text(f"elastic CB tip_z : {de:+.2f} mm")
        window.GUI.text(f"hybrid     tip_z : {dp:+.2f} mm")
        window.GUI.text("")
        window.GUI.text(f"hybrid regions : K={N_REGIONS}")
        window.GUI.text(f"  active         : {n_active}")
        window.GUI.text(f"  rebuild pending: {n_pending}")
        window.GUI.text(f"  rebuilds done  : {bar_p._rebuild_count}")
        window.GUI.end()
        window.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--gui", action="store_true",
                    help="Open Taichi GGUI window (default if not --headless).")
    args = ap.parse_args()
    if args.headless:
        run_headless()
    else:
        run_gui()


if __name__ == "__main__":
    main()
