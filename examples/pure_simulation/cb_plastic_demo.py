"""Hybrid CB-plastic vs full-FEM plastic — side-by-side cantilever.

Same loading scenario as ``fem_plastic_demo.py`` (clamped bar, tip pull,
release, settle), but two backends running in parallel:

  • Left  bar — full-FEM ``CorotationalPlastic`` (every step is full
                 implicit Euler over the whole mesh).
  • Right bar — :class:`HybridCBPlasticBody`. METIS K-way partition;
                 ELASTIC regions step through Craig-Bampton (cheap),
                 PLASTIC_ACTIVE regions fall to full FEM only where
                 yielding occurs, REBUILD_PENDING regions wait for
                 settle. When all regions reach REBUILD_PENDING the
                 deformed shape is absorbed into the reference and the
                 body returns to all-CB at the new rest config.

GUI mode colours the hybrid bar's surface elements by state:

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

from robosim.physics.fem.materials import (                          # noqa: E402
    CorotationalElastic, CorotationalPlastic,
)
from robosim.physics.fem.mesh import TetMesh                         # noqa: E402
from robosim.physics.fem.solver import DeformableBody, FEMSolver     # noqa: E402
from robosim.physics.fem.hybrid import (                             # noqa: E402
    HybridCBPlasticBody, RegionState,
)


# ── Geometry & material (sized so CB reduction is meaningful) ────────────────
# Bar mesh chosen large enough that CB's ~10 reduced DOFs really cut work
# vs full-FEM (~ 1.5k DOFs). Fewer divisions ⇒ CB looks the same speed
# as FEM and the demo tells you nothing.
L, W, H = 0.40, 0.05, 0.05
DIVISIONS = (24, 4, 4)        # ~1920 tets, 625 nodes, 1875 dofs
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
N_REGIONS = 6                       # METIS k-way along the bar's long axis
N_MODES = 12
REBUILD_AFTER_STEADY_STEPS = 30


# ── Mesh helper (same as fem_plastic_demo) ───────────────────────────────────

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


# ── Body builders ────────────────────────────────────────────────────────────

def _make_full_fem_bar() -> tuple[DeformableBody, FEMSolver]:
    mesh = _hex_to_5tet(*DIVISIONS, lx=L, ly=W, lz=H)
    fixed = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    body = DeformableBody(
        name="fem_plastic", mesh=mesh,
        material=CorotationalPlastic(
            young=YOUNG, poisson=POISSON,
            yield_stress=YIELD_STRESS, hardening=HARDENING,
        ),
        density=DENSITY, fixed_nodes=fixed,
    )
    solver = FEMSolver(bodies=[body], dt=DT, gravity=np.zeros(3),
                       damping=DAMPING, max_newton_iters=10)
    solver.initialize(DT)
    return body, solver


def _make_hybrid_bar() -> HybridCBPlasticBody:
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
        name="hybrid_plastic",
    )
    body.initialize(dt=DT)
    return body


# ── Common diagnostics ───────────────────────────────────────────────────────

def _tip_force_vector(mesh: TetMesh, total_force_z: float) -> np.ndarray:
    n_dof = mesh.n_nodes * 3
    f = np.zeros(n_dof)
    tip_x = mesh.nodes[:, 0].max()
    tip = np.where(mesh.nodes[:, 0] > tip_x - 1e-9)[0]
    per_node = total_force_z / len(tip)
    for n in tip:
        f[n * 3 + 2] = per_node
    return f


def _tip_z_displacement(x: np.ndarray, ref_nodes: np.ndarray) -> float:
    tip_x = ref_nodes[:, 0].max()
    tip = np.where(ref_nodes[:, 0] > tip_x - 1e-9)[0]
    return float(np.mean(x[tip, 2] - ref_nodes[tip, 2]))


# ── Headless ─────────────────────────────────────────────────────────────────

def run_headless() -> None:
    bar_fem, solver_fem = _make_full_fem_bar()
    bar_hyb = _make_hybrid_bar()

    # Reference geometries for tip deflection (hybrid mutates mesh.nodes
    # at every rebuild, so capture the original now).
    ref_fem = bar_fem.mesh.nodes.copy()
    ref_hyb = bar_hyb.mesh.nodes.copy()

    f_pull_fem = _tip_force_vector(bar_fem.mesh, -TIP_FORCE)
    f_pull_hyb = _tip_force_vector(bar_hyb.mesh, -TIP_FORCE)

    print(f"[init] mesh        : {bar_fem.mesh.n_nodes} nodes / {bar_fem.mesh.n_elements} tets / {bar_fem.mesh.n_nodes*3} dofs")
    print(f"[init] hybrid n_r  : {bar_hyb._cb._n_r} reduced dofs over {N_REGIONS} regions, {N_MODES} modes")
    print(f"[init] dt={DT}s    load={N_LOAD_STEPS} steps  release={N_RELEASE_STEPS} steps")
    print(f"[init] young={YOUNG:.1e}  yield={YIELD_STRESS:.1e}  H={HARDENING:.1e}")
    print()
    print(f"{'phase':<10}{'t (s)':>8}{'tip_z FEM (mm)':>18}{'tip_z HYB (mm)':>18}"
          f"{'rebuilds':>11}{'states (HYB)':>26}")

    def _log(phase: str, step: int) -> None:
        d_fem = _tip_z_displacement(bar_fem.x, ref_fem) * 1000
        d_hyb = _tip_z_displacement(bar_hyb.x, ref_hyb) * 1000
        states = ''.join(s.value[0].upper() for s in bar_hyb.region_state)  # E/P/R
        print(f"{phase:<10}{step*DT:>8.3f}{d_fem:>18.3f}{d_hyb:>18.3f}"
              f"{bar_hyb._rebuild_count:>11d}{states:>26s}")

    # ── Loading ──
    t_fem = t_hyb = 0.0
    for k in range(N_LOAD_STEPS):
        t0 = time.perf_counter()
        solver_fem.step(DT, extra_forces={0: f_pull_fem})
        t_fem += time.perf_counter() - t0
        t0 = time.perf_counter()
        bar_hyb.step(DT, extra_forces={0: f_pull_hyb})
        t_hyb += time.perf_counter() - t0
        if k % 100 == 0:
            _log("load", k)
    _log("load_end", N_LOAD_STEPS)

    # ── Release ──
    for k in range(N_RELEASE_STEPS):
        t0 = time.perf_counter()
        solver_fem.step(DT)
        t_fem += time.perf_counter() - t0
        t0 = time.perf_counter()
        bar_hyb.step(DT)
        t_hyb += time.perf_counter() - t0
        if k % 200 == 0:
            _log("release", N_LOAD_STEPS + k)
    _log("settle", N_LOAD_STEPS + N_RELEASE_STEPS)

    # ── Summary ──
    n_steps_total = N_LOAD_STEPS + N_RELEASE_STEPS
    print()
    print(f"Wall time per step:")
    print(f"  FEM    : {t_fem*1000/n_steps_total:7.3f} ms/step  (total {t_fem:.2f} s)")
    print(f"  HYBRID : {t_hyb*1000/n_steps_total:7.3f} ms/step  (total {t_hyb:.2f} s)")
    if t_hyb > 0:
        print(f"  speedup: {t_fem/t_hyb:.2f}× (HYBRID vs FEM)")
    print()
    print(f"Final tip-z displacement (after release):")
    print(f"  FEM    : {_tip_z_displacement(bar_fem.x, ref_fem)*1000:+.3f} mm")
    print(f"  HYBRID : {_tip_z_displacement(bar_hyb.x, ref_hyb)*1000:+.3f} mm  "
          f"(rebuilds={bar_hyb._rebuild_count})")
    eps_p_max_fem = float(np.linalg.norm(bar_fem.eps_p, axis=(1, 2)).max())
    eps_p_max_hyb = float(np.linalg.norm(bar_hyb.eps_p, axis=(1, 2)).max())
    print(f"  ‖eps_p‖ peak: FEM={eps_p_max_fem:.4f}  HYBRID={eps_p_max_hyb:.4f}")


# ── GUI ──────────────────────────────────────────────────────────────────────

def _state_color(state: RegionState) -> tuple[float, float, float]:
    if state == RegionState.ELASTIC:         return (0.30, 0.55, 0.95)   # cool blue
    if state == RegionState.PLASTIC_ACTIVE:  return (0.95, 0.30, 0.20)   # hot red
    return (0.95, 0.85, 0.20)                                            # yellow


def run_gui() -> None:
    import taichi as ti
    ti.init(arch=ti.metal)

    bar_fem, solver_fem = _make_full_fem_bar()
    bar_hyb = _make_hybrid_bar()
    ref_fem = bar_fem.mesh.nodes.copy()
    ref_hyb = bar_hyb.mesh.nodes.copy()

    f_pull_fem = _tip_force_vector(bar_fem.mesh, -TIP_FORCE)
    f_pull_hyb = _tip_force_vector(bar_hyb.mesh, -TIP_FORCE)

    surf_fem = bar_fem.mesh.extract_surface()
    surf_hyb = bar_hyb.mesh.extract_surface()
    y_shift = W * 1.6        # plastic bar offset for visualisation only

    n_v_fem = bar_fem.mesh.n_nodes
    n_v_hyb = bar_hyb.mesh.n_nodes
    v_fem = ti.Vector.field(3, dtype=ti.f32, shape=n_v_fem)
    c_fem = ti.Vector.field(3, dtype=ti.f32, shape=n_v_fem)
    i_fem = ti.field(dtype=ti.i32, shape=surf_fem.shape[0] * 3)
    v_hyb = ti.Vector.field(3, dtype=ti.f32, shape=n_v_hyb)
    c_hyb = ti.Vector.field(3, dtype=ti.f32, shape=n_v_hyb)
    i_hyb = ti.field(dtype=ti.i32, shape=surf_hyb.shape[0] * 3)

    i_fem.from_numpy(surf_fem.ravel().astype(np.int32))
    i_hyb.from_numpy(surf_hyb.ravel().astype(np.int32))
    c_fem.from_numpy(np.tile([0.30, 0.55, 0.95],
                             (n_v_fem, 1)).astype(np.float32))

    def _hybrid_node_colors() -> np.ndarray:
        """Per-node colour from the region containing the most elements
        touching that node; ties broken by lowest region index. Cheap,
        good-enough visual."""
        n_nodes = bar_hyb.mesh.n_nodes
        # per-node colour: start cool blue, paint per-element colours
        # over each surface node so PLASTIC_ACTIVE visibly bleeds into
        # neighbour nodes.
        cols = np.tile([0.30, 0.55, 0.95], (n_nodes, 1)).astype(np.float32)
        # Per-element state from element_region.
        for r, elem_idx in enumerate(bar_hyb.partition.region_elements):
            col = _state_color(bar_hyb.region_state[r])
            for e in elem_idx:
                for n in bar_hyb.mesh.elements[e]:
                    cols[n] = col
        return cols

    window = ti.ui.Window("CB-hybrid plastic vs full-FEM plastic",
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
            ext_fem = {0: f_pull_fem} if k < N_LOAD_STEPS else None
            ext_hyb = {0: f_pull_hyb} if k < N_LOAD_STEPS else None
            solver_fem.step(DT, extra_forces=ext_fem)
            bar_hyb.step(DT, extra_forces=ext_hyb)
            step_count[0] += 1
            fps_window_steps[0] += 1

        v_fem.from_numpy(bar_fem.x.astype(np.float32))
        # Hybrid bar: shift Y for visualisation, recolour by region state.
        v_hyb_arr = bar_hyb.x.astype(np.float32).copy()
        v_hyb_arr[:, 1] += y_shift
        v_hyb.from_numpy(v_hyb_arr)
        c_hyb.from_numpy(_hybrid_node_colors())

        # FPS over a 0.25 s sliding window.
        now = time.perf_counter()
        if now - last_wall[0] > 0.25:
            fps_value[0] = fps_window_steps[0] / (now - last_wall[0])
            last_wall[0] = now
            fps_window_steps[0] = 0

        cam.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
        scene.set_camera(cam)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(1.0, 1.0, 1.0), color=(0.9, 0.9, 0.9))
        scene.mesh(v_fem, indices=i_fem, per_vertex_color=c_fem, two_sided=True)
        scene.mesh(v_hyb, indices=i_hyb, per_vertex_color=c_hyb, two_sided=True)
        canvas.scene(scene)

        d_fem = _tip_z_displacement(bar_fem.x, ref_fem) * 1000
        d_hyb = _tip_z_displacement(bar_hyb.x, ref_hyb) * 1000
        phase = "LOADING" if step_count[0] < N_LOAD_STEPS else "RELEASE"
        n_active = sum(1 for s in bar_hyb.region_state
                       if s == RegionState.PLASTIC_ACTIVE)
        n_pending = sum(1 for s in bar_hyb.region_state
                        if s == RegionState.REBUILD_PENDING)
        window.GUI.begin("status", 0.02, 0.02, 0.34, 0.28)
        window.GUI.text(f"phase    : {phase}")
        window.GUI.text(f"step     : {step_count[0]}  t = {step_count[0]*DT:.3f} s")
        window.GUI.text(f"sim FPS  : {fps_value[0]:.1f}")
        window.GUI.text("")
        window.GUI.text(f"FEM    tip_z : {d_fem:+.2f} mm")
        window.GUI.text(f"HYBRID tip_z : {d_hyb:+.2f} mm")
        window.GUI.text("")
        window.GUI.text(f"hybrid regions : K={N_REGIONS}")
        window.GUI.text(f"  active         : {n_active}")
        window.GUI.text(f"  rebuild pending: {n_pending}")
        window.GUI.text(f"  rebuilds done  : {bar_hyb._rebuild_count}")
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
