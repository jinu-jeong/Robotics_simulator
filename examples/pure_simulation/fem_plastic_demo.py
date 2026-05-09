"""FEM corotational J2 plasticity — elastic vs plastic side-by-side.

Two clamped cantilever bars, identical mesh and geometry. A downward
tip load bends both for the loading phase, then the load is released
and the bars relax under their own dynamics.

  - Elastic bar  (CorotationalElastic)   → recovers nearly fully.
  - Plastic bar  (CorotationalPlastic +  → keeps a permanent bend
    linear isotropic hardening)            ("permanent set").

Headless mode prints tip deflection per bar over time. GUI mode (opt-in
via ``--gui``) renders both bars in a Taichi GGUI window.

Usage:
  python examples/pure_simulation/fem_plastic_demo.py --headless
  python examples/pure_simulation/fem_plastic_demo.py --gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.physics.fem.materials import (                          # noqa: E402
    CorotationalElastic, CorotationalPlastic,
)
from robosim.physics.fem.mesh import TetMesh                         # noqa: E402
from robosim.physics.fem.solver import DeformableBody, FEMSolver     # noqa: E402


# ── Geometry ─────────────────────────────────────────────────────────────────
# Cantilever beam: tip deflection δ ≈ F·L³/(3·E·I), I = W·H³/12.
# Picked so δ_at_load ≈ 30 mm (visible bend, but still small-strain).
L, W, H = 0.40, 0.05, 0.05      # bar length × width × height (m)
DIVISIONS = (16, 3, 3)
DENSITY = 1000.0
YOUNG = 5e7
POISSON = 0.3
# Yield stress just below the bending stress at the load case so the
# plastic bar yields visibly while the elastic bar stays linear.
YIELD_STRESS = 2.5e5
HARDENING = 8e6                 # large enough to keep clear residual set

DT = 5e-4
DAMPING = 5.0                   # heavy mass-proportional damping → quasi-static
N_LOAD_STEPS = 400
N_RELEASE_STEPS = 800
TIP_FORCE = 35.0                # downward total force on tip face (N)


def _hex_to_5tet(nx, ny, nz, lx, ly, lz, x_offset=0.0) -> TetMesh:
    """Hexahedral grid split into 5 tets per hex — matches integration test."""
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


def _build_body(name: str, plastic: bool, x_offset: float) -> DeformableBody:
    mesh = _hex_to_5tet(*DIVISIONS, lx=L, ly=W, lz=H, x_offset=x_offset)
    fixed = np.where(mesh.nodes[:, 0] < x_offset + 1e-9)[0]
    if plastic:
        material = CorotationalPlastic(
            young=YOUNG, poisson=POISSON,
            yield_stress=YIELD_STRESS, hardening=HARDENING,
        )
    else:
        material = CorotationalElastic(young=YOUNG, poisson=POISSON)
    return DeformableBody(
        name=name, mesh=mesh, material=material,
        density=DENSITY, fixed_nodes=fixed,
    )


def _tip_force_vector(body: DeformableBody, total_force_z: float) -> np.ndarray:
    n_dof = body.mesh.n_nodes * 3
    f = np.zeros(n_dof)
    tip_x = body.mesh.nodes[:, 0].max()
    tip = np.where(body.mesh.nodes[:, 0] > tip_x - 1e-9)[0]
    per_node = total_force_z / len(tip)
    for n in tip:
        f[n * 3 + 2] = per_node
    return f


def _tip_z_displacement(body: DeformableBody) -> float:
    tip_x = body.mesh.nodes[:, 0].max()
    tip = np.where(body.mesh.nodes[:, 0] > tip_x - 1e-9)[0]
    return float(np.mean(body.x[tip, 2] - body.mesh.nodes[tip, 2]))


def run_headless() -> None:
    bar_e = _build_body("elastic", plastic=False, x_offset=0.0)
    bar_p = _build_body("plastic", plastic=True, x_offset=0.0)

    solver_e = FEMSolver(bodies=[bar_e], dt=DT,
                         gravity=np.zeros(3), damping=DAMPING,
                         max_newton_iters=15)
    solver_p = FEMSolver(bodies=[bar_p], dt=DT,
                         gravity=np.zeros(3), damping=DAMPING,
                         max_newton_iters=15)
    solver_e.initialize(DT)
    solver_p.initialize(DT)

    f_pull_e = _tip_force_vector(bar_e, -TIP_FORCE)
    f_pull_p = _tip_force_vector(bar_p, -TIP_FORCE)

    print(f"[init] mesh: {bar_e.mesh.n_nodes} nodes / {bar_e.mesh.n_elements} tets")
    print(f"[init] dt={DT}s  load_steps={N_LOAD_STEPS}  release_steps={N_RELEASE_STEPS}")
    print(f"[init] young={YOUNG:.1e} Pa  yield={YIELD_STRESS:.1e}  H={HARDENING:.1e}")
    print(f"{'phase':<10}{'t (s)':>8}{'tip_z elastic (mm)':>22}{'tip_z plastic (mm)':>22}")

    def _log(phase, step):
        de = _tip_z_displacement(bar_e) * 1000
        dp = _tip_z_displacement(bar_p) * 1000
        print(f"{phase:<10}{step*DT:>8.3f}{de:>22.3f}{dp:>22.3f}")

    # Loading phase
    for k in range(N_LOAD_STEPS):
        solver_e.step(DT, extra_forces={0: f_pull_e})
        solver_p.step(DT, extra_forces={0: f_pull_p})
        if k % 100 == 0:
            _log("load", k)
    _log("load_end", N_LOAD_STEPS)

    # Release phase
    for k in range(N_RELEASE_STEPS):
        solver_e.step(DT)
        solver_p.step(DT)
        if k % 200 == 0:
            _log("release", N_LOAD_STEPS + k)
    _log("settle", N_LOAD_STEPS + N_RELEASE_STEPS)

    # Summary
    final_e = _tip_z_displacement(bar_e) * 1000
    final_p = _tip_z_displacement(bar_p) * 1000
    print()
    print(f"Final tip-z displacement after release:")
    print(f"  elastic : {final_e:+.3f} mm  (recovery ratio {1 - abs(final_e/final_p) if abs(final_p) > 1e-6 else 1.0:.2f})")
    print(f"  plastic : {final_p:+.3f} mm  (permanent set)")
    if bar_p.eps_p is not None:
        eps_p_max = float(np.linalg.norm(bar_p.eps_p, axis=(1, 2)).max())
        eps_p_mean = float(np.linalg.norm(bar_p.eps_p, axis=(1, 2)).mean())
        print(f"  plastic eps_p (Frobenius): max={eps_p_max:.4f}  mean={eps_p_mean:.4f}")


def run_gui() -> None:
    """Minimal Taichi GGUI viewer: both bars rendered side-by-side."""
    import taichi as ti
    ti.init(arch=ti.metal)

    # Place plastic bar offset along Y so the two are side-by-side.
    bar_e = _build_body("elastic", plastic=False, x_offset=0.0)
    bar_p = _build_body("plastic", plastic=True, x_offset=0.0)
    # Shift plastic bar in Y for visualisation only (not in physics).
    y_shift = W * 1.6

    solver_e = FEMSolver(bodies=[bar_e], dt=DT, gravity=np.zeros(3),
                         damping=DAMPING, max_newton_iters=15)
    solver_p = FEMSolver(bodies=[bar_p], dt=DT, gravity=np.zeros(3),
                         damping=DAMPING, max_newton_iters=15)
    solver_e.initialize(DT)
    solver_p.initialize(DT)

    f_pull_e = _tip_force_vector(bar_e, -TIP_FORCE)
    f_pull_p = _tip_force_vector(bar_p, -TIP_FORCE)

    surf_e = bar_e.mesh.extract_surface()
    surf_p = bar_p.mesh.extract_surface()

    def _to_field(verts, faces, color, y_off=0.0):
        n_v = verts.shape[0]
        n_f = faces.shape[0]
        v_field = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        c_field = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
        i_field = ti.field(dtype=ti.i32, shape=n_f * 3)
        v0 = verts.astype(np.float32).copy()
        v0[:, 1] += y_off
        v_field.from_numpy(v0)
        c_field.from_numpy(np.tile(color, (n_v, 1)).astype(np.float32))
        i_field.from_numpy(faces.ravel().astype(np.int32))
        return v_field, c_field, i_field, n_v, n_f, y_off

    ve, ce, ie_, _, nfe, _ = _to_field(bar_e.x, surf_e, [0.30, 0.55, 0.95])
    vp, cp, ip_, _, nfp, _ = _to_field(bar_p.x, surf_p, [0.95, 0.45, 0.30], y_shift)

    window = ti.ui.Window("FEM plastic vs elastic", (1280, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.08, 0.08, 0.10))
    scene = window.get_scene()
    cam = ti.ui.Camera()
    cam.position(0.5, -0.5, 0.4)
    cam.lookat(0.20, y_shift / 2, 0.0)
    cam.up(0, 0, 1)

    sub = 4
    step_count = [0]

    while window.running:
        for _ in range(sub):
            k = step_count[0]
            ext_e = {0: f_pull_e} if k < N_LOAD_STEPS else None
            ext_p = {0: f_pull_p} if k < N_LOAD_STEPS else None
            solver_e.step(DT, extra_forces=ext_e)
            solver_p.step(DT, extra_forces=ext_p)
            step_count[0] += 1

        ve.from_numpy(bar_e.x.astype(np.float32))
        vp_arr = bar_p.x.astype(np.float32).copy()
        vp_arr[:, 1] += y_shift
        vp.from_numpy(vp_arr)

        cam.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
        scene.set_camera(cam)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(1.0, 1.0, 1.0), color=(0.9, 0.9, 0.9))
        scene.mesh(ve, indices=ie_, per_vertex_color=ce, two_sided=True)
        scene.mesh(vp, indices=ip_, per_vertex_color=cp, two_sided=True)
        canvas.scene(scene)

        de = _tip_z_displacement(bar_e) * 1000
        dp = _tip_z_displacement(bar_p) * 1000
        phase = "LOADING" if step_count[0] < N_LOAD_STEPS else "RELEASE"
        window.GUI.begin("status", 0.02, 0.02, 0.32, 0.18)
        window.GUI.text(f"phase: {phase}")
        window.GUI.text(f"step: {step_count[0]}  t={step_count[0]*DT:.3f} s")
        window.GUI.text(f"elastic tip_z: {de:+.2f} mm")
        window.GUI.text(f"plastic tip_z: {dp:+.2f} mm")
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
