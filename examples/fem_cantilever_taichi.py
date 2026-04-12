"""Phase 3 검증 데모: FEM 외팔보 시뮬레이션 — Taichi GGUI 뷰어.

Metal/Vulkan 네이티브 렌더링으로 PyVista 대비 훨씬 빠릅니다.

실행:
  python examples/fem_cantilever_taichi.py
  python examples/fem_cantilever_taichi.py --no-stress
  python examples/fem_cantilever_taichi.py --substeps 10

조작:
  - 마우스 좌클릭 드래그: 카메라 회전
  - 스크롤: 줌 인/아웃
  - ESC: 종료
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import taichi as ti

from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.solver import FEMSolver, DeformableBody
from robosim.physics.fem.assembly import batch_von_mises


def stress_to_color(vm: np.ndarray, vmin: float = 0.0, vmax: float = 500.0) -> np.ndarray:
    """Map von Mises stress to jet-like RGB colors (n_nodes, 3)."""
    t = np.clip((vm - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    # Simplified jet colormap: blue -> cyan -> green -> yellow -> red
    r = np.clip(1.5 - np.abs(t - 0.75) * 4.0, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(t - 0.50) * 4.0, 0.0, 1.0)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4.0, 0.0, 1.0)
    return np.column_stack([r, g, b]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--substeps", type=int, default=5, help="프레임당 물리 서브스텝 (기본: 5)")
    args = parser.parse_args()
    substeps = args.substeps

    ti.init(arch=ti.metal)

    print("=" * 60)
    print("  RoboSim Phase 3: FEM Cantilever — Taichi GGUI")
    print("=" * 60)

    # ── Mesh & solver setup ──
    L, W, H = 1.0, 0.1, 0.1
    mesh = TetMesh.create_box(
        origin=np.zeros(3),
        size=np.array([L, W, H]),
        divisions=(10, 2, 2),
    )
    fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-10)[0]

    body = DeformableBody(
        name="cantilever",
        mesh=mesh,
        material=CorotationalElastic(young=1e7, poisson=0.3),
        density=1000.0,
        fixed_nodes=fixed_nodes,
    )
    solver = FEMSolver(
        bodies=[body],
        gravity=np.array([0.0, 0.0, -9.81]),
        damping=0.5,
    )
    solver.initialize(dt=0.005)

    print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    print(f"Fixed nodes: {len(fixed_nodes)}")
    print(f"Sub-steps per frame: {substeps}")

    # Analytical reference
    I_moment = W * H**3 / 12.0
    A = W * H
    w = body.density * 9.81 * A
    delta_analytical = -w * L**4 / (8 * body.material.young * I_moment)
    print(f"Analytical tip deflection: {delta_analytical * 1000:.2f} mm")

    # ── Surface mesh for rendering ──
    surface_tri = mesh.extract_surface()  # (n_faces, 3) node indices
    n_nodes = mesh.n_nodes
    n_tri = surface_tri.shape[0]

    # Taichi fields
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_nodes)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=n_nodes)
    indices = ti.field(dtype=ti.i32, shape=n_tri * 3)

    # Upload indices (static)
    indices.from_numpy(surface_tri.ravel().astype(np.int32))

    # Initial vertex upload
    vertices.from_numpy(body.x.astype(np.float32))

    # Default color (steel blue)
    default_color = np.full((n_nodes, 3), [0.35, 0.55, 0.75], dtype=np.float32)
    colors.from_numpy(default_color)

    # Fixed node markers
    fixed_pos_np = mesh.nodes[fixed_nodes].astype(np.float32)
    fixed_vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(fixed_nodes))
    fixed_vertices.from_numpy(fixed_pos_np)

    # Ground plane (simple quad as 2 triangles)
    ground_verts = ti.Vector.field(3, dtype=ti.f32, shape=4)
    ground_indices = ti.field(dtype=ti.i32, shape=6)
    gv = np.array([[-0.5, -0.25, -0.15], [1.5, -0.25, -0.15],
                    [1.5, 0.35, -0.15], [-0.5, 0.35, -0.15]], dtype=np.float32)
    gi = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    ground_verts.from_numpy(gv)
    ground_indices.from_numpy(gi)

    tip_nodes = np.where(mesh.nodes[:, 0] > 0.95)[0]

    # ── GGUI ──
    window = ti.ui.Window("RoboSim — FEM Cantilever", (1200, 800), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.5, -0.8, 0.3)
    camera.lookat(0.5, 0.05, 0.0)
    camera.up(0, 0, 1)

    frame_count = 0
    frame_times = []

    print("\nSimulation running... (ESC to quit)")

    while window.running:
        t0 = time.perf_counter()

        # Physics sub-steps
        for _ in range(substeps):
            solver.step()

        # Upload updated positions
        vertices.from_numpy(body.x.astype(np.float32))

        # Stress colormap (auto-scale)
        vm = batch_von_mises(body.mesh, body.x, body.material,
                             body._dN_list, body._volumes)
        vm_max = max(vm.max(), 1.0)
        colors.from_numpy(stress_to_color(vm, vmin=0.0, vmax=vm_max))

        # Camera
        camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        # Lighting
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, -1.0, 1.5), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(0.5, 1.0, 1.0), color=(0.5, 0.5, 0.5))

        # Draw ground
        scene.mesh(ground_verts, ground_indices, color=(0.45, 0.45, 0.45))

        # Draw beam
        scene.mesh(vertices, indices, per_vertex_color=colors, two_sided=True)

        # Draw fixed nodes
        scene.particles(fixed_vertices, radius=0.005, color=(1.0, 0.2, 0.2))

        # Background BEFORE scene render
        canvas.set_background_color((0.15, 0.15, 0.2))
        canvas.scene(scene)

        # HUD
        tip_dz = (body.x[tip_nodes, 2] - mesh.nodes[tip_nodes, 2]).mean()
        dt_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(dt_ms)
        avg_ms = np.mean(frame_times[-30:])
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0

        gui = window.get_gui()
        with gui.sub_window("Info", x=0.01, y=0.01, width=0.4, height=0.15):
            gui.text(f"t={solver.time:.3f}s  FPS:{fps:.0f}")
            gui.text(f"Tip: {tip_dz*1000:.2f} mm (analytical: {delta_analytical*1000:.1f} mm)")
            gui.text(f"Stress: 0 ~ {vm_max:.0f} Pa")

        window.show()
        frame_count += 1

    # Final report
    tip_dz = (body.x[tip_nodes, 2] - mesh.nodes[tip_nodes, 2]).mean()
    ratio = tip_dz / delta_analytical
    print(f"\nFinal tip deflection: {tip_dz * 1000:.2f} mm")
    print(f"Analytical: {delta_analytical * 1000:.2f} mm")
    print(f"Ratio (FEM/analytical): {ratio:.2f}")
    if frame_times:
        print(f"Avg frame time: {np.mean(frame_times):.1f} ms ({1000/np.mean(frame_times):.0f} FPS)")
    print("Phase 3 FEM Solver: OK")


if __name__ == "__main__":
    main()
