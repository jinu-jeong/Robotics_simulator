"""Phase 3 검증 데모: FEM 외팔보 시뮬레이션 — Taichi GGUI 뷰어.

Metal/Vulkan 네이티브 렌더링으로 PyVista 대비 훨씬 빠릅니다.
요소 타입(Tet4/Tet10/Hex8) 비교 가능, 실시간 렌더모드 전환.

실행:
  python examples/fem_cantilever_taichi.py                      # 기본 Tet4 FEM
  python examples/fem_cantilever_taichi.py --mode cb            # C-B 축소 (Tet4)
  python examples/fem_cantilever_taichi.py --element-type hex8
  python examples/fem_cantilever_taichi.py --element-type tet10
  python examples/fem_cantilever_taichi.py --mode cb --element-type tet4 --n-modes 20

조작:
  - 마우스 좌클릭 드래그: 카메라 회전
  - 스크롤: 줌 인/아웃
  - 1: Solid (채움만)
  - 2: Wireframe (엣지만)
  - 3: Solid + Wire (채움+엣지)
  - 4: Stress + Wire (응력컬러+엣지)
  - ESC: 종료
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import taichi as ti

from robosim.physics.fem.mesh import TetMesh, FEMesh
from robosim.physics.fem.elements import ElementType
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.solver import FEMSolver, DeformableBody
from robosim.physics.fem.reduced import CraigBamptonBody, CraigBamptonSolver
from robosim.physics.fem.assembly import batch_von_mises
from robosim.viz.colormap import (
    stress_to_color, stress_with_element_tint,
    element_type_color, element_type_hud,
    ELEMENT_TYPE_SHORT, ELEMENT_TYPE_COLORS,
)
from robosim.viz.wireframe import (
    RenderMode, RENDER_MODE_NAMES,
    extract_edges, edges_to_line_indices, handle_mode_events,
)


def create_mesh(element_type: str, L, W, H, divisions):
    """Create mesh of the specified element type."""
    origin = np.zeros(3)
    size = np.array([L, W, H])

    if element_type == "tet4":
        mesh = TetMesh.create_box(origin=origin, size=size, divisions=divisions)
        etype = ElementType.TET4
    elif element_type == "tet10":
        mesh = FEMesh.create_tet10_box(origin=origin, size=size, divisions=divisions)
        etype = ElementType.TET10
    elif element_type == "hex8":
        mesh = FEMesh.create_hex_box(origin=origin, size=size, divisions=divisions)
        etype = ElementType.HEX8
    else:
        raise ValueError(f"Unknown element type: {element_type}")

    return mesh, etype


def main():
    parser = argparse.ArgumentParser(description="FEM cantilever beam demo")
    parser.add_argument("--mode", choices=["fem", "cb"], default="fem",
                        help="솔버: 'fem' (전체 FEM) 또는 'cb' (Craig-Bampton 축소)")
    parser.add_argument(
        "--element-type", "-e",
        choices=["tet4", "tet10", "hex8"],
        default="tet4",
        help="요소 타입 선택 (기본: tet4)",
    )
    parser.add_argument("--n-modes", type=int, default=10,
                        help="C-B 고정 경계 고유 모드 수 (--mode cb 시 사용, 기본: 10)")
    parser.add_argument("--substeps", type=int, default=5, help="프레임당 물리 서브스텝 (기본: 5)")
    parser.add_argument("--divisions", type=str, default=None,
                        help="메시 분할 수 (예: 10,2,2). 미지정 시 요소 타입에 맞게 자동 설정")
    parser.add_argument("--render", type=int, choices=[1, 2, 3, 4], default=4,
                        help="초기 렌더 모드: 1=Solid, 2=Wire, 3=Solid+Wire, 4=Stress+Wire (기본: 4)")
    args = parser.parse_args()

    substeps = args.substeps
    etype_str = args.element_type

    # Auto-select divisions based on element type
    if args.divisions:
        divisions = tuple(int(x) for x in args.divisions.split(","))
    else:
        if etype_str == "tet4":
            divisions = (10, 2, 2)
        elif etype_str == "tet10":
            divisions = (6, 2, 2)
        elif etype_str == "hex8":
            divisions = (10, 2, 2)
        else:
            divisions = (10, 2, 2)

    ti.init(arch=ti.metal)

    solver_mode = args.mode
    n_modes_cb = args.n_modes

    print("=" * 60)
    print(f"  RoboSim FEM Cantilever — {etype_str.upper()}  [{solver_mode.upper()}]")
    print("=" * 60)

    # ── Mesh & solver setup ──
    L, W, H = 1.0, 0.1, 0.1
    mesh, etype = create_mesh(etype_str, L, W, H, divisions)
    fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-10)[0]
    material = CorotationalElastic(young=1e7, poisson=0.3)

    if solver_mode == "cb":
        body = CraigBamptonBody(
            mesh=mesh,
            material=material,
            density=1000.0,
            n_modes=n_modes_cb,
            fixed_nodes=fixed_nodes,
            gravity=np.array([0.0, 0.0, -9.81]),
            damping=0.5,
            name="cantilever",
        )
        solver = CraigBamptonSolver(bodies=[body])
    else:
        body = DeformableBody(
            name="cantilever",
            mesh=mesh,
            material=material,
            density=1000.0,
            fixed_nodes=fixed_nodes,
        )
        solver = FEMSolver(
            bodies=[body],
            gravity=np.array([0.0, 0.0, -9.81]),
            damping=0.5,
        )
    solver.initialize(dt=0.005)

    print(f"Solver: {solver_mode.upper()}"
          + (f"  n_modes={n_modes_cb}  n_r={body._n_r}" if solver_mode == "cb" else ""))
    print(f"Element type: {etype_str.upper()}")
    print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    print(f"DOFs: {mesh.n_nodes * 3}")
    print(f"Fixed nodes: {len(fixed_nodes)}")
    print(f"Divisions: {divisions}")

    # Analytical reference
    I_moment = W * H**3 / 12.0
    A = W * H
    w = body.density * 9.81 * A
    delta_analytical = -w * L**4 / (8 * body.material.young * I_moment)
    print(f"Analytical tip deflection: {delta_analytical * 1000:.2f} mm")

    # ── Surface & edge data ──
    surface_tri = mesh.extract_surface()
    n_nodes = mesh.n_nodes
    n_tri = surface_tri.shape[0]

    edges = extract_edges(surface_tri)
    n_edges = edges.shape[0]
    line_idx_np = edges_to_line_indices(edges)

    print(f"Surface: {n_tri} triangles, {n_edges} edges")

    # ── Taichi fields ──
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_nodes)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=n_nodes)
    indices = ti.field(dtype=ti.i32, shape=n_tri * 3)
    line_indices = ti.field(dtype=ti.i32, shape=n_edges * 2)

    indices.from_numpy(surface_tri.ravel().astype(np.int32))
    line_indices.from_numpy(line_idx_np)
    vertices.from_numpy(body.x.astype(np.float32))

    # Colors
    base_color_np = element_type_color(etype, n_nodes)
    colors.from_numpy(base_color_np)

    # Fixed node markers
    fixed_pos_np = mesh.nodes[fixed_nodes].astype(np.float32)
    fixed_vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(fixed_nodes))
    fixed_vertices.from_numpy(fixed_pos_np)

    # Ground plane
    ground_verts = ti.Vector.field(3, dtype=ti.f32, shape=4)
    ground_indices = ti.field(dtype=ti.i32, shape=6)
    gv = np.array([[-0.5, -0.25, -0.15], [1.5, -0.25, -0.15],
                    [1.5, 0.35, -0.15], [-0.5, 0.35, -0.15]], dtype=np.float32)
    ground_verts.from_numpy(gv)
    ground_indices.from_numpy(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))

    tip_nodes = np.where(mesh.nodes[:, 0] > 0.95)[0]
    etype_info = element_type_hud(etype, mesh)

    # ── Render state ──
    render_mode = RenderMode(args.render)
    etype_rgb = tuple(float(c) for c in ELEMENT_TYPE_COLORS[etype])

    # ── GGUI ──
    window = ti.ui.Window(f"RoboSim — FEM Cantilever [{etype_str.upper()}]", (1200, 800), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.5, -0.8, 0.3)
    camera.lookat(0.5, 0.05, 0.0)
    camera.up(0, 0, 1)

    frame_times = []
    vm_max = 1.0

    print(f"\nRender mode: {RENDER_MODE_NAMES[render_mode]} (press 1-4 to switch)")
    print("Simulation running... (ESC to quit)")

    while window.running:
        t0 = time.perf_counter()

        # ── Mode switch ──
        render_mode = handle_mode_events(window, render_mode)

        # ── Physics ──
        for _ in range(substeps):
            solver.step()

        vertices.from_numpy(body.x.astype(np.float32))

        # ── Color update ──
        need_stress = render_mode == RenderMode.STRESS_WIRE
        if need_stress:
            vm = batch_von_mises(body.mesh, body.x, body.material,
                                 body._dN_list, body._volumes)
            vm_max = max(vm.max(), 1.0)
            color_np = stress_with_element_tint(vm, etype, vmin=0.0, vmax=vm_max, tint_strength=0.15)
        else:
            color_np = base_color_np
        colors.from_numpy(color_np)

        # ── Render ──
        camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, -1.0, 1.5), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(0.5, 1.0, 1.0), color=(0.5, 0.5, 0.5))

        # Ground
        scene.mesh(ground_verts, ground_indices, color=(0.45, 0.45, 0.45))

        # FEM mesh — draw based on mode
        show_solid = render_mode in (RenderMode.SOLID, RenderMode.SOLID_WIRE, RenderMode.STRESS_WIRE)
        show_wire = render_mode in (RenderMode.WIREFRAME, RenderMode.SOLID_WIRE, RenderMode.STRESS_WIRE)

        if show_solid:
            scene.mesh(vertices, indices, per_vertex_color=colors, two_sided=True)

        if show_wire:
            scene.lines(vertices, width=1.5, indices=line_indices, color=(0.1, 0.1, 0.1))

        # Fixed nodes
        scene.particles(fixed_vertices, radius=0.005, color=(1.0, 0.2, 0.2))

        canvas.set_background_color((0.15, 0.15, 0.2))
        canvas.scene(scene)

        # ── HUD ──
        tip_dz = (body.x[tip_nodes, 2] - mesh.nodes[tip_nodes, 2]).mean()
        dt_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(dt_ms)
        avg_ms = np.mean(frame_times[-30:])
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0

        gui = window.get_gui()
        with gui.sub_window("Info", x=0.01, y=0.01, width=0.45, height=0.38):
            gui.text(f"Solver: {solver_mode.upper()}  t={solver.time:.3f}s  FPS:{fps:.0f}")
            gui.text(f"Tip: {tip_dz*1000:.2f}mm (analytical: {delta_analytical*1000:.1f}mm)")
            if need_stress:
                gui.text(f"Stress: 0 ~ {vm_max:.0f} Pa")
            gui.text(f"---")
            for line in etype_info.split("\n"):
                gui.text(line)
            gui.text(f"---")
            gui.text(f"Mode: [{int(render_mode)}] {RENDER_MODE_NAMES[render_mode]}")
            gui.text("Keys: 1=Solid 2=Wire 3=Solid+Wire 4=Stress+Wire")

        window.show()

    # Final report
    tip_dz = (body.x[tip_nodes, 2] - mesh.nodes[tip_nodes, 2]).mean()
    ratio = tip_dz / delta_analytical
    print(f"\n[{etype_str.upper()}] Final tip deflection: {tip_dz * 1000:.2f} mm")
    print(f"Analytical: {delta_analytical * 1000:.2f} mm")
    print(f"Ratio (FEM/analytical): {ratio:.2f}")
    if frame_times:
        print(f"Avg frame time: {np.mean(frame_times):.1f} ms ({1000/np.mean(frame_times):.0f} FPS)")


if __name__ == "__main__":
    main()
