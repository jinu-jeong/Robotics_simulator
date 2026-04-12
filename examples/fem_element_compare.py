"""요소 타입 비교 데모: Tet4 vs Tet10 vs Hex8.

같은 외팔보를 세 가지 요소 타입으로 나란히 시뮬레이션합니다.
각 요소는 고유 색상으로 구분됩니다:
  - Tet4  : 파란색 (steel blue)
  - Tet10 : 녹색 (green)
  - Hex8  : 주황색 (orange)

실행:
  python examples/fem_element_compare.py

조작:
  - 좌클릭 드래그(회전), 스크롤(줌)
  - 1: Solid  2: Wire  3: Solid+Wire  4: Stress+Wire
  - ESC(종료)
"""

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
from robosim.physics.fem.assembly import batch_von_mises
from robosim.viz.colormap import (
    stress_with_element_tint, element_type_color,
    ELEMENT_TYPE_SHORT, ELEMENT_TYPE_COLORS,
)
from robosim.viz.wireframe import (
    RenderMode, RENDER_MODE_NAMES,
    extract_edges, edges_to_line_indices, handle_mode_events,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--substeps", type=int, default=3)
    parser.add_argument("--render", type=int, choices=[1, 2, 3, 4], default=3,
                        help="초기 렌더 모드 (기본: 3=Solid+Wire)")
    args = parser.parse_args()
    substeps = args.substeps

    ti.init(arch=ti.metal)

    print("=" * 60)
    print("  RoboSim: Element Type Comparison")
    print("  Tet4 (blue) | Tet10 (green) | Hex8 (orange)")
    print("=" * 60)

    L, W, H = 0.6, 0.06, 0.06
    E, nu, rho = 1e7, 0.3, 1000.0
    material = CorotationalElastic(young=E, poisson=nu)

    spacing = 0.15

    configs = [
        ("Tet4",  ElementType.TET4,  (8, 2, 2), -spacing),
        ("Tet10", ElementType.TET10, (5, 1, 1),  0.0),
        ("Hex8",  ElementType.HEX8,  (8, 2, 2),  spacing),
    ]

    bodies = []
    meshes = []
    etypes = []
    solvers = []

    for name, etype, divs, y_offset in configs:
        origin = np.array([0.0, y_offset, 0.0])
        size = np.array([L, W, H])

        if etype == ElementType.TET4:
            mesh = TetMesh.create_box(origin=origin, size=size, divisions=divs)
        elif etype == ElementType.TET10:
            mesh = FEMesh.create_tet10_box(origin=origin, size=size, divisions=divs)
        elif etype == ElementType.HEX8:
            mesh = FEMesh.create_hex_box(origin=origin, size=size, divisions=divs)

        fixed = np.where(mesh.nodes[:, 0] < 1e-10)[0]

        body = DeformableBody(
            name=f"beam_{name}", mesh=mesh, material=material,
            density=rho, fixed_nodes=fixed,
        )
        fem = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]), damping=0.5)
        fem.initialize(dt=0.005)

        bodies.append(body)
        meshes.append(mesh)
        etypes.append(etype)
        solvers.append(fem)

        print(f"  {name}: {mesh.n_nodes} nodes, {mesh.n_elements} elem, "
              f"{mesh.n_nodes * 3} DOFs, div={divs}")

    # Analytical reference
    I_moment = W * H**3 / 12.0
    A = W * H
    w = rho * 9.81 * A
    delta_analytical = -w * L**4 / (8 * E * I_moment)
    print(f"\nAnalytical tip: {delta_analytical * 1000:.2f} mm")

    # ── Build combined rendering data ──
    all_surface = []
    all_edges = []
    all_n_nodes = []
    all_tip_nodes = []

    for mesh, body in zip(meshes, bodies):
        surf = mesh.extract_surface()
        all_surface.append(surf)
        all_edges.append(extract_edges(surf))
        all_n_nodes.append(mesh.n_nodes)
        all_tip_nodes.append(np.where(mesh.nodes[:, 0] > L * 0.95)[0])

    total_verts = sum(all_n_nodes)
    total_tri_idx = sum(s.shape[0] * 3 for s in all_surface)
    total_edge_idx = sum(e.shape[0] * 2 for e in all_edges)

    vertices = ti.Vector.field(3, dtype=ti.f32, shape=total_verts)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=total_verts)
    indices = ti.field(dtype=ti.i32, shape=total_tri_idx)
    line_indices = ti.field(dtype=ti.i32, shape=total_edge_idx)

    # Upload indices (with offsets)
    all_tri_idx = []
    all_line_idx = []
    v_off = 0
    for surf, edg, nn in zip(all_surface, all_edges, all_n_nodes):
        all_tri_idx.append(surf.ravel().astype(np.int32) + v_off)
        all_line_idx.append(edg.ravel().astype(np.int32) + v_off)
        v_off += nn
    indices.from_numpy(np.concatenate(all_tri_idx))
    line_indices.from_numpy(np.concatenate(all_line_idx))

    # Base colors per beam
    base_colors_list = []
    for etype, nn in zip(etypes, all_n_nodes):
        base_colors_list.append(element_type_color(etype, nn))
    base_color_np = np.concatenate(base_colors_list)

    # Fixed node markers
    all_fixed = []
    for body, mesh in zip(bodies, meshes):
        all_fixed.append(mesh.nodes[body.fixed_nodes])
    fixed_all = np.concatenate(all_fixed).astype(np.float32)
    fixed_verts = ti.Vector.field(3, dtype=ti.f32, shape=len(fixed_all))
    fixed_verts.from_numpy(fixed_all)

    # Ground
    ground_v = ti.Vector.field(3, dtype=ti.f32, shape=4)
    ground_i = ti.field(dtype=ti.i32, shape=6)
    gv = np.array([[-0.3, -0.3, -0.1], [1.0, -0.3, -0.1],
                    [1.0, 0.3, -0.1], [-0.3, 0.3, -0.1]], dtype=np.float32)
    ground_v.from_numpy(gv)
    ground_i.from_numpy(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))

    n_total_edges = sum(e.shape[0] for e in all_edges)
    print(f"Total: {total_verts} verts, {sum(s.shape[0] for s in all_surface)} tris, {n_total_edges} edges")

    # ── Render state ──
    render_mode = RenderMode(args.render)

    window = ti.ui.Window("RoboSim — Element Comparison [Tet4|Tet10|Hex8]",
                          (1400, 800), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.3, -0.5, 0.25)
    camera.lookat(0.3, 0.0, 0.0)
    camera.up(0, 0, 1)

    frame_times = []

    print(f"\nRender mode: {RENDER_MODE_NAMES[render_mode]} (press 1-4)")
    print("Simulation running... (ESC to quit)")

    while window.running:
        t0 = time.perf_counter()

        render_mode = handle_mode_events(window, render_mode)

        # Physics
        for fem in solvers:
            for _ in range(substeps):
                fem.step()

        # Upload vertices & colors
        all_verts_np = []
        all_colors_np = []
        tip_deflections = []
        need_stress = render_mode == RenderMode.STRESS_WIRE

        for i, (body, mesh, etype) in enumerate(zip(bodies, meshes, etypes)):
            all_verts_np.append(body.x.astype(np.float32))

            if need_stress:
                vm = batch_von_mises(body.mesh, body.x, body.material,
                                     body._dN_list, body._volumes)
                vm_max = max(vm.max(), 1.0)
                c = stress_with_element_tint(vm, etype, vmin=0.0, vmax=vm_max, tint_strength=0.25)
                all_colors_np.append(c)
            else:
                all_colors_np.append(base_colors_list[i])

            tip = all_tip_nodes[i]
            dz = (body.x[tip, 2] - mesh.nodes[tip, 2]).mean() if len(tip) else 0.0
            tip_deflections.append(dz)

        vertices.from_numpy(np.concatenate(all_verts_np))
        colors.from_numpy(np.concatenate(all_colors_np))

        # Render
        camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.3, -0.8, 1.0), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(0.3, 0.8, 0.5), color=(0.4, 0.4, 0.4))

        scene.mesh(ground_v, ground_i, color=(0.45, 0.45, 0.45))

        show_solid = render_mode in (RenderMode.SOLID, RenderMode.SOLID_WIRE, RenderMode.STRESS_WIRE)
        show_wire = render_mode in (RenderMode.WIREFRAME, RenderMode.SOLID_WIRE, RenderMode.STRESS_WIRE)

        if show_solid:
            scene.mesh(vertices, indices, per_vertex_color=colors, two_sided=True)
        if show_wire:
            scene.lines(vertices, width=1.5, indices=line_indices, color=(0.1, 0.1, 0.1))

        scene.particles(fixed_verts, radius=0.004, color=(1.0, 0.2, 0.2))

        canvas.set_background_color((0.12, 0.12, 0.18))
        canvas.scene(scene)

        # HUD
        dt_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(dt_ms)
        avg_ms = np.mean(frame_times[-30:])
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0

        gui = window.get_gui()
        with gui.sub_window("Element Comparison", x=0.01, y=0.01, width=0.42, height=0.40):
            gui.text(f"t={solvers[0].time:.3f}s  FPS:{fps:.0f}")
            gui.text(f"Analytical tip: {delta_analytical*1000:.2f}mm")
            gui.text("---")
            for i, (name, etype, divs, _) in enumerate(configs):
                m = meshes[i]
                dz_mm = tip_deflections[i] * 1000
                ratio = tip_deflections[i] / delta_analytical if abs(delta_analytical) > 1e-12 else 0
                short = ELEMENT_TYPE_SHORT[etype]
                gui.text(f"{short}: {dz_mm:+.2f}mm (r={ratio:.2f}) "
                         f"[{m.n_nodes}n {m.n_elements}e]")
            gui.text("---")
            gui.text(f"Mode: [{int(render_mode)}] {RENDER_MODE_NAMES[render_mode]}")
            gui.text("Keys: 1=Solid 2=Wire 3=S+W 4=Stress+W")
            gui.text("Tet4=blue  Tet10=green  Hex8=orange")

        window.show()

    # Final report
    print("\n" + "=" * 60)
    for i, (name, etype, divs, _) in enumerate(configs):
        dz = tip_deflections[i]
        ratio = dz / delta_analytical if abs(delta_analytical) > 1e-12 else 0
        m = meshes[i]
        print(f"  {name:6s}: {dz*1000:+7.2f}mm (ratio={ratio:.3f}) "
              f"{m.n_nodes}n {m.n_elements}e")
    if frame_times:
        print(f"\nAvg: {np.mean(frame_times):.1f}ms ({1000/np.mean(frame_times):.0f} FPS)")


if __name__ == "__main__":
    main()
