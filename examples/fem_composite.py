"""Composite Mesh 데모: Hex8 코어 + Tet4 셸 혼합 요소.

실무 FEM 솔버처럼 하나의 메시 안에 서로 다른 요소 타입이 섞인 composite mesh를
시뮬레이션합니다. 코어(내부)는 Hex8, 셸(외부)은 Tet4로 구성됩니다.

색상:
  - 주황색 영역 → Hex8 elements
  - 파란색 영역 → Tet4 elements
  - 경계에서 자연스럽게 블렌딩

실행:
  python examples/fem_composite.py
  python examples/fem_composite.py --mesh merge

조작:
  - 마우스 좌클릭 드래그: 카메라 회전
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

from robosim.physics.fem.mesh import CompositeMesh, FEMesh, TetMesh
from robosim.physics.fem.elements import ElementType
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.solver import FEMSolver, DeformableBody
from robosim.physics.fem.assembly import batch_von_mises
from robosim.viz.colormap import (
    composite_element_color, element_type_hud,
)
from robosim.viz.wireframe import (
    RenderMode, RENDER_MODE_NAMES,
    extract_edges, edges_to_line_indices, handle_mode_events,
)


def create_hex_tet_mesh(L, W, H):
    """Create hex-core + tet-shell composite box."""
    return CompositeMesh.create_hex_tet_box(
        origin=np.zeros(3),
        size=np.array([L, W, H]),
        hex_divisions=(8, 3, 3),
        tet_layers=1,
    )


def create_merged_mesh(L, W, H):
    """Create composite by merging separate hex and tet meshes."""
    hex_m = FEMesh.create_hex_box(
        origin=np.zeros(3),
        size=np.array([L / 2, W, H]),
        divisions=(4, 2, 2),
    )
    tet_m = TetMesh.create_box(
        origin=np.array([L / 2, 0, 0]),
        size=np.array([L / 2, W, H]),
        divisions=(4, 2, 2),
    )
    return CompositeMesh.merge_meshes(
        [hex_m, tet_m],
        names=["hex_half", "tet_half"],
        merge_tol=1e-8,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", choices=["hex-tet", "merge"], default="hex-tet",
                        help="Composite 생성 방식 (hex-tet: 코어+셸, merge: 좌우 병합)")
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--render", type=int, choices=[1, 2, 3, 4], default=3,
                        help="초기 렌더 모드: 1=Solid, 2=Wire, 3=Solid+Wire, 4=Stress+Wire (기본: 3)")
    args = parser.parse_args()

    ti.init(arch=ti.metal)

    L, W, H = 0.8, 0.12, 0.12

    if args.mesh == "hex-tet":
        mesh = create_hex_tet_mesh(L, W, H)
        title_suffix = "Hex Core + Tet Shell"
    else:
        mesh = create_merged_mesh(L, W, H)
        title_suffix = "Merged Hex + Tet"

    print("=" * 60)
    print(f"  RoboSim: Composite Mesh — {title_suffix}")
    print("=" * 60)
    print(mesh)
    print(mesh.block_summary())

    fixed = np.where(mesh.nodes[:, 0] < 1e-6)[0]

    body = DeformableBody(
        name="composite_beam",
        mesh=mesh,
        material=CorotationalElastic(young=5e6, poisson=0.3),
        density=1000.0,
        fixed_nodes=fixed,
    )

    fem = FEMSolver(
        bodies=[body],
        gravity=np.array([0, 0, -9.81]),
        damping=0.5,
    )
    fem.initialize(dt=0.003)

    print(f"Fixed: {len(fixed)} nodes, DOFs: {mesh.n_dof}")

    # Analytical reference
    I_moment = W * H**3 / 12.0
    A = W * H
    w = body.density * 9.81 * A
    delta_analytical = -w * L**4 / (8 * body.material.young * I_moment)
    print(f"Analytical tip: {delta_analytical*1000:.2f} mm")

    # ── Surface & edges ──
    surface_tri = mesh.extract_surface()
    n_nodes = mesh.n_nodes
    n_tri = surface_tri.shape[0]
    edges = extract_edges(surface_tri)
    n_edges = edges.shape[0]

    print(f"Surface: {n_tri} triangles, {n_edges} edges")

    # ── Taichi fields ──
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_nodes)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=n_nodes)
    indices = ti.field(dtype=ti.i32, shape=n_tri * 3)
    line_indices = ti.field(dtype=ti.i32, shape=n_edges * 2)

    indices.from_numpy(surface_tri.ravel().astype(np.int32))
    line_indices.from_numpy(edges_to_line_indices(edges))
    vertices.from_numpy(body.x.astype(np.float32))

    # Element-type base colors
    base_color_np = composite_element_color(mesh)
    colors.from_numpy(base_color_np)

    # Fixed nodes
    fixed_pos = mesh.nodes[fixed].astype(np.float32)
    fixed_verts = ti.Vector.field(3, dtype=ti.f32, shape=len(fixed))
    fixed_verts.from_numpy(fixed_pos)

    # Ground
    ground_v = ti.Vector.field(3, dtype=ti.f32, shape=4)
    ground_i = ti.field(dtype=ti.i32, shape=6)
    gv = np.array([[-0.3, -0.2, -0.1], [1.2, -0.2, -0.1],
                    [1.2, 0.35, -0.1], [-0.3, 0.35, -0.1]], dtype=np.float32)
    ground_v.from_numpy(gv)
    ground_i.from_numpy(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))

    tip_nodes = np.where(mesh.nodes[:, 0] > L * 0.95)[0]
    hud_info = element_type_hud(None, mesh)

    # ── Render state ──
    render_mode = RenderMode(args.render)

    # ── Window ──
    window = ti.ui.Window(f"RoboSim — Composite [{title_suffix}]", (1200, 800), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.4, -0.6, 0.25)
    camera.lookat(0.4, 0.06, 0.02)
    camera.up(0, 0, 1)

    frame_times = []
    vm_max = 1.0

    print(f"\nRender mode: {RENDER_MODE_NAMES[render_mode]} (press 1-4 to switch)")
    print("Simulation running... (ESC to quit)")

    while window.running:
        t0 = time.perf_counter()

        # Mode switch
        render_mode = handle_mode_events(window, render_mode)

        # Physics
        for _ in range(args.substeps):
            fem.step()

        vertices.from_numpy(body.x.astype(np.float32))

        # Color
        need_stress = render_mode == RenderMode.STRESS_WIRE
        if need_stress:
            vm = batch_von_mises(body.mesh, body.x, body.material,
                                 body._dN_list, body._volumes)
            vm_max = max(vm.max(), 1.0)
            color_np = composite_element_color(mesh, vm=vm, vmin=0, vmax=vm_max, tint_strength=0.3)
        else:
            color_np = base_color_np
        colors.from_numpy(color_np)

        # Render
        camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.4, -0.8, 1.0), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(0.4, 0.8, 0.5), color=(0.4, 0.4, 0.4))

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
        tip_dz = (body.x[tip_nodes, 2] - mesh.nodes[tip_nodes, 2]).mean() if len(tip_nodes) else 0

        gui = window.get_gui()
        with gui.sub_window("Composite Mesh", x=0.01, y=0.01, width=0.48, height=0.38):
            gui.text(f"t={fem.time:.3f}s  FPS:{fps:.0f}")
            gui.text(f"Tip: {tip_dz*1000:.2f}mm (analytical: {delta_analytical*1000:.1f}mm)")
            if need_stress:
                gui.text(f"Stress max: {vm_max:.0f} Pa")
            gui.text("---")
            for line in hud_info.split("\n"):
                gui.text(line)
            gui.text("---")
            gui.text(f"Mode: [{int(render_mode)}] {RENDER_MODE_NAMES[render_mode]}")
            gui.text("Keys: 1=Solid 2=Wire 3=Solid+Wire 4=Stress+Wire")
            gui.text("Color: Orange=Hex8  Blue=Tet4")

        window.show()

    # Report
    if tip_nodes.size:
        tip_dz = (body.x[tip_nodes, 2] - mesh.nodes[tip_nodes, 2]).mean()
        ratio = tip_dz / delta_analytical if abs(delta_analytical) > 1e-12 else 0
        print(f"\nTip: {tip_dz*1000:.2f}mm (analytical: {delta_analytical*1000:.2f}mm, ratio={ratio:.2f})")
    if frame_times:
        print(f"Avg frame: {np.mean(frame_times):.1f}ms ({1000/np.mean(frame_times):.0f} FPS)")


if __name__ == "__main__":
    main()
