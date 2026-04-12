"""FEM-FEM 접촉 데모: 변형 큐브가 변형 블록 위에 낙하.

두 개의 변형체(소프트 큐브) 사이의 접촉을 시연합니다.
위의 큐브가 중력으로 낙하하여 아래 블록에 부딪히며,
FEM-FEM impulse 기반 접촉이 관통을 방지합니다.

실행:
  python examples/fem_fem_drop.py                 # 기본
  python examples/fem_fem_drop.py --tilt 20       # 기울여서 낙하
  python examples/fem_fem_drop.py --render 3      # 솔리드+와이어프레임

조작: 좌클릭 드래그(회전), W/S(줌), 키보드 1-4(렌더 모드), ESC(종료)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).parent.parent))


def stress_to_color(vm, vmax):
    """Von Mises stress -> heat-map RGB."""
    t = np.clip(vm / max(vmax, 1.0), 0, 1)
    r = np.clip(1.5 - np.abs(t - 0.75) * 4.0, 0, 1)
    g = np.clip(1.5 - np.abs(t - 0.50) * 4.0, 0, 1)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4.0, 0, 1)
    return np.column_stack([r, g, b]).astype(np.float32)


def run(tilt_deg: float = 0.0, render_mode: int = 1):
    from robosim.physics.fem.mesh import TetMesh
    from robosim.physics.fem.materials import CorotationalElastic
    from robosim.physics.fem.solver import FEMSolver, DeformableBody
    from robosim.physics.fem.assembly import batch_von_mises
    from robosim.physics.contact.detection import GroundPlane
    from robosim.physics.contact.solver import ContactSolver
    from robosim.viz.viewer import SimViewer
    from robosim.viz.wireframe import RenderMode, extract_edges, edges_to_line_indices, handle_mode_events
    from scipy.spatial.transform import Rotation

    dt = 0.001
    substeps = 10

    # ── Bottom block: resting on ground, fixed bottom face ──
    bottom_size = np.array([0.3, 0.3, 0.1])
    mesh_bottom = TetMesh.create_box(
        origin=np.array([-0.15, -0.15, 0.0]),
        size=bottom_size,
        divisions=(4, 4, 2),
    )
    fixed = np.where(mesh_bottom.nodes[:, 2] < 1e-6)[0]
    body_bottom = DeformableBody(
        name="bottom_block",
        mesh=mesh_bottom,
        material=CorotationalElastic(young=3e5, poisson=0.3),
        density=1200.0,
        fixed_nodes=fixed,
    )

    # ── Top cube: drops from height ──
    top_side = 0.15
    drop_h = 0.25
    mesh_top = TetMesh.create_box(
        origin=np.array([-top_side / 2, -top_side / 2, drop_h]),
        size=np.array([top_side, top_side, top_side]),
        divisions=(3, 3, 3),
    )

    # Optional tilt
    if tilt_deg != 0.0:
        com = mesh_top.nodes.mean(axis=0)
        R = Rotation.from_euler("y", tilt_deg, degrees=True).as_matrix()
        mesh_top.nodes[:] = (R @ (mesh_top.nodes - com).T).T + com
        print(f"  Top cube tilt: {tilt_deg} deg about Y")

    body_top = DeformableBody(
        name="top_cube",
        mesh=mesh_top,
        material=CorotationalElastic(young=5e5, poisson=0.3),
        density=1000.0,
    )

    # ── Solvers ──
    fem = FEMSolver(
        bodies=[body_bottom, body_top],
        gravity=np.array([0.0, 0.0, -9.81]),
        damping=0.05,
        max_newton_iters=3,
    )
    fem.initialize(dt=dt)

    contact = ContactSolver(ground=GroundPlane(height=0.0))
    contact.register_fem(fem)

    # ── Info ──
    for i, b in enumerate(fem.bodies):
        m = b.density * b._volumes.sum()
        print(f"  Body {i} ({b.name}): {b.mesh.n_nodes} nodes, "
              f"{b.mesh.n_elements} elems, mass={m:.3f}kg")

    # ── Viewer ──
    viewer = SimViewer(title="RoboSim - FEM-FEM Drop Test", window_size=(1200, 800))
    viewer.initialize()

    colors_default = [
        np.array([0.3, 0.65, 0.9]),   # bottom: blue
        np.array([0.9, 0.45, 0.2]),    # top: orange
    ]

    surfaces = []
    edge_data = []
    for i, body in enumerate(fem.bodies):
        surf = body.mesh.extract_surface()
        surfaces.append(surf)
        edges = extract_edges(surf)
        edge_data.append((edges, edges_to_line_indices(edges)))
        viewer.add_mesh(f"body_{i}", body.x.copy(), surf, color=colors_default[i])

    # Ground plane
    gnd_v = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float64)
    gnd_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    viewer.add_mesh("ground", gnd_v, gnd_f, color=np.array([0.85, 0.85, 0.8]), opacity=0.5)

    mode = RenderMode(render_mode)
    sim_time = 0.0
    frame_count = 0
    total_fem_contacts = 0

    def step_callback(step):
        nonlocal sim_time, frame_count, mode, total_fem_contacts

        # Mode switching (1-4)
        mode = handle_mode_events(viewer._window, mode)

        for _ in range(substeps):
            fem.step()

            # Ground contact for each body
            for body in fem.bodies:
                contact.resolve_fem_contact(body, restitution=0.2, friction_mu=0.5)

            # FEM-FEM contact
            n_c = contact.resolve_fem_fem_all(fem, restitution=0.1, friction_mu=0.5, d_hat=0.008)
            total_fem_contacts += n_c

            sim_time += dt

        # Update visuals
        for i, body in enumerate(fem.bodies):
            viewer.update_mesh_vertices(f"body_{i}", body.x.copy())

            # Stress coloring for stress modes
            if mode in (RenderMode.STRESS_WIRE,):
                vm = batch_von_mises(body.mesh, body.x, body.material,
                                     body._dN_list, body._volumes)
                vm_max = max(vm.max(), 1.0)
                viewer.update_mesh_color(f"body_{i}", stress_to_color(vm, vm_max))
            else:
                # Reset to default color
                n_verts = body.x.shape[0]
                c = np.tile(colors_default[i], (n_verts, 1)).astype(np.float32)
                viewer.update_mesh_color(f"body_{i}", c)

        # Metrics
        top_z_min = body_top.x[:, 2].min()
        top_com = body_top.x.mean(axis=0)
        bottom_z_max = body_bottom.x[:, 2].max()
        gap = top_z_min - bottom_z_max

        vm_top = batch_von_mises(body_top.mesh, body_top.x, body_top.material,
                                 body_top._dN_list, body_top._volumes)
        vm_bot = batch_von_mises(body_bottom.mesh, body_bottom.x, body_bottom.material,
                                 body_bottom._dN_list, body_bottom._volumes)

        info = (
            f"t = {sim_time:.3f}s | Render: {mode.name}\n"
            f"Top CoM: ({top_com[0]:+.3f}, {top_com[1]:+.3f}, {top_com[2]:+.3f})\n"
            f"Gap: {gap:+.4f} m (negative = overlap)\n"
            f"Top Zmin: {top_z_min:.4f} | Bot Zmax: {bottom_z_max:.4f}\n"
            f"Stress: top={vm_top.max():.0f} Pa, bot={vm_bot.max():.0f} Pa\n"
            f"FEM-FEM contacts total: {total_fem_contacts}"
        )
        viewer.add_text(info)
        frame_count += 1

    viewer.add_callback(step_callback)
    print("\nFEM-FEM Drop Test - ESC to quit, keys 1-4 for render mode")
    viewer.show()

    # Final report
    top_z_min = body_top.x[:, 2].min()
    bottom_z_max = body_bottom.x[:, 2].max()
    print(f"\n{'='*50}")
    print(f"Final top Zmin: {top_z_min:.4f} m")
    print(f"Final bot Zmax: {bottom_z_max:.4f} m")
    print(f"Gap: {top_z_min - bottom_z_max:+.4f} m")
    print(f"Total FEM-FEM contacts resolved: {total_fem_contacts}")
    finite_ok = all(np.all(np.isfinite(b.x)) for b in fem.bodies)
    print(f"Finite check: {'OK' if finite_ok else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description="FEM-FEM contact drop test")
    parser.add_argument("--tilt", type=float, default=0.0,
                        help="Tilt angle for top cube (degrees)")
    parser.add_argument("--render", type=int, choices=[1, 2, 3, 4], default=1,
                        help="Render mode: 1=solid 2=wire 3=solid+wire 4=stress+wire")
    args = parser.parse_args()

    ti.init(arch=ti.metal)

    print("=" * 60)
    print("  RoboSim: FEM-FEM Contact Drop Test")
    print("  Soft cube drops onto soft block")
    print("=" * 60)

    run(tilt_deg=args.tilt, render_mode=args.render)


if __name__ == "__main__":
    main()
