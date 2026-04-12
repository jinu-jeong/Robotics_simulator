"""Phase 3 검증 데모: FEM 외팔보 시뮬레이션 (3D).

한쪽 끝이 고정된 탄성 빔이 중력 하에 처지는 모습을 실시간 3D로 확인합니다.

실행:
  python examples/fem_cantilever.py            # 응력 컬러맵 ON
  python examples/fem_cantilever.py --no-stress # 응력 컬러맵 OFF (더 빠름)

조작:
  - 마우스 좌클릭 드래그: 카메라 회전
  - 스크롤: 줌 인/아웃
  - 'q' 키: 종료
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.solver import FEMSolver, DeformableBody
from robosim.physics.fem.assembly import batch_von_mises


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-stress", action="store_true", help="응력 컬러맵 비활성화 (더 빠름)")
    parser.add_argument("--substeps", type=int, default=5, help="렌더 프레임당 물리 서브스텝 수 (기본: 5)")
    args = parser.parse_args()
    show_stress = not args.no_stress
    substeps = args.substeps

    print("=" * 60)
    print("  RoboSim Phase 3: FEM Cantilever Beam (3D)")
    print("=" * 60)

    # Create cantilever beam
    L, W, H = 1.0, 0.1, 0.1
    mesh = TetMesh.create_box(
        origin=np.zeros(3),
        size=np.array([L, W, H]),
        divisions=(10, 2, 2),
    )

    # Fix nodes at x=0
    fixed_nodes = np.where(mesh.nodes[:, 0] < 1e-10)[0]

    body = DeformableBody(
        name="cantilever",
        mesh=mesh,
        material=CorotationalElastic(young=1e6, poisson=0.3),
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
    print(f"Fixed nodes: {len(fixed_nodes)} (x=0 face)")
    print(f"Material: E={body.material.young:.0f} Pa, nu={body.material.poisson}")
    print(f"Stress colormap: {'ON' if show_stress else 'OFF'}")
    print(f"Sub-steps per frame: {substeps}")

    # Analytical reference
    I_moment = W * H**3 / 12.0
    A = W * H
    w = body.density * 9.81 * A
    delta_analytical = -w * L**4 / (8 * body.material.young * I_moment)
    print(f"Analytical tip deflection: {delta_analytical * 1000:.2f} mm")

    # Setup PyVista
    plotter = pv.Plotter(title="RoboSim - FEM Cantilever", window_size=(1200, 800))
    plotter.set_background("white")

    # Ground plane
    ground = pv.Plane(center=(0.5, 0, -0.15), direction=(0, 0, 1), i_size=2, j_size=0.5)
    plotter.add_mesh(ground, color="lightgray", opacity=0.3)

    # Surface faces for rendering
    surface_faces = mesh.extract_surface()

    # Build PyVista surface mesh
    n_faces = surface_faces.shape[0]
    faces = np.column_stack([np.full(n_faces, 3), surface_faces]).ravel()
    surf = pv.PolyData(body.x.copy(), faces)

    if show_stress:
        vm_stress = batch_von_mises(body.mesh, body.x, body.material, body._dN_list, body._volumes)
        surf.point_data["VonMises"] = vm_stress
        plotter.add_mesh(
            surf, scalars="VonMises", cmap="jet", clim=[0, 500],
            show_edges=True, edge_color="gray", line_width=0.5,
            name="beam", scalar_bar_args={"title": "Von Mises (Pa)", "color": "black"},
        )
    else:
        plotter.add_mesh(
            surf, color="steelblue",
            show_edges=True, edge_color="gray", line_width=0.5,
            name="beam",
        )

    # Fixed face indicator
    fixed_cloud = pv.PolyData(mesh.nodes[fixed_nodes])
    plotter.add_mesh(fixed_cloud, color="red", point_size=8, name="fixed_pts",
                     render_points_as_spheres=True)

    # Camera
    plotter.camera.position = (0.5, -0.8, 0.3)
    plotter.camera.focal_point = (0.5, 0.05, 0.0)
    plotter.camera.up = (0, 0, 1)

    # Precompute tip node indices
    tip_nodes = np.where(mesh.nodes[:, 0] > 0.95)[0]

    # Timing
    frame_times = []

    def step_callback(step):
        t0 = time.perf_counter()

        # 여러 물리 스텝을 한 렌더 프레임에 처리
        for _ in range(substeps):
            solver.step()

        # In-place vertex update (렌더는 1회만)
        surf.points = body.x.copy()

        if show_stress:
            surf.point_data["VonMises"] = batch_von_mises(
                body.mesh, body.x, body.material, body._dN_list, body._volumes
            )

        tip_dz = (body.x[tip_nodes, 2] - mesh.nodes[tip_nodes, 2]).mean()

        dt_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(dt_ms)
        avg_ms = np.mean(frame_times[-30:])

        info = (
            f"t = {solver.time:.3f}s | step {step}\n"
            f"Tip: {tip_dz * 1000:.2f} mm (analytical: {delta_analytical * 1000:.2f} mm)\n"
            f"Frame: {dt_ms:.0f} ms (avg {avg_ms:.0f} ms)"
        )
        plotter.add_text(info, position="upper_left", font_size=9, name="info", color="black")

    plotter.add_timer_event(max_steps=500, duration=50, callback=step_callback)

    print("\nSimulation running... (close window or press 'q' to quit)")
    plotter.show()

    # Final report
    tip_dz = (body.x[tip_nodes, 2] - mesh.nodes[tip_nodes, 2]).mean()
    ratio = tip_dz / delta_analytical
    print(f"\nFinal tip deflection: {tip_dz * 1000:.2f} mm")
    print(f"Analytical: {delta_analytical * 1000:.2f} mm")
    print(f"Ratio (FEM/analytical): {ratio:.2f}")
    if frame_times:
        print(f"Avg callback time: {np.mean(frame_times):.1f} ms")
    print("Phase 3 FEM Solver: OK")


if __name__ == "__main__":
    main()
