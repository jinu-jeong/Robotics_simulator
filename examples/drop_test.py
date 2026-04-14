"""낙하 충돌 테스트 — 다중 박스 / 솔버 FPS 비교.

rbd / fem / cb 모드에서 N개 박스를 공중에서 낙하시켜 FPS를 비교합니다.
--num-boxes > 1 이면 각 박스의 tilt 각도는 랜덤.  단일 박스일 때만 --tilt 사용 가능.

실행:
  python examples/drop_test.py                              # RBD 박스 1개
  python examples/drop_test.py --num-boxes 4                # RBD 박스 4개
  python examples/drop_test.py --mode fem --num-boxes 4     # FEM 박스 4개
  python examples/drop_test.py --mode cb  --num-boxes 4     # C-B 박스 4개
  python examples/drop_test.py --mode fem --tilt 30         # FEM 단일 박스 30° 기울임

조작: 좌클릭 드래그(카메라 회전), 스크롤(줌), ESC(종료)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────
# 색상 팔레트
# ─────────────────────────────────────────────
BOX_COLORS = [
    np.array([0.95, 0.35, 0.25]),  # 빨강
    np.array([0.25, 0.60, 0.95]),  # 파랑
    np.array([0.30, 0.88, 0.45]),  # 초록
    np.array([0.95, 0.82, 0.22]),  # 노랑
    np.array([0.82, 0.32, 0.95]),  # 보라
    np.array([0.95, 0.60, 0.22]),  # 주황
    np.array([0.25, 0.85, 0.85]),  # 청록
    np.array([0.95, 0.42, 0.75]),  # 핑크
]

SIDE   = 0.2      # 박스 한 변 (m)
MASS   = 1.0      # 박스 질량 (kg)
DROP_H = 0.8      # 낙하 시작 높이 (m)
DT     = 0.001    # 시뮬 타임스텝
SUBSTEPS = 15     # 프레임당 서브스텝


def _box_positions(num_boxes: int) -> list[np.ndarray]:
    """N개 박스를 격자로 배치. 간격 = SIDE * 1.6."""
    n_side = max(1, int(np.ceil(np.sqrt(num_boxes))))
    spacing = SIDE * 1.6
    positions = []
    for i in range(num_boxes):
        row = i // n_side
        col = i % n_side
        x = (col - (n_side - 1) / 2.0) * spacing
        y = (row - (n_side - 1) / 2.0) * spacing
        positions.append(np.array([x, y, DROP_H]))
    return positions


def _box_euler_angles(num_boxes: int, tilt_deg: float,
                      rng: np.random.Generator) -> list[np.ndarray]:
    """각 박스의 초기 Euler 각도 (degrees)."""
    if num_boxes == 1 and tilt_deg != 0.0:
        return [np.array([0.0, tilt_deg, 0.0])]
    # 랜덤 ±45° 내 세 축
    angles = rng.uniform(-45, 45, size=(num_boxes, 3))
    return [angles[i] for i in range(num_boxes)]


def _color(i: int) -> np.ndarray:
    return BOX_COLORS[i % len(BOX_COLORS)]


# ─────────────────────────────────────────────
# von Mises stress → 색상 (FEM/CB 공용)
# ─────────────────────────────────────────────
def _stress_color(vm: np.ndarray, vmax: float) -> np.ndarray:
    t = np.clip(vm / max(vmax, 1.0), 0, 1)
    r = np.clip(1.5 - np.abs(t - 0.75) * 4.0, 0, 1)
    g = np.clip(1.5 - np.abs(t - 0.50) * 4.0, 0, 1)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4.0, 0, 1)
    return np.column_stack([r, g, b]).astype(np.float32)


# ═══════════════════════════════════════════════════════
# RBD 강체 박스
# ═══════════════════════════════════════════════════════

def run_rbd_drop(num_boxes: int, tilt_deg: float, seed: int) -> None:
    from robosim.model.factory import create_free_box
    from robosim.physics.rbd.solver import RBDSolver
    from robosim.physics.contact.detection import GroundPlane
    from robosim.physics.contact.response import ContactParams
    from robosim.physics.contact.solver import ContactSolver
    from robosim.viz.viewer import SimViewer, geometry_to_trimesh

    rng = np.random.default_rng(seed)
    positions = _box_positions(num_boxes)
    euler_angles = _box_euler_angles(num_boxes, tilt_deg, rng)

    contact = ContactSolver(
        ground=GroundPlane(height=0.0),
        params=ContactParams(stiffness=5e4, damping=500, friction_mu=0.5),
    )

    robots, solvers, box_verts_list = [], [], []
    for i in range(num_boxes):
        col = np.append(_color(i), 1.0)
        robot = create_free_box(
            name=f"box_{i}", size=(SIDE, SIDE, SIDE),
            mass=MASS, position=positions[i], color=col,
        )
        # 초기 회전 설정 (q[3]=rx, q[4]=ry, q[5]=rz, degrees→radians)
        ea = np.deg2rad(euler_angles[i])
        robot.q[3:6] = ea
        robot.qd = np.zeros(robot.n_dof)

        solver_i = RBDSolver(robot=robot)
        solver_i.initialize(dt=DT)
        contact.register_rbd(solver_i, robot_id=f"box_{i}")

        # 박스 기준 메시 버텍스 (local)
        box_body = robot.links[-1]
        bv, bf = geometry_to_trimesh(box_body.visuals[0].geometry)
        box_verts_list.append((bv, bf))

        robots.append(robot)
        solvers.append(solver_i)

    title = f"RoboSim — Drop Test [RBD]  {num_boxes} box{'es' if num_boxes>1 else ''}"
    viewer = SimViewer(title=title, window_size=(1280, 800))
    viewer.initialize()

    # 지면 메시
    gv = np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]], dtype=np.float32)
    gf = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
    viewer.add_mesh("ground", gv, gf, color=np.array([0.45, 0.45, 0.45]))

    # 박스 메시 등록
    for i, (robot, (bv, bf)) in enumerate(zip(robots, box_verts_list)):
        fk = robot.forward_kinematics()
        mat = fk[-1].to_matrix()
        R_init, t_init = mat[:3, :3], mat[:3, 3]
        viewer.add_mesh(f"box_{i}", (R_init @ bv.T).T + t_init, bf,
                        color=_color(i))

    frame_times: list[float] = []
    sim_time = [0.0]

    def step_callback(_step):
        t0 = time.perf_counter()

        for _ in range(SUBSTEPS):
            for solver_i in solvers:
                wrenches = contact.compute_rbd_contact_forces(solver_i)
                solver_i.clear_external_forces()
                for li, w in wrenches.items():
                    solver_i.set_external_force(li, w)
                solver_i.step()
            sim_time[0] += DT

        phys_ms = (time.perf_counter() - t0) * 1000

        for i, (robot, (bv, _)) in enumerate(zip(robots, box_verts_list)):
            fk = robot.forward_kinematics()
            mat = fk[-1].to_matrix()
            R_i, t_i = mat[:3, :3], mat[:3, 3]
            viewer.update_mesh_vertices(f"box_{i}", (R_i @ bv.T).T + t_i)

        frame_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(frame_ms)
        fps = 1000 / np.mean(frame_times[-30:]) if frame_times else 0

        zs = [r.q[2] for r in robots]
        info = (
            f"[RBD]  {num_boxes} boxes\n"
            f"t = {sim_time[0]:.3f}s\n"
            f"FPS: {fps:.1f}  phys: {phys_ms:.1f}ms\n"
            f"Z min: {min(zs):.4f}m"
        )
        viewer.add_text(info)

    viewer.add_callback(step_callback)
    print(f"RBD drop — {num_boxes} box(es), tilt={'random' if num_boxes>1 else tilt_deg}°"
          f" — ESC to quit")
    viewer.show()

    avg_fps = 1000 / np.mean(frame_times) if frame_times else 0
    print(f"\n[RBD] {num_boxes} boxes  Avg FPS: {avg_fps:.1f}  "
          f"({np.mean(frame_times):.1f} ms/frame)")


# ═══════════════════════════════════════════════════════
# FEM / C-B 변형 박스 (공용)
# ═══════════════════════════════════════════════════════

def run_deformable_drop(mode: str, num_boxes: int,
                        tilt_deg: float, seed: int) -> None:
    from robosim.physics.fem.mesh import TetMesh
    from robosim.physics.fem.materials import CorotationalElastic
    from robosim.physics.fem.assembly import batch_von_mises
    from robosim.physics.contact.detection import GroundPlane
    from robosim.physics.contact.solver import ContactSolver
    from robosim.viz.viewer import SimViewer
    from scipy.spatial.transform import Rotation

    use_cb = (mode == "cb")

    if use_cb:
        from robosim.physics.fem.reduced import CraigBamptonBody, CraigBamptonSolver
    else:
        from robosim.physics.fem.solver import DeformableBody, FEMSolver

    rng = np.random.default_rng(seed)
    positions = _box_positions(num_boxes)
    euler_angles = _box_euler_angles(num_boxes, tilt_deg, rng)

    bodies, surface_tris = [], []
    for i in range(num_boxes):
        pos = positions[i]
        mesh = TetMesh.create_box(
            origin=pos - SIDE / 2,
            size=np.full(3, SIDE),
            divisions=(3, 3, 3),
        )
        # 기울임 적용
        ea = euler_angles[i]
        R_tilt = (Rotation.from_euler('xyz', ea, degrees=True).as_matrix()
                  if np.any(ea != 0) else np.eye(3))
        if np.any(ea != 0):
            com = mesh.nodes.mean(axis=0)
            mesh.nodes[:] = (R_tilt @ (mesh.nodes - com).T).T + com

        mat = CorotationalElastic(young=5e5, poisson=0.3)

        if use_cb:
            body = CraigBamptonBody(
                mesh=mesh, material=mat, density=1000.0,
                n_modes=6, gravity=np.array([0., 0., -9.81]),
                damping=0.02, name=f"cb_box_{i}",
            )
        else:
            body = DeformableBody(
                name=f"fem_box_{i}", mesh=mesh,
                material=mat, density=1000.0,
            )

        bodies.append(body)
        surface_tris.append(mesh.extract_surface())

        if use_cb and i < num_boxes - 1:
            print()   # 빈 줄 구분

    # 솔버 생성
    if use_cb:
        solver = CraigBamptonSolver(bodies=bodies)
    else:
        solver = FEMSolver(
            bodies=bodies,
            gravity=np.array([0., 0., -9.81]),
            damping=0.02,
        )
    solver.initialize(dt=DT)

    contact = ContactSolver(ground=GroundPlane(height=0.0))

    mode_str = "C-B" if use_cb else "FEM"
    title = (f"RoboSim — Drop Test [{mode_str}]  "
             f"{num_boxes} box{'es' if num_boxes>1 else ''}")
    viewer = SimViewer(title=title, window_size=(1280, 800))
    viewer.initialize()

    # 지면
    gv = np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]], dtype=np.float32)
    gf = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
    viewer.add_mesh("ground", gv, gf, color=np.array([0.45, 0.45, 0.45]))

    # 박스 메시 등록
    for i, (body, stri) in enumerate(zip(bodies, surface_tris)):
        viewer.add_mesh(f"box_{i}", body.x.copy(), stri, color=_color(i))

    frame_times: list[float] = []

    def step_callback(_step):
        t0 = time.perf_counter()

        for _ in range(SUBSTEPS):
            solver.step()
            for body in bodies:
                contact.resolve_fem_contact(body, restitution=0.3, friction_mu=0.5)

        phys_ms = (time.perf_counter() - t0) * 1000

        vm_max_all = 1.0
        for i, (body, stri) in enumerate(zip(bodies, surface_tris)):
            viewer.update_mesh_vertices(f"box_{i}", body.x.copy())
            vm = batch_von_mises(body.mesh, body.x, body.material,
                                 body._dN_list, body._volumes)
            vm_max = max(float(vm.max()), 1.0)
            vm_max_all = max(vm_max_all, vm_max)
            viewer.update_mesh_color(f"box_{i}", _stress_color(vm, vm_max))

        frame_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(frame_ms)
        fps = 1000 / np.mean(frame_times[-30:]) if frame_times else 0

        z_mins = [body.x[:, 2].min() for body in bodies]
        extra = ""
        if use_cb:
            b0 = bodies[0]
            extra = f"\nn_r={b0._n_r} ({b0._n_b}b+{b0._n_modes_actual}m)"

        info = (
            f"[{mode_str}]  {num_boxes} boxes\n"
            f"t = {solver.time:.3f}s\n"
            f"FPS: {fps:.1f}  phys: {phys_ms:.1f}ms\n"
            f"Z min: {min(z_mins):.4f}m  stress: {vm_max_all:.0f}Pa"
            + extra
        )
        viewer.add_text(info)

    viewer.add_callback(step_callback)
    print(f"{mode_str} drop — {num_boxes} box(es), "
          f"tilt={'random' if num_boxes>1 else tilt_deg}° — ESC to quit")
    viewer.show()

    avg_fps = 1000 / np.mean(frame_times) if frame_times else 0
    print(f"\n[{mode_str}] {num_boxes} boxes  Avg FPS: {avg_fps:.1f}  "
          f"({np.mean(frame_times):.1f} ms/frame)")


# ═══════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="다중 박스 낙하 테스트 — rbd/fem/cb FPS 비교")
    parser.add_argument("--mode", choices=["rbd", "fem", "cb"], default="rbd",
                        help="솔버 모드 (기본: rbd)")
    parser.add_argument("--num-boxes", "-n", type=int, default=1,
                        metavar="N",
                        help="낙하 박스 수 (기본: 1).  N>1 이면 tilt 랜덤.")
    parser.add_argument("--tilt", type=float, default=0.0,
                        metavar="DEG",
                        help="단일 박스 Y축 기울기 (degrees).  --num-boxes>1 이면 무시됨.")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (기본: 42)")
    args = parser.parse_args()

    if args.num_boxes > 1 and args.tilt != 0.0:
        print("[경고] --num-boxes > 1 이면 --tilt 는 무시됩니다. tilt는 랜덤으로 설정됩니다.")

    if args.num_boxes < 1:
        parser.error("--num-boxes 는 1 이상이어야 합니다.")

    ti.init(arch=ti.metal)

    print("=" * 60)
    print(f"  RoboSim: Drop Test")
    print(f"  mode={args.mode.upper()}  boxes={args.num_boxes}"
          f"  tilt={'random' if args.num_boxes > 1 else args.tilt}°"
          f"  seed={args.seed}")
    print("=" * 60)

    if args.mode == "rbd":
        run_rbd_drop(args.num_boxes, args.tilt, args.seed)
    else:
        run_deformable_drop(args.mode, args.num_boxes, args.tilt, args.seed)


if __name__ == "__main__":
    main()
