"""Phase 6 최종 데모: 강체 팔 + FEM 소프트 핑거 그리퍼.

2-DOF 강체 팔 끝에 두 개의 FEM 변형 블록(손가락)이 페널티 커플링으로
연결되어 있습니다. 팔이 아래로 내려가 목표 물체(강체 큐브) 근처에서
손가락을 닫아 잡는 시뮬레이션입니다.

실행:
  python examples/soft_gripper.py

조작: 좌클릭 드래그(회전), W/S(줌), ESC(종료)
"""

import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.math.spatial import SpatialInertia
from robosim.math.transforms import Transform
from robosim.model.geometry import Geometry
from robosim.model.joint import Joint, JointType
from robosim.model.link import Collision, Link, Visual
from robosim.model.robot import Robot
from robosim.physics.rbd.solver import RBDSolver
from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.solver import FEMSolver, DeformableBody
from robosim.physics.fem.assembly import batch_von_mises
from robosim.physics.contact.solver import ContactSolver
from robosim.physics.contact.detection import GroundPlane
from robosim.physics.coupling.interface import create_boundary_map, select_face_nodes
from robosim.physics.coupling.penalty import PenaltyCoupling
from robosim.control.pid import PIDController
from robosim.control.trajectory import TrajectoryController, Waypoint
from robosim.io.logger import SimLogger
from robosim.viz.viewer import SimViewer
from robosim.viz.scene_renderer import RobotRenderer


# ── Robot: 2-DOF arm (shoulder + elbow) ──

def make_gripper_arm(upper_len=0.3, lower_len=0.25):
    """2-DOF arm: base -> shoulder(revolute Y) -> upper -> elbow(revolute Y) -> lower."""
    base = Link(
        name="base",
        inertial=SpatialInertia(mass=0, com=np.zeros(3), inertia=np.zeros((3, 3))),
        visuals=[Visual(
            geometry=Geometry.sphere(0.03),
            origin=Transform.identity(),
            color=np.array([0.3, 0.3, 0.3, 1.0]),
        )],
    )

    upper = Link(
        name="upper_arm",
        inertial=SpatialInertia(
            mass=1.5,
            com=np.array([0, 0, -upper_len / 2]),
            inertia=np.diag([1.5 * upper_len**2 / 12, 1.5 * upper_len**2 / 12, 0.001]),
        ),
        visuals=[Visual(
            geometry=Geometry.box(0.04, 0.04, upper_len),
            origin=Transform.from_translation(np.array([0, 0, -upper_len / 2])),
            color=np.array([0.2, 0.5, 0.8, 1.0]),
        )],
    )

    lower = Link(
        name="lower_arm",
        inertial=SpatialInertia(
            mass=1.0,
            com=np.array([0, 0, -lower_len / 2]),
            inertia=np.diag([1.0 * lower_len**2 / 12, 1.0 * lower_len**2 / 12, 0.001]),
        ),
        visuals=[Visual(
            geometry=Geometry.box(0.04, 0.04, lower_len),
            origin=Transform.from_translation(np.array([0, 0, -lower_len / 2])),
            color=np.array([0.3, 0.6, 0.9, 1.0]),
        )],
    )

    shoulder = Joint(
        name="shoulder",
        joint_type=JointType.REVOLUTE,
        parent_link="base",
        child_link="upper_arm",
        axis=np.array([0.0, 1.0, 0.0]),
        origin=Transform.from_translation(np.array([0, 0, 0.6])),
    )

    elbow = Joint(
        name="elbow",
        joint_type=JointType.REVOLUTE,
        parent_link="upper_arm",
        child_link="lower_arm",
        axis=np.array([0.0, 1.0, 0.0]),
        origin=Transform.from_translation(np.array([0, 0, -upper_len])),
    )

    robot = Robot(name="gripper_arm", links=[base, upper, lower], joints=[shoulder, elbow])
    robot.build()
    return robot


# ── FEM fingers ──

def make_finger(origin, size=(0.02, 0.04, 0.06)):
    """Create a small FEM finger block."""
    mesh = TetMesh.create_box(
        origin=origin,
        size=np.array(size),
        divisions=(2, 3, 4),
    )
    return DeformableBody(
        name="finger",
        mesh=mesh,
        material=CorotationalElastic(young=2e5, poisson=0.35),
        density=600.0,
    )


def stress_to_color(vm, vmax):
    t = np.clip(vm / max(vmax, 1.0), 0, 1)
    r = np.clip(1.5 - np.abs(t - 0.75) * 4.0, 0, 1)
    g = np.clip(1.5 - np.abs(t - 0.50) * 4.0, 0, 1)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4.0, 0, 1)
    return np.column_stack([r, g, b]).astype(np.float32)


def run():
    dt = 0.001
    substeps = 8
    upper_len, lower_len = 0.3, 0.25

    # ── Build robot ──
    robot = make_gripper_arm(upper_len, lower_len)
    robot.gravity = np.array([0.0, 0.0, -9.81])
    robot.q = np.array([0.3, -0.6])  # initial pose: arm slightly bent
    robot.qd = np.array([0.0, 0.0])

    # ── FK to place fingers at arm tip ──
    fk = robot.forward_kinematics()
    tip = fk[2].apply_point(np.array([0, 0, -lower_len]))

    finger_h = 0.06
    finger_w = 0.02
    gap = 0.05  # gap between fingers

    left_origin = tip + np.array([-finger_w, -gap / 2 - 0.04, -finger_h])
    right_origin = tip + np.array([-finger_w, gap / 2, -finger_h])

    left_finger = make_finger(left_origin)
    right_finger = make_finger(right_origin)

    # ── Solvers ──
    rbd = RBDSolver(robot=robot)
    rbd.initialize(dt=dt)

    fem = FEMSolver(
        bodies=[left_finger, right_finger],
        gravity=np.array([0, 0, -9.81]),
        damping=0.1,
        max_newton_iters=2,  # fast interactive mode
    )
    fem.initialize(dt=dt)

    contact = ContactSolver(ground=GroundPlane(height=0.0))

    # ── Coupling: top of each finger -> lower arm link ──
    boundary_maps = []
    for body_idx, body in enumerate([left_finger, right_finger]):
        top = select_face_nodes(body.x, axis=2, side="max")
        bmap = create_boundary_map(robot, link_idx=2, fem_body=body,
                                   fem_body_idx=body_idx, boundary_nodes=top)
        boundary_maps.append(bmap)

    coupling = PenaltyCoupling(stiffness=1e4, damping=300, max_force=500)
    coupling.setup(rbd, fem, boundary_maps)

    # ── Controller: trajectory to reach down then hold ──
    # Phase 1 (0-1s): reach down (open pose)
    # Phase 2 (1-3s): hold position (fingers close via gravity/coupling)
    waypoints = [
        Waypoint(t=0.0, q=np.array([0.3, -0.6])),
        Waypoint(t=0.8, q=np.array([0.8, -1.2])),     # reach lower
        Waypoint(t=1.5, q=np.array([0.8, -1.2])),     # hold
        Waypoint(t=3.0, q=np.array([0.5, -0.8])),     # lift slightly
        Waypoint(t=5.0, q=np.array([0.5, -0.8])),     # hold
    ]
    traj = TrajectoryController(
        n_dof=2, waypoints=waypoints,
        kp=np.array([300.0, 200.0]),
        kd=np.array([40.0, 30.0]),
        max_torque=50.0,
    )

    # ── Logger ──
    logger = SimLogger(interval=10)

    # ── Info ──
    print(f"Left finger:  {left_finger.mesh.n_nodes} nodes, {left_finger.mesh.n_elements} elements")
    print(f"Right finger: {right_finger.mesh.n_nodes} nodes, {right_finger.mesh.n_elements} elements")
    lm = left_finger.density * left_finger._volumes.sum()
    rm = right_finger.density * right_finger._volumes.sum()
    print(f"Finger mass: L={lm:.3f}kg  R={rm:.3f}kg")
    print(f"Trajectory duration: {traj.duration:.1f}s")

    # ── Viewer ──
    viewer = SimViewer(title="RoboSim — Soft Gripper", window_size=(1200, 800))
    viewer.initialize()

    renderer = RobotRenderer(robot, viewer)
    renderer.setup()

    for i, (body, color) in enumerate([
        (left_finger, np.array([0.9, 0.4, 0.3])),
        (right_finger, np.array([0.3, 0.8, 0.4])),
    ]):
        surface = body.mesh.extract_surface()
        viewer.add_mesh(f"finger_{i}", body.x.copy(), surface, color=color)

    # Ground plane visual (large flat quad)
    gnd_v = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], dtype=np.float64)
    gnd_f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    viewer.add_mesh("ground", gnd_v, gnd_f, color=np.array([0.85, 0.85, 0.8]), opacity=0.5)

    sim_time = 0.0
    frame_count = 0

    def step_callback(step):
        nonlocal sim_time, frame_count

        for _ in range(substeps):
            # Controller torque
            tau = traj.compute(sim_time, robot.q, robot.qd)

            # Gravity compensation
            g_tau = rbd.compute_gravity_torques()

            # Coupling
            rbd_w, fem_f = coupling.compute_forces()

            # Step RBD
            rbd.clear_external_forces()
            rbd.tau = tau + g_tau
            for li, w in rbd_w.items():
                rbd.set_external_force(li, w)
            rbd.step()

            # Step FEM
            fem.step(extra_forces=fem_f if fem_f else None)

            # Ground contact
            for body in fem.bodies:
                contact.resolve_fem_contact(body, restitution=0.1, friction_mu=0.6)

            sim_time += dt

            # Log
            logger.log(
                step=int(sim_time / dt),
                time=sim_time,
                rbd_q=robot.q,
                rbd_qd=robot.qd,
                fem_positions=[b.x for b in fem.bodies],
            )

        # Update visuals
        renderer.update()
        for i, body in enumerate([left_finger, right_finger]):
            viewer.update_mesh_vertices(f"finger_{i}", body.x.copy())
            vm = batch_von_mises(body.mesh, body.x, body.material,
                                 body._dN_list, body._volumes)
            vm_max = max(vm.max(), 1.0)
            viewer.update_mesh_color(f"finger_{i}", stress_to_color(vm, vm_max))

        # Drift
        max_drift = 0.0
        fk_now = robot.forward_kinematics()
        for bmap in boundary_maps:
            T = fk_now[bmap.rigid_link_idx]
            targets = np.array([T.apply_point(p) for p in bmap.local_positions])
            actual = fem.bodies[bmap.fem_body_idx].x[bmap.fem_boundary_nodes]
            d = np.linalg.norm(targets - actual, axis=1).max()
            max_drift = max(max_drift, d)

        q_des, _ = traj.evaluate(sim_time)
        q_deg = np.degrees(robot.q)
        q_des_deg = np.degrees(q_des)
        tip_pos = fk_now[2].apply_point(np.array([0, 0, -lower_len]))

        info = (
            f"t = {sim_time:.2f}s\n"
            f"q  = [{q_deg[0]:+6.1f}, {q_deg[1]:+6.1f}] deg\n"
            f"ref= [{q_des_deg[0]:+6.1f}, {q_des_deg[1]:+6.1f}] deg\n"
            f"Tip: ({tip_pos[0]:+.3f}, {tip_pos[1]:+.3f}, {tip_pos[2]:+.3f})\n"
            f"Boundary drift: {max_drift:.4f} m\n"
            f"Frames logged: {logger.n_frames}"
        )
        viewer.add_text(info)
        frame_count += 1

    viewer.add_callback(step_callback)
    print("\nSoft Gripper Demo — ESC to quit")
    viewer.show()

    # Save log
    log_path = Path(__file__).parent / "soft_gripper_log.npz"
    logger.save(log_path)
    print(f"\nLog saved to {log_path} ({logger.n_frames} frames)")
    print(f"Simulated {sim_time:.2f}s in {frame_count} render frames")
    finite_ok = all(np.all(np.isfinite(b.x)) and np.all(np.isfinite(b.v)) for b in fem.bodies)
    print(f"Finite check: {'OK' if finite_ok else 'FAIL'}")


def main():
    ti.init(arch=ti.metal)

    print("=" * 60)
    print("  RoboSim Phase 6: Soft Gripper Demo")
    print("  강체 2-DOF 팔 + FEM 소프트 핑거")
    print("=" * 60)

    run()


if __name__ == "__main__":
    main()
