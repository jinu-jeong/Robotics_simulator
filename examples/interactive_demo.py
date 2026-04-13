"""인터랙티브 시뮬레이션 데모.

UI 패널에서 물리 파라미터를 실시간으로 조절하고,
접촉력 화살표와 관절 축 오버레이를 토글할 수 있습니다.

모드:
  pendulum  — 2-링크 진자 (관절 댐핑, 중력 슬라이더)
  drop      — 구 낙하 (접촉 강성, 마찰, 반발 계수)
  arm       — 2-DOF 팔 관절 각도 직접 제어

실행:
  python examples/interactive_demo.py                # 기본: pendulum
  python examples/interactive_demo.py --mode drop
  python examples/interactive_demo.py --mode arm

조작:
  좌클릭 드래그: 카메라 회전, W/S: 줌, ESC: 종료
  UI 패널 슬라이더로 파라미터 실시간 조절
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.math.spatial import SpatialInertia
from robosim.math.transforms import Transform
from robosim.model.geometry import Geometry
from robosim.model.joint import Joint, JointType, JointLimits
from robosim.model.link import Collision, Link, Visual
from robosim.model.robot import Robot
from robosim.physics.rbd.solver import RBDSolver
from robosim.physics.contact.detection import GroundPlane
from robosim.physics.contact.response import ContactParams
from robosim.physics.contact.solver import ContactSolver
from robosim.viz.viewer import SimViewer
from robosim.viz.scene_renderer import RobotRenderer
from robosim.viz.ui_panels import UIPanel
from robosim.viz.overlays import ContactForceOverlay, JointAxisOverlay


# ════════════════════════════════════════════════════════════════
# Mode 1: Interactive Double Pendulum
# ════════════════════════════════════════════════════════════════

def run_pendulum():
    """Double pendulum with real-time damping and gravity control."""
    L1, L2 = 0.5, 0.4
    m1, m2 = 1.5, 1.0

    base = Link(
        name="base",
        inertial=SpatialInertia(mass=0, com=np.zeros(3), inertia=np.zeros((3, 3))),
        visuals=[Visual(
            geometry=Geometry.sphere(0.03),
            origin=Transform.identity(),
            color=np.array([0.3, 0.3, 0.3, 1.0]),
        )],
    )

    link1 = Link(
        name="link1",
        inertial=SpatialInertia(
            mass=m1, com=np.array([0, 0, -L1/2]),
            inertia=np.diag([m1*L1**2/3, m1*L1**2/3, 0.001]),
        ),
        visuals=[
            Visual(geometry=Geometry.cylinder(0.015, L1),
                   origin=Transform.from_translation(np.array([0, 0, -L1/2])),
                   color=np.array([0.3, 0.6, 0.9, 1.0])),
            Visual(geometry=Geometry.sphere(0.03),
                   origin=Transform.from_translation(np.array([0, 0, -L1])),
                   color=np.array([0.9, 0.5, 0.2, 1.0])),
        ],
    )

    link2 = Link(
        name="link2",
        inertial=SpatialInertia(
            mass=m2, com=np.array([0, 0, -L2/2]),
            inertia=np.diag([m2*L2**2/3, m2*L2**2/3, 0.001]),
        ),
        visuals=[
            Visual(geometry=Geometry.cylinder(0.012, L2),
                   origin=Transform.from_translation(np.array([0, 0, -L2/2])),
                   color=np.array([0.9, 0.3, 0.5, 1.0])),
            Visual(geometry=Geometry.sphere(0.025),
                   origin=Transform.from_translation(np.array([0, 0, -L2])),
                   color=np.array([0.2, 0.9, 0.3, 1.0])),
        ],
    )

    j1 = Joint(name="j1", joint_type=JointType.REVOLUTE,
               parent_link="base", child_link="link1",
               axis=np.array([0, 1, 0]),
               origin=Transform.from_translation(np.array([0, 0, 1.0])))
    j2 = Joint(name="j2", joint_type=JointType.REVOLUTE,
               parent_link="link1", child_link="link2",
               axis=np.array([0, 1, 0]),
               origin=Transform.from_translation(np.array([0, 0, -L1])))

    robot = Robot(name="double_pendulum", links=[base, link1, link2], joints=[j1, j2])
    robot.gravity = np.array([0, 0, -9.81])
    robot.build()
    robot.q = np.array([math.radians(90), math.radians(45)])
    robot.qd = np.zeros(2)

    dt = 0.002
    solver = RBDSolver(robot=robot)
    solver.initialize(dt=dt)
    E0 = solver.total_energy()

    # ── Viewer ──
    viewer = SimViewer(title="RoboSim — Interactive Double Pendulum",
                       window_size=(1200, 800))
    viewer.initialize()

    renderer = RobotRenderer(robot, viewer)
    renderer.setup()

    # ── UI Panel ──
    panel = UIPanel("Pendulum Controls")
    panel.add_slider("Gravity", 0.0, 20.0, 9.81)
    panel.add_slider("Damping", 0.0, 5.0, 0.0, fmt="%.2f")
    panel.add_slider("Init Angle 1", -180, 180, 90)
    panel.add_slider("Init Angle 2", -180, 180, 45)
    panel.add_checkbox("Show Joint Axes", False)
    panel.add_button("Reset")

    # ── Overlays ──
    joint_overlay = JointAxisOverlay(viewer, length=0.1)

    substeps = 5

    def step_callback(step):
        nonlocal E0

        # Read UI parameters
        robot.gravity[2] = -panel.get("Gravity")
        j1.damping = panel.get("Damping")
        j2.damping = panel.get("Damping")
        joint_overlay.active = panel.get_bool("Show Joint Axes")

        # Playback
        import time as _t
        panel.playback.update_fps(_t.time())

        # Reset button
        if panel.was_clicked("Reset"):
            a1 = math.radians(panel.get("Init Angle 1"))
            a2 = math.radians(panel.get("Init Angle 2"))
            robot.q[:] = [a1, a2]
            robot.qd[:] = 0
            solver._time = 0.0
            E0 = solver.total_energy()
            panel.playback.sim_time = 0.0
            panel.playback.frame_count = 0

        if panel.should_step():
            for _ in range(substeps):
                solver.step()
            panel.playback.sim_time = solver.time
            panel.playback.frame_count += 1

        renderer.update()

        # Joint axis overlay
        if joint_overlay.active:
            fk = robot.forward_kinematics()
            joint_overlay.update(robot, fk)

        # Status
        E = solver.total_energy()
        drift = abs(E - E0) / max(abs(E0), 1e-6) * 100
        panel.set_info([
            f"q1={math.degrees(robot.q[0]):+.1f} deg  "
            f"q2={math.degrees(robot.q[1]):+.1f} deg",
            f"KE={solver.kinetic_energy():.3f}  "
            f"PE={solver.potential_energy():.3f}  "
            f"E={E:.3f} J",
            f"Drift: {drift:.4f}%",
        ])

        # Render UI
        panel.render(viewer._window)

    viewer.add_callback(step_callback)
    print("Interactive Double Pendulum — ESC to quit")
    viewer.show()


# ════════════════════════════════════════════════════════════════
# Mode 2: Interactive Drop Test
# ════════════════════════════════════════════════════════════════

def run_drop():
    """Sphere drop with adjustable contact parameters."""
    radius = 0.12
    mass = 2.0
    drop_h = 1.2

    base = Link(
        name="base",
        inertial=SpatialInertia(mass=0, com=np.zeros(3), inertia=np.zeros((3, 3))),
    )
    sphere_link = Link(
        name="sphere",
        inertial=SpatialInertia(
            mass=mass, com=np.zeros(3),
            inertia=np.eye(3) * 0.4 * mass * radius**2,
        ),
        visuals=[Visual(
            geometry=Geometry.sphere(radius),
            origin=Transform.identity(),
            color=np.array([0.9, 0.4, 0.2, 1.0]),
        )],
        collisions=[Collision(
            geometry=Geometry.sphere(radius),
            origin=Transform.identity(),
        )],
    )

    joint = Joint(name="drop_z", joint_type=JointType.PRISMATIC,
                  parent_link="base", child_link="sphere",
                  axis=np.array([0, 0, 1]),
                  origin=Transform.identity())

    robot = Robot(name="drop_sphere", links=[base, sphere_link], joints=[joint])
    robot.gravity = np.array([0, 0, -9.81])
    robot.build()
    robot.q = np.array([drop_h])
    robot.qd = np.array([0.0])

    dt = 0.001
    solver = RBDSolver(robot=robot)
    solver.initialize(dt=dt)

    contact_params = ContactParams(stiffness=5e4, damping=500, friction_mu=0.5)
    contact = ContactSolver(ground=GroundPlane(height=0.0), params=contact_params)
    contact.register_rbd(solver)

    # ── Viewer ──
    viewer = SimViewer(title="RoboSim — Interactive Drop Test",
                       window_size=(1200, 800))
    viewer.initialize()
    renderer = RobotRenderer(robot, viewer)
    renderer.setup()

    # ── UI Panel ──
    panel = UIPanel("Drop Controls")
    panel.add_slider("Stiffness", 1e3, 1e6, 5e4)
    panel.add_slider("Damping", 10, 5000, 500)
    panel.add_slider("Friction", 0.0, 2.0, 0.5)
    panel.add_slider("Drop Height", 0.3, 3.0, drop_h)
    panel.add_checkbox("Show Forces", True)
    panel.add_button("Drop Again")

    # ── Overlays ──
    force_overlay = ContactForceOverlay(viewer, scale=0.0005, min_force=0.5,
                                        color=np.array([1.0, 0.8, 0.0]))

    max_height = drop_h
    substeps = 10

    def step_callback(step):
        nonlocal max_height
        import time as _t
        panel.playback.update_fps(_t.time())

        # Apply UI parameters to contact
        contact.params.stiffness = panel.get("Stiffness")
        contact.params.damping = panel.get("Damping")
        contact.params.friction_mu = panel.get("Friction")
        force_overlay.active = panel.get_bool("Show Forces")

        # Reset
        if panel.was_clicked("Drop Again"):
            h = panel.get("Drop Height")
            robot.q[:] = [h]
            robot.qd[:] = 0
            solver._time = 0.0
            max_height = h
            panel.playback.sim_time = 0.0

        if panel.should_step():
            for _ in range(substeps):
                solver.clear_external_forces()
                wrenches = contact.compute_rbd_contact_forces(solver)
                for li, w in wrenches.items():
                    solver.set_external_force(li, w)
                solver.step()
            panel.playback.sim_time = solver.time

        renderer.update()
        force_overlay.update(contact.last_forces)

        z = robot.q[0]
        vz = robot.qd[0]
        max_height = max(max_height, z)
        n_c = len(contact.last_forces)

        panel.set_info([
            f"Height: {z:.4f} m",
            f"Velocity: {vz:+.3f} m/s",
            f"Max height: {max_height:.3f} m",
            f"Contacts: {n_c}",
        ])

        panel.render(viewer._window)

    viewer.add_callback(step_callback)
    print("Interactive Drop Test — ESC to quit")
    viewer.show()


# ════════════════════════════════════════════════════════════════
# Mode 3: Interactive Arm with Joint Control
# ════════════════════════════════════════════════════════════════

def run_arm():
    """2-DOF arm with direct joint angle control sliders."""
    L1, L2 = 0.4, 0.35
    m1, m2 = 2.0, 1.5
    r = 0.025

    base = Link(
        name="base",
        inertial=SpatialInertia(mass=0, com=np.zeros(3), inertia=np.zeros((3, 3))),
        visuals=[Visual(
            geometry=Geometry.cylinder(0.04, 0.1),
            origin=Transform.identity(),
            color=np.array([0.4, 0.4, 0.4, 1.0]),
        )],
    )

    link1 = Link(
        name="upper_arm",
        inertial=SpatialInertia(
            mass=m1, com=np.array([0, 0, L1/2]),
            inertia=np.diag([m1*L1**2/12, m1*L1**2/12, 0.001]),
        ),
        visuals=[
            Visual(geometry=Geometry.cylinder(r, L1),
                   origin=Transform.from_translation(np.array([0, 0, L1/2])),
                   color=np.array([0.2, 0.5, 0.8, 1.0])),
            Visual(geometry=Geometry.sphere(r*1.5),
                   origin=Transform.from_translation(np.array([0, 0, L1])),
                   color=np.array([0.7, 0.7, 0.3, 1.0])),
        ],
    )

    link2 = Link(
        name="forearm",
        inertial=SpatialInertia(
            mass=m2, com=np.array([0, 0, L2/2]),
            inertia=np.diag([m2*L2**2/12, m2*L2**2/12, 0.001]),
        ),
        visuals=[
            Visual(geometry=Geometry.cylinder(r*0.8, L2),
                   origin=Transform.from_translation(np.array([0, 0, L2/2])),
                   color=np.array([0.8, 0.3, 0.3, 1.0])),
            Visual(geometry=Geometry.sphere(r*1.2),
                   origin=Transform.from_translation(np.array([0, 0, L2])),
                   color=np.array([0.3, 0.8, 0.3, 1.0])),
        ],
    )

    j1 = Joint(name="shoulder", joint_type=JointType.REVOLUTE,
               parent_link="base", child_link="upper_arm",
               axis=np.array([0, 1, 0]),
               origin=Transform.from_translation(np.array([0, 0, 0.05])),
               limits=JointLimits(lower=-np.pi, upper=np.pi))
    j2 = Joint(name="elbow", joint_type=JointType.REVOLUTE,
               parent_link="upper_arm", child_link="forearm",
               axis=np.array([0, 1, 0]),
               origin=Transform.from_translation(np.array([0, 0, L1])),
               limits=JointLimits(lower=-np.pi*0.9, upper=np.pi*0.9))

    robot = Robot(name="2dof_arm", links=[base, link1, link2], joints=[j1, j2])
    robot.gravity = np.array([0, 0, -9.81])
    robot.build()
    robot.q = np.array([0.0, math.radians(30)])
    robot.qd = np.zeros(2)

    dt = 0.002
    solver = RBDSolver(robot=robot)
    solver.initialize(dt=dt)

    # PD gains
    kp = np.array([200.0, 150.0])
    kd = np.array([20.0, 15.0])

    # ── Viewer ──
    viewer = SimViewer(title="RoboSim — Interactive 2-DOF Arm",
                       window_size=(1200, 800))
    viewer.initialize()
    renderer = RobotRenderer(robot, viewer)
    renderer.setup()

    # ── UI Panel ──
    panel = UIPanel("Arm Controls")
    panel.add_slider("Kp", 10, 1000, 200)
    panel.add_slider("Kd", 1, 100, 20)
    panel.add_checkbox("Show Joint Axes", True)
    panel.add_checkbox("Gravity Comp", True)

    panel.setup_joint_control(
        joint_names=["shoulder", "elbow"],
        initial_values=np.array([0.0, math.radians(30)]),
        limits=[(-180, 180), (-162, 162)],  # degrees
    )

    # ── Overlays ──
    joint_overlay = JointAxisOverlay(viewer, length=0.12)

    substeps = 5

    def step_callback(step):
        import time as _t
        panel.playback.update_fps(_t.time())

        # Read gains
        kp_val = panel.get("Kp")
        kd_val = panel.get("Kd")
        joint_overlay.active = panel.get_bool("Show Joint Axes")
        grav_comp = panel.get_bool("Gravity Comp")

        # Joint targets (from slider, in degrees → radians)
        targets_deg = panel.get_joint_targets()
        if targets_deg is not None:
            q_des = np.radians(targets_deg)
        else:
            q_des = robot.q.copy()

        if panel.should_step():
            for _ in range(substeps):
                tau = kp_val * (q_des - robot.q) + kd_val * (-robot.qd)
                if grav_comp:
                    tau += solver.compute_gravity_torques()
                tau = np.clip(tau, -100, 100)
                solver.tau = tau
                solver.step()
            panel.playback.sim_time = solver.time

        renderer.update()

        if joint_overlay.active:
            fk = robot.forward_kinematics()
            joint_overlay.update(robot, fk)

        fk = robot.forward_kinematics()
        ee_pos = fk[2].translation  # forearm tip
        # Approximate end-effector (L2 further along)
        ee = fk[2].apply_point(np.array([0, 0, L2]))

        panel.set_info([
            f"q: [{math.degrees(robot.q[0]):+.1f}, "
            f"{math.degrees(robot.q[1]):+.1f}] deg",
            f"qd: [{robot.qd[0]:+.2f}, {robot.qd[1]:+.2f}] rad/s",
            f"EE: ({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f})",
            f"Kp={kp_val:.0f}  Kd={kd_val:.0f}",
        ])

        # Update displayed joint values
        panel.update_joint_values(np.degrees(robot.q))
        panel.render(viewer._window)

    viewer.add_callback(step_callback)
    print("Interactive 2-DOF Arm — ESC to quit")
    viewer.show()

    print(f"\nFinal q: [{math.degrees(robot.q[0]):.1f}, "
          f"{math.degrees(robot.q[1]):.1f}] deg")


# ════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Interactive simulation demo with UI panels")
    parser.add_argument("--mode", choices=["pendulum", "drop", "arm"],
                        default="pendulum",
                        help="Demo mode (default: pendulum)")
    args = parser.parse_args()

    ti.init(arch=ti.metal)

    print("=" * 60)
    print(f"  RoboSim: Interactive Demo ({args.mode})")
    print("  UI Panel: adjust parameters in real-time")
    print("=" * 60)

    if args.mode == "pendulum":
        run_pendulum()
    elif args.mode == "drop":
        run_drop()
    elif args.mode == "arm":
        run_arm()


if __name__ == "__main__":
    main()
