"""Phase 5 검증 데모: 강체 팔 + FEM 변형 블록 커플링.

1-DOF revolute 팔에 FEM 변형 블록이 페널티 스프링으로 연결되어
함께 움직이는 모습을 확인합니다.

실행:
  python examples/hybrid_test.py
  python examples/hybrid_test.py --no-gravity   # 무중력 모드

조작: 좌클릭 드래그(회전), W/S(줌), ESC(종료)
"""

import argparse
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
from robosim.physics.world import PhysicsWorld
from robosim.viz.viewer import SimViewer
from robosim.viz.scene_renderer import RobotRenderer


def make_arm(arm_len=0.5):
    """Revolute arm: base(fixed) -> shoulder(revolute Y) -> arm link."""
    base = Link(
        name="base",
        inertial=SpatialInertia(mass=0, com=np.zeros(3), inertia=np.zeros((3, 3))),
        visuals=[Visual(
            geometry=Geometry.sphere(0.04),
            origin=Transform.identity(),
            color=np.array([0.4, 0.4, 0.4, 1.0]),
        )],
    )

    arm = Link(
        name="arm",
        inertial=SpatialInertia(
            mass=2.0,
            com=np.array([arm_len / 2, 0, 0]),
            inertia=np.diag([0.001, 2.0 * arm_len**2 / 3, 2.0 * arm_len**2 / 3]),
        ),
        visuals=[Visual(
            geometry=Geometry.box(arm_len, 0.04, 0.04),
            origin=Transform.from_translation(np.array([arm_len / 2, 0, 0])),
            color=np.array([0.2, 0.5, 0.8, 1.0]),
        )],
        collisions=[Collision(
            geometry=Geometry.box(arm_len, 0.04, 0.04),
            origin=Transform.from_translation(np.array([arm_len / 2, 0, 0])),
        )],
    )

    joint = Joint(
        name="shoulder",
        joint_type=JointType.REVOLUTE,
        parent_link="base",
        child_link="arm",
        axis=np.array([0.0, 1.0, 0.0]),
        origin=Transform.from_translation(np.array([0, 0, 0.8])),
    )

    robot = Robot(name="arm", links=[base, arm], joints=[joint])
    robot.build()
    return robot


def stress_to_color(vm, vmax):
    """Von Mises stress -> jet colormap."""
    t = np.clip(vm / max(vmax, 1.0), 0, 1)
    r = np.clip(1.5 - np.abs(t - 0.75) * 4.0, 0, 1)
    g = np.clip(1.5 - np.abs(t - 0.50) * 4.0, 0, 1)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4.0, 0, 1)
    return np.column_stack([r, g, b]).astype(np.float32)


def run(use_gravity=True):
    arm_len = 0.5
    dt = 0.001
    substeps = 10  # physics substeps per render frame
    block_size = 0.08

    gravity = np.array([0.0, 0.0, -9.81]) if use_gravity else np.zeros(3)

    # --- Robot ---
    robot = make_arm(arm_len)
    robot.gravity = gravity.copy()
    robot.q = np.array([0.3])   # initial angle ~17 degrees
    robot.qd = np.array([0.0])

    # --- FEM block at arm tip ---
    fk = robot.forward_kinematics()
    tip = fk[1].apply_point(np.array([arm_len, 0, 0]))
    origin = tip + np.array([-block_size / 2, -block_size / 2, -block_size])
    mesh = TetMesh.create_box(
        origin=origin,
        size=np.array([block_size, block_size, block_size]),
        divisions=(3, 3, 3),
    )
    body = DeformableBody(
        name="block",
        mesh=mesh,
        material=CorotationalElastic(young=5e5, poisson=0.3),
        density=800.0,
    )

    # --- Solvers ---
    rbd = RBDSolver(robot=robot)
    rbd.initialize(dt=dt)
    fem = FEMSolver(bodies=[body], gravity=gravity, damping=0.05)
    fem.initialize(dt=dt)

    contact = ContactSolver(ground=GroundPlane(height=0.0))

    # --- Coupling: top face of FEM block -> arm link ---
    top_nodes = select_face_nodes(body.x, axis=2, side="max")
    bmap = create_boundary_map(robot, link_idx=1, fem_body=body,
                               fem_body_idx=0, boundary_nodes=top_nodes)

    coupling = PenaltyCoupling(stiffness=2e4, damping=500, max_force=1000)
    coupling.setup(rbd, fem, [bmap])

    print(f"FEM mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    print(f"Boundary nodes (top face): {len(top_nodes)}")
    print(f"Block mass: {body.density * body._volumes.sum():.3f} kg")
    print(f"Coupling: k={coupling.stiffness:.0f} N/m, c={coupling.damping:.0f}")

    # --- Viewer setup ---
    viewer = SimViewer(title="RoboSim — Hybrid RBD + FEM", window_size=(1200, 800))
    viewer.initialize()

    # Robot meshes
    renderer = RobotRenderer(robot, viewer)
    renderer.setup()

    # FEM surface mesh
    surface_tri = mesh.extract_surface()
    viewer.add_mesh(
        "fem_block",
        body.x.copy(),
        surface_tri,
        color=np.array([0.3, 0.7, 0.4]),
    )

    # Boundary node markers (small green dots)
    boundary_verts = body.x[top_nodes].copy()

    frame_count = 0
    sim_time = 0.0

    def step_callback(step):
        nonlocal frame_count, sim_time

        for _ in range(substeps):
            # 1. Coupling forces
            rbd_wrenches, fem_forces = coupling.compute_forces()

            # 2. Step RBD
            rbd.clear_external_forces()
            for li, w in rbd_wrenches.items():
                rbd.set_external_force(li, w)
            rbd.step()

            # 3. Step FEM
            fem.step(extra_forces=fem_forces if fem_forces else None)

            # 4. Ground contact for FEM
            if use_gravity:
                contact.resolve_fem_contact(body, restitution=0.2, friction_mu=0.5)

            sim_time += dt

        # Update renderer
        renderer.update()
        viewer.update_mesh_vertices("fem_block", body.x.copy())

        # Stress coloring
        vm = batch_von_mises(body.mesh, body.x, body.material,
                             body._dN_list, body._volumes)
        vm_max = max(vm.max(), 1.0)
        viewer.update_mesh_color("fem_block", stress_to_color(vm, vm_max))

        # Measure boundary drift
        fk_now = robot.forward_kinematics()
        T = fk_now[bmap.rigid_link_idx]
        targets = np.array([T.apply_point(p) for p in bmap.local_positions])
        actual = body.x[bmap.fem_boundary_nodes]
        drift = np.linalg.norm(targets - actual, axis=1).max()

        # HUD
        com = body.x.mean(axis=0)
        q_deg = np.degrees(robot.q[0])
        info = (
            f"t = {sim_time:.3f}s\n"
            f"Joint q = {q_deg:+.1f} deg  qd = {robot.qd[0]:+.2f} rad/s\n"
            f"FEM CoM: ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f})\n"
            f"Boundary drift: {drift:.4f} m\n"
            f"Stress max: {vm_max:.0f} Pa\n"
            f"Gravity: {'ON' if use_gravity else 'OFF'}"
        )
        viewer.add_text(info)
        frame_count += 1

    viewer.add_callback(step_callback)
    mode = "Gravity" if use_gravity else "No-gravity"
    print(f"\nHybrid test ({mode}) — ESC to quit")
    viewer.show()

    # Final report
    print(f"\nSimulated {sim_time:.2f}s in {frame_count} frames")
    print(f"Final joint angle: {np.degrees(robot.q[0]):.1f} deg")
    print(f"FEM CoM: {body.x.mean(axis=0)}")
    print(f"NaN check: positions={'OK' if np.all(np.isfinite(body.x)) else 'FAIL'}, "
          f"velocities={'OK' if np.all(np.isfinite(body.v)) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Hybrid RBD + FEM coupling demo")
    parser.add_argument("--no-gravity", action="store_true",
                        help="Run without gravity")
    args = parser.parse_args()

    ti.init(arch=ti.metal)

    print("=" * 60)
    print("  RoboSim Phase 5: Hybrid RBD + FEM Coupling")
    print("=" * 60)

    run(use_gravity=not args.no_gravity)


if __name__ == "__main__":
    main()
