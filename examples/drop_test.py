"""Phase 4 검증 데모: 낙하 충돌 테스트.

구(sphere)가 중력으로 낙하하여 바닥에 충돌·반발하는 모습을 확인합니다.
FEM 모드에서는 변형 큐브가 바닥에 떨어집니다.

실행:
  python examples/drop_test.py              # RBD 구 낙하
  python examples/drop_test.py --mode fem   # FEM 큐브 낙하

조작: 좌클릭 드래그(회전), W/S(줌), ESC(종료)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_rbd_drop():
    """RBD sphere dropping onto ground plane."""
    from robosim.math.spatial import SpatialInertia
    from robosim.math.transforms import Transform
    from robosim.model.geometry import Geometry
    from robosim.model.joint import Joint, JointType
    from robosim.model.link import Collision, Link, Visual
    from robosim.model.robot import Robot
    from robosim.physics.rbd.solver import RBDSolver
    from robosim.physics.contact.detection import GroundPlane
    from robosim.physics.contact.response import ContactParams
    from robosim.physics.contact.solver import ContactSolver
    from robosim.viz.viewer import SimViewer
    from robosim.viz.scene_renderer import RobotRenderer

    radius = 0.15
    mass = 2.0
    drop_height = 1.5

    # Robot: base (fixed) → prismatic Z → sphere link
    base = Link(
        name="base",
        inertial=SpatialInertia(mass=0, com=np.zeros(3), inertia=np.zeros((3, 3))),
    )
    sphere_link = Link(
        name="sphere",
        inertial=SpatialInertia(
            mass=mass, com=np.zeros(3),
            inertia=np.eye(3) * (2.0 / 5.0) * mass * radius**2,
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

    joint = Joint(
        name="drop_z",
        joint_type=JointType.PRISMATIC,
        parent_link="base",
        child_link="sphere",
        axis=np.array([0.0, 0.0, 1.0]),
        origin=Transform.identity(),
    )

    robot = Robot(name="drop_sphere", links=[base, sphere_link], joints=[joint])
    robot.gravity = np.array([0.0, 0.0, -9.81])
    robot.build()
    robot.q = np.array([drop_height])
    robot.qd = np.array([0.0])

    dt = 0.001
    solver = RBDSolver(robot=robot)
    solver.initialize(dt=dt)

    contact = ContactSolver(
        ground=GroundPlane(height=0.0),
        params=ContactParams(stiffness=5e4, damping=500, friction_mu=0.5),
    )
    contact.register_rbd(solver)

    # Viewer
    viewer = SimViewer(title="RoboSim — Drop Test (RBD)", window_size=(1200, 800))
    viewer.initialize()
    renderer = RobotRenderer(robot, viewer)
    renderer.setup()

    substeps = 10

    def step_callback(step):
        for _ in range(substeps):
            solver.clear_external_forces()
            wrenches = contact.compute_rbd_contact_forces(solver)
            for link_idx, wrench in wrenches.items():
                solver.set_external_force(link_idx, wrench)
            solver.step()

        renderer.update()

        z = robot.q[0]
        vz = robot.qd[0]
        n_contacts = len(contact.last_forces)
        info = (
            f"t = {solver.time:.3f}s\n"
            f"Height: {z:.4f} m | Vel: {vz:.3f} m/s\n"
            f"Contacts: {n_contacts}"
        )
        viewer.add_text(info)

    viewer.add_callback(step_callback)
    print("RBD sphere drop test — ESC to quit")
    viewer.show()

    print(f"\nFinal height: {robot.q[0]:.4f} m (expected: ~{radius:.4f})")
    print(f"Final velocity: {robot.qd[0]:.6f} m/s")


def run_fem_drop(tilt_deg: float = 0.0):
    """FEM soft cube dropping onto ground plane."""
    from robosim.physics.fem.mesh import TetMesh
    from robosim.physics.fem.materials import CorotationalElastic
    from robosim.physics.fem.solver import FEMSolver, DeformableBody
    from robosim.physics.fem.assembly import batch_von_mises
    from robosim.physics.contact.detection import GroundPlane
    from robosim.physics.contact.solver import ContactSolver
    from robosim.viz.viewer import SimViewer
    from scipy.spatial.transform import Rotation

    side = 0.2
    drop_height = 0.5

    mesh = TetMesh.create_box(
        origin=np.array([-side/2, -side/2, drop_height]),
        size=np.array([side, side, side]),
        divisions=(3, 3, 3),
    )

    # Apply tilt: rotate mesh nodes about CoM
    if tilt_deg != 0.0:
        com = mesh.nodes.mean(axis=0)
        R = Rotation.from_euler('y', tilt_deg, degrees=True).as_matrix()
        mesh.nodes[:] = (R @ (mesh.nodes - com).T).T + com
        print(f"  Tilt: {tilt_deg}° about Y axis")

    body = DeformableBody(
        name="cube",
        mesh=mesh,
        material=CorotationalElastic(young=5e5, poisson=0.3),
        density=1000.0,
    )

    fem_solver = FEMSolver(
        bodies=[body],
        gravity=np.array([0.0, 0.0, -9.81]),
        damping=0.02,
    )
    fem_solver.initialize(dt=0.001)

    # Impulse-based contact — no stiffness/damping tuning needed
    contact = ContactSolver(ground=GroundPlane(height=0.0))

    # Viewer
    viewer = SimViewer(title="RoboSim — Drop Test (FEM)", window_size=(1200, 800))
    viewer.initialize()

    surface_tri = mesh.extract_surface()
    n_faces = surface_tri.shape[0]
    faces = np.column_stack([np.full(n_faces, 3), surface_tri]).ravel()

    # Helper to build color from stress
    def stress_to_color(vm, vmax):
        t = np.clip(vm / max(vmax, 1.0), 0, 1)
        r = np.clip(1.5 - np.abs(t - 0.75) * 4.0, 0, 1)
        g = np.clip(1.5 - np.abs(t - 0.50) * 4.0, 0, 1)
        b = np.clip(1.5 - np.abs(t - 0.25) * 4.0, 0, 1)
        return np.column_stack([r, g, b]).astype(np.float32)

    viewer.add_mesh(
        "cube",
        body.x.copy(),
        surface_tri,
        color=np.array([0.3, 0.6, 0.9]),
    )

    substeps = 15

    def step_callback(step):
        for _ in range(substeps):
            fem_solver.step()
            contact.resolve_fem_contact(body, restitution=0.3, friction_mu=0.5)

        viewer.update_mesh_vertices("cube", body.x.copy())

        vm = batch_von_mises(body.mesh, body.x, body.material,
                             body._dN_list, body._volumes)
        vm_max = max(vm.max(), 1.0)
        viewer.update_mesh_color("cube", stress_to_color(vm, vm_max))

        z_min = body.x[:, 2].min()
        com = body.x.mean(axis=0)
        n_contacts = len(contact.last_forces)
        info = (
            f"t = {fem_solver.time:.3f}s\n"
            f"CoM: ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f})\n"
            f"Z_min: {z_min:.4f} m | Contacts: {n_contacts}\n"
            f"Stress max: {vm_max:.0f} Pa"
        )
        viewer.add_text(info)

    viewer.add_callback(step_callback)
    print("FEM cube drop test — ESC to quit")
    viewer.show()

    print(f"\nFinal Z_min: {body.x[:, 2].min():.4f} m")
    print(f"Final Z_com: {body.x[:, 2].mean():.4f} m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rbd", "fem"], default="rbd")
    parser.add_argument("--tilt", type=float, default=0.0,
                        help="FEM: tilt angle in degrees (e.g. 30)")
    args = parser.parse_args()

    ti.init(arch=ti.metal)

    print("=" * 60)
    print(f"  RoboSim Phase 4: Drop Test ({args.mode.upper()})")
    print("=" * 60)

    if args.mode == "rbd":
        run_rbd_drop()
    else:
        run_fem_drop(tilt_deg=args.tilt)

    print("Phase 4 Contact: OK")


if __name__ == "__main__":
    main()
