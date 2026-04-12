"""Phase 2 검증 데모: 3D 단진자 시뮬레이션.

단진자가 중력 하에 자유 진동하는 모습을 3D 뷰어에서 실시간으로 확인합니다.
에너지 보존 상태도 오버레이로 표시됩니다.

실행: python examples/simple_pendulum.py

조작:
  - 마우스 좌클릭 드래그: 카메라 회전
  - W/S: 줌 인/아웃
  - A/D: 좌/우, E/Q: 상/하
  - ESC: 종료
"""

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
from robosim.model.joint import Joint, JointType
from robosim.model.link import Collision, Link, Visual
from robosim.model.robot import Robot
from robosim.physics.rbd.solver import RBDSolver
from robosim.viz.viewer import SimViewer
from robosim.viz.scene_renderer import RobotRenderer


def make_visual_pendulum(length: float = 0.8, mass: float = 2.0) -> Robot:
    """Create a pendulum with visual geometries for 3D rendering."""
    # Base (fixed support)
    base = Link(
        name="base",
        inertial=SpatialInertia(mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3))),
        visuals=[
            Visual(
                geometry=Geometry.sphere(0.04),
                origin=Transform.identity(),
                color=np.array([0.3, 0.3, 0.3, 1.0]),
            )
        ],
    )

    # Pendulum rod + bob
    rod_inertia = (1.0 / 3.0) * mass * length**2
    bob = Link(
        name="bob",
        inertial=SpatialInertia(
            mass=mass,
            com=np.array([0.0, 0.0, -length / 2]),
            inertia=np.diag([rod_inertia, rod_inertia, 0.001]),
        ),
        visuals=[
            # Rod (cylinder along -Z)
            Visual(
                geometry=Geometry.cylinder(0.015, length),
                origin=Transform.from_translation(np.array([0.0, 0.0, -length / 2])),
                color=np.array([0.4, 0.6, 0.9, 1.0]),
            ),
            # Bob sphere at tip
            Visual(
                geometry=Geometry.sphere(0.04),
                origin=Transform.from_translation(np.array([0.0, 0.0, -length])),
                color=np.array([0.9, 0.3, 0.2, 1.0]),
            ),
        ],
    )

    joint = Joint(
        name="pivot",
        joint_type=JointType.REVOLUTE,
        parent_link="base",
        child_link="bob",
        axis=np.array([0.0, 1.0, 0.0]),
        origin=Transform.from_translation(np.array([0.0, 0.0, 1.0])),  # pivot at z=1
    )

    robot = Robot(name="pendulum", links=[base, bob], joints=[joint])
    robot.gravity = np.array([0.0, 0.0, -9.81])
    robot.build()
    return robot


def main():
    ti.init(arch=ti.metal)

    print("=" * 60)
    print("  RoboSim Phase 2: Simple Pendulum (3D)")
    print("=" * 60)

    # Create pendulum
    robot = make_visual_pendulum(length=0.8, mass=2.0)
    robot.q = np.array([math.radians(60)])  # start at 60 degrees
    robot.qd = np.array([0.0])

    # Physics solver
    dt = 0.002
    solver = RBDSolver(robot=robot)
    solver.initialize(dt=dt)

    E0 = solver.total_energy()
    print(f"Initial angle:  {math.degrees(robot.q[0]):.1f} deg")
    print(f"Initial energy: {E0:.4f} J")

    # Visualization
    viewer = SimViewer(title="RoboSim - Simple Pendulum", window_size=(1200, 800))
    viewer.initialize()

    renderer = RobotRenderer(robot, viewer)
    renderer.setup()

    # Simulation parameters
    physics_substeps = 5

    def step_callback(step):
        for _ in range(physics_substeps):
            solver.step()

        renderer.update()

        E = solver.total_energy()
        KE = solver.kinetic_energy()
        PE = solver.potential_energy()
        q_deg = math.degrees(robot.q[0])
        drift = abs(E - E0) / abs(E0) * 100

        info = (
            f"t = {solver.time:.2f}s | angle = {q_deg:+.1f} deg\n"
            f"KE = {KE:.3f} | PE = {PE:.3f} | Total = {E:.3f} J\n"
            f"Energy drift: {drift:.4f}%"
        )
        viewer.add_text(info)

    viewer.add_callback(step_callback, interval=16)

    print("\nSimulation running... (ESC to quit)")
    viewer.show()

    # Final report
    E_final = solver.total_energy()
    drift_final = abs(E_final - E0) / abs(E0) * 100
    print(f"\nSimulation ended at t = {solver.time:.2f}s")
    print(f"Final energy: {E_final:.4f} J (drift: {drift_final:.4f}%)")
    print("Phase 2 RBD + Visualization: OK")


if __name__ == "__main__":
    main()
