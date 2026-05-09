"""Phase 1 검증 데모: URDF 로드 및 kinematic tree 출력.

실행: python examples/load_urdf_demo.py
"""

import math
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.model.urdf_parser import parse_urdf


def main():
    urdf_path = Path(__file__).parent / "urdf" / "simple_arm.urdf"
    print(f"Loading URDF: {urdf_path}")
    print("=" * 60)

    robot = parse_urdf(urdf_path)
    print(f"\nRobot: {robot}")
    print(f"  Links:  {robot.n_links}")
    print(f"  Joints: {robot.n_joints}")
    print(f"  DOF:    {robot.n_dof}")

    print("\n--- Kinematic Tree ---")
    robot.print_tree()

    print("\n--- Link Details ---")
    for link in robot.links:
        print(f"  {link.name}: mass={link.mass:.3f} kg, "
              f"visuals={len(link.visuals)}, collisions={len(link.collisions)}")

    print("\n--- Joint Details ---")
    for joint in robot.joints:
        print(f"  {joint.name}: type={joint.joint_type.value}, "
              f"axis={joint.axis}, damping={joint.damping}")
        if joint.limits:
            print(f"    limits: [{joint.limits.lower:.2f}, {joint.limits.upper:.2f}]")

    # Forward kinematics at zero configuration
    print("\n--- Forward Kinematics (q=0) ---")
    fk = robot.forward_kinematics()
    for i, link in enumerate(robot.links):
        t = fk[i].translation
        r, p, y = fk[i].to_rpy()
        print(f"  {link.name}: pos=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}], "
              f"rpy=[{math.degrees(r):.1f}, {math.degrees(p):.1f}, {math.degrees(y):.1f}]°")

    # Forward kinematics with some joint angles
    print("\n--- Forward Kinematics (shoulder=45°, elbow=30°, wrist=-20°) ---")
    robot.q = np.array([math.radians(45), math.radians(30), math.radians(-20)])
    fk = robot.forward_kinematics()
    for i, link in enumerate(robot.links):
        t = fk[i].translation
        print(f"  {link.name}: pos=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")

    # End-effector position
    hand_idx = robot.link_index("hand")
    ee_pos = fk[hand_idx].translation
    print(f"\n  End-effector (hand) position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")

    print("\n" + "=" * 60)
    print("Phase 1 Foundation: OK")


if __name__ == "__main__":
    main()
