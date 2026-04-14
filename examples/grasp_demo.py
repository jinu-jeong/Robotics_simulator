"""Panda 로봇 그리퍼 파지 데모.

Franka Panda 7-DOF 팔이 테이블 위의 박스를 잡고 들어올립니다.
URDF 메시 렌더링, mimic 관절, mesh-box 충돌, 다중 강체 접촉을 시연.

실행:
  python examples/grasp_demo.py                       # GUI primitive
  python examples/grasp_demo.py --mode fem             # GUI FEM (변형 박스)
  python examples/grasp_demo.py --mode cb              # GUI C-B reduced FEM
  python examples/grasp_demo.py --headless             # headless primitive
  python examples/grasp_demo.py --headless --mode fem  # headless FEM
  python examples/grasp_demo.py --headless --mode cb   # headless C-B reduced FEM

조작 (GUI): 좌클릭 드래그(회전), W/S(줌), ESC(종료)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.model.urdf_parser import parse_urdf
from robosim.model.factory import create_free_box
from robosim.physics.rbd.solver import RBDSolver
from robosim.physics.contact.detection import GroundPlane
from robosim.physics.contact.solver import ContactSolver
from robosim.physics.contact.response import ContactParams


PANDA_URDF = Path(__file__).parent / "urdf" / "panda" / "panda.urdf"
TABLE_HEIGHT = 0.15  # table surface height in meters
BOX_SIZE = (0.04, 0.04, 0.04)

# ── Joint configurations ──
HOME_Q = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785, 0.04, 0.04])
PRE_GRASP_Q = np.array([0.0, 0.5, 0.0, -1.6, 0.0, 2.0, 0.785, 0.04, 0.04])
GRASP_Q = np.array([0.0, 0.7, 0.0, -1.6, 0.0, 2.4, 0.785, 0.04, 0.04])
GRASP_CLOSED_Q = np.array([0.0, 0.7, 0.0, -1.6, 0.0, 2.4, 0.785, 0.005, 0.005])
LIFT_Q = np.array([0.0, 0.5, 0.0, -1.6, 0.0, 2.0, 0.785, 0.005, 0.005])


def interpolate_q(q_start, q_end, t, duration):
    """Smooth interpolation between two configurations."""
    s = np.clip(t / duration, 0, 1)
    s = 3 * s**2 - 2 * s**3
    return q_start + s * (q_end - q_start)


# ═══════════════════════════════════════════════════════════════════
# Position-projection finger-box contact (primitive mode)
# ═══════════════════════════════════════════════════════════════════

def _project_finger_generic(robot, fk, box_center, box_rot, box_half, on_push_box):
    """Core position-projection engine shared by rigid-box and FEM modes.

    Parameters
    ----------
    box_center  : (3,) world-frame centre of the effective contact box
    box_rot     : (3,3) box orientation (identity for axis-aligned FEM AABB)
    box_half    : (3,) half-extents of the contact box
    on_push_box : callable(shift: float, grip_axis: ndarray)
                  Translates the box by ``shift * grip_axis`` and clamps
                  its re-penetrating velocity.  Called only when |shift|>0.

    Returns
    -------
    (left_c, right_c, b_ext_grip, grip_axis)
    """
    hand_idx = robot.link_index("panda_hand")
    hand_R = fk[hand_idx].rotation
    grip_axis = hand_R[:, 1]
    side_axes = [hand_R[:, 0], hand_R[:, 2]]

    b_ext_grip = sum(box_half[k] * abs(np.dot(box_rot[:, k], grip_axis))
                     for k in range(3))
    TOUCH = 0.0

    def _overlap(fname):
        link_idx = robot.link_index(fname)
        col = robot.links[link_idx].collisions[0]
        T_col = fk[link_idx].compose(col.origin)
        fc, fR = T_col.translation, T_col.rotation

        verts = col.geometry.mesh_vertices
        if verts is not None:
            v_min = verts.min(axis=0)
            v_max = verts.max(axis=0)
            mesh_center_local = (v_min + v_max) * 0.5
            fh = (v_max - v_min) * 0.5
        else:
            mesh_center_local = np.array([7.68e-6, 0.01314, 0.02699])
            fh = np.array([0.01049, 0.01327, 0.02686])
        fc = fc + fR @ mesh_center_local  # shift to actual mesh centre

        d_to_box = fc - box_center
        d_grip = np.dot(d_to_box, grip_axis)
        f_ext = sum(abs(np.dot(fR[:, k], grip_axis)) * fh[k] for k in range(3))
        for ax in side_axes:
            d_ax = abs(np.dot(d_to_box, ax))
            fe = sum(abs(np.dot(fR[:, k], ax)) * fh[k] for k in range(3))
            be = sum(box_half[k] * abs(np.dot(box_rot[:, k], ax)) for k in range(3))
            if d_ax > fe + be:
                return -np.inf, 1.0
        return f_ext + b_ext_grip - abs(d_grip), (1.0 if d_grip >= 0 else -1.0)

    ov_l, sl = _overlap("panda_leftfinger")
    ov_r, sr = _overlap("panda_rightfinger")

    box_shift = 0.0
    q7_corr   = 0.0

    if ov_l >= 0 and ov_r >= 0 and (ov_l > 0 or ov_r > 0):
        box_shift = (ov_r - ov_l) * 0.5
        q7_corr   = (ov_l + ov_r) * 0.5
    elif ov_r > 0:
        push_cap  = -ov_l
        box_shift = min(ov_r, push_cap)
        q7_corr   = max(0.0, ov_r - push_cap)
    elif ov_l > 0:
        push_cap  = -ov_r
        box_shift = -min(ov_l, push_cap)
        q7_corr   = max(0.0, ov_l - push_cap)

    if abs(box_shift) > 1e-9:
        on_push_box(box_shift, grip_axis)

    if q7_corr > 1e-9:
        robot.q[7] += q7_corr
        robot.qd[7] = 0.0
    robot.enforce_mimic()

    ov_l_f = ov_l + box_shift - q7_corr
    ov_r_f = ov_r - box_shift - q7_corr
    return (ov_l_f >= -TOUCH), (ov_r_f >= -TOUCH), b_ext_grip, grip_axis


def _project_finger_box(robot, fk, target_box, box_fk):
    """Position projection against a rigid free-body box."""
    box_body_idx = len(target_box.links) - 1
    T_box = box_fk[box_body_idx]

    def on_push(shift, grip_ax):
        target_box.q[:3] += shift * grip_ax
        v_n = float(np.dot(target_box.qd[:3], grip_ax))
        if (shift > 0 and v_n < 0) or (shift < 0 and v_n > 0):
            target_box.qd[:3] -= v_n * grip_ax

    return _project_finger_generic(
        robot, fk,
        T_box.translation, T_box.rotation, np.array(BOX_SIZE) / 2.0,
        on_push,
    )


def _project_finger_fem(robot, fk, fem_body):
    """Position projection against a FEM deformable mesh.

    Uses the mesh's current AABB as the effective contact box, so it works
    even when the box has deformed under gripping forces.
    Asymmetric corrections rigidly translate all nodes (no strain introduced).
    """
    nodes = fem_body.x   # (n_nodes, 3) — current node world positions
    box_center = (nodes.min(axis=0) + nodes.max(axis=0)) * 0.5

    def on_push(shift, grip_ax):
        fem_body.x += shift * grip_ax          # rigid translation of all nodes
        v_mean_n = float(np.mean(fem_body.v @ grip_ax))
        if (shift > 0 and v_mean_n < 0) or (shift < 0 and v_mean_n > 0):
            fem_body.v -= v_mean_n * grip_ax   # clamp mean re-penetrating velocity

    return _project_finger_generic(
        robot, fk,
        box_center, np.eye(3), np.array(BOX_SIZE) / 2.0,
        on_push,
    )




# ═══════════════════════════════════════════════════════════════════
# Physics setup (shared robot, mode-specific box)
# ═══════════════════════════════════════════════════════════════════

def setup_physics(mode="primitive"):
    """Create robot, box (rigid or FEM), solvers, contacts."""
    dt = 0.002
    substeps = 5

    print("Loading Panda URDF...")
    robot = parse_urdf(str(PANDA_URDF))
    robot.gravity = np.array([0.0, 0.0, -9.81])
    hand_idx = robot.link_index("panda_hand")

    robot.q = GRASP_Q.copy()
    robot.enforce_mimic()
    fk_grasp = robot.forward_kinematics()
    hand_x = fk_grasp[hand_idx].translation[0]

    box_center_z = TABLE_HEIGHT + BOX_SIZE[2] / 2
    box_pos = np.array([hand_x, 0.0, box_center_z])
    table_pos = np.array([hand_x, 0.0, TABLE_HEIGHT / 2])
    print(f"Table height: {TABLE_HEIGHT}m at x={hand_x:.3f}")
    print(f"Box center:   ({box_pos[0]:.3f}, {box_pos[1]:.3f}, {box_pos[2]:.3f})")

    robot.q = HOME_Q.copy()
    robot.qd = np.zeros(robot.n_dof)
    robot.enforce_mimic()

    # ── Arm solver + ground contact ──
    arm_solver = RBDSolver(robot=robot)
    arm_solver.initialize(dt=dt)

    contact_params = ContactParams(stiffness=5e3, damping=200, friction_mu=0.6)
    contact = ContactSolver(
        ground=GroundPlane(height=0.0),
        params=contact_params,
    )
    contact.register_rbd(arm_solver, robot_id="panda")

    kp = np.array([600, 600, 600, 600, 250, 150, 50, 100, 100], dtype=np.float64)
    kd = np.array([50, 50, 50, 50, 30, 25, 10, 10, 10], dtype=np.float64)
    max_tau = np.array([87, 87, 87, 87, 12, 12, 12, 20, 20], dtype=np.float64)

    total_mass = sum(l.mass for l in robot.links)
    print(f"\nPanda: {robot.n_dof} DOF, {total_mass:.2f} kg total")

    phases = [
        ("home",       0.0,  1.0, HOME_Q,         HOME_Q),
        ("pre_grasp",  1.0,  3.0, HOME_Q,         PRE_GRASP_Q),
        ("reach",      3.0,  5.0, PRE_GRASP_Q,    GRASP_Q),
        ("close",      5.0,  7.0, GRASP_Q,        GRASP_CLOSED_Q),
        ("lift",       7.0, 10.0, GRASP_CLOSED_Q, LIFT_Q),
        ("hold",      10.0, 15.0, LIFT_Q,         LIFT_Q),
    ]

    def get_target_q(t):
        for name, t_start, t_end, q_start, q_end in phases:
            if t < t_end:
                return interpolate_q(q_start, q_end,
                                     t - t_start, t_end - t_start), name
        return LIFT_Q.copy(), "done"

    ctx = dict(
        mode=mode, dt=dt, substeps=substeps,
        robot=robot, arm_solver=arm_solver,
        contact=contact, contact_params=contact_params,
        kp=kp, kd=kd, max_tau=max_tau,
        hand_idx=hand_idx, box_pos=box_pos, table_pos=table_pos,
        get_target_q=get_target_q,
        _finger_contacts=(False, False),
    )

    # ── Mode-specific box setup ──
    if mode == "primitive":
        target_box = create_free_box(
            name="target_box", size=BOX_SIZE, mass=0.2,
            position=box_pos, color=np.array([0.9, 0.3, 0.2, 1.0]),
        )
        box_solver = RBDSolver(robot=target_box)
        box_solver.initialize(dt=dt)

        ctx.update(
            target_box=target_box, box_solver=box_solver,
        )
        print(f"Table: static at z={TABLE_HEIGHT}m, Box: rigid 0.2 kg")

    elif mode == "fem":
        from robosim.physics.fem.mesh import TetMesh
        from robosim.physics.fem.solver import DeformableBody, FEMSolver
        from robosim.physics.fem.materials import CorotationalElastic

        fem_origin = box_pos - np.array(BOX_SIZE) / 2.0
        mesh = TetMesh.create_box(
            origin=fem_origin,
            size=np.array(BOX_SIZE),
            divisions=(3, 3, 3),
        )
        fem_body = DeformableBody(
            name="fem_box",
            mesh=mesh,
            material=CorotationalElastic(young=1e4, poisson=0.3),
            density=1000.0,
        )
        fem_solver = FEMSolver(
            bodies=[fem_body],
            gravity=np.array([0.0, 0.0, -9.81]),
            damping=20.0,   # heavy Rayleigh damping — damps rigid-body rotation in ~2s
            max_newton_iters=3,
        )
        fem_solver.initialize(dt=dt)

        # Register FEM with robot contact solver for RBD-FEM detection
        contact.register_fem(fem_solver)

        # Separate ground contact for FEM at table height
        fem_ground_contact = ContactSolver(
            ground=GroundPlane(height=TABLE_HEIGHT),
            params=ContactParams(),
        )

        surface_tri = mesh.extract_surface()

        surface_nodes = np.unique(surface_tri)   # indices of surface nodes
        ctx.update(
            fem_solver=fem_solver, fem_body=fem_body,
            fem_ground_contact=fem_ground_contact,
            surface_tri=surface_tri,
            surface_nodes=surface_nodes,
        )
        print(f"Table: static at z={TABLE_HEIGHT}m, "
              f"Box: FEM {mesh.n_nodes} nodes, {mesh.n_elements} tets, "
              f"E=1e4")

    elif mode == "cb":
        from robosim.physics.fem.mesh import TetMesh
        from robosim.physics.fem.materials import CorotationalElastic
        from robosim.physics.fem.reduced import CraigBamptonBody, CraigBamptonSolver

        fem_origin = box_pos - np.array(BOX_SIZE) / 2.0
        mesh = TetMesh.create_box(
            origin=fem_origin,
            size=np.array(BOX_SIZE),
            divisions=(3, 3, 3),
        )
        fem_body = CraigBamptonBody(
            mesh=mesh,
            material=CorotationalElastic(young=1e4, poisson=0.3),
            density=1000.0,
            n_modes=6,
            gravity=np.array([0.0, 0.0, -9.81]),
            damping=20.0,
            name="cb_box",
        )
        fem_solver = CraigBamptonSolver(bodies=[fem_body])
        fem_solver.initialize(dt=dt)

        # Register with robot contact solver (duck-types as FEMSolver)
        contact.register_fem(fem_solver)

        # Separate ground contact for FEM at table height
        fem_ground_contact = ContactSolver(
            ground=GroundPlane(height=TABLE_HEIGHT),
            params=ContactParams(),
        )

        surface_tri = mesh.extract_surface()
        surface_nodes = np.unique(surface_tri)   # indices of surface nodes
        ctx.update(
            fem_solver=fem_solver, fem_body=fem_body,
            fem_ground_contact=fem_ground_contact,
            surface_tri=surface_tri,
            surface_nodes=surface_nodes,
        )
        print(f"Table: static at z={TABLE_HEIGHT}m, "
              f"Box: C-B {mesh.n_nodes} nodes, {mesh.n_elements} tets, "
              f"n_r={fem_body._n_r} ({fem_body._n_b} bdry + {fem_body._n_modes_actual} modes)")

    return ctx


# ═══════════════════════════════════════════════════════════════════
# Physics step — primitive (rigid body box)
# ═══════════════════════════════════════════════════════════════════

def step_physics_primitive(ctx, sim_time):
    """One frame of primitive (rigid box) physics — position-projection contact."""
    from robosim.physics.contact.solver import _link_point_velocity

    robot = ctx["robot"]
    target_box = ctx["target_box"]
    arm_solver = ctx["arm_solver"]
    box_solver = ctx["box_solver"]
    contact = ctx["contact"]
    kp, kd, max_tau = ctx["kp"], ctx["kd"], ctx["max_tau"]
    dt = ctx["dt"]
    get_target_q = ctx["get_target_q"]
    box_body_idx = len(target_box.links) - 1
    hand_idx = ctx["hand_idx"]
    l_idx = robot.link_index("panda_leftfinger")
    r_idx = robot.link_index("panda_rightfinger")
    table_min_z = TABLE_HEIGHT + BOX_SIZE[2] / 2.0

    phase_name = "home"
    for _ in range(ctx["substeps"]):
        q_des, phase_name = get_target_q(sim_time)

        # 1. PD control + gravity compensation
        tau = kp * (q_des - robot.q) + kd * (-robot.qd)
        tau += arm_solver.compute_gravity_torques()
        tau = np.clip(tau, -max_tau, max_tau)
        arm_solver.tau = tau

        # 2. Ground contact forces + arm dynamics step
        panda_ground = contact.compute_rbd_contact_forces(arm_solver)
        arm_solver.clear_external_forces()
        for li, w in panda_ground.items():
            arm_solver.set_external_force(li, w)
        arm_solver.step()

        # 3. Position projection: push master finger joint outward to eliminate
        #    any box penetration. enforce_mimic() inside _project_finger_box
        #    keeps the follower joint consistent.
        #    Active from "reach" onward so any accidental contact during approach
        #    is also resolved (fingers are wide open then, so no false triggers).
        fk_now = robot.forward_kinematics()
        box_fk_now = target_box.forward_kinematics()

        if phase_name not in ("home", "pre_grasp"):
            left_c, right_c, b_ext_grip, grip_ax = _project_finger_box(
                robot, fk_now, target_box, box_fk_now)
            if left_c or right_c:
                fk_now = robot.forward_kinematics()        # refresh after q correction
                box_fk_now = target_box.forward_kinematics()  # refresh after box push
        else:
            left_c = right_c = False
            b_ext_grip = float(np.array(BOX_SIZE)[1]) / 2.0   # fallback, unused
            grip_ax = fk_now[hand_idx].rotation[:, 1]          # fallback, unused

        ctx["_finger_contacts"] = (left_c, right_c)

        # 4. Box dynamics
        if left_c and right_c:
            # Both fingers in contact: kinematic coupling — box tracks fingers.
            # Velocity is evaluated at the actual contact points (box edges along
            # the grip axis) rather than the box centre, which is more accurate
            # when the wrist rotates during the lift.
            box_c = box_fk_now[box_body_idx].translation
            cp_l = box_c + b_ext_grip * grip_ax   # left-finger contact point
            cp_r = box_c - b_ext_grip * grip_ax   # right-finger contact point
            vl = _link_point_velocity(robot, l_idx, cp_l, fk=fk_now)
            vr = _link_point_velocity(robot, r_idx, cp_r, fk=fk_now)
            avg_v = 0.5 * (vl + vr)
            target_box.q[:3] += avg_v * dt
            target_box.qd[:3] = avg_v
            target_box.qd[3:] = 0.0

            # Re-project after kinematic move to keep contact tight within the
            # same substep (corrects any asymmetric drift between the two fingers).
            box_fk_post = target_box.forward_kinematics()
            _project_finger_box(robot, fk_now, target_box, box_fk_post)
        else:
            # No full grip: free fall + table
            box_solver.clear_external_forces()
            box_solver.tau = np.zeros(6)
            box_solver.step()

        # 5. Table constraint — inelastic, no bounce
        if target_box.q[2] < table_min_z:
            target_box.q[2] = table_min_z
            if target_box.qd[2] < 0:
                target_box.qd[2] = 0.0
            v_xy = target_box.qd[:2]
            v_mag = float(np.linalg.norm(v_xy))
            if v_mag > 1e-6:
                decel = min(0.5 * 9.81, v_mag / dt)
                target_box.qd[:2] -= decel * (v_xy / v_mag) * dt

        sim_time += dt

    return sim_time, phase_name


# ═══════════════════════════════════════════════════════════════════
# Physics step — FEM (deformable box)
# ═══════════════════════════════════════════════════════════════════

def step_physics_fem(ctx, sim_time):
    """One frame of FEM (deformable box) physics.

    Finger contact uses the same position-projection + kinematic-coupling
    strategy as primitive mode:
      • Position projection (_project_finger_fem) prevents gross penetration
        by clamping q[7] outward and rigidly translating FEM nodes.
      • When both fingers contact, kinematic coupling translates all FEM nodes
        with the average finger velocity (same rigid-body transport used for
        the primitive box).  The FEM integrator is skipped during full contact
        to avoid double-counting the transport displacement.
      • When only one finger (or no finger) contacts, the FEM integrator runs
        freely (gravity + elasticity + table ground contact).

    Note: detect_rbd_fem skips MESH collision geometry, so the penalty-based
    FEM contact path is not used for the finger STLs.  Position projection
    handles all finger-box non-penetration.
    """
    from robosim.physics.contact.solver import _link_point_velocity

    robot = ctx["robot"]
    arm_solver = ctx["arm_solver"]
    contact = ctx["contact"]
    fem_solver = ctx["fem_solver"]
    fem_body = ctx["fem_body"]
    fem_ground_contact = ctx["fem_ground_contact"]
    surface_nodes = ctx["surface_nodes"]
    hand_idx = ctx["hand_idx"]
    kp, kd, max_tau = ctx["kp"], ctx["kd"], ctx["max_tau"]
    dt = ctx["dt"]
    get_target_q = ctx["get_target_q"]
    l_idx = robot.link_index("panda_leftfinger")
    r_idx = robot.link_index("panda_rightfinger")

    phase_name = "home"
    for _ in range(ctx["substeps"]):
        q_des, phase_name = get_target_q(sim_time)

        # 1. Robot PD + gravity comp (fingers PD-controlled, no kinematic override)
        tau = kp * (q_des - robot.q) + kd * (-robot.qd)
        tau += arm_solver.compute_gravity_torques()
        tau = np.clip(tau, -max_tau, max_tau)
        arm_solver.tau = tau

        # 2. Robot ground contact + arm dynamics step
        panda_ground = contact.compute_rbd_contact_forces(arm_solver)
        arm_solver.clear_external_forces()
        for li, w in panda_ground.items():
            arm_solver.set_external_force(li, w)
        arm_solver.step()

        # 3. Coarse AABB projection: keeps fingers from grossly penetrating the
        #    FEM mesh and detects left/right contact state.
        fk_now = robot.forward_kinematics()
        if phase_name not in ("home", "pre_grasp"):
            left_c, right_c, b_ext_grip, grip_ax = _project_finger_fem(
                robot, fk_now, fem_body)
            if left_c or right_c:
                fk_now = robot.forward_kinematics()
        else:
            left_c = right_c = False
            b_ext_grip = float(np.array(BOX_SIZE)[1]) / 2.0
            grip_ax = fk_now[hand_idx].rotation[:, 1]

        ctx["_finger_contacts"] = (left_c, right_c)

        # 4. Penalty contact forces: apply inward force to surface nodes on each
        #    finger contact face, proportional to how much the PD controller
        #    wants to close further (robot.q[7] − q_des[7]).  This drives
        #    elastic deformation through the FEM integrator rather than hard
        #    position snapping, so the elastic response is physically consistent.
        #    k_press = 30 N/m per node gives ~3 mm squeeze for E=5e4.
        f_contact = None
        if left_c and right_c:
            box_c_pre = fem_body.x.mean(axis=0)   # centroid — stable under deformation
            cp_l = box_c_pre + b_ext_grip * grip_ax
            cp_r = box_c_pre - b_ext_grip * grip_ax
            vl = _link_point_velocity(robot, l_idx, cp_l, fk=fk_now)
            vr = _link_point_velocity(robot, r_idx, cp_r, fk=fk_now)
            avg_v = 0.5 * (vl + vr)

            # Kinematic coupling: shift mean node velocity to arm transport.
            v_mean = fem_body.v.mean(axis=0)
            fem_body.v += (avg_v - v_mean)

            press = max(0.0, float(robot.q[7] - q_des[7]))
            if press > 1e-4:
                k_press = 6.0           # N/(m·node) — tuned for E=1e4 (~3mm squeeze)
                threshold = b_ext_grip * 0.5  # outer-half filter along grip axis
                n_nodes = len(fem_body.x)
                f_arr = np.zeros(n_nodes * 3)
                for ni in surface_nodes:
                    d = float(np.dot(fem_body.x[ni] - box_c_pre, grip_ax))
                    if d > threshold:   # left-finger contact face
                        f_arr[ni*3:ni*3+3] -= k_press * press * grip_ax
                    elif d < -threshold:  # right-finger contact face
                        f_arr[ni*3:ni*3+3] += k_press * press * grip_ax
                f_contact = {0: f_arr}

        # 5. FEM step: elastic + gravity (+ optional contact forces).
        fem_solver.step(extra_forces=f_contact)

        if left_c and right_c:
            # Correct transport drift so the box centroid follows the arm precisely.
            box_c_post = fem_body.x.mean(axis=0)
            box_c_target = box_c_pre + avg_v * dt
            fem_body.x += box_c_target - box_c_post

        # 6. FEM ground contact at table height (restitution=0 → no bounce)
        fem_ground_contact.resolve_fem_contact(
            fem_body, restitution=0.0, friction_mu=0.5)

        sim_time += dt

    ctx["_n_fem_contacts"] = int(left_c) + int(right_c)
    return sim_time, phase_name


def step_physics(ctx, sim_time, **kwargs):
    """Dispatch to primitive or FEM/CB step."""
    if ctx["mode"] in ("fem", "cb"):
        return step_physics_fem(ctx, sim_time, **kwargs)
    else:
        return step_physics_primitive(ctx, sim_time, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# Headless mode
# ═══════════════════════════════════════════════════════════════════

def run_headless(mode: str, n_frames: int = 1500):
    """Run physics-only loop without visualization. Reports FPS."""
    ctx = setup_physics(mode=mode)
    dt = ctx["dt"]
    substeps = ctx["substeps"]
    robot = ctx["robot"]
    hand_idx = ctx["hand_idx"]

    print(f"\nMode: HEADLESS {mode.upper()} ({n_frames} frames, "
          f"dt={dt}, substeps={substeps})")
    print("-" * 60)

    sim_time = 0.0
    log_interval = 500

    wall_start = time.perf_counter()
    lap_start = wall_start
    lap_frames = 0

    for frame in range(n_frames):
        sim_time, phase_name = step_physics(ctx, sim_time)
        lap_frames += 1

        if (frame + 1) % log_interval == 0 or frame == n_frames - 1:
            now = time.perf_counter()
            lap_dt = now - lap_start
            lap_fps = lap_frames / lap_dt if lap_dt > 0 else 0
            rt_factor = (lap_frames * substeps * dt) / lap_dt if lap_dt > 0 else 0

            if mode == "primitive":
                box_z = ctx["target_box"].q[2]
            else:
                box_z = ctx["fem_body"].x.mean(axis=0)[2]

            extra = ""
            if mode in ("fem", "cb"):
                extra = f"  fem_contacts={ctx.get('_n_fem_contacts', 0)}"

            print(f"  frame {frame+1:5d}/{n_frames}  "
                  f"sim_t={sim_time:.3f}s  "
                  f"phase={phase_name:<10s}  "
                  f"FPS={lap_fps:.1f}  "
                  f"RT={rt_factor:.2f}x  "
                  f"box_z={box_z:.4f}m{extra}")

            lap_start = now
            lap_frames = 0

    wall_total = time.perf_counter() - wall_start
    avg_fps = n_frames / wall_total
    total_sim_time = n_frames * substeps * dt

    if mode == "primitive":
        box_z = ctx["target_box"].q[2]
    else:
        box_z = ctx["fem_body"].x.mean(axis=0)[2]

    print("-" * 60)
    print(f"Done: {n_frames} frames in {wall_total:.2f}s")
    print(f"  Avg FPS:       {avg_fps:.1f}")
    print(f"  Avg frame:     {wall_total/n_frames*1000:.2f} ms")
    print(f"  Sim time:      {total_sim_time:.2f}s")
    print(f"  Real-time:     {total_sim_time/wall_total:.2f}x")
    print(f"  Final box Z:   {box_z:.4f}m")
    finite_ok = np.all(np.isfinite(robot.q))
    if mode in ("fem", "cb"):
        finite_ok = finite_ok and np.all(np.isfinite(ctx["fem_body"].x))
    else:
        finite_ok = finite_ok and np.all(np.isfinite(ctx["target_box"].q))
    print(f"  Finite check:  {'OK' if finite_ok else 'FAIL'}")


# ═══════════════════════════════════════════════════════════════════
# GUI mode
# ═══════════════════════════════════════════════════════════════════

def run_gui(mode: str):
    """Interactive GUI loop with visualization."""
    import taichi as ti
    from robosim.viz.viewer import SimViewer, geometry_to_trimesh
    from robosim.viz.scene_renderer import RobotRenderer
    from robosim.viz.ui_panels import UIPanel
    from robosim.viz.overlays import ContactForceOverlay
    from robosim.model.geometry import Geometry as _Geo

    ti.init(arch=ti.metal)

    ctx = setup_physics(mode=mode)
    dt = ctx["dt"]
    robot = ctx["robot"]
    contact = ctx["contact"]
    hand_idx = ctx["hand_idx"]
    box_pos = ctx["box_pos"]
    table_pos = ctx["table_pos"]

    viewer = SimViewer(
        title=f"RoboSim — Panda Grasp ({mode.upper()})",
        window_size=(1280, 800),
    )
    viewer.initialize()

    renderer = RobotRenderer(robot, viewer)
    renderer.setup()
    # Only show finger collision meshes — the only links that directly
    # contact the box (arm links only hit the ground plane).
    renderer.set_contact_links(["panda_leftfinger", "panda_rightfinger"])

    # Table
    _table_geo = _Geo.box(0.30, 0.30, TABLE_HEIGHT)
    tv, tf = geometry_to_trimesh(_table_geo)
    viewer.add_mesh("table", tv + table_pos, tf,
                    color=np.array([0.6, 0.5, 0.4]))

    # Box visualization (mode-specific)
    if mode == "primitive":
        target_box = ctx["target_box"]
        box_body = target_box.links[-1]
        bv, bf = geometry_to_trimesh(box_body.visuals[0].geometry)
        viewer.add_mesh("target_box", bv + box_pos, bf,
                        color=np.array([0.9, 0.3, 0.2]))
        # Collision mesh for target box (wireframe, hidden by default)
        col = box_body.collisions[0]
        cv, cf = geometry_to_trimesh(col.geometry)
        col_mat = col.origin.to_matrix()
        cv = (col_mat[:3, :3] @ cv.T).T + col_mat[:3, 3]
        viewer.add_mesh("target_box_col", cv + box_pos, cf,
                        color=np.array([0.2, 0.8, 0.3]))
        viewer.set_meshes_wireframe_by_prefix("target_box_col", True)
        viewer.set_mesh_visible("target_box_col", False)
    else:
        fem_body = ctx["fem_body"]
        surface_tri = ctx["surface_tri"]
        viewer.add_mesh("fem_box", fem_body.x.copy(), surface_tri,
                        color=np.array([1.0, 0.35, 0.1]))

    # UI
    panel = UIPanel("Grasp Controls")
    panel.add_checkbox("Show Forces", False)
    panel.add_checkbox("Show Collision", False)

    force_overlay = ContactForceOverlay(
        viewer, scale=0.0003, max_arrows=20,
        color=np.array([1.0, 0.8, 0.0]), min_force=0.5,
    )

    sim_time = 0.0
    frame_count = 0

    def step_callback(step):
        nonlocal sim_time, frame_count

        force_overlay.active = panel.get_bool("Show Forces")
        show_col = panel.get_bool("Show Collision")
        renderer.show_collision(show_col)

        import time as _t
        panel.playback.update_fps(_t.time())

        kwargs = {}

        sim_time_new, phase_name = step_physics(ctx, sim_time, **kwargs)
        sim_time = sim_time_new

        # ── Update visuals ──
        fk_render = robot.forward_kinematics()
        renderer.update(fk=fk_render)

        if mode == "primitive":
            target_box = ctx["target_box"]
            fk_b = target_box.forward_kinematics()
            T_box = fk_b[len(target_box.links) - 1]
            mat = T_box.to_matrix()
            R, t = mat[:3, :3], mat[:3, 3]
            viewer.update_mesh_vertices("target_box", (R @ bv.T).T + t)
            viewer.set_mesh_visible("target_box", not show_col)
            viewer.update_mesh_vertices("target_box_col", (R @ cv.T).T + t)
            viewer.set_mesh_visible("target_box_col", show_col)
            box_z = target_box.q[2]
            left_c, right_c = ctx.get("_finger_contacts", (False, False))
            grip_str = ("L" if left_c else "-") + ("R" if right_c else "-")
            extra_info = f"Grip: {grip_str}  Ground: {len(contact.last_forces)}"
        else:
            viewer.update_mesh_vertices("fem_box", ctx["fem_body"].x)
            box_z = ctx["fem_body"].x.mean(axis=0)[2]
            extra_info = f"FEM contacts: {ctx.get('_n_fem_contacts', 0)}"

        force_overlay.update(contact.last_forces)

        hand_now = fk_render[hand_idx].translation
        finger_q = robot.q[7]

        panel.playback.sim_time = sim_time
        panel.set_info([
            f"Phase: {phase_name}",
            f"Hand: ({hand_now[0]:+.3f}, {hand_now[1]:+.3f}, {hand_now[2]:+.3f})",
            f"Finger: {finger_q*1000:.1f}mm",
            f"Box Z: {box_z:.4f}m",
            extra_info,
        ])
        panel.render(viewer._window)
        frame_count += 1

    viewer.add_callback(step_callback)
    print(f"\nPanda Grasp Demo ({mode.upper()}) — ESC to quit")
    viewer.show()

    if mode == "primitive":
        box_z = ctx["target_box"].q[2]
    else:
        box_z = ctx["fem_body"].x.mean(axis=0)[2]
    print(f"\nFinal box Z: {box_z:.4f}m")
    print(f"Simulated {sim_time:.2f}s in {frame_count} frames")



def main():
    parser = argparse.ArgumentParser(
        description="Panda gripper grasp demo")
    parser.add_argument(
        "--mode", choices=["primitive", "fem", "cb"], default="primitive",
        help="Box type: 'primitive' (rigid), 'fem' (full deformable), "
             "or 'cb' (Craig-Bampton reduced deformable)")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without visualization (physics only)")
    parser.add_argument(
        "-n", "--frames", type=int, default=1500,
        help="Number of frames in headless mode (default: 1500)")
    args = parser.parse_args()

    print("=" * 60)
    print("  RoboSim: Panda Gripper Grasp Demo")
    print(f"  Mode: {args.mode.upper()}"
          f"{' (headless)' if args.headless else ''}")
    print("=" * 60)

    if args.headless:
        run_headless(mode=args.mode, n_frames=args.frames)
    else:
        run_gui(mode=args.mode)


if __name__ == "__main__":
    main()
