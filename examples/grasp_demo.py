"""Panda 로봇 그리퍼 파지 데모.

Franka Panda 7-DOF 팔이 테이블 위의 박스를 잡고 들어올립니다.
URDF 메시 렌더링, mimic 관절, mesh-box 충돌, 다중 강체 접촉을 시연.

실행:
  python examples/grasp_demo.py                       # GUI primitive
  python examples/grasp_demo.py --mode fem             # GUI FEM (변형 박스)
  python examples/grasp_demo.py --headless             # headless primitive
  python examples/grasp_demo.py --headless --mode fem  # headless FEM

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
# Analytical finger-box forces (primitive mode only)
# ═══════════════════════════════════════════════════════════════════

def compute_finger_box_forces(
    robot, fk, target_box, box_solver,
    stiffness=5e3, damping=200, friction_mu=0.6,
    box_fk=None,
):
    """Analytical parallel-jaw grasp forces (primitive mode)."""
    from robosim.physics.contact.solver import _link_point_velocity

    if box_fk is None:
        box_fk = target_box.forward_kinematics()
    box_body_idx = len(target_box.links) - 1
    T_box = box_fk[box_body_idx]
    box_center = T_box.translation
    box_half = np.array(BOX_SIZE) / 2.0
    fh = np.array([0.021, 0.026, 0.054]) / 2.0

    finger_data = {}
    for fname, sign in [("panda_leftfinger", -1), ("panda_rightfinger", +1)]:
        idx = robot.link_index(fname)
        col = robot.links[idx].collisions[0]
        T_col = fk[idx].compose(col.origin)
        fc = T_col.translation
        fR = T_col.rotation

        y_ext = sum(abs(fR[:, k][1]) * fh[k] for k in range(3))
        f_y_min = fc[1] - y_ext
        f_y_max = fc[1] + y_ext
        b_y_min = box_center[1] - box_half[1]
        b_y_max = box_center[1] + box_half[1]
        overlap_y = min(f_y_max, b_y_max) - max(f_y_min, b_y_min)
        if overlap_y <= 0:
            continue

        ok = True
        for ax in [0, 2]:
            f_ext = sum(abs(fR[:, k][ax]) * fh[k] for k in range(3))
            if abs(fc[ax] - box_center[ax]) > f_ext + box_half[ax]:
                ok = False
                break
        if not ok:
            continue
        finger_data[fname] = (idx, fc, fR, overlap_y, sign)

    if len(finger_data) < 2:
        return {}, np.zeros(6)

    finger_wrenches = {}
    box_wrench = np.zeros(6)

    for fname, (idx, fc, fR, overlap_y, sign) in finger_data.items():
        pen = min(overlap_y, 0.01)
        contact_y = (box_center[1] + box_half[1]) if sign > 0 else (box_center[1] - box_half[1])
        contact_pt = np.array([box_center[0], contact_y, box_center[2]])
        normal = np.array([0.0, float(sign), 0.0])

        v_finger = _link_point_velocity(robot, idx, contact_pt, fk=fk)
        v_box = _link_point_velocity(target_box, box_body_idx, contact_pt, fk=box_fk)
        v_rel = v_finger - v_box
        v_n = float(np.dot(v_rel, normal))

        eff_mass = min(robot.links[idx].mass,
                       target_box.links[box_body_idx].mass)
        c = min(damping, 2.0 * np.sqrt(stiffness * eff_mass))

        fn_mag = stiffness * pen - c * v_n
        if fn_mag <= 0:
            continue
        f_normal = fn_mag * normal

        friction_ramp = min(1.0, pen / 0.005)
        effective_mu = friction_mu * friction_ramp
        v_t = v_rel - v_n * normal
        v_t_norm = np.linalg.norm(v_t)
        f_friction = np.zeros(3)
        if v_t_norm > 1e-12:
            scale = min(1.0, v_t_norm / 1e-3)
            f_friction = -effective_mu * fn_mag * scale * (v_t / v_t_norm)

        f_on_finger = f_normal + f_friction
        f_on_box = -f_on_finger

        T_inv = fk[idx].inverse()
        fl = T_inv.apply_vector(f_on_finger)
        pl = T_inv.apply_point(contact_pt)
        finger_wrenches[idx] = np.concatenate([np.cross(pl, fl), fl])

        T_inv_b = box_fk[box_body_idx].inverse()
        fb = T_inv_b.apply_vector(f_on_box)
        pb = T_inv_b.apply_point(contact_pt)
        box_wrench += np.concatenate([np.cross(pb, fb), fb])

    return finger_wrenches, box_wrench


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

    from robosim.model.geometry import Geometry
    from robosim.model.link import Collision
    from robosim.math.transforms import Transform
    finger_box_size = (0.021, 0.026, 0.054)
    finger_origins = {
        "panda_leftfinger": Transform(
            translation=np.array([0.0, 0.013, 0.027]),
            rotation=np.eye(3),
        ),
        "panda_rightfinger": Transform(
            translation=np.array([0.0, -0.013, 0.027]),
            rotation=np.eye(3),
        ),
    }
    for finger_name, origin in finger_origins.items():
        idx = robot.link_index(finger_name)
        robot.links[idx].collisions = [Collision(
            geometry=Geometry.box(*finger_box_size),
            origin=origin,
        )]

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
    )

    # ── Mode-specific box setup ──
    if mode == "primitive":
        target_box = create_free_box(
            name="target_box", size=BOX_SIZE, mass=0.2,
            position=box_pos, color=np.array([0.9, 0.3, 0.2, 1.0]),
        )
        box_solver = RBDSolver(robot=target_box)
        box_solver.initialize(dt=dt)

        box_table_params = ContactParams(stiffness=5e3, damping=200, friction_mu=0.6)
        box_table_contact = ContactSolver(
            ground=GroundPlane(height=TABLE_HEIGHT),
            params=box_table_params,
        )
        box_table_contact.register_rbd(box_solver, robot_id="target_box")

        ctx.update(
            target_box=target_box, box_solver=box_solver,
            box_table_contact=box_table_contact,
            box_table_params=box_table_params,
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
            material=CorotationalElastic(young=5e4, poisson=0.3),
            density=1000.0,
        )
        fem_solver = FEMSolver(
            bodies=[fem_body],
            gravity=np.array([0.0, 0.0, -9.81]),
            damping=0.05,
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

        ctx.update(
            fem_solver=fem_solver, fem_body=fem_body,
            fem_ground_contact=fem_ground_contact,
            surface_tri=surface_tri,
        )
        print(f"Table: static at z={TABLE_HEIGHT}m, "
              f"Box: FEM {mesh.n_nodes} nodes, {mesh.n_elements} tets, "
              f"E=5e4")

    return ctx


# ═══════════════════════════════════════════════════════════════════
# Physics step — primitive (rigid body box)
# ═══════════════════════════════════════════════════════════════════

def step_physics_primitive(ctx, sim_time, grasp_k=5e3, grasp_c=200, grasp_mu=0.6):
    """One frame of primitive (rigid box) physics."""
    from robosim.physics.contact.solver import _link_point_velocity

    robot = ctx["robot"]
    target_box = ctx["target_box"]
    arm_solver = ctx["arm_solver"]
    box_solver = ctx["box_solver"]
    contact = ctx["contact"]
    box_table_contact = ctx["box_table_contact"]
    kp, kd, max_tau = ctx["kp"], ctx["kd"], ctx["max_tau"]
    dt = ctx["dt"]
    get_target_q = ctx["get_target_q"]
    finger_dof = [7, 8]
    box_body_idx = len(target_box.links) - 1

    ctx["contact_params"].stiffness = grasp_k
    ctx["contact_params"].damping = grasp_c
    ctx["contact_params"].friction_mu = grasp_mu
    ctx["box_table_params"].stiffness = grasp_k
    ctx["box_table_params"].damping = grasp_c

    phase_name = "home"
    for _ in range(ctx["substeps"]):
        q_des, phase_name = get_target_q(sim_time)

        tau = kp * (q_des - robot.q) + kd * (-robot.qd)
        tau += arm_solver.compute_gravity_torques()
        tau = np.clip(tau, -max_tau, max_tau)
        arm_solver.tau = tau

        panda_ground = contact.compute_rbd_contact_forces(arm_solver)
        box_table_forces = box_table_contact.compute_rbd_contact_forces(box_solver)

        fk_now = robot.forward_kinematics()
        box_fk_now = target_box.forward_kinematics()
        if phase_name in ("close", "lift", "hold", "done"):
            finger_wrenches, box_grasp_wrench = compute_finger_box_forces(
                robot, fk_now, target_box, box_solver,
                stiffness=grasp_k, damping=grasp_c, friction_mu=grasp_mu,
                box_fk=box_fk_now,
            )
        else:
            finger_wrenches, box_grasp_wrench = {}, np.zeros(6)

        arm_solver.clear_external_forces()
        panda_forces = dict(panda_ground)
        for li, w in finger_wrenches.items():
            panda_forces[li] = panda_forces.get(li, np.zeros(6)) + w
        for li, w in panda_forces.items():
            arm_solver.set_external_force(li, w)
        arm_solver.step()

        for fi in finger_dof:
            robot.q[fi] = q_des[fi]
            robot.qd[fi] = 0.0
        robot.enforce_mimic()

        if len(finger_wrenches) == 2:
            _l_idx = robot.link_index("panda_leftfinger")
            _r_idx = robot.link_index("panda_rightfinger")
            box_c = box_fk_now[box_body_idx].translation
            vl = _link_point_velocity(robot, _l_idx, box_c, fk=fk_now)
            vr = _link_point_velocity(robot, _r_idx, box_c, fk=fk_now)
            avg_v = 0.5 * (vl + vr)
            target_box.q[:3] += avg_v * dt
            target_box.qd[:3] = avg_v
            target_box.qd[3:] = 0.0
        else:
            box_solver.clear_external_forces()
            box_solver.tau = np.zeros(6)
            combined = np.zeros(6)
            for li, w in box_table_forces.items():
                combined += w
            combined += box_grasp_wrench
            if np.any(combined != 0):
                box_solver.set_external_force(box_body_idx, combined)
            box_solver.step()

        sim_time += dt

    return sim_time, phase_name


# ═══════════════════════════════════════════════════════════════════
# Physics step — FEM (deformable box)
# ═══════════════════════════════════════════════════════════════════

def step_physics_fem(ctx, sim_time, d_hat=0.008, penalty_k=30.0, friction_mu=0.5):
    """One frame of FEM (deformable box) physics."""
    robot = ctx["robot"]
    arm_solver = ctx["arm_solver"]
    contact = ctx["contact"]
    fem_solver = ctx["fem_solver"]
    fem_body = ctx["fem_body"]
    fem_ground_contact = ctx["fem_ground_contact"]
    kp, kd, max_tau = ctx["kp"], ctx["kd"], ctx["max_tau"]
    dt = ctx["dt"]
    get_target_q = ctx["get_target_q"]
    finger_dof = [7, 8]

    n_fem_contacts = 0
    phase_name = "home"
    for _ in range(ctx["substeps"]):
        q_des, phase_name = get_target_q(sim_time)

        # 1. Robot PD + gravity comp
        tau = kp * (q_des - robot.q) + kd * (-robot.qd)
        tau += arm_solver.compute_gravity_torques()
        tau = np.clip(tau, -max_tau, max_tau)
        arm_solver.tau = tau

        # 2. Robot ground contact
        panda_ground = contact.compute_rbd_contact_forces(arm_solver)
        arm_solver.clear_external_forces()
        for li, w in panda_ground.items():
            arm_solver.set_external_force(li, w)
        arm_solver.step()

        # 3. Kinematic fingers
        for fi in finger_dof:
            robot.q[fi] = q_des[fi]
            robot.qd[fi] = 0.0
        robot.enforce_mimic()

        # 4. RBD-FEM contact forces (compute BEFORE FEM step)
        fem_forces = contact.compute_rbd_fem_forces(
            arm_solver, fem_solver,
            d_hat=d_hat,
            stiffness=penalty_k,
            friction_mu=friction_mu,
            damping_ratio=1.0,
        )
        n_fem_contacts = sum(
            int(np.count_nonzero(f)) // 3
            for f in fem_forces.values()
        )

        # 5. FEM step (implicit Euler with contact forces)
        fem_solver.step(extra_forces=fem_forces if fem_forces else None)

        # 6. FEM ground contact at table height (post-step projection)
        fem_ground_contact.resolve_fem_contact(
            fem_body, restitution=0.3, friction_mu=0.5)

        sim_time += dt

    ctx["_n_fem_contacts"] = n_fem_contacts
    return sim_time, phase_name


def step_physics(ctx, sim_time, **kwargs):
    """Dispatch to primitive or FEM step."""
    if ctx["mode"] == "fem":
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
            if mode == "fem":
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
    if mode == "fem":
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
    else:
        fem_body = ctx["fem_body"]
        surface_tri = ctx["surface_tri"]
        viewer.add_mesh("fem_box", fem_body.x.copy(), surface_tri,
                        color=np.array([1.0, 0.35, 0.1]))

    # UI
    panel = UIPanel("Grasp Controls")
    if mode == "primitive":
        panel.add_slider("Stiffness", 1e3, 2e4, 5e3)
        panel.add_slider("Damping", 50, 1000, 200)
        panel.add_slider("Friction", 0.1, 1.5, 0.6)
    else:
        panel.add_slider("Contact d_hat", 0.002, 0.02, 0.008)
        panel.add_slider("Penalty K", 5.0, 200.0, 30.0)
        panel.add_slider("Friction", 0.1, 1.5, 0.5)
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
        renderer.show_collision(panel.get_bool("Show Collision"))

        import time as _t
        panel.playback.update_fps(_t.time())

        if mode == "primitive":
            kwargs = dict(
                grasp_k=panel.get("Stiffness"),
                grasp_c=panel.get("Damping"),
                grasp_mu=panel.get("Friction"),
            )
        else:
            kwargs = dict(
                d_hat=panel.get("Contact d_hat"),
                penalty_k=panel.get("Penalty K"),
                friction_mu=panel.get("Friction"),
            )

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
            viewer.update_mesh_vertices("target_box",
                                        (mat[:3, :3] @ bv.T).T + mat[:3, 3])
            box_z = target_box.q[2]
            extra_info = f"Contacts: {len(contact.last_forces)}"
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
        "--mode", choices=["primitive", "fem"], default="primitive",
        help="Box type: 'primitive' (rigid) or 'fem' (deformable)")
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
