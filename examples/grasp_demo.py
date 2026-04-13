"""Panda 로봇 그리퍼 파지 데모.

Franka Panda 7-DOF 팔이 테이블 위의 박스를 잡고 들어올립니다.
URDF 메시 렌더링, mimic 관절, mesh-box 충돌, 다중 강체 접촉을 시연.

실행:
  python examples/grasp_demo.py                # GUI 모드
  python examples/grasp_demo.py --headless     # 헤드리스 (물리만)
  python examples/grasp_demo.py --headless -n 2000  # 2000프레임

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
# Panda: j1(base), j2(shoulder), j3(upper_rot), j4(elbow),
#        j5(forearm_rot), j6(wrist), j7(wrist_rot), finger_l, finger_r

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


def compute_finger_box_forces(
    robot, fk, target_box, box_solver,
    stiffness=5e3, damping=200, friction_mu=0.6,
    box_fk=None,
):
    """Analytical parallel-jaw grasp forces between finger boxes and target box.

    Uses a 1D squeeze model: each finger contacts the box along the
    world y-axis with a horizontal normal.  Coulomb friction in x and z
    provides the vertical lift force.  This avoids SAT normal tilt that
    launches light objects.

    Forces are only generated when BOTH fingers overlap the box (a true
    grip).  A single-finger contact would push the box away, not grip it.

    Parameters
    ----------
    box_fk : optional pre-computed FK for target_box (avoids redundant call).

    Returns: (finger_wrenches, box_wrench)
        finger_wrenches: dict {link_idx: wrench(6)} for each finger
        box_wrench:      (6,) combined wrench on the box
    """
    from robosim.physics.contact.solver import _link_point_velocity

    if box_fk is None:
        box_fk = target_box.forward_kinematics()
    box_body_idx = len(target_box.links) - 1
    T_box = box_fk[box_body_idx]
    box_center = T_box.translation
    box_half = np.array(BOX_SIZE) / 2.0
    fh = np.array([0.021, 0.026, 0.054]) / 2.0

    # ── Phase 1: check which fingers overlap the box ──
    finger_data = {}  # fname -> (idx, fc, fR, overlap_y)
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

        # x and z overlap check
        ok = True
        for ax in [0, 2]:
            f_ext = sum(abs(fR[:, k][ax]) * fh[k] for k in range(3))
            if abs(fc[ax] - box_center[ax]) > f_ext + box_half[ax]:
                ok = False
                break
        if not ok:
            continue

        finger_data[fname] = (idx, fc, fR, overlap_y, sign)

    # ── Require both fingers in contact for a real grip ──
    if len(finger_data) < 2:
        return {}, np.zeros(6)

    # ── Phase 2: compute forces for each finger ──
    finger_wrenches = {}
    box_wrench = np.zeros(6)

    for fname, (idx, fc, fR, overlap_y, sign) in finger_data.items():
        pen = min(overlap_y, 0.01)

        contact_y = (box_center[1] + box_half[1]) if sign > 0 else (box_center[1] - box_half[1])
        # Apply grip force at the box center's x,z to prevent tipping torques.
        # The finger z-centre is ~25mm below the box centre — using the midpoint
        # would create a lever arm that tips the light box into the table.
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

        # Ramp friction with penetration depth (prevent launch on marginal contact)
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


def setup_physics():
    """Create robot, box, solvers, and contacts. Returns all state as a dict."""
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

    target_box = create_free_box(
        name="target_box",
        size=BOX_SIZE,
        mass=0.2,
        position=box_pos,
        color=np.array([0.9, 0.3, 0.2, 1.0]),
    )

    arm_solver = RBDSolver(robot=robot)
    arm_solver.initialize(dt=dt)

    box_solver = RBDSolver(robot=target_box)
    box_solver.initialize(dt=dt)

    contact_params = ContactParams(stiffness=5e3, damping=200, friction_mu=0.6)
    contact = ContactSolver(
        ground=GroundPlane(height=0.0),
        params=contact_params,
    )
    contact.register_rbd(arm_solver, robot_id="panda")

    box_table_params = ContactParams(stiffness=5e3, damping=200, friction_mu=0.6)
    box_table_contact = ContactSolver(
        ground=GroundPlane(height=TABLE_HEIGHT),
        params=box_table_params,
    )
    box_table_contact.register_rbd(box_solver, robot_id="target_box")

    kp = np.array([600, 600, 600, 600, 250, 150, 50, 100, 100], dtype=np.float64)
    kd = np.array([50, 50, 50, 50, 30, 25, 10, 10, 10], dtype=np.float64)
    max_tau = np.array([87, 87, 87, 87, 12, 12, 12, 20, 20], dtype=np.float64)

    total_mass = sum(l.mass for l in robot.links)
    print(f"\nPanda: {robot.n_dof} DOF, {total_mass:.2f} kg total")
    print(f"Table: static at z={TABLE_HEIGHT}m, Box: mass=0.2 kg")

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

    return dict(
        dt=dt, substeps=substeps,
        robot=robot, target_box=target_box,
        arm_solver=arm_solver, box_solver=box_solver,
        contact=contact, box_table_contact=box_table_contact,
        contact_params=contact_params, box_table_params=box_table_params,
        kp=kp, kd=kd, max_tau=max_tau,
        hand_idx=hand_idx, box_pos=box_pos, table_pos=table_pos,
        get_target_q=get_target_q,
    )


def step_physics(
    ctx, sim_time,
    grasp_k=5e3, grasp_c=200, grasp_mu=0.6,
):
    """Run one frame (substeps) of pure physics. Returns (sim_time, phase_name)."""
    from robosim.physics.contact.solver import _link_point_velocity

    robot = ctx["robot"]
    target_box = ctx["target_box"]
    arm_solver = ctx["arm_solver"]
    box_solver = ctx["box_solver"]
    contact = ctx["contact"]
    box_table_contact = ctx["box_table_contact"]
    kp, kd, max_tau = ctx["kp"], ctx["kd"], ctx["max_tau"]
    dt = ctx["dt"]
    substeps = ctx["substeps"]
    get_target_q = ctx["get_target_q"]
    finger_dof = [7, 8]
    box_body_idx = len(target_box.links) - 1

    ctx["contact_params"].stiffness = grasp_k
    ctx["contact_params"].damping = grasp_c
    ctx["contact_params"].friction_mu = grasp_mu
    ctx["box_table_params"].stiffness = grasp_k
    ctx["box_table_params"].damping = grasp_c

    phase_name = "home"
    for _ in range(substeps):
        q_des, phase_name = get_target_q(sim_time)

        tau = kp * (q_des - robot.q) + kd * (-robot.qd)
        tau += arm_solver.compute_gravity_torques()
        tau = np.clip(tau, -max_tau, max_tau)
        arm_solver.tau = tau

        panda_ground = contact.compute_rbd_contact_forces(arm_solver)
        box_table_forces = box_table_contact.compute_rbd_contact_forces(
            box_solver)

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
# Headless mode
# ═══════════════════════════════════════════════════════════════════

def run_headless(n_frames: int = 1500):
    """Run physics-only loop without visualization. Reports FPS."""
    ctx = setup_physics()
    dt = ctx["dt"]
    substeps = ctx["substeps"]
    robot = ctx["robot"]
    target_box = ctx["target_box"]
    hand_idx = ctx["hand_idx"]

    print(f"\nMode: HEADLESS ({n_frames} frames, "
          f"dt={dt}, substeps={substeps})")
    print("-" * 50)

    sim_time = 0.0
    log_interval = 500  # frames between status prints

    wall_start = time.perf_counter()
    lap_start = wall_start
    lap_frames = 0

    for frame in range(n_frames):
        sim_time, phase_name = step_physics(ctx, sim_time)
        lap_frames += 1

        # Periodic status
        if (frame + 1) % log_interval == 0 or frame == n_frames - 1:
            now = time.perf_counter()
            lap_dt = now - lap_start
            lap_fps = lap_frames / lap_dt if lap_dt > 0 else 0
            rt_factor = (lap_frames * substeps * dt) / lap_dt if lap_dt > 0 else 0

            fk_now = robot.forward_kinematics()
            hand_pos = fk_now[hand_idx].translation
            box_z = target_box.q[2]

            print(f"  frame {frame+1:5d}/{n_frames}  "
                  f"sim_t={sim_time:.3f}s  "
                  f"phase={phase_name:<10s}  "
                  f"FPS={lap_fps:.1f}  "
                  f"RT={rt_factor:.2f}x  "
                  f"box_z={box_z:.4f}m")

            lap_start = now
            lap_frames = 0

    wall_total = time.perf_counter() - wall_start
    avg_fps = n_frames / wall_total
    total_sim_time = n_frames * substeps * dt
    rt_factor = total_sim_time / wall_total

    print("-" * 50)
    print(f"Done: {n_frames} frames in {wall_total:.2f}s")
    print(f"  Avg FPS:       {avg_fps:.1f}")
    print(f"  Avg frame:     {wall_total/n_frames*1000:.2f} ms")
    print(f"  Sim time:      {total_sim_time:.2f}s")
    print(f"  Real-time:     {rt_factor:.2f}x")
    print(f"  Final box Z:   {target_box.q[2]:.4f}m")
    finite_ok = (np.all(np.isfinite(robot.q)) and
                 np.all(np.isfinite(target_box.q)))
    print(f"  Finite check:  {'OK' if finite_ok else 'FAIL'}")


# ═══════════════════════════════════════════════════════════════════
# GUI mode
# ═══════════════════════════════════════════════════════════════════

def run_gui():
    """Interactive GUI loop with visualization."""
    from robosim.viz.viewer import SimViewer
    from robosim.viz.scene_renderer import RobotRenderer
    from robosim.viz.ui_panels import UIPanel
    from robosim.viz.overlays import ContactForceOverlay

    ctx = setup_physics()
    dt = ctx["dt"]
    robot = ctx["robot"]
    target_box = ctx["target_box"]
    arm_solver = ctx["arm_solver"]
    contact = ctx["contact"]
    contact_params = ctx["contact_params"]
    box_table_params = ctx["box_table_params"]
    hand_idx = ctx["hand_idx"]
    box_pos = ctx["box_pos"]
    table_pos = ctx["table_pos"]

    # ── Viewer ──
    viewer = SimViewer(
        title="RoboSim — Panda Gripper Grasp Demo",
        window_size=(1280, 800),
    )
    viewer.initialize()

    renderer = RobotRenderer(robot, viewer)
    renderer.setup()

    from robosim.viz.viewer import geometry_to_trimesh
    from robosim.model.geometry import Geometry as _Geo
    _table_geo = _Geo.box(0.30, 0.30, TABLE_HEIGHT)
    tv, tf = geometry_to_trimesh(_table_geo)
    viewer.add_mesh("table", tv + table_pos, tf,
                    color=np.array([0.6, 0.5, 0.4]))

    box_body = target_box.links[-1]
    bv, bf = geometry_to_trimesh(box_body.visuals[0].geometry)
    viewer.add_mesh("target_box", bv + box_pos, bf,
                    color=np.array([0.9, 0.3, 0.2]))

    panel = UIPanel("Grasp Controls")
    panel.add_slider("Stiffness", 1e3, 2e4, 5e3)
    panel.add_slider("Damping", 50, 1000, 200)
    panel.add_slider("Friction", 0.1, 1.5, 0.6)
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

        grasp_k = panel.get("Stiffness")
        grasp_c = panel.get("Damping")
        grasp_mu = panel.get("Friction")
        force_overlay.active = panel.get_bool("Show Forces")
        renderer.show_collision(panel.get_bool("Show Collision"))

        import time as _t
        panel.playback.update_fps(_t.time())

        sim_time_new, phase_name = step_physics(
            ctx, sim_time,
            grasp_k=grasp_k, grasp_c=grasp_c, grasp_mu=grasp_mu,
        )
        sim_time = sim_time_new

        # ── Update visuals ──
        fk_render = robot.forward_kinematics()
        renderer.update(fk=fk_render)

        fk_b = target_box.forward_kinematics()
        T_box = fk_b[len(target_box.links) - 1]
        mat = T_box.to_matrix()
        viewer.update_mesh_vertices("target_box",
                                    (mat[:3, :3] @ bv.T).T + mat[:3, 3])

        force_overlay.update(contact.last_forces)

        hand_now = fk_render[hand_idx].translation
        finger_q = robot.q[7]
        box_z = target_box.q[2]

        panel.playback.sim_time = sim_time
        panel.set_info([
            f"Phase: {phase_name}",
            f"Hand: ({hand_now[0]:+.3f}, {hand_now[1]:+.3f}, {hand_now[2]:+.3f})",
            f"Finger: {finger_q*1000:.1f}mm",
            f"Box Z: {box_z:.4f}m",
            f"Contacts: {len(contact.last_forces)}",
        ])
        panel.render(viewer._window)
        frame_count += 1

    viewer.add_callback(step_callback)
    print("\nPanda Grasp Demo — ESC to quit")
    viewer.show()

    box_z = target_box.q[2]
    print(f"\nFinal box Z: {box_z:.4f}m")
    print(f"Simulated {sim_time:.2f}s in {frame_count} frames")
    finite_ok = (np.all(np.isfinite(robot.q)) and
                 np.all(np.isfinite(target_box.q)))
    print(f"Finite check: {'OK' if finite_ok else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(
        description="Panda gripper grasp demo")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without visualization (physics only)")
    parser.add_argument(
        "-n", "--frames", type=int, default=1500,
        help="Number of frames to simulate in headless mode (default: 1500)")
    args = parser.parse_args()

    if not args.headless:
        ti.init(arch=ti.metal)

    print("=" * 60)
    print("  RoboSim: Panda Gripper Grasp Demo")
    print("  Franka Panda 7-DOF + Parallel Jaw Gripper")
    print("=" * 60)

    if args.headless:
        run_headless(n_frames=args.frames)
    else:
        run_gui()


if __name__ == "__main__":
    main()
