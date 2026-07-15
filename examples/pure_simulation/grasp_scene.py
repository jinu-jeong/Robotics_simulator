"""Grasp demo via the Scene API.

Robot picks up a box (rigid / FEM / CB) with optional deformable CB fingers:
  Scene → add robot + box → register contact/grip → build trajectory → run

Uses penalty contact + kinematic grip lock (Scene's default). The
constraint-based Coulomb-cone solver is also available — see
``Scene(contact_solver="constraint")`` and tests/test_constraint_friction.py
— but it's not exposed here because penalty + grip-lock is the cleaner
visual for this demo.

Usage:
  python examples/pure_simulation/grasp_scene.py
  python examples/pure_simulation/grasp_scene.py --object rigid --finger cb
  python examples/pure_simulation/grasp_scene.py --object cb --finger cb --headless
  python examples/pure_simulation/grasp_scene.py --grasp-depth 0.0   # tip at near face
  python examples/pure_simulation/grasp_scene.py --grasp-depth 1.0   # tip at far face
  python examples/pure_simulation/grasp_scene.py --mode fem --headless   # legacy

``--grasp-depth`` ∈ [0, 1] places the box along the finger X span:
  0 = fingertip at the object's near face (barely gripping)
  1 = fingertip at the object's far face (full-span grasp)

Always writes sim ground-truth finger deflection / normal-force CSV under
``runs/grasp_d<depth>/finger_force_log.csv`` (override with ``--log PATH``)
and saves the matching PNG plot. With ``--finger cb`` (or ``--capture``),
PNG frames go to ``runs/grasp_d<depth>/frames/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim import Scene, Robot, Box, Ground, FEM, CB, Rigid, JointPD

# ── URDF path ─────────────────────────────────────────────────────────────────
_URDF = str(Path(__file__).parent / "urdf" / "arm6_gripper" / "arm6_gripper.urdf")

# ── Grasp poses ───────────────────────────────────────────────────────────────
HOME_Q     = np.array([ 0.0, -1.571,  0.000, +1.571,  0.040, -0.040])
FOLD_Q     = np.array([ 0.0, -1.400,  2.000, -0.600,  0.040, -0.040])
APPROACH_Q = np.array([ 0.0, -0.365,  2.047, -1.682,  0.040, -0.040])
NEAR_Q     = np.array([ 0.0, -0.370,  1.909, -1.539,  0.040, -0.040])
# Finger close target: joint_left origin at palm tip (x=0.025, y=0.043);
# visual half-width is 0.005, so the finger inner face sits at
# Y = 0.043 + q[4] - 0.005 (palm frame).  The box's +Y face is at
# palm-Y 0.040 (box half-extent 0.04, centered on palm Y=0).
# Solving inner_face == +Y face gives q[4] ≈ +0.002.  We slightly
# undershoot at q[4]=-0.002 so the finger PD still drives it just
# 4 mm past the surface (enough load for the kinematic grip threshold
# q[4] < 0.003 to fire and lock the grip on), without the previous
# 13 mm "fork through tofu" overshoot the old -0.011 target produced.
CLOSE_Q    = np.array([ 0.0, -0.370,  1.909, -1.539, -0.002,  0.002])
LIFT_Q     = np.array([ 0.0, -0.878,  2.080, -1.202, -0.002,  0.002])
# RELEASE: same arm pose as the LIFT hold but fingers wide open (HOME
# finger q = ±0.040). Crossing ``release_val=0.020`` snaps the kinematic
# grip off and the box inherits the palm's velocity at that instant.
RELEASE_Q  = np.array([ 0.0, -0.878,  2.080, -1.202,  0.040, -0.040])

_LIFT_START_T = 1.5 + 1.2 + 5.0 + 1.5 + 1.5   # = 10.7 s (APPROACH @ 0.5× speed)
# Trajectory totals: HOME+FOLD+APPROACH+NEAR+CLOSE+LIFT+RELEASE = 16.2 s.
# After RELEASE the trajectory holds and physics runs free (box falls).
_RELEASE_END_T = _LIFT_START_T + 3.0 + 0.5     # = 14.2 s
_DROP_END_T    = _RELEASE_END_T + 2.5          # let the box settle

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FORCE_LOG_NAME = "finger_force_log.csv"
_DEFAULT_CAPTURE_SUBDIR = "frames"

_FINGER_LINKS = ["left_finger", "right_finger"]
_BOX_SIZE = 0.08
# Legacy box X ≈ grasp_depth 0.55 with the current NEAR pose / 160 mm fingers.
_DEFAULT_GRASP_DEPTH = 0.55


def _grasp_tag(depth: float) -> str:
    """Filesystem tag for a grasp depth, e.g. ``d0.55``."""
    return f"d{float(np.clip(depth, 0.0, 1.0)):.2f}"


def _default_run_dir(depth: float) -> Path:
    """Per-depth output root: ``runs/grasp_d0.55/``."""
    return _REPO_ROOT / "runs" / f"grasp_{_grasp_tag(depth)}"


def _default_force_log(depth: float) -> Path:
    return _default_run_dir(depth) / _DEFAULT_FORCE_LOG_NAME


def _default_capture_dir(depth: float) -> Path:
    return _default_run_dir(depth) / _DEFAULT_CAPTURE_SUBDIR


def _finger_tip_world_x(robot, q: np.ndarray) -> float:
    """World +X of the left-finger tip at joint configuration ``q``."""
    model = robot._model
    q_prev = np.asarray(model.q, dtype=float).copy()
    model.q[:] = np.asarray(q, dtype=float)
    try:
        fk = model.forward_kinematics()
        lf = fk[model.link_index("left_finger")]
        vis = model.links[model.link_index("left_finger")].visuals[0]
        size = np.asarray(vis.geometry.size, dtype=float)
        origin = np.asarray(vis.origin.translation, dtype=float)
        tip_local = origin + np.array([0.5 * size[0], 0.0, 0.0])
        tip_w = lf.translation + lf.rotation @ tip_local
        return float(tip_w[0])
    finally:
        model.q[:] = q_prev


def box_x_for_grasp_depth(tip_x: float, box_size: float, depth: float) -> float:
    """Map grasp depth ∈ [0, 1] → box centre X.

    ``depth=0``: fingertip at the object's near face (closest to the palm —
    tip-only / barely gripping).
    ``depth=1``: fingertip at the object's far face (full object under the
    finger span).
    """
    d = float(np.clip(depth, 0.0, 1.0))
    half = 0.5 * float(box_size)
    return float(tip_x - half * (2.0 * d - 1.0))


# ── Contact params ────────────────────────────────────────────────────────────
_OBJECT_CONTACT_PARAMS = {
    "fem": dict(k=1e3,  c=30.0),
    "cb":  dict(k=1e4,  c=50.0),
}
# Softer finger↔rigid penalty so the CB mesh can bend (cantilever visual).
_FINGER_RIGID_CONTACT = dict(k=300.0, c=8.0)

# Shared Young's modulus for deformable grasp objects [Pa].
_OBJECT_YOUNG = 3.0e4

# ── Object physics ────────────────────────────────────────────────────────────
def _box_physics(object_mode: str):
    if object_mode == "fem":
        return FEM(young=_OBJECT_YOUNG, poisson=0.45, mesh=(8, 8, 8), dt_scale=5)
    if object_mode == "cb":
        return CB(young=_OBJECT_YOUNG, poisson=0.45, mesh=(6, 6, 6), n_modes=10)
    return Rigid()


class FingerForceLogger:
    """Records sim ground-truth finger deflection / normal force each frame."""

    _HEADER = (
        "t,phase,object_mode,finger_mode,"
        "left_deflection_mm,right_deflection_mm,"
        "left_deflection_y_mm,right_deflection_y_mm,"
        "left_fn_n,right_fn_n,"
        "left_contact_s,right_contact_s,"
        "left_deflection_cv_mm,right_deflection_cv_mm,"
        "left_fn_cv_n,right_fn_cv_n,"
        "left_contact_s_cv,right_contact_s_cv"
    )

    def __init__(
        self,
        path: Path,
        object_mode: str,
        finger_mode: str,
    ) -> None:
        self.path = path
        self.object_mode = object_mode
        self.finger_mode = finger_mode
        self._rows: list[str] = []

    def record(
        self,
        t: float,
        phase: str | None,
        probe: dict,
        cv: dict | None = None,
    ) -> None:
        lf = probe["left_finger"]
        rf = probe["right_finger"]
        ph = phase or ""
        cv = cv or {}
        self._rows.append(
            f"{t:.6f},{ph},{self.object_mode},{self.finger_mode},"
            f"{lf['deflection_m'] * 1e3:.6f},{rf['deflection_m'] * 1e3:.6f},"
            f"{lf['deflection_y_m'] * 1e3:.6f},{rf['deflection_y_m'] * 1e3:.6f},"
            f"{lf['fn_n']:.6f},{rf['fn_n']:.6f},"
            f"{lf.get('contact_s', float('nan')):.6f},"
            f"{rf.get('contact_s', float('nan')):.6f},"
            f"{cv.get('left_deflection_cv_mm', 0.0):.6f},"
            f"{cv.get('right_deflection_cv_mm', 0.0):.6f},"
            f"{cv.get('left_fn_cv_n', 0.0):.6f},"
            f"{cv.get('right_fn_cv_n', 0.0):.6f},"
            f"{cv.get('left_contact_s', float('nan')):.6f},"
            f"{cv.get('right_contact_s', float('nan')):.6f}"
        )

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self._HEADER + "\n" + "\n".join(self._rows) + "\n")
        return self.path


def main(
    object_mode: str = "rigid",
    finger_mode: str = "rigid",
    headless: bool = False,
    log_path: Path | None = None,
    capture: bool = False,
    capture_dir: Path | None = None,
    capture_every: int = 5,
    grasp_depth: float = _DEFAULT_GRASP_DEPTH,
) -> None:
    # ── Build scene ───────────────────────────────────────────────────────────
    scene = Scene(dt=0.001, gravity=[0, 0, -9.81], substeps=10)

    ground = scene.add(Ground(height=0.0))

    robot = scene.add(
        Robot(_URDF)
        .controller(JointPD(
            kp=[280.0, 480.0, 200.0, 75.0,  800,  800],
            kd=[ 35.0,  50.0,  20.0,  2.0,   30,   30],
        ))
        .initial_q(HOME_Q)
        .name("arm")
    )

    tip_x = _finger_tip_world_x(robot, NEAR_Q)
    box_x = box_x_for_grasp_depth(tip_x, _BOX_SIZE, grasp_depth)
    print(
        f"grasp_depth={float(np.clip(grasp_depth, 0.0, 1.0)):.3f}  "
        f"box_x={box_x:.4f}  finger_tip_x@NEAR={tip_x:.4f}"
    )

    box = scene.add(
        Box(size=_BOX_SIZE, mass=1.0, pos=[box_x, 0.0, 0.5 * _BOX_SIZE])
        .physics(_box_physics(object_mode))
        .name("box")
    )

    # ── Contact + deformable fingers ──────────────────────────────────────────
    # Rigid URDF finger slabs ↔ deformable object (skip when fingers are CB).
    if object_mode != "rigid" and finger_mode != "cb":
        scene.contact(
            robot, box,
            links=_FINGER_LINKS,
            **_OBJECT_CONTACT_PARAMS[object_mode],
        )

    if finger_mode == "cb":
        # Soft CB prong: anchored at palm (-X face), free tip bends under load.
        scene.attach_deformable_gripper(
            robot, links=_FINGER_LINKS,
            divisions=(4, 2, 2),
            young=3.0e4,
            poisson=0.45,
            n_modes=8,
            damping=0.12,
        )
        scene.contact_deformable_gripper(
            robot, box,
            **_FINGER_RIGID_CONTACT,
        )

    scene.grip(
        robot, box,
        trigger_q=4,
        trigger_val=0.003,
        lift_start_t=_LIFT_START_T,
        release_val=0.020,
    )

    # ── Trajectory ────────────────────────────────────────────────────────────
    traj = robot.trajectory()
    traj.phase("HOME",     target=HOME_Q,     duration=1.5)
    traj.phase("FOLD",     target=FOLD_Q,     duration=1.2)
    traj.phase("APPROACH", target=APPROACH_Q, duration=5.0)  # 0.5× approach speed
    traj.phase("NEAR",     target=NEAR_Q,     duration=1.5)
    traj.phase("CLOSE",    target=CLOSE_Q,    duration=1.5)
    traj.phase("LIFT",     target=LIFT_Q,     duration=3.0)
    traj.phase("RELEASE",  target=RELEASE_Q,  duration=0.5)

    # ── Finger force / deflection log (sim ground truth for CV validation) ───
    out_log = Path(log_path) if log_path is not None else _default_force_log(grasp_depth)
    out_cap = (
        Path(capture_dir) if capture_dir is not None
        else _default_capture_dir(grasp_depth)
    )
    logger = FingerForceLogger(
        out_log,
        object_mode=object_mode,
        finger_mode=finger_mode,
    )

    capture_session = None
    if capture:
        from robosim.viz.finger_capture import FingerCaptureSession
        capture_session = FingerCaptureSession(
            out_cap,
            every=capture_every,
        )
        scene._finger_capture = capture_session

    def on_step(sc: Scene, t: float) -> None:
        probe = sc.sample_finger_probe()
        cv = capture_session.latest if capture_session is not None else None
        logger.record(t, traj.current_phase, probe, cv=cv)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(scene)
    scene.run(
        trajectories=traj,
        duration=_DROP_END_T,
        viewer=not headless or capture,
        headless=headless,
        on_step=on_step,
    )
    out_path = logger.write()
    print(f"Finger log → {out_path}")
    if capture_session is not None:
        manifest = capture_session.write_manifest()
        profiles = capture_session.write_profiles()
        n_frames = len(capture_session._manifest_rows)
        print(f"Capture    → {capture_session.out_dir}  ({n_frames} frames)")
        print(f"Manifest   → {manifest}")
        print(f"Profiles   → {profiles}")
        try:
            from robosim.viz.finger_force_plot import (
                plot_deflection_profiles,
                plot_station_trajectories,
            )
            prof_plot = plot_deflection_profiles(profiles)
            print(f"Profile plot → {prof_plot}")
            st_plot = plot_station_trajectories(
                profiles,
                title=(
                    f"Deflection vs time at x=0, 0.5L, L — grasp_{_grasp_tag(grasp_depth)}\n"
                    f"(sim red solid, CV blue dashed)"
                ),
            )
            print(f"Station plot → {st_plot}")
        except ImportError:
            print("Profile plot skipped (install matplotlib to enable)")

    try:
        from robosim.viz.finger_force_plot import plot_finger_force_log
        plot_path = plot_finger_force_log(out_path)
        print(f"Finger plot → {plot_path}")
    except ImportError:
        print("Finger plot skipped (install matplotlib to enable)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp demo — Scene API")
    parser.add_argument(
        "--object", default=None, choices=["rigid", "fem", "cb"],
        help="physics model for the grasped box",
    )
    parser.add_argument(
        "--finger", default=None, choices=["rigid", "cb"],
        help="physics model for gripper fingers (cb = deformable Craig-Bampton)",
    )
    parser.add_argument(
        "--mode", default=None, choices=["rigid", "fem", "cb"],
        help="legacy alias: sets --object and auto-enables deformable fingers "
             "for fem/cb (use --object/--finger instead)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--log",
        default=None,
        help="CSV path for finger deflection / normal-force log "
             f"(default: runs/grasp_d<depth>/{_DEFAULT_FORCE_LOG_NAME})",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="save viewer PNGs + gradient CV deflection (needs GPU window)",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="disable frame capture (overrides default for --finger cb)",
    )
    parser.add_argument(
        "--capture-dir",
        default=None,
        help="directory for PNG frames "
             f"(default: runs/grasp_d<depth>/{_DEFAULT_CAPTURE_SUBDIR})",
    )
    parser.add_argument(
        "--capture-every",
        type=int,
        default=5,
        help="save one PNG every N viewer frames (default: 5)",
    )
    parser.add_argument(
        "--grasp-depth",
        type=float,
        default=_DEFAULT_GRASP_DEPTH,
        help="where along the object X the fingertip sits: 0 = near face "
             "(tip-only), 1 = far face (full grasp). Default: "
             f"{_DEFAULT_GRASP_DEPTH}",
    )
    args = parser.parse_args()

    if not 0.0 <= args.grasp_depth <= 1.0:
        parser.error("--grasp-depth must be in [0, 1]")

    if args.mode is not None:
        object_mode = args.object or args.mode
        if args.finger is not None:
            finger_mode = args.finger
        else:
            finger_mode = "cb" if args.mode in ("fem", "cb") else "rigid"
    else:
        object_mode = args.object or "rigid"
        finger_mode = args.finger or "rigid"

    do_capture = (args.capture or finger_mode == "cb") and not args.no_capture

    main(
        object_mode=object_mode,
        finger_mode=finger_mode,
        headless=args.headless,
        log_path=Path(args.log) if args.log else None,
        capture=do_capture,
        capture_dir=Path(args.capture_dir) if args.capture_dir else None,
        capture_every=args.capture_every,
        grasp_depth=args.grasp_depth,
    )
