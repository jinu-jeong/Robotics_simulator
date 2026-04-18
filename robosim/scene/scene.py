"""Scene — high-level simulation scene assembly and execution.

The Scene is the primary entry point for the RoboSim Scene API.

Workflow::

    from robosim import Scene, Robot, Box, Ground, FEM, JointPD

    scene = Scene(dt=0.001)
    ground = scene.add(Ground())
    robot  = scene.add(Robot("urdf/arm.urdf").controller(JointPD(...)).initial_q(HOME_Q))
    box    = scene.add(Box(size=0.08, mass=1.0, pos=[0.6, 0, 0.04]).physics(FEM()))

    scene.contact(robot, box, links=["left_finger", "right_finger"])
    scene.grip(robot, box, trigger_q=4, trigger_val=0.003, lift_start_t=8.2)

    traj = robot.trajectory()
    traj.phase("HOME",  target=HOME_Q,  duration=1.5)
    traj.phase("LIFT",  target=LIFT_Q,  duration=3.0)

    scene.run(traj, duration=12.0, headless=True)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from robosim.scene.objects import Robot as RobotDef, Box as BoxDef, Ground as GroundDef
from robosim.scene.objects import FEM, CB, Rigid
from robosim.scene.handles import RobotHandle, FEMBodyHandle, CBBodyHandle, RigidBodyHandle
from robosim.scene.runner import SimRunner, _ContactPair

if TYPE_CHECKING:
    from robosim.control.phases import Trajectory


class Scene:
    """High-level simulation scene.

    Parameters
    ----------
    dt       : master simulation timestep [s]
    gravity  : world gravity vector [m/s²]
    substeps : physics substeps per viewer frame (higher = smoother GUI)
    """

    def __init__(
        self,
        dt:       float = 0.001,
        gravity:  list | np.ndarray | None = None,
        substeps: int = 10,
    ):
        self.dt       = dt
        self.gravity  = np.asarray(gravity if gravity is not None else [0.0, 0.0, -9.81])
        self.substeps = substeps

        # Registered objects
        self._robot_handles: dict[str, RobotHandle]  = {}
        self._body_handles:  dict[str, object]        = {}   # any body handle
        self._ground:        GroundDef | None          = None

        # Physics metadata (needed by runner)
        self._body_physics:  dict[str, FEM | CB | Rigid] = {}
        self._body_colors:   dict[str, np.ndarray]        = {}
        self._body_ground_xy: dict[str, np.ndarray]       = {}
        self._arm_contact_links: list[str] = [
            "base_link", "link_1", "link_2", "link_3", "link_4", "palm_link"
        ]

        # Contact
        self._contact_pairs: list[_ContactPair] = []
        self._contact = None   # ContactSolver — created in _build_contact()

        # Grip state (per body name)
        self._grip_state:     dict[str, dict]      = {}
        self._grip_active:    dict[str, bool]       = {}
        self._grip_palm_pos0: dict[str, np.ndarray] = {}
        self._grip_body_x0:   dict[str, np.ndarray] = {}
        self._grip_box_pos0:  dict[str, np.ndarray] = {}
        self._grip_offset:    dict[str, np.ndarray] = {}
        self._grip_prev_palm: dict[str, np.ndarray] = {}

    # ══════════════════════════════════════════════════════════════════════════
    # add()
    # ══════════════════════════════════════════════════════════════════════════

    def add(self, obj) -> object:
        """Add an object to the scene. Returns a live handle.

        Parameters
        ----------
        obj : Robot, Box, or Ground declaration (from ``robosim.scene.objects``)

        Returns
        -------
        RobotHandle, FEMBodyHandle, CBBodyHandle, or RigidBodyHandle
        """
        if isinstance(obj, RobotDef):
            return self._add_robot(obj)
        if isinstance(obj, BoxDef):
            return self._add_box(obj)
        if isinstance(obj, GroundDef):
            self._ground = obj
            return obj      # Ground has no live handle
        raise TypeError(f"Scene.add(): unsupported object type {type(obj).__name__}")

    def _add_robot(self, defn: RobotDef) -> RobotHandle:
        from robosim.model.urdf_parser import parse_urdf
        from robosim.physics.rbd.solver import RBDSolver

        model = parse_urdf(defn._urdf_path)
        model.gravity = self.gravity.copy()

        if defn._initial_q is not None:
            model.q[:]  = defn._initial_q
            if hasattr(model, 'enforce_mimic'):
                model.enforce_mimic()
            model.qd[:] = 0.0

        name = defn._name or model.name or f"robot_{len(self._robot_handles)}"

        # Attach robot model to controller for gravity compensation
        ctrl = defn._controller
        if ctrl is not None and hasattr(ctrl, '_robot'):
            ctrl._robot = model

        solver = RBDSolver(robot=model)
        handle = RobotHandle(name=name, model=model, solver=solver, controller=ctrl)
        self._robot_handles[name] = handle
        return handle

    def _add_box(self, defn: BoxDef) -> "FEMBodyHandle | CBBodyHandle | RigidBodyHandle":
        name = defn._name or f"box_{len(self._body_handles)}"
        phys = defn._physics

        ground_z = self._ground.height if self._ground else 0.0

        if isinstance(phys, FEM):
            handle = self._build_fem_box(defn, name, phys, ground_z)
        elif isinstance(phys, CB):
            handle = self._build_cb_box(defn, name, phys, ground_z)
        else:
            handle = self._build_rigid_box(defn, name, ground_z)

        self._body_handles[name]  = handle
        self._body_physics[name]  = phys
        self._body_colors[name]   = defn._color[:3].copy()

        # Ground XY hold position
        self._body_ground_xy[name]  = defn._pos[:2].copy()
        self._grip_active[name]     = False

        return handle

    def _build_fem_box(self, defn: BoxDef, name: str, phys: FEM, ground_z: float):
        from robosim.physics.fem.materials import CorotationalElastic
        from robosim.physics.fem.mesh import FEMesh
        from robosim.physics.fem.solver import DeformableBody, FEMSolver

        origin = defn._pos - defn._size * 0.5
        origin[2] = ground_z

        mesh     = FEMesh.create_hex_box(origin=origin, size=defn._size, divisions=phys.mesh)
        density  = defn._mass / float(np.prod(defn._size))
        material = CorotationalElastic(young=phys.young, poisson=phys.poisson)

        body = DeformableBody(name=name, mesh=mesh, material=material, density=density)
        fem_dt = self.dt * phys.dt_scale
        solver = FEMSolver(
            bodies=[body],
            gravity=self.gravity.copy(),
            damping=phys.damping,
            max_newton_iters=phys.max_newton_iters,
        )
        # Attach fem_dt and fem_every for runner
        handle = FEMBodyHandle(name=name, body=body, solver=solver)
        handle._fem_dt         = fem_dt
        handle._fem_every      = phys.dt_scale
        handle._arm_step_count = 0
        return handle

    def _build_cb_box(self, defn: BoxDef, name: str, phys: CB, ground_z: float):
        import time as _time
        from robosim.physics.fem.materials import CorotationalElastic
        from robosim.physics.fem.mesh import FEMesh
        from robosim.physics.fem.reduced import CraigBamptonBody, CraigBamptonSolver

        origin = defn._pos - defn._size * 0.5
        origin[2] = ground_z

        mesh     = FEMesh.create_hex_box(origin=origin, size=defn._size, divisions=phys.mesh)
        density  = defn._mass / float(np.prod(defn._size))
        material = CorotationalElastic(young=phys.young, poisson=phys.poisson)

        print(f"\n  Building Craig-Bampton basis  "
              f"(n_nodes={mesh.n_nodes}, n_modes={phys.n_modes}) …")
        t0   = _time.time()
        body = CraigBamptonBody(
            mesh=mesh, material=material, density=density,
            n_modes=phys.n_modes,
            gravity=self.gravity.copy(),
            damping=phys.damping,
            name=name,
        )
        print(f"  CB basis built in {_time.time()-t0:.2f} s  (reduced DOFs: {body._n_r})")

        solver = CraigBamptonSolver(bodies=[body])
        handle = CBBodyHandle(name=name, body=body, solver=solver)
        handle._fem_dt         = self.dt
        handle._fem_every      = 1
        handle._arm_step_count = 0
        return handle

    def _build_rigid_box(self, defn: BoxDef, name: str, ground_z: float):
        from robosim.model.factory import create_free_box
        from robosim.physics.rbd.solver import RBDSolver

        center = defn._pos.copy()
        center[2] = ground_z + defn._size[2] / 2.0   # sit on ground

        model = create_free_box(
            name=name,
            size=tuple(defn._size),
            mass=defn._mass,
            position=center,
            color=defn._color.copy(),
        )
        model.gravity = self.gravity.copy()
        solver = RBDSolver(robot=model)
        handle = RigidBodyHandle(name=name, model=model, solver=solver)
        return handle

    # ══════════════════════════════════════════════════════════════════════════
    # contact() and grip()
    # ══════════════════════════════════════════════════════════════════════════

    def contact(
        self,
        robot:  RobotHandle,
        body:   "FEMBodyHandle | CBBodyHandle | RigidBodyHandle",
        links:  list[str] | None = None,
        k:      float = 1e4,
        c:      float = 50.0,
    ) -> None:
        """Register a contact pair between a robot and a body.

        Parameters
        ----------
        robot  : RobotHandle
        body   : body handle (FEM, CB, or Rigid)
        links  : robot link names used for contact detection.
                 If None, auto-detects all links whose collision geometry
                 is a box (suitable for finger links).
        k      : contact stiffness [N/m] (CB/Rigid only; FEM uses projection)
        c      : contact damping  [N·s/m]
        """
        if isinstance(body, RigidBodyHandle):
            # Rigid–rigid contact is handled by ContactSolver; nothing to register here.
            return

        contact_links = links or self._auto_detect_contact_links(robot)

        cp = _ContactPair(
            robot=robot,
            body=body,
            contact_links=contact_links,
            k_contact=k,
            c_contact=c,
        )
        self._contact_pairs.append(cp)

    def grip(
        self,
        robot:       RobotHandle,
        body,
        trigger_q:   int   = 4,
        trigger_val: float = 0.003,
        lift_start_t: float | None = None,
    ) -> None:
        """Register a kinematic grip constraint.

        Once ``robot.q[trigger_q] < trigger_val`` AND
        ``sim_time >= lift_start_t``, the body is kinematically coupled to
        the robot's palm link and moves with it.

        Parameters
        ----------
        trigger_q    : joint index whose value triggers grip (typically the
                       finger prismatic joint)
        trigger_val  : threshold value (grip activates when q < trigger_val)
        lift_start_t : earliest simulation time grip can activate [s].
                       Defaults to the start of the last trajectory phase.
        """
        if lift_start_t is None:
            lift_start_t = 0.0   # runner can override from trajectory

        self._grip_state[body.name] = {
            "robot":        robot,
            "trigger_q_idx": trigger_q,
            "trigger_q_val": trigger_val,
            "lift_start_t":  lift_start_t,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # run()
    # ══════════════════════════════════════════════════════════════════════════

    def run(
        self,
        trajectories: "Trajectory | list[Trajectory] | None" = None,
        duration:  float = 12.0,
        viewer:    bool  = False,
        headless:  bool  = False,
        on_step: Callable | None = None,
    ) -> dict:
        """Run the simulation.

        Parameters
        ----------
        trajectories : single Trajectory or list of Trajectories
        duration     : total simulated time [s]
        viewer       : open an interactive GUI window
        headless     : suppress GUI (for testing / CI)
        on_step      : optional ``(scene, t) -> None`` callback per frame

        Returns
        -------
        dict with ``'sim_time'`` and ``'wall_time'`` keys
        """
        # Normalise to list
        if trajectories is None:
            trajs = []
        elif isinstance(trajectories, list):
            trajs = trajectories
        else:
            trajs = [trajectories]

        # Build ContactSolver
        self._build_contact()

        runner = SimRunner(scene=self, dt=self.dt, substeps=self.substeps)
        return runner.run(
            trajectories=trajs,
            duration=duration,
            viewer=viewer,
            headless=headless,
            on_step=on_step,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Item access and display
    # ══════════════════════════════════════════════════════════════════════════

    def __getitem__(self, name: str):
        """Access any registered object by name::

            scene["arm"]        → RobotHandle
            scene["target_box"] → FEMBodyHandle
        """
        if name in self._robot_handles:
            return self._robot_handles[name]
        if name in self._body_handles:
            return self._body_handles[name]
        raise KeyError(f"Scene has no object named {name!r}.  "
                       f"Available: {self.names}")

    @property
    def robots(self) -> list[RobotHandle]:
        return list(self._robot_handles.values())

    @property
    def bodies(self) -> list:
        return list(self._body_handles.values())

    @property
    def names(self) -> list[str]:
        return list(self._robot_handles) + list(self._body_handles)

    def status(self) -> str:
        """Current simulation state snapshot (call during or after run)."""
        lines = [f"Scene  dt={self.dt}  gravity={self.gravity.tolist()}",
                 "─" * 55]
        for rh in self._robot_handles.values():
            q_str = np.array2string(rh._model.q, precision=3, suppress_small=True)
            lines.append(f"  [{rh.name}]  q={q_str}")
        for bh in self._body_handles.values():
            c   = bh.com
            ext = bh.extents * 1e3 if hasattr(bh, 'extents') else np.zeros(3)
            lines.append(
                f"  [{bh.name}]  "
                f"com=({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f})  "
                f"bot_z={bh.bottom_z:+.4f}  "
                f"ext=X{ext[0]:.1f}/Y{ext[1]:.1f}/Z{ext[2]:.1f}mm"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        from robosim.scene.handles import FEMBodyHandle, CBBodyHandle, RigidBodyHandle
        lines = [f"Scene  dt={self.dt}  gravity={self.gravity.tolist()}",
                 "─" * 55]

        if self._robot_handles:
            lines.append(f"  Robots ({len(self._robot_handles)})")
            for rh in self._robot_handles.values():
                ctrl = f"  ctrl={rh._controller!r}" if rh._controller else ""
                lines.append(f"    [{rh.name}]  {rh.n_dof}-DOF{ctrl}")

        if self._body_handles:
            lines.append(f"  Bodies ({len(self._body_handles)})")
            for bh in self._body_handles.values():
                phys = self._body_physics.get(bh.name, Rigid())
                lines.append(f"    [{bh.name}]  {type(bh).__name__}  {phys!r}")

        if self._ground:
            lines.append(f"  Ground  z={self._ground.height}")

        if self._contact_pairs:
            lines.append(f"  Contact pairs ({len(self._contact_pairs)})")
            for cp in self._contact_pairs:
                lines.append(f"    {cp.robot.name} ↔ {cp.body.name}"
                             f"  links={cp.contact_links}")

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════════
    # Internals
    # ══════════════════════════════════════════════════════════════════════════

    def _build_contact(self) -> None:
        from robosim.physics.contact.detection import GroundPlane
        from robosim.physics.contact.response import ContactParams
        from robosim.physics.contact.solver import ContactSolver

        g_height = self._ground.height if self._ground else 0.0
        self._contact = ContactSolver(
            ground=GroundPlane(height=g_height),
            params=ContactParams(stiffness=1e3, damping=50, friction_mu=0.0),
        )

    def _auto_detect_contact_links(self, robot: RobotHandle) -> list[str]:
        """Return robot link names that have box collision geometry (likely fingers)."""
        from robosim.model.geometry import GeometryType
        result = []
        for link in robot._model.links:
            for col in link.collisions:
                if col.geometry.geometry_type == GeometryType.BOX:
                    if any(k in link.name.lower() for k in ("finger", "tip", "pad")):
                        result.append(link.name)
                        break
        return result
