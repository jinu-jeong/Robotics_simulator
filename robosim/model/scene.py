"""Scene: manages ground plane, robots, free objects, and deformable bodies.

Provides a high-level container that:
 - Registers multiple robots and free rigid objects.
 - Optionally holds deformable (FEM) bodies.
 - Configures a ground plane and static obstacles.
 - Builds the complete PhysicsWorld with contact and coupling.

Usage::

    scene = Scene(ground_height=0.0)
    scene.add_robot(robot, solver)
    scene.add_free_object(box, box_solver)
    scene.add_deformable(fem_solver)
    world = scene.build_world(dt=0.001)

    for _ in range(n_steps):
        world.step()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from robosim.math.transforms import Transform
from robosim.model.robot import Robot
from robosim.model.geometry import Geometry
from robosim.model.link import Visual, Collision, Link


@dataclass
class StaticObstacle:
    """A non-moving object in the scene (visual + collision only)."""

    name: str
    geometry: Geometry
    transform: Transform
    color: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 1.0]))


@dataclass
class RobotEntry:
    """A robot registered in the scene."""

    robot: Robot
    solver: object  # RBDSolver
    robot_id: str
    initial_q: np.ndarray | None = None
    controller: Callable | None = None  # (robot, t) -> tau


@dataclass
class FreeObjectEntry:
    """A free rigid object (6-DOF floating body)."""

    robot: Robot
    solver: object  # RBDSolver
    object_id: str


class Scene:
    """High-level scene container for multi-body simulation.

    Aggregates robots, free objects, deformable bodies, ground, and obstacles
    into a coherent setup that can be turned into a PhysicsWorld.
    """

    def __init__(
        self,
        ground_height: float = 0.0,
        gravity: np.ndarray | None = None,
    ):
        self.ground_height = ground_height
        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])

        self._robots: dict[str, RobotEntry] = {}
        self._free_objects: dict[str, FreeObjectEntry] = {}
        self._obstacles: list[StaticObstacle] = []
        self._fem_solver = None
        self._coupling_maps: list = []
        self._collision_filters: list[tuple[str, str, str, str]] = []

    # ── Registration ──

    def add_robot(
        self,
        robot: Robot,
        solver=None,
        robot_id: str | None = None,
        initial_q: np.ndarray | None = None,
        controller: Callable | None = None,
    ) -> str:
        """Add a robot to the scene.

        Parameters
        ----------
        robot : Robot model
        solver : RBDSolver (created automatically if None)
        robot_id : unique identifier (defaults to robot.name)
        initial_q : initial joint configuration
        controller : optional (robot, t) -> tau callback

        Returns
        -------
        robot_id : the registered ID
        """
        if robot_id is None:
            robot_id = robot.name
        robot.gravity = self.gravity.copy()

        if solver is None:
            from robosim.physics.rbd.solver import RBDSolver
            solver = RBDSolver(robot=robot)

        if initial_q is not None:
            robot.q = initial_q.copy()
            robot.qd = np.zeros(robot.n_dof)

        self._robots[robot_id] = RobotEntry(
            robot=robot, solver=solver, robot_id=robot_id,
            initial_q=initial_q, controller=controller,
        )
        return robot_id

    def add_free_object(
        self,
        robot: Robot,
        solver=None,
        object_id: str | None = None,
    ) -> str:
        """Add a free rigid object (6-DOF floating body).

        Parameters
        ----------
        robot : Robot model (e.g. from create_free_box)
        solver : RBDSolver (created automatically if None)
        object_id : unique identifier (defaults to robot.name)
        """
        if object_id is None:
            object_id = robot.name
        robot.gravity = self.gravity.copy()

        if solver is None:
            from robosim.physics.rbd.solver import RBDSolver
            solver = RBDSolver(robot=robot)

        self._free_objects[object_id] = FreeObjectEntry(
            robot=robot, solver=solver, object_id=object_id,
        )
        return object_id

    def add_deformable(self, fem_solver) -> None:
        """Register an FEM solver with deformable bodies."""
        self._fem_solver = fem_solver

    def add_obstacle(
        self,
        name: str,
        geometry: Geometry,
        transform: Transform,
        color: np.ndarray | None = None,
    ) -> None:
        """Add a static obstacle (visual only, collision TODO)."""
        if color is None:
            color = np.array([0.5, 0.5, 0.5, 1.0])
        self._obstacles.append(StaticObstacle(
            name=name, geometry=geometry, transform=transform, color=color,
        ))

    def add_collision_filter(
        self,
        robot_id_a: str, link_name_a: str,
        robot_id_b: str, link_name_b: str,
    ) -> None:
        """Register a collision filter between specific links."""
        self._collision_filters.append(
            (robot_id_a, link_name_a, robot_id_b, link_name_b))

    def add_coupling(self, boundary_map) -> None:
        """Register an RBD-FEM coupling boundary map."""
        self._coupling_maps.append(boundary_map)

    # ── Accessors ──

    def get_robot(self, robot_id: str) -> Robot:
        """Get a registered robot by ID."""
        if robot_id in self._robots:
            return self._robots[robot_id].robot
        raise KeyError(f"Robot '{robot_id}' not found in scene")

    def get_solver(self, robot_id: str):
        """Get the RBDSolver for a registered robot."""
        if robot_id in self._robots:
            return self._robots[robot_id].solver
        if robot_id in self._free_objects:
            return self._free_objects[robot_id].solver
        raise KeyError(f"Robot/object '{robot_id}' not found in scene")

    @property
    def robot_ids(self) -> list[str]:
        return list(self._robots.keys())

    @property
    def object_ids(self) -> list[str]:
        return list(self._free_objects.keys())

    @property
    def all_robots(self) -> list[RobotEntry]:
        return list(self._robots.values())

    @property
    def all_objects(self) -> list[FreeObjectEntry]:
        return list(self._free_objects.values())

    @property
    def obstacles(self) -> list[StaticObstacle]:
        return self._obstacles

    # ── Build ──

    def build_world(
        self,
        dt: float = 0.001,
        contact_stiffness: float = 5e4,
        contact_damping: float = 500,
        contact_friction: float = 0.5,
        coupling_stiffness: float = 1e5,
        coupling_damping: float = 1e3,
    ):
        """Build a PhysicsWorld from the scene configuration.

        Returns
        -------
        SceneWorld with step(), reset(), and accessor methods.
        """
        from robosim.physics.contact.detection import GroundPlane
        from robosim.physics.contact.response import ContactParams
        from robosim.physics.contact.solver import ContactSolver

        contact = ContactSolver(
            ground=GroundPlane(height=self.ground_height),
            params=ContactParams(
                stiffness=contact_stiffness,
                damping=contact_damping,
                friction_mu=contact_friction,
            ),
        )

        # Initialize all solvers
        all_solvers = {}
        for entry in self._robots.values():
            entry.solver.initialize(dt=dt)
            contact.register_rbd(entry.solver, robot_id=entry.robot_id)
            all_solvers[entry.robot_id] = entry.solver

        for entry in self._free_objects.values():
            entry.solver.initialize(dt=dt)
            contact.register_rbd(entry.solver, robot_id=entry.object_id)
            all_solvers[entry.object_id] = entry.solver

        # Apply collision filters
        for rid_a, link_a, rid_b, link_b in self._collision_filters:
            contact.add_cross_filter(rid_a, link_a, rid_b, link_b)

        # FEM
        if self._fem_solver is not None:
            self._fem_solver.initialize(dt=dt)
            contact.register_fem(self._fem_solver)

        # Coupling
        coupling = None
        if self._coupling_maps and self._fem_solver is not None:
            from robosim.physics.coupling.penalty import PenaltyCoupling
            coupling = PenaltyCoupling(
                stiffness=coupling_stiffness,
                damping=coupling_damping,
            )
            # Use the first robot's solver for coupling
            first_solver = list(self._robots.values())[0].solver
            coupling.setup(first_solver, self._fem_solver, self._coupling_maps)

        return SceneWorld(
            scene=self,
            contact=contact,
            coupling=coupling,
            solvers=all_solvers,
            dt=dt,
        )

    def reset(self) -> None:
        """Reset all robots and objects to initial state."""
        for entry in self._robots.values():
            if entry.initial_q is not None:
                entry.robot.q = entry.initial_q.copy()
            else:
                entry.robot.q = np.zeros(entry.robot.n_dof)
            entry.robot.qd = np.zeros(entry.robot.n_dof)

        for entry in self._free_objects.values():
            entry.robot.q = np.zeros(entry.robot.n_dof)
            entry.robot.qd = np.zeros(entry.robot.n_dof)

    def summary(self) -> str:
        """Return a text summary of the scene."""
        lines = [
            f"Scene: ground_z={self.ground_height}, "
            f"gravity={self.gravity}",
            f"  Robots: {len(self._robots)}",
        ]
        for rid, entry in self._robots.items():
            r = entry.robot
            lines.append(
                f"    [{rid}] {r.n_dof} DOF, {r.n_links} links, "
                f"{sum(l.mass for l in r.links):.2f} kg"
            )
        lines.append(f"  Free objects: {len(self._free_objects)}")
        for oid, entry in self._free_objects.items():
            r = entry.robot
            lines.append(
                f"    [{oid}] mass={sum(l.mass for l in r.links):.2f} kg"
            )
        if self._fem_solver:
            lines.append(f"  Deformable bodies: {len(self._fem_solver.bodies)}")
        lines.append(f"  Obstacles: {len(self._obstacles)}")
        return "\n".join(lines)


class SceneWorld:
    """Simulation world built from a Scene.

    Handles the simulation loop, including multi-robot contact,
    FEM physics, and coupling.
    """

    def __init__(
        self,
        scene: Scene,
        contact: object,  # ContactSolver
        coupling: object | None,
        solvers: dict[str, object],  # id -> RBDSolver
        dt: float,
    ):
        self.scene = scene
        self.contact = contact
        self.coupling = coupling
        self.solvers = solvers
        self.dt = dt
        self._time = 0.0

    @property
    def time(self) -> float:
        return self._time

    def step(self) -> None:
        """Advance all physics by one time step."""
        # 1. Compute controllers
        for entry in self.scene.all_robots:
            if entry.controller is not None:
                tau = entry.controller(entry.robot, self._time)
                entry.solver.tau = tau

        # 2. Coupling forces
        rbd_extra: dict[str, dict[int, np.ndarray]] = {}
        fem_extra: dict[int, np.ndarray] = {}

        if self.coupling is not None:
            rbd_wrenches, fem_extra = self.coupling.compute_forces()
            # coupling returns per-link wrenches for the first robot
            first_rid = list(self.scene._robots.keys())[0]
            rbd_extra[first_rid] = rbd_wrenches

        # 3. Contact forces (computed once for all robots)
        all_contact_forces = self.contact.compute_all_rbd_contact_forces()

        # 4. Apply forces and step all robots
        for rid, solver in self.solvers.items():
            solver.clear_external_forces()

            # Contact
            for li, w in all_contact_forces.get(rid, {}).items():
                solver.set_external_force(li, w)

            # Coupling
            for li, w in rbd_extra.get(rid, {}).items():
                current = solver._f_ext.get(li, np.zeros(6))
                solver.set_external_force(li, current + w)

            solver.step()

        # 5. FEM step
        fem_solver = self.scene._fem_solver
        if fem_solver is not None:
            fem_solver.step(extra_forces=fem_extra if fem_extra else None)

            # FEM ground contact
            for body in fem_solver.bodies:
                self.contact.resolve_fem_contact(body, restitution=0.3, friction_mu=0.5)

            # FEM-FEM contact
            if len(fem_solver.bodies) > 1:
                self.contact.resolve_fem_fem_all(fem_solver)

            # RBD-FEM contact
            first_solver = list(self.solvers.values())[0] if self.solvers else None
            if first_solver is not None:
                self.contact.resolve_rbd_fem_all(
                    first_solver, fem_solver,
                    restitution=0.1, friction_mu=0.5,
                    dt=self.dt,
                )

        self._time += self.dt

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.scene.reset()
        self._time = 0.0
        for solver in self.solvers.values():
            solver._time = 0.0
