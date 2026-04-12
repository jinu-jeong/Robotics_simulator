"""Penalty-based RBD-FEM coupling.

Each boundary FEM node is connected to its target position on the RBD
link by a stiff spring.  The spring force pulls the FEM node toward the
link, and Newton's 3rd law pushes back on the RBD link as a spatial
wrench.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosim.math.transforms import Transform
from robosim.physics.coupling.interface import BoundaryMap


@dataclass
class PenaltyCoupling:
    """Penalty (spring) coupling between RBD links and FEM boundary nodes."""

    stiffness: float = 1e5      # spring constant (N/m)
    damping: float = 1e3        # velocity damping  (N·s/m)
    max_force: float = 1e4      # per-node force clamp (N), 0 = auto from mass
    boundary_maps: list[BoundaryMap] = field(default_factory=list)

    # References set by setup()
    _rbd_solver: object | None = field(default=None, repr=False)
    _fem_solver: object | None = field(default=None, repr=False)
    _node_masses: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)

    def setup(self, rbd_solver, fem_solver, boundary_maps: list[BoundaryMap]):
        self._rbd_solver = rbd_solver
        self._fem_solver = fem_solver
        self.boundary_maps = list(boundary_maps)
        # Cache per-node masses for mass-proportional damping/force clamping
        self._node_masses = {}
        for bmap in boundary_maps:
            body = fem_solver.bodies[bmap.fem_body_idx]
            M_diag = body._M.diagonal()
            for node_idx in bmap.fem_boundary_nodes:
                mass = M_diag[node_idx * 3]  # lumped mass (same for x,y,z)
                self._node_masses[(bmap.fem_body_idx, node_idx)] = mass

    def compute_forces(self) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Compute coupling forces for one time step.

        Returns
        -------
        rbd_wrenches : {link_idx: (6,) wrench [torque; force] in link frame}
        fem_forces   : {body_idx: (n_dof,) force vector}
        """
        robot = self._rbd_solver.robot
        fk = robot.forward_kinematics()
        self._precompute_link_velocities(fk)

        rbd_wrenches: dict[int, np.ndarray] = {}
        fem_forces: dict[int, np.ndarray] = {}

        for bmap in self.boundary_maps:
            link_idx = bmap.rigid_link_idx
            body_idx = bmap.fem_body_idx
            body = self._fem_solver.bodies[body_idx]

            T_world = fk[link_idx]
            T_inv = T_world.inverse()

            n_dof = body.mesh.n_nodes * 3
            if body_idx not in fem_forces:
                fem_forces[body_idx] = np.zeros(n_dof)
            if link_idx not in rbd_wrenches:
                rbd_wrenches[link_idx] = np.zeros(6)

            link_origin = T_world.translation

            for i, node_idx in enumerate(bmap.fem_boundary_nodes):
                # Target position in world frame
                target = T_world.apply_point(bmap.local_positions[i])
                actual = body.x[node_idx]
                disp = target - actual  # displacement: FEM node → target

                # Velocity damping (mass-proportional limit)
                v_node = body.v[node_idx]
                v_target = self._link_point_velocity(link_idx, target, fk=fk)
                v_rel = v_target - v_node

                # Clamp damping at critical damping for this node's mass
                node_mass = self._node_masses.get((body_idx, node_idx), 1.0)
                c_crit = 2.0 * np.sqrt(self.stiffness * node_mass)
                c_eff = min(self.damping, c_crit)

                # Spring-damper force on FEM node (toward target)
                f_world = self.stiffness * disp + c_eff * v_rel

                # Clamp force: limit velocity change to 2 m/s per step
                dt = self._fem_solver.dt
                f_mass_limit = node_mass * 2.0 / dt  # Δv < 2 m/s
                f_max = min(self.max_force, f_mass_limit)
                f_mag = np.linalg.norm(f_world)
                if f_mag > f_max:
                    f_world *= f_max / f_mag

                # Apply to FEM node
                fem_forces[body_idx][node_idx * 3: node_idx * 3 + 3] += f_world

                # Reaction on RBD link (Newton's 3rd): -f in link frame
                f_react_world = -f_world
                f_react_link = T_inv.apply_vector(f_react_world)
                r_link = bmap.local_positions[i]  # lever arm in link frame
                tau_link = np.cross(r_link, f_react_link)

                rbd_wrenches[link_idx][:3] += tau_link
                rbd_wrenches[link_idx][3:] += f_react_link

        return rbd_wrenches, fem_forces

    def _precompute_link_velocities(self, fk):
        """Precompute (omega, v_origin) per link for velocity queries."""
        from robosim.model.joint import JointType
        robot = self._rbd_solver.robot

        if not hasattr(self, '_vel_cache') or self._vel_cache is None:
            self._vel_cache = {}

        self._vel_cache.clear()

        for link_idx in range(robot.n_links):
            omega = np.zeros(3)
            v_origin = np.zeros(3)

            path = []
            idx = link_idx
            while idx >= 0:
                path.append(idx)
                idx = robot.parent_index(idx)
            path.reverse()

            for i in path:
                j_idx = robot.joint_index_for_link(i)
                if j_idx is None or j_idx < 0:
                    continue
                joint = robot.joints[j_idx]
                qd_j = robot.qd[j_idx]
                T_world_link = fk[i]

                if joint.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
                    axis_world = T_world_link.rotation @ joint.axis
                    omega += qd_j * axis_world
                elif joint.joint_type == JointType.PRISMATIC:
                    axis_world = T_world_link.rotation @ joint.axis
                    v_origin += qd_j * axis_world

            self._vel_cache[link_idx] = (omega, v_origin)

    def _link_point_velocity(self, link_idx: int, point_world: np.ndarray,
                             fk=None) -> np.ndarray:
        """Compute world velocity of a point on an RBD link (uses cache)."""
        omega, v_origin = self._vel_cache[link_idx]
        if fk is None:
            fk = self._rbd_solver.robot.forward_kinematics()
        r = point_world - fk[link_idx].translation
        return v_origin + np.cross(omega, r)
