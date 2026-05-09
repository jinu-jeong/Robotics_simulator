"""Robot model: kinematic tree with joint state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from robosim.math.spatial import (
    SpatialInertia,
    joint_transform,
    motion_subspace_prismatic,
    motion_subspace_revolute,
)
from robosim.math.transforms import Transform
from robosim.model.joint import Joint, JointType
from robosim.model.link import Link


@dataclass
class Robot:
    """Articulated rigid body system defined as a kinematic tree.

    The tree is stored in a flat array indexed by link index.
    Link 0 is the root (base) link. Joints connect parent -> child.

    Attributes
    ----------
    name : Robot identifier
    links : Ordered list of links (index 0 = root)
    joints : Ordered list of joints
    gravity : (3,) gravity vector in world frame
    """

    name: str
    links: list[Link] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81])
    )

    # Internal mappings (built by _build_tree)
    _link_name_to_idx: dict[str, int] = field(default_factory=dict, repr=False)
    _joint_name_to_idx: dict[str, int] = field(default_factory=dict, repr=False)
    _parent: list[int] = field(default_factory=list, repr=False)
    _children: list[list[int]] = field(default_factory=list, repr=False)
    _joint_for_link: list[int] = field(default_factory=list, repr=False)
    _dof_index: list[int] = field(default_factory=list, repr=False)

    # State
    _q: Optional[np.ndarray] = field(default=None, repr=False)
    _qd: Optional[np.ndarray] = field(default=None, repr=False)

    # FK cache — invalidated when ``q`` changes.
    _fk_cache: Optional[list] = field(default=None, repr=False)
    _fk_q_snapshot: Optional[np.ndarray] = field(default=None, repr=False)

    # Per-link world-frame spatial velocity cache: list of (omega, v_origin),
    # where v_origin is the linear velocity of the link's frame origin.
    # Recomputed lazily by :meth:`link_world_velocities` and invalidated when
    # ``q`` or ``qd`` changes.
    _vel_cache: Optional[list] = field(default=None, repr=False)
    _vel_q_snapshot: Optional[np.ndarray] = field(default=None, repr=False)
    _vel_qd_snapshot: Optional[np.ndarray] = field(default=None, repr=False)

    # Per-joint local transform cache: list[Transform], one per joint.
    # ABA / RNEA / gravity_torques each iterate every joint computing the
    # same transform from the same q — this caches across those calls.
    _jlt_cache: Optional[list] = field(default=None, repr=False)
    _jlt_q_snapshot: Optional[np.ndarray] = field(default=None, repr=False)

    # Per-joint spatial transform matrix cache (X = X(joint_local_transform^-1)).
    # Filled on first request via a (Robot, j_idx) -> (6, 6) ndarray map.
    _xspatial_cache: Optional[list] = field(default=None, repr=False)
    _xspatial_q_snapshot: Optional[np.ndarray] = field(default=None, repr=False)

    def build(self) -> Robot:
        """Build internal data structures from links and joints. Call after loading."""
        self._link_name_to_idx = {link.name: i for i, link in enumerate(self.links)}
        self._joint_name_to_idx = {j.name: i for i, j in enumerate(self.joints)}

        n_links = len(self.links)
        self._parent = [-1] * n_links
        self._children = [[] for _ in range(n_links)]
        self._joint_for_link = [-1] * n_links

        for j_idx, joint in enumerate(self.joints):
            parent_idx = self._link_name_to_idx[joint.parent_link]
            child_idx = self._link_name_to_idx[joint.child_link]
            self._parent[child_idx] = parent_idx
            self._children[parent_idx].append(child_idx)
            self._joint_for_link[child_idx] = j_idx

        # Compute DOF offsets for each joint
        self._dof_index = [0] * len(self.joints)
        dof_count = 0
        for j_idx, joint in enumerate(self.joints):
            self._dof_index[j_idx] = dof_count
            dof_count += joint.num_dof

        # Initialize state
        self._q = np.zeros(dof_count)
        self._qd = np.zeros(dof_count)

        # Build mimic joint map: follower_joint_idx -> (leader_joint_idx, multiplier, offset)
        self._mimic_map: dict[int, tuple[int, float, float]] = {}
        for j_idx, joint in enumerate(self.joints):
            if joint.mimic_joint is not None:
                leader_idx = self._joint_name_to_idx.get(joint.mimic_joint)
                if leader_idx is not None:
                    self._mimic_map[j_idx] = (
                        leader_idx,
                        joint.mimic_multiplier,
                        joint.mimic_offset,
                    )

        return self

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    @property
    def n_dof(self) -> int:
        """Total number of degrees of freedom (actuated joints only)."""
        return sum(j.num_dof for j in self.joints)

    @property
    def q(self) -> np.ndarray:
        """Joint positions vector."""
        if self._q is None:
            self._q = np.zeros(self.n_dof)
        return self._q

    @q.setter
    def q(self, value: np.ndarray):
        self._q = np.asarray(value, dtype=np.float64)

    @property
    def qd(self) -> np.ndarray:
        """Joint velocities vector."""
        if self._qd is None:
            self._qd = np.zeros(self.n_dof)
        return self._qd

    @qd.setter
    def qd(self, value: np.ndarray):
        self._qd = np.asarray(value, dtype=np.float64)

    def parent_index(self, link_idx: int) -> int:
        """Get parent link index (-1 for root)."""
        return self._parent[link_idx]

    def children_indices(self, link_idx: int) -> list[int]:
        """Get child link indices."""
        return self._children[link_idx]

    def joint_index_for_link(self, link_idx: int) -> int:
        """Get the joint index connecting parent to this link (-1 for root)."""
        return self._joint_for_link[link_idx]

    def link_index(self, name: str) -> int:
        return self._link_name_to_idx[name]

    def joint_index(self, name: str) -> int:
        return self._joint_name_to_idx[name]

    def get_joint_q(self, joint_idx: int) -> float:
        """Get position of a single joint."""
        j = self.joints[joint_idx]
        if j.num_dof == 0:
            return 0.0
        return float(self.q[self._dof_index[joint_idx]])

    def set_joint_q(self, joint_idx: int, value: float):
        """Set position of a single joint."""
        j = self.joints[joint_idx]
        if j.num_dof > 0:
            self.q[self._dof_index[joint_idx]] = value

    def get_joint_qd(self, joint_idx: int) -> float:
        """Get velocity of a single joint."""
        j = self.joints[joint_idx]
        if j.num_dof == 0:
            return 0.0
        return float(self.qd[self._dof_index[joint_idx]])

    def joint_motion_subspace(self, joint_idx: int) -> np.ndarray:
        """Get the 6xn_dof motion subspace matrix S for a joint."""
        j = self.joints[joint_idx]
        if j.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
            return motion_subspace_revolute(j.axis)
        elif j.joint_type == JointType.PRISMATIC:
            return motion_subspace_prismatic(j.axis)
        elif j.joint_type == JointType.FIXED:
            return np.zeros((6, 0))
        else:
            raise NotImplementedError(f"Motion subspace for {j.joint_type}")

    def joint_local_transform(self, joint_idx: int) -> Transform:
        """Get the transform across the joint: origin * joint_motion(q).

        Cached across joints + reused while ``q`` is unchanged.
        """
        cache = self._jlt_cache
        if cache is not None and self._jlt_q_snapshot is not None:
            snap = self._jlt_q_snapshot
            q = self._q
            if snap.shape == q.shape and (snap == q).all():
                return cache[joint_idx]
            cache = None  # stale

        # Rebuild full cache (cheap and avoids per-joint snapshot bookkeeping)
        n_joints = len(self.joints)
        new_cache = [None] * n_joints
        for j_idx in range(n_joints):
            j = self.joints[j_idx]
            q_val = self.get_joint_q(j_idx)
            jtype = ("revolute" if j.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS)
                     else j.joint_type.value)
            T_joint = joint_transform(jtype, j.axis, q_val)
            new_cache[j_idx] = j.origin.compose(T_joint)
        self._jlt_cache = new_cache
        self._jlt_q_snapshot = self._q.copy()
        return new_cache[joint_idx]

    def forward_kinematics(self) -> list[Transform]:
        """Compute world-frame transforms for all links.

        Cached against the current ``q``: subsequent calls without a
        configuration change reuse the result. Callers that mutate the
        returned Transforms should call :meth:`invalidate_fk_cache`.
        """
        # Cache hit: q unchanged since last computation.  Fast path uses
        # ``ndarray.tobytes()`` equality which beats ``np.array_equal``
        # (which spends ~6 µs/call on dispatch overhead alone).
        if self._fk_cache is not None and self._fk_q_snapshot is not None:
            snap = self._fk_q_snapshot
            q = self._q
            if snap.shape == q.shape and (snap == q).all():
                return self._fk_cache

        T_world = [Transform.identity() for _ in range(self.n_links)]

        # BFS from root
        from collections import deque

        queue = deque()
        # Find root(s) — links with no parent
        for i in range(self.n_links):
            if self._parent[i] == -1:
                T_world[i] = Transform.identity()
                queue.append(i)

        while queue:
            link_idx = queue.popleft()
            for child_idx in self._children[link_idx]:
                j_idx = self._joint_for_link[child_idx]
                T_local = self.joint_local_transform(j_idx)
                T_world[child_idx] = T_world[link_idx].compose(T_local)
                queue.append(child_idx)

        self._fk_cache = T_world
        self._fk_q_snapshot = self._q.copy() if self._q is not None else None
        return T_world

    def invalidate_fk_cache(self) -> None:
        """Force the next ``forward_kinematics()`` call to recompute.

        Use after directly mutating an FK Transform returned by an earlier
        call. Mutating ``robot.q`` does not require an explicit invalidation
        — the cache compares ``q`` snapshots automatically.
        """
        self._fk_cache = None
        self._fk_q_snapshot = None
        self._vel_cache = None
        self._vel_q_snapshot = None
        self._vel_qd_snapshot = None
        self._jlt_cache = None
        self._jlt_q_snapshot = None
        self._xspatial_cache = None
        self._xspatial_q_snapshot = None

    def joint_spatial_transforms(self) -> list[np.ndarray]:
        """Return cached ``X(joint_local_transform^-1)`` for every joint.

        ABA, RNEA, and CRBA each rebuild this transform per joint per call.
        Cache it across calls (within an unchanged ``q``) since they all run
        in the same substep on the same ``q``.
        """
        cache = self._xspatial_cache
        if cache is not None and self._xspatial_q_snapshot is not None:
            snap = self._xspatial_q_snapshot
            q = self._q
            if snap.shape == q.shape and (snap == q).all():
                return cache

        # Build for every joint.  Imported here to avoid circular import at
        # module load time.
        from robosim.physics.rbd.algorithms import _spatial_transform_matrix

        n_joints = len(self.joints)
        new_cache: list[np.ndarray] = [None] * n_joints  # type: ignore
        for j_idx in range(n_joints):
            T_local = self.joint_local_transform(j_idx)
            T_inv = T_local.inverse()
            new_cache[j_idx] = _spatial_transform_matrix(T_inv)
        self._xspatial_cache = new_cache
        self._xspatial_q_snapshot = self._q.copy()
        return new_cache

    def link_world_velocities(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per-link world-frame spatial velocity (omega, v_origin).

        Returns ``[(omega_i, v_origin_i)]`` for each link i, where
        ``omega_i`` is the link's angular velocity in the world frame and
        ``v_origin_i`` is the linear velocity of the link's frame origin.

        Cached against ``(q, qd)``.  Use the velocity at a body-fixed point
        ``p_world`` via ``v = v_origin + omega × (p_world - link_origin)``.
        """
        from robosim.model.joint import JointType

        if (self._vel_cache is not None
                and self._vel_q_snapshot is not None
                and self._vel_qd_snapshot is not None):
            qs, qds = self._vel_q_snapshot, self._vel_qd_snapshot
            q, qd = self._q, self._qd
            if (qs.shape == q.shape and (qs == q).all()
                    and qds.shape == qd.shape and (qds == qd).all()):
                return self._vel_cache

        fk = self.forward_kinematics()
        n = self.n_links
        result: list[tuple[np.ndarray, np.ndarray]] = [None] * n  # type: ignore

        # BFS from root, propagating spatial velocity outward.
        from collections import deque
        roots = [i for i in range(n) if self._parent[i] == -1]
        for r in roots:
            result[r] = (np.zeros(3), np.zeros(3))
        queue = deque(roots)

        while queue:
            link_idx = queue.popleft()
            omega_p, v_origin_p = result[link_idx]
            origin_p = fk[link_idx].translation
            for child_idx in self._children[link_idx]:
                origin_c = fk[child_idx].translation
                # Propagate from parent: v_c = v_p + ω_p × (origin_c - origin_p)
                d = origin_c - origin_p
                v_c = v_origin_p + np.array([
                    omega_p[1]*d[2] - omega_p[2]*d[1],
                    omega_p[2]*d[0] - omega_p[0]*d[2],
                    omega_p[0]*d[1] - omega_p[1]*d[0],
                ])
                omega_c = omega_p

                j_idx = self._joint_for_link[child_idx]
                if j_idx >= 0:
                    joint = self.joints[j_idx]
                    jt = joint.joint_type
                    if jt != JointType.FIXED:
                        dof_idx = self._dof_index[j_idx]
                        qd_j = float(self._qd[dof_idx])
                        # Joint axis in WORLD frame uses the CHILD link transform
                        axis_world = fk[child_idx].rotation @ joint.axis
                        if jt in (JointType.REVOLUTE, JointType.CONTINUOUS):
                            omega_c = omega_p + qd_j * axis_world
                        elif jt == JointType.PRISMATIC:
                            v_c = v_c + qd_j * axis_world

                result[child_idx] = (omega_c, v_c)
                queue.append(child_idx)

        # Fallback: any link without an explicit parent (disconnected) gets zero.
        for i in range(n):
            if result[i] is None:
                result[i] = (np.zeros(3), np.zeros(3))

        self._vel_cache = result
        self._vel_q_snapshot = self._q.copy()
        self._vel_qd_snapshot = self._qd.copy()
        return result

    def enforce_mimic(self):
        """Synchronize mimic (follower) joints to their leader joints.

        For each mimic joint:  q_follower = multiplier * q_leader + offset
                               qd_follower = multiplier * qd_leader
        """
        for follower_idx, (leader_idx, mult, offset) in self._mimic_map.items():
            f_dof = self._dof_index[follower_idx]
            l_dof = self._dof_index[leader_idx]
            if self.joints[follower_idx].num_dof > 0 and self.joints[leader_idx].num_dof > 0:
                self._q[f_dof] = mult * self._q[l_dof] + offset
                self._qd[f_dof] = mult * self._qd[l_dof]

    @property
    def has_mimic(self) -> bool:
        """True if any mimic joints are defined."""
        return len(self._mimic_map) > 0

    def print_tree(self, indent: int = 0, link_idx: int = 0):
        """Print the kinematic tree structure."""
        prefix = "  " * indent
        link = self.links[link_idx]
        j_idx = self._joint_for_link[link_idx]
        if j_idx >= 0:
            joint = self.joints[j_idx]
            print(f"{prefix}[{joint.joint_type.value}] {joint.name}")
        print(f"{prefix}  -> {link.name} (mass={link.mass:.3f})")

        for child_idx in self._children[link_idx]:
            self.print_tree(indent + 2, child_idx)

    def __repr__(self) -> str:
        return (
            f"Robot('{self.name}', links={self.n_links}, "
            f"joints={self.n_joints}, dof={self.n_dof})"
        )
