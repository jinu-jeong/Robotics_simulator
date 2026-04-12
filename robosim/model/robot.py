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
        """Get the transform across the joint: origin * joint_motion(q)."""
        j = self.joints[joint_idx]
        q_val = self.get_joint_q(joint_idx)
        jtype = "revolute" if j.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS) else j.joint_type.value
        T_joint = joint_transform(jtype, j.axis, q_val)
        return j.origin.compose(T_joint)

    def forward_kinematics(self) -> list[Transform]:
        """Compute world-frame transforms for all links.

        Returns a list of Transform objects, one per link, representing
        the transform from link frame to world frame.
        """
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

        return T_world

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
