"""Boundary map: defines which FEM nodes attach to which RBD link."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robosim.math.transforms import Transform


@dataclass
class BoundaryMap:
    """Connection between an RBD link and a set of FEM boundary nodes.

    Attributes
    ----------
    rigid_link_idx : index into robot.links
    fem_body_idx : index into FEMSolver.bodies
    fem_boundary_nodes : (n_boundary,) FEM node indices that are glued to the link
    local_positions : (n_boundary, 3) positions of those nodes expressed
                      in the **link frame** at setup time.  At each step the
                      solver computes target_world = FK[link] @ local_pos.
    """

    rigid_link_idx: int
    fem_body_idx: int
    fem_boundary_nodes: np.ndarray  # (n_boundary,)
    local_positions: np.ndarray  # (n_boundary, 3)


def create_boundary_map(
    robot,
    link_idx: int,
    fem_body,
    fem_body_idx: int,
    boundary_nodes: np.ndarray,
) -> BoundaryMap:
    """Build a BoundaryMap from explicit node indices.

    Computes *local_positions* by transforming the current FEM node
    world positions into the link frame using FK.
    """
    fk = robot.forward_kinematics()
    T_world_link = fk[link_idx]
    T_link_world = T_world_link.inverse()

    world_pos = fem_body.x[boundary_nodes]  # (n, 3)
    local_pos = np.array([T_link_world.apply_point(p) for p in world_pos])

    return BoundaryMap(
        rigid_link_idx=link_idx,
        fem_body_idx=fem_body_idx,
        fem_boundary_nodes=boundary_nodes.copy(),
        local_positions=local_pos,
    )


def select_face_nodes(
    mesh_nodes: np.ndarray,
    axis: int,
    side: str = "max",
    tol: float = 1e-6,
) -> np.ndarray:
    """Select nodes on a face of an axis-aligned box mesh.

    Parameters
    ----------
    mesh_nodes : (n_nodes, 3)
    axis : 0=X, 1=Y, 2=Z
    side : "min" or "max"
    tol : tolerance for face membership

    Returns
    -------
    node_indices : (n_face,) integer array
    """
    vals = mesh_nodes[:, axis]
    target = vals.max() if side == "max" else vals.min()
    return np.where(np.abs(vals - target) < tol)[0]
