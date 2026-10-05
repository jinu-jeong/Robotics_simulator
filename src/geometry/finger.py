"""Rectangular compliant-finger geometry and structured hex-dominant meshing.

Coordinate convention (project-wide)
------------------------------------
* ``x`` in ``[0, L]`` : finger length. The root surface ``x = 0`` is clamped.
* ``y`` in ``[0, W]`` : finger width. The inner grasp face is ``y = W``.
* ``z`` in ``[0, H]`` : finger thickness, ``+z`` is "up".

Units: SI meters.

Meshing
-------
A structured ``nx × ny × nz`` hexahedral grid is generated. Cells are
20-node serendipity hexahedra (H20); the ``tet_fraction`` of the length at
the finger tip (default 20 %) is split into 10-node tetrahedra (T10) with the
Kuhn subdivision (6 tets per cell), mirroring what a hex-dominant mesher does
where hexes cannot be swept. The hex↔tet interface is made conforming with
hanging-node tie constraints: the mid-diagonal node of each interface quad is
slaved to the quad's 8 nodes (``u_s = Σ N_i(0,0) u_i``, corner weight −¼,
mid-edge weight ½).

The boundary is exposed twice: as quadratic parent faces (for bookkeeping)
and as ``surface_faces`` — a triangulation using *all* boundary nodes
(corner + mid-edge). Contact loads, markers, cameras and the viewer work on
that triangulation exactly as they did on linear tets.

Node ordering: corner grid ``node(i, j, k) = (i (ny+1) + j)(nz+1) + k`` first,
then one mid-edge node per unique element edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Mapping

import numpy as np

from .mesh_utils import select_nodes_on_plane, tet_signed_volumes, unique_edges


@dataclass(frozen=True)
class FingerGeometry:
    """Rectangular finger dimensions in meters."""

    length: float = 0.10
    width: float = 0.02
    height: float = 0.01

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "FingerGeometry":
        g = cfg.get("geometry", cfg)
        return cls(float(g["length"]), float(g["width"]), float(g["height"]))

    @property
    def size(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height], dtype=float)

    def point_from_relative(self, rel) -> np.ndarray:
        """Map fractions ``(fx, fy, fz)`` in [0,1]^3 to a point in meters."""
        return np.asarray(rel, dtype=float) * self.size


@dataclass
class FEMesh:
    """Quadratic hex-dominant mesh (H20 + T10) with pre-extracted boundary data.

    Attributes
    ----------
    nodes : (N, 3) float64 — corner nodes first, then mid-edge nodes
    tets : (Mt, 10) int64, positively oriented T10 (may be empty)
    hexes : (Mh, 20) int64, H20 in VTK order (may be empty)
    surface_faces : (F, 3) int64, outward boundary triangulation over all boundary nodes
    surface_quads : (Fq, 8) int64, boundary quad8 parent faces (corners then mid-edges)
    surface_tris : (Ft, 6) int64, boundary tri6 parent faces
    surface_edges : (Es, 2) int64, corner edges of the boundary parent faces (wireframe)
    element_edges : (Ee, 2) int64, corner edges of all elements
    tie_slaves : (S,) hanging nodes; tie_masters (S, 8); tie_weights (S, 8)
    meta : free-form dictionary (grid resolution, geometry, ...)
    """

    nodes: np.ndarray
    tets: np.ndarray
    hexes: np.ndarray
    surface_faces: np.ndarray
    surface_quads: np.ndarray
    surface_tris: np.ndarray
    surface_edges: np.ndarray
    element_edges: np.ndarray
    tie_slaves: np.ndarray
    tie_masters: np.ndarray
    tie_weights: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ props
    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_tets(self) -> int:
        return int(self.tets.shape[0])

    @property
    def n_hexes(self) -> int:
        return int(self.hexes.shape[0])

    @property
    def n_elements(self) -> int:
        return self.n_tets + self.n_hexes

    @property
    def n_dofs(self) -> int:
        return 3 * self.n_nodes

    @property
    def n_ties(self) -> int:
        return int(self.tie_slaves.shape[0])

    @property
    def tet_edges(self) -> np.ndarray:  # backwards-compatible alias
        return self.element_edges

    @property
    def surface_nodes(self) -> np.ndarray:
        return np.unique(self.surface_faces.ravel())

    @property
    def element_blocks(self) -> list[np.ndarray]:
        """Non-empty connectivity blocks in assembly order (tets, hexes)."""
        return [c for c in (self.tets, self.hexes) if len(c)]

    def volumes(self) -> np.ndarray:
        """Element volumes in assembly order (tets, then hexes)."""
        from ..fem.elements import element_volumes

        parts = [element_volumes(self.nodes, c) for c in self.element_blocks]
        return np.concatenate(parts) if parts else np.zeros(0)

    def total_volume(self) -> float:
        return float(self.volumes().sum())

    def hex_volume_fraction(self) -> float:
        v = self.volumes()
        if v.size == 0:
            return 0.0
        return float(v[self.n_tets:].sum() / v.sum())

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    def nodes_on_plane(self, axis: int, value: float, tol: float = 1e-9) -> np.ndarray:
        return select_nodes_on_plane(self.nodes, axis, value, tol)

    def summary(self) -> str:
        return (f"{self.n_nodes} nodes, {self.n_hexes} hex20 + {self.n_tets} tet10 "
                f"({100 * self.hex_volume_fraction():.0f} % hex by volume), {self.n_ties} ties")


TetMesh = FEMesh  # legacy name used across the code base


# --------------------------------------------------------------------------- #
# Kuhn subdivision of the unit cube into 6 tetrahedra.
# Corner index c(dx,dy,dz) = dx*4 + dy*2 + dz.
# --------------------------------------------------------------------------- #
def _kuhn_tets() -> np.ndarray:
    tets = []
    for perm in permutations(range(3)):
        corner = np.zeros(3, dtype=int)
        path = [corner.copy()]
        for axis in perm:
            corner = corner.copy()
            corner[axis] = 1
            path.append(corner)
        tets.append([int(c[0] * 4 + c[1] * 2 + c[2]) for c in path])
    return np.array(tets, dtype=np.int64)


_KUHN = _kuhn_tets()
# grid corner c(dx,dy,dz) -> VTK hex corner order (-,-,-) (+,-,-) (+,+,-) (-,+,-) (-,-,+) (+,-,+) (+,+,+) (-,+,+)
_VTK_FROM_GRID = np.array([0, 4, 6, 2, 1, 5, 7, 3], dtype=np.int64)


def _sorted_pairs(a: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(a, dtype=np.int64).reshape(-1, 2), axis=1)


def _row_keys(a: np.ndarray) -> np.ndarray:
    """Sorted-row view usable as dictionary keys (tuple per row)."""
    return np.sort(np.asarray(a, dtype=np.int64), axis=1)


def build_quadratic_mesh(
    nodes_c: np.ndarray,
    hex8: np.ndarray,
    tet4: np.ndarray,
    meta: dict | None = None,
) -> FEMesh:
    """Promote corner-node hex8 (VTK order) / tet4 connectivity to H20 / T10 and
    extract boundary, wireframe and hanging-node ties."""
    from ..fem.elements import H20_EDGES, H20_FACES, T10_EDGES, T10_FACES

    nodes_c = np.ascontiguousarray(nodes_c, dtype=np.float64).reshape(-1, 3)
    hex8 = np.asarray(hex8, dtype=np.int64).reshape(-1, 8)
    tet4 = np.asarray(tet4, dtype=np.int64).reshape(-1, 4)
    if len(tet4):  # positive orientation (swap last two nodes of inverted tets)
        neg = tet_signed_volumes(nodes_c, tet4) < 0
        tet4 = tet4.copy()
        tet4[neg, 2], tet4[neg, 3] = tet4[neg, 3].copy(), tet4[neg, 2].copy()
    Nc = nodes_c.shape[0]

    # ---- unique corner edges -> one mid node each ------------------------------------
    hex_e = _sorted_pairs(hex8[:, H20_EDGES]) if len(hex8) else np.zeros((0, 2), np.int64)
    tet_e = _sorted_pairs(tet4[:, T10_EDGES]) if len(tet4) else np.zeros((0, 2), np.int64)
    all_e = np.concatenate([hex_e, tet_e], axis=0)
    edges, inv = np.unique(all_e, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    mid_of = Nc + inv  # mid node index for every (element, local edge) in all_e order
    nodes = np.concatenate([nodes_c, 0.5 * (nodes_c[edges[:, 0]] + nodes_c[edges[:, 1]])], axis=0)
    hex_mid = mid_of[: hex_e.shape[0]].reshape(-1, 12)
    tet_mid = mid_of[hex_e.shape[0]:].reshape(-1, 6)
    hex20 = np.concatenate([hex8, hex_mid], axis=1) if len(hex8) else np.zeros((0, 20), np.int64)
    tet10 = np.concatenate([tet4, tet_mid], axis=1) if len(tet4) else np.zeros((0, 10), np.int64)
    edge_id = {(int(a), int(b)): Nc + k for k, (a, b) in enumerate(edges)}

    def mid(a: int, b: int) -> int:
        return edge_id[(min(a, b), max(a, b))]

    # ---- corner faces: boundary detection with hex↔tet interface matching -------------
    quads = hex8[:, H20_FACES].reshape(-1, 4) if len(hex8) else np.zeros((0, 4), np.int64)  # (6Mh, 4) oriented
    quad_owner = np.repeat(np.arange(len(hex8)), 6)
    tris = tet4[:, T10_FACES].reshape(-1, 3) if len(tet4) else np.zeros((0, 3), np.int64)  # (4Mt, 3) oriented
    tri_owner = np.repeat(np.arange(len(tet4)), 4)

    qkey = _row_keys(quads)
    _, q_first, q_inv, q_cnt = np.unique(qkey, axis=0, return_index=True, return_inverse=True, return_counts=True)
    q_inv = q_inv.reshape(-1)
    tkey = _row_keys(tris)
    _, t_first, t_inv, t_cnt = np.unique(tkey, axis=0, return_index=True, return_inverse=True, return_counts=True)
    t_inv = t_inv.reshape(-1)
    tri_set = {tuple(r) for r in tkey}
    # halves of every hex quad along both diagonals (as sorted tri keys)
    half_tris: set[tuple] = set()
    quad_halves = []
    for q in quads:
        a, b, c, d = (int(v) for v in q)
        h = [tuple(sorted((a, b, c))), tuple(sorted((a, c, d))), tuple(sorted((a, b, d))), tuple(sorted((b, c, d)))]
        quad_halves.append(h)
        half_tris.update(h)

    quad_boundary = np.zeros(len(quads), dtype=bool)
    for i in range(len(quads)):
        if q_cnt[q_inv[i]] != 1:
            continue
        h = quad_halves[i]
        covered = (h[0] in tri_set and h[1] in tri_set) or (h[2] in tri_set and h[3] in tri_set)
        quad_boundary[i] = not covered
    tri_boundary = np.array([t_cnt[t_inv[i]] == 1 and tuple(tkey[i]) not in half_tris for i in range(len(tris))], dtype=bool)

    # ---- boundary parent faces (quad8 / tri6) and their display triangulation ---------
    bq = quads[quad_boundary]
    bt = tris[tri_boundary]
    surface_quads = np.zeros((len(bq), 8), np.int64)
    surface_tris = np.zeros((len(bt), 6), np.int64)
    disp = []
    wire = []
    for r, q in enumerate(bq):
        c0, c1, c2, c3 = (int(v) for v in q)
        m01, m12, m23, m30 = mid(c0, c1), mid(c1, c2), mid(c2, c3), mid(c3, c0)
        surface_quads[r] = [c0, c1, c2, c3, m01, m12, m23, m30]
        disp += [[c0, m01, m30], [c1, m12, m01], [c2, m23, m12], [c3, m30, m23], [m01, m12, m23], [m01, m23, m30]]
        wire += [[c0, c1], [c1, c2], [c2, c3], [c3, c0]]
    for r, t in enumerate(bt):
        c0, c1, c2 = (int(v) for v in t)
        m01, m12, m02 = mid(c0, c1), mid(c1, c2), mid(c0, c2)
        surface_tris[r] = [c0, c1, c2, m01, m12, m02]
        disp += [[c0, m01, m02], [m01, c1, m12], [m02, m12, c2], [m01, m12, m02]]
        wire += [[c0, c1], [c1, c2], [c2, c0]]
    surface_faces = np.asarray(disp, dtype=np.int64).reshape(-1, 3)
    # orientation: parent faces were emitted outward for positively oriented elements; enforce
    # geometrically against the owning element centroid to be safe
    owner_centroid = np.concatenate([
        np.repeat(nodes_c[bq].mean(axis=1), 6, axis=0) if len(bq) else np.zeros((0, 3)),
        np.repeat(nodes_c[bt].mean(axis=1), 4, axis=0) if len(bt) else np.zeros((0, 3)),
    ], axis=0)
    owners = np.concatenate([
        nodes_c[hex8[quad_owner[quad_boundary]]].mean(axis=1).repeat(6, axis=0) if len(bq) else np.zeros((0, 3)),
        nodes_c[tet4[tri_owner[tri_boundary]]].mean(axis=1).repeat(4, axis=0) if len(bt) else np.zeros((0, 3)),
    ], axis=0)
    p0, p1, p2 = nodes[surface_faces[:, 0]], nodes[surface_faces[:, 1]], nodes[surface_faces[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    inward = np.einsum("ij,ij->i", nrm, owner_centroid - owners) < 0
    surface_faces[inward] = surface_faces[inward][:, [0, 2, 1]]
    surface_edges = np.unique(_sorted_pairs(np.asarray(wire, dtype=np.int64)), axis=0) if wire else np.zeros((0, 2), np.int64)

    # ---- hanging-node ties at the hex↔tet interface -----------------------------------
    diag_face: dict[tuple, list[int]] = {}
    for q in quads:
        c0, c1, c2, c3 = (int(v) for v in q)
        face8 = [c0, c1, c2, c3, mid(c0, c1), mid(c1, c2), mid(c2, c3), mid(c3, c0)]
        diag_face[(min(c0, c2), max(c0, c2))] = face8
        diag_face[(min(c1, c3), max(c1, c3))] = face8
    hex_edge_set = {(int(a), int(b)) for a, b in hex_e}
    slaves, masters = [], []
    for a, b in np.unique(tet_e, axis=0) if len(tet_e) else []:
        key = (int(a), int(b))
        if key in hex_edge_set:
            continue
        f8 = diag_face.get(key)
        if f8 is not None:
            slaves.append(edge_id[key])
            masters.append(f8)
    tie_slaves = np.asarray(slaves, dtype=np.int64)
    tie_masters = np.asarray(masters, dtype=np.int64).reshape(-1, 8)
    tie_weights = np.tile(np.array([-0.25] * 4 + [0.5] * 4), (len(slaves), 1))

    element_edges = edges.copy()
    return FEMesh(
        nodes=nodes, tets=tet10, hexes=hex20,
        surface_faces=surface_faces, surface_quads=surface_quads, surface_tris=surface_tris,
        surface_edges=surface_edges, element_edges=element_edges,
        tie_slaves=tie_slaves, tie_masters=tie_masters, tie_weights=tie_weights,
        meta=dict(meta or {}),
    )


def resolution_from_element_size(geometry: FingerGeometry, element_size: float) -> tuple[int, int, int]:
    """Cells per axis for a target edge length (at least 2 through the thickness)."""
    h = float(element_size)
    if h <= 0.0:
        raise ValueError("element_size must be positive")
    return tuple(max(2, int(round(d / h))) for d in geometry.size)  # type: ignore[return-value]


def default_element_size(geometry: FingerGeometry) -> float:
    """Commercial-mesher style default: a quarter of the smallest dimension."""
    return float(geometry.size.min()) / 4.0


def make_rectangular_finger_mesh(
    geometry: FingerGeometry,
    nx: int,
    ny: int,
    nz: int,
    tet_fraction: float = 0.2,
) -> FEMesh:
    """Structured hex-dominant quadratic mesh of a rectangular finger.

    Parameters
    ----------
    geometry : FingerGeometry
    nx, ny, nz : number of cells along x, y, z (each >= 1).
    tet_fraction : fraction of the length at the tip meshed with T10 (0 → all
        H20, 1 → all T10 via the Kuhn split). Default 0.2.

    Returns
    -------
    FEMesh with ``(nx+1)(ny+1)(nz+1)`` corner nodes plus one node per edge.
    """
    if min(nx, ny, nz) < 1:
        raise ValueError("nx, ny, nz must be >= 1")
    if not 0.0 <= tet_fraction <= 1.0:
        raise ValueError("tet_fraction must be in [0, 1]")

    xs = np.linspace(0.0, geometry.length, nx + 1)
    ys = np.linspace(0.0, geometry.width, ny + 1)
    zs = np.linspace(0.0, geometry.height, nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes_c = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    corners = np.stack(
        [nid(I + dx, J + dy, K + dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
        axis=1,
    )  # (n_cell, 8) in grid corner order
    n_tet_layers = int(round(tet_fraction * nx))
    is_tet = I >= nx - n_tet_layers
    hex8 = corners[~is_tet][:, _VTK_FROM_GRID]
    tet4 = corners[is_tet][:, _KUHN].reshape(-1, 4)

    meta = {
        "type": "rectangular_finger",
        "length": geometry.length, "width": geometry.width, "height": geometry.height,
        "nx": nx, "ny": ny, "nz": nz,
        "tet_fraction": float(tet_fraction), "tet_layers": n_tet_layers,
        "elements": "hex20 + tet10 (Kuhn split at the tip), hanging-node ties at the interface",
    }
    return build_quadratic_mesh(nodes_c, hex8, tet4, meta)


def make_finger_mesh_from_config(cfg: Mapping[str, Any]) -> tuple[FingerGeometry, FEMesh]:
    """Build geometry + mesh from a ``configs/fem.yaml``-style dictionary.

    ``mesh.element_size`` [m] sets the resolution (default: smallest dimension / 4);
    explicit ``mesh.nx / ny / nz`` override individual axes. ``mesh.tet_fraction``
    controls the T10 tip region (default 0.2).
    """
    geom = FingerGeometry.from_config(cfg)
    m = cfg.get("mesh", {}) or {}
    nx, ny, nz = resolution_from_element_size(geom, float(m.get("element_size", default_element_size(geom))))
    nx, ny, nz = int(m.get("nx", nx)), int(m.get("ny", ny)), int(m.get("nz", nz))
    mesh = make_rectangular_finger_mesh(geom, nx, ny, nz, float(m.get("tet_fraction", 0.2)))
    return geom, mesh


def root_fixed_nodes(mesh: FEMesh, tol: float = 1e-9) -> np.ndarray:
    """Indices of all nodes on the clamped root surface ``x = 0``."""
    return mesh.nodes_on_plane(0, 0.0, tol)
