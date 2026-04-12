"""Finite element mesh data structures.

Supports:
  - TetMesh       : Tet4 (backward compatible)
  - FEMesh        : General mesh (Tet4, Tet10, Hex8)
  - CompositeMesh : Mixed element types (e.g. Hex8 + Tet4 in one body)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosim.physics.fem.elements import ElementType, NODES_PER_ELEMENT


# ═══════════════════════════════════════════════════════════════
# General FE Mesh
# ═══════════════════════════════════════════════════════════════

@dataclass
class FEMesh:
    """General finite element mesh supporting multiple element types.

    Attributes
    ----------
    nodes : (n_nodes, 3) rest positions
    elements : (n_elements, nodes_per_elem) connectivity
    element_type : ElementType enum
    """

    nodes: np.ndarray
    elements: np.ndarray
    element_type: ElementType = ElementType.TET4

    _surface_faces: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        self.elements = np.asarray(self.elements, dtype=np.int64)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def n_dof(self) -> int:
        return self.n_nodes * 3

    @property
    def nodes_per_element(self) -> int:
        return NODES_PER_ELEMENT[self.element_type]

    def extract_surface(self) -> np.ndarray:
        """Extract boundary triangle faces for visualization."""
        if self._surface_faces is not None:
            return self._surface_faces

        if self.element_type in (ElementType.TET4, ElementType.TET10):
            self._surface_faces = _extract_surface_tet(self.elements, self.element_type)
        elif self.element_type == ElementType.HEX8:
            self._surface_faces = _extract_surface_hex(self.elements)
        return self._surface_faces

    # ── Factory methods ──

    @staticmethod
    def create_hex_box(
        origin: np.ndarray = np.zeros(3),
        size: np.ndarray = np.ones(3),
        divisions: tuple[int, int, int] = (3, 3, 3),
    ) -> FEMesh:
        """Create a Hex8 box mesh.

        Parameters
        ----------
        origin : lower corner
        size : extent in x, y, z
        divisions : cells per direction
        """
        nx, ny, nz = divisions
        origin = np.asarray(origin, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)

        xs = np.linspace(origin[0], origin[0] + size[0], nx + 1)
        ys = np.linspace(origin[1], origin[1] + size[1], ny + 1)
        zs = np.linspace(origin[2], origin[2] + size[2], nz + 1)

        nodes = []
        for iz in range(nz + 1):
            for iy in range(ny + 1):
                for ix in range(nx + 1):
                    nodes.append([xs[ix], ys[iy], zs[iz]])
        nodes = np.array(nodes, dtype=np.float64)

        def idx(ix, iy, iz):
            return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

        elements = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    # 8 corners in standard Hex8 ordering:
                    #   0-3: bottom face (z), 4-7: top face (z+1)
                    #   counterclockwise when viewed from outside
                    v = [
                        idx(ix,     iy,     iz),
                        idx(ix + 1, iy,     iz),
                        idx(ix + 1, iy + 1, iz),
                        idx(ix,     iy + 1, iz),
                        idx(ix,     iy,     iz + 1),
                        idx(ix + 1, iy,     iz + 1),
                        idx(ix + 1, iy + 1, iz + 1),
                        idx(ix,     iy + 1, iz + 1),
                    ]
                    elements.append(v)

        elements = np.array(elements, dtype=np.int64)
        return FEMesh(nodes=nodes, elements=elements, element_type=ElementType.HEX8)

    @staticmethod
    def create_tet10_box(
        origin: np.ndarray = np.zeros(3),
        size: np.ndarray = np.ones(3),
        divisions: tuple[int, int, int] = (2, 2, 2),
    ) -> FEMesh:
        """Create a Tet10 box mesh.

        First creates a Tet4 mesh, then adds edge midpoint nodes.
        """
        # Start with Tet4 mesh
        tet4 = TetMesh.create_box(origin=origin, size=size, divisions=divisions)
        nodes = list(tet4.nodes)
        n_orig = len(nodes)

        # Map edge -> midpoint node index
        edge_to_mid: dict[tuple[int, int], int] = {}

        def get_midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in edge_to_mid:
                mid = 0.5 * (np.array(nodes[a]) + np.array(nodes[b]))
                edge_to_mid[key] = len(nodes)
                nodes.append(mid)
            return edge_to_mid[key]

        # Tet4 edges: (0,1), (1,2), (0,2), (0,3), (1,3), (2,3)
        elements10 = []
        for e in range(tet4.n_elements):
            n0, n1, n2, n3 = tet4.elements[e]
            m01 = get_midpoint(n0, n1)  # node 4
            m12 = get_midpoint(n1, n2)  # node 5
            m02 = get_midpoint(n0, n2)  # node 6
            m03 = get_midpoint(n0, n3)  # node 7
            m13 = get_midpoint(n1, n3)  # node 8
            m23 = get_midpoint(n2, n3)  # node 9
            elements10.append([n0, n1, n2, n3, m01, m12, m02, m03, m13, m23])

        nodes = np.array(nodes, dtype=np.float64)
        elements10 = np.array(elements10, dtype=np.int64)
        return FEMesh(nodes=nodes, elements=elements10, element_type=ElementType.TET10)

    def __repr__(self) -> str:
        return f"FEMesh({self.element_type.value}, nodes={self.n_nodes}, elements={self.n_elements})"


# ═══════════════════════════════════════════════════════════════
# Element Block & Composite Mesh
# ═══════════════════════════════════════════════════════════════

@dataclass
class ElementBlock:
    """A group of elements sharing the same element type within a composite mesh.

    Attributes
    ----------
    element_type : element type for this block
    elements : (n_elements, nodes_per_elem) node indices (global)
    name : optional label (e.g. "hex_core", "tet_shell")
    """

    element_type: ElementType
    elements: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.elements = np.asarray(self.elements, dtype=np.int64)

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def nodes_per_element(self) -> int:
        return NODES_PER_ELEMENT[self.element_type]


@dataclass
class CompositeMesh:
    """Finite element mesh with multiple element blocks sharing a single node array.

    Like ABAQUS *ELEMENT blocks or CalculiX element sets: one global node pool,
    multiple connectivity tables (one per element type).

    Attributes
    ----------
    nodes : (n_nodes, 3) rest positions (shared by all blocks)
    blocks : list of ElementBlock (each with its own element_type & connectivity)
    """

    nodes: np.ndarray
    blocks: list[ElementBlock] = field(default_factory=list)

    _surface_faces: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=np.float64)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return sum(b.n_elements for b in self.blocks)

    @property
    def n_dof(self) -> int:
        return self.n_nodes * 3

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def element_types(self) -> list[ElementType]:
        """Unique element types across all blocks."""
        return list(dict.fromkeys(b.element_type for b in self.blocks))

    def block_summary(self) -> str:
        """Human-readable summary of all blocks."""
        lines = []
        for i, b in enumerate(self.blocks):
            name = b.name or f"block_{i}"
            lines.append(f"  [{i}] {name}: {b.element_type.value} "
                         f"({b.n_elements} elements, {b.nodes_per_element} nodes/elem)")
        return "\n".join(lines)

    def extract_surface(self) -> np.ndarray:
        """Extract boundary triangle faces for visualization (across all blocks)."""
        if self._surface_faces is not None:
            return self._surface_faces
        self._surface_faces = _extract_surface_composite(self.blocks)
        return self._surface_faces

    # ── Factory methods ──

    @classmethod
    def from_blocks(
        cls,
        nodes: np.ndarray,
        blocks: list[ElementBlock],
    ) -> "CompositeMesh":
        """Create a CompositeMesh from a shared node array and pre-built blocks."""
        return cls(nodes=nodes, blocks=blocks)

    @staticmethod
    def create_hex_tet_box(
        origin: np.ndarray = np.zeros(3),
        size: np.ndarray = np.ones(3),
        hex_divisions: tuple[int, int, int] = (2, 2, 2),
        tet_layers: int = 1,
    ) -> "CompositeMesh":
        """Create a composite box: hex core surrounded by tet shell.

        The box is split into an inner hex region and outer tet region.
        Outer cells (within `tet_layers` of any face) are subdivided into
        5 tets each. Inner cells remain as Hex8.

        Parameters
        ----------
        origin : lower corner
        size : box extent
        hex_divisions : total cells in each direction (must be > 2*tet_layers)
        tet_layers : number of outer cell layers converted to tets
        """
        nx, ny, nz = hex_divisions
        origin = np.asarray(origin, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)

        # Must have enough cells for both hex core and tet shell
        # Directions with cells <= 2*tet_layers become all-tet (no hex core in that dim)
        need_inner = [
            nx > 2 * tet_layers,
            ny > 2 * tet_layers,
            nz > 2 * tet_layers,
        ]
        if not all(need_inner):
            # Relax: allow slim meshes where some dims are all-tet
            pass

        # Build shared node grid
        xs = np.linspace(origin[0], origin[0] + size[0], nx + 1)
        ys = np.linspace(origin[1], origin[1] + size[1], ny + 1)
        zs = np.linspace(origin[2], origin[2] + size[2], nz + 1)

        nodes = []
        for iz in range(nz + 1):
            for iy in range(ny + 1):
                for ix in range(nx + 1):
                    nodes.append([xs[ix], ys[iy], zs[iz]])
        nodes = np.array(nodes, dtype=np.float64)

        def idx(ix, iy, iz):
            return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

        hex_elements = []
        tet_elements = []

        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    # 8 hex corners
                    v = [
                        idx(ix,     iy,     iz),
                        idx(ix + 1, iy,     iz),
                        idx(ix + 1, iy + 1, iz),
                        idx(ix,     iy + 1, iz),
                        idx(ix,     iy,     iz + 1),
                        idx(ix + 1, iy,     iz + 1),
                        idx(ix + 1, iy + 1, iz + 1),
                        idx(ix,     iy + 1, iz + 1),
                    ]

                    # Is this cell in the outer shell?
                    # If a dimension has <= 2*tet_layers cells, all cells in
                    # that dimension are outer (no hex core in that direction).
                    is_outer = (
                        ix < tet_layers or ix >= nx - tet_layers or
                        iy < tet_layers or iy >= ny - tet_layers or
                        iz < tet_layers or iz >= nz - tet_layers or
                        nx <= 2 * tet_layers or
                        ny <= 2 * tet_layers or
                        nz <= 2 * tet_layers
                    )

                    if is_outer:
                        # Subdivide into 5 tets (parity-dependent)
                        parity = (ix + iy + iz) % 2
                        if parity == 0:
                            tet_elements.append([v[0], v[1], v[3], v[4]])
                            tet_elements.append([v[1], v[2], v[3], v[6]])
                            tet_elements.append([v[1], v[4], v[5], v[6]])
                            tet_elements.append([v[3], v[4], v[6], v[7]])
                            tet_elements.append([v[1], v[3], v[4], v[6]])
                        else:
                            tet_elements.append([v[0], v[1], v[2], v[5]])
                            tet_elements.append([v[0], v[2], v[3], v[7]])
                            tet_elements.append([v[0], v[4], v[5], v[7]])
                            tet_elements.append([v[2], v[5], v[6], v[7]])
                            tet_elements.append([v[0], v[2], v[5], v[7]])
                    else:
                        hex_elements.append(v)

        blocks = []
        if hex_elements:
            blocks.append(ElementBlock(
                element_type=ElementType.HEX8,
                elements=np.array(hex_elements, dtype=np.int64),
                name="hex_core",
            ))
        if tet_elements:
            blocks.append(ElementBlock(
                element_type=ElementType.TET4,
                elements=np.array(tet_elements, dtype=np.int64),
                name="tet_shell",
            ))

        return CompositeMesh(nodes=nodes, blocks=blocks)

    @staticmethod
    def merge_meshes(
        meshes: list,
        names: list[str] | None = None,
        merge_tol: float = 1e-10,
    ) -> "CompositeMesh":
        """Merge multiple FEMesh/TetMesh instances into a single CompositeMesh.

        Nodes within `merge_tol` distance are merged (shared at block boundaries).

        Parameters
        ----------
        meshes : list of FEMesh or TetMesh
        names : optional block names
        merge_tol : distance tolerance for node merging
        """
        if names is None:
            names = [f"block_{i}" for i in range(len(meshes))]

        # Collect all nodes
        all_nodes = [m.nodes for m in meshes]
        offsets = [0]
        for ns in all_nodes:
            offsets.append(offsets[-1] + ns.shape[0])
        combined = np.vstack(all_nodes)

        # Merge duplicate nodes (KD-tree)
        from scipy.spatial import cKDTree
        tree = cKDTree(combined)
        pairs = tree.query_pairs(merge_tol)

        # Union-find for node merging
        parent = list(range(len(combined)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for a, b in pairs:
            union(a, b)

        # Build reindex map: old index -> new index
        root_to_new = {}
        new_count = 0
        remap = np.zeros(len(combined), dtype=np.int64)
        for i in range(len(combined)):
            r = find(i)
            if r not in root_to_new:
                root_to_new[r] = new_count
                new_count += 1
            remap[i] = root_to_new[r]

        # Build merged node array (use root node position)
        merged_nodes = np.zeros((new_count, 3), dtype=np.float64)
        for i in range(len(combined)):
            r = find(i)
            if remap[i] == remap[r]:
                merged_nodes[remap[i]] = combined[r]

        # Build blocks with remapped connectivity
        blocks = []
        for mesh_idx, (mesh, name) in enumerate(zip(meshes, names)):
            off = offsets[mesh_idx]
            etype = mesh.element_type if hasattr(mesh, 'element_type') else ElementType.TET4
            old_elems = mesh.elements
            new_elems = remap[old_elems + off]
            blocks.append(ElementBlock(
                element_type=etype,
                elements=new_elems,
                name=name,
            ))

        return CompositeMesh(nodes=merged_nodes, blocks=blocks)

    def __repr__(self) -> str:
        etypes = "+".join(b.element_type.value for b in self.blocks)
        return (f"CompositeMesh({etypes}, nodes={self.n_nodes}, "
                f"elements={self.n_elements}, blocks={self.n_blocks})")


# ═══════════════════════════════════════════════════════════════
# Surface extraction helpers
# ═══════════════════════════════════════════════════════════════

def _extract_surface_tet(elements, etype):
    """Extract surface triangles from tet mesh (Tet4 or Tet10)."""
    face_count: dict[tuple, int] = {}
    face_to_orig: dict[tuple, tuple] = {}

    for e in range(elements.shape[0]):
        n = elements[e]
        # 4 faces, using corner nodes only (first 4)
        faces = [
            (n[0], n[1], n[2]),
            (n[0], n[1], n[3]),
            (n[0], n[2], n[3]),
            (n[1], n[2], n[3]),
        ]
        for f in faces:
            key = tuple(sorted(f))
            face_count[key] = face_count.get(key, 0) + 1
            if key not in face_to_orig:
                face_to_orig[key] = f

    surface = [face_to_orig[k] for k, c in face_count.items() if c == 1]
    return np.array(surface, dtype=np.int64) if surface else np.zeros((0, 3), dtype=np.int64)


def _extract_surface_hex(elements):
    """Extract surface triangles from Hex8 mesh (each quad → 2 triangles)."""
    # 6 faces of a hex, each defined by 4 corner indices (local)
    hex_faces = [
        (0, 3, 2, 1),  # bottom (-z)
        (4, 5, 6, 7),  # top (+z)
        (0, 1, 5, 4),  # front (-y)
        (2, 3, 7, 6),  # back (+y)
        (0, 4, 7, 3),  # left (-x)
        (1, 2, 6, 5),  # right (+x)
    ]

    face_count: dict[tuple, int] = {}
    face_to_quad: dict[tuple, tuple] = {}

    for e in range(elements.shape[0]):
        n = elements[e]
        for lf in hex_faces:
            quad = tuple(n[i] for i in lf)
            key = tuple(sorted(quad))
            face_count[key] = face_count.get(key, 0) + 1
            if key not in face_to_quad:
                face_to_quad[key] = quad

    triangles = []
    for key, count in face_count.items():
        if count == 1:
            q = face_to_quad[key]
            triangles.append((q[0], q[1], q[2]))
            triangles.append((q[0], q[2], q[3]))

    return np.array(triangles, dtype=np.int64) if triangles else np.zeros((0, 3), dtype=np.int64)


def _extract_surface_composite(blocks: list) -> np.ndarray:
    """Extract surface triangles from a composite mesh with mixed element types.

    All faces (tet triangles and hex quads→2 triangles) are collected into
    a unified face count. Faces appearing once are surface faces.
    At hex-tet interfaces, the hex quad is split into the same two triangles
    that the adjacent tets produce, so they cancel correctly.
    """
    # Count all triangular faces across all blocks
    face_count: dict[tuple, int] = {}
    face_to_orig: dict[tuple, tuple] = {}

    for block in blocks:
        elements = block.elements
        etype = block.element_type

        if etype in (ElementType.TET4, ElementType.TET10):
            # 4 triangular faces per tet (corner nodes only)
            for e in range(elements.shape[0]):
                n = elements[e]
                faces = [
                    (n[0], n[1], n[2]),
                    (n[0], n[1], n[3]),
                    (n[0], n[2], n[3]),
                    (n[1], n[2], n[3]),
                ]
                for f in faces:
                    key = tuple(sorted(f))
                    face_count[key] = face_count.get(key, 0) + 1
                    if key not in face_to_orig:
                        face_to_orig[key] = f

        elif etype == ElementType.HEX8:
            hex_face_local = [
                (0, 3, 2, 1), (4, 5, 6, 7),
                (0, 1, 5, 4), (2, 3, 7, 6),
                (0, 4, 7, 3), (1, 2, 6, 5),
            ]
            for e in range(elements.shape[0]):
                n = elements[e]
                for lf in hex_face_local:
                    q = tuple(n[i] for i in lf)
                    # Split quad into 2 triangles using consistent convention:
                    # (q0,q1,q2) and (q0,q2,q3)
                    tris = [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
                    for tri in tris:
                        key = tuple(sorted(tri))
                        face_count[key] = face_count.get(key, 0) + 1
                        if key not in face_to_orig:
                            face_to_orig[key] = tri

    surface = [face_to_orig[k] for k, c in face_count.items() if c == 1]
    return np.array(surface, dtype=np.int64) if surface else np.zeros((0, 3), dtype=np.int64)


# ═══════════════════════════════════════════════════════════════
# TetMesh (backward compatible)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TetMesh:
    """Linear tetrahedral (Tet4) mesh — backward compatible API.

    Attributes
    ----------
    nodes : (n_nodes, 3) rest positions
    elements : (n_elements, 4) node indices per tetrahedron
    """

    nodes: np.ndarray
    elements: np.ndarray

    _surface_faces: np.ndarray | None = field(default=None, repr=False)
    _volumes: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        self.elements = np.asarray(self.elements, dtype=np.int64)

    @property
    def element_type(self) -> ElementType:
        return ElementType.TET4

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def n_dof(self) -> int:
        return self.n_nodes * 3

    @property
    def nodes_per_element(self) -> int:
        return 4

    def compute_volumes(self) -> np.ndarray:
        if self._volumes is not None:
            return self._volumes
        vols = np.zeros(self.n_elements)
        for e in range(self.n_elements):
            n0, n1, n2, n3 = self.elements[e]
            d1 = self.nodes[n1] - self.nodes[n0]
            d2 = self.nodes[n2] - self.nodes[n0]
            d3 = self.nodes[n3] - self.nodes[n0]
            vols[e] = abs(np.dot(d1, np.cross(d2, d3))) / 6.0
        self._volumes = vols
        return vols

    def total_volume(self) -> float:
        return float(self.compute_volumes().sum())

    def extract_surface(self) -> np.ndarray:
        if self._surface_faces is not None:
            return self._surface_faces
        self._surface_faces = _extract_surface_tet(self.elements, ElementType.TET4)
        return self._surface_faces

    @staticmethod
    def create_box(
        origin: np.ndarray = np.zeros(3),
        size: np.ndarray = np.ones(3),
        divisions: tuple[int, int, int] = (3, 3, 3),
    ) -> TetMesh:
        """Create a box-shaped tet mesh (5 tets per hex cell)."""
        nx, ny, nz = divisions
        origin = np.asarray(origin, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)

        xs = np.linspace(origin[0], origin[0] + size[0], nx + 1)
        ys = np.linspace(origin[1], origin[1] + size[1], ny + 1)
        zs = np.linspace(origin[2], origin[2] + size[2], nz + 1)

        nodes = []
        for iz in range(nz + 1):
            for iy in range(ny + 1):
                for ix in range(nx + 1):
                    nodes.append([xs[ix], ys[iy], zs[iz]])
        nodes = np.array(nodes, dtype=np.float64)

        def node_idx(ix, iy, iz):
            return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

        elements = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    v = [
                        node_idx(ix, iy, iz), node_idx(ix+1, iy, iz),
                        node_idx(ix+1, iy+1, iz), node_idx(ix, iy+1, iz),
                        node_idx(ix, iy, iz+1), node_idx(ix+1, iy, iz+1),
                        node_idx(ix+1, iy+1, iz+1), node_idx(ix, iy+1, iz+1),
                    ]
                    parity = (ix + iy + iz) % 2
                    if parity == 0:
                        elements.append([v[0], v[1], v[3], v[4]])
                        elements.append([v[1], v[2], v[3], v[6]])
                        elements.append([v[1], v[4], v[5], v[6]])
                        elements.append([v[3], v[4], v[6], v[7]])
                        elements.append([v[1], v[3], v[4], v[6]])
                    else:
                        elements.append([v[0], v[1], v[2], v[5]])
                        elements.append([v[0], v[2], v[3], v[7]])
                        elements.append([v[0], v[4], v[5], v[7]])
                        elements.append([v[2], v[5], v[6], v[7]])
                        elements.append([v[0], v[2], v[5], v[7]])

        elements = np.array(elements, dtype=np.int64)
        return TetMesh(nodes=nodes, elements=elements)

    def __repr__(self) -> str:
        return f"TetMesh(nodes={self.n_nodes}, elements={self.n_elements})"
