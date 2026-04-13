"""Tetrahedral mesh I/O: load and save tet meshes in various formats.

Wraps meshio for .msh (Gmsh), .vtu (VTK), and raw .npz formats.

Usage::

    mesh = load_tet_mesh("model.msh")
    save_tet_mesh("output.vtu", mesh)
    save_tet_mesh("output.npz", mesh)  # fast NumPy format
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robosim.physics.fem.mesh import TetMesh


def load_tet_mesh(
    path: str | Path,
    scale: float = 1.0,
    translate: np.ndarray | None = None,
) -> TetMesh:
    """Load a tetrahedral mesh from file.

    Supported formats:
      - .npz  — NumPy archive (nodes + elements)
      - .msh  — Gmsh format (requires meshio)
      - .vtu  — VTK unstructured grid (requires meshio)
      - .mesh — Medit/TetGen format (requires meshio)

    Parameters
    ----------
    path : file path
    scale : uniform scaling factor
    translate : (3,) translation to apply after scaling

    Returns
    -------
    TetMesh instance
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        return _load_npz(path, scale, translate)
    else:
        return _load_meshio(path, scale, translate)


def save_tet_mesh(path: str | Path, mesh: TetMesh) -> None:
    """Save a tetrahedral mesh to file.

    Supported formats:
      - .npz  — NumPy archive (fast, compact)
      - .vtu  — VTK unstructured grid (requires meshio)
      - .msh  — Gmsh format (requires meshio)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        _save_npz(path, mesh)
    else:
        _save_meshio(path, mesh)


def _load_npz(
    path: Path, scale: float, translate: np.ndarray | None,
) -> TetMesh:
    """Load from NumPy .npz archive."""
    data = np.load(path)
    nodes = data["nodes"].astype(np.float64) * scale
    elements = data["elements"].astype(np.int32)

    if translate is not None:
        nodes += translate

    return TetMesh(nodes=nodes, elements=elements)


def _save_npz(path: Path, mesh: TetMesh) -> None:
    """Save to NumPy .npz archive."""
    np.savez_compressed(
        path,
        nodes=mesh.nodes,
        elements=mesh.elements,
    )


def _load_meshio(
    path: Path, scale: float, translate: np.ndarray | None,
) -> TetMesh:
    """Load via meshio (supports .msh, .vtu, .mesh, etc.)."""
    try:
        import meshio
    except ImportError:
        raise ImportError(
            "meshio is required for loading non-.npz mesh files. "
            "Install with: pip install meshio"
        )

    mesh_data = meshio.read(str(path))
    nodes = mesh_data.points.astype(np.float64) * scale

    if translate is not None:
        nodes += translate

    # Find tetrahedral cells
    elements = None
    for cell_block in mesh_data.cells:
        if cell_block.type == "tetra":
            elements = cell_block.data.astype(np.int32)
            break
        elif cell_block.type == "tetra10":
            # Tet10: take first 4 nodes (corners) for linear Tet4
            elements = cell_block.data[:, :4].astype(np.int32)
            break

    if elements is None:
        cell_types = [cb.type for cb in mesh_data.cells]
        raise ValueError(
            f"No tetrahedral cells found in {path}. "
            f"Available cell types: {cell_types}"
        )

    return TetMesh(nodes=nodes, elements=elements)


def _save_meshio(path: Path, mesh: TetMesh) -> None:
    """Save via meshio."""
    try:
        import meshio
    except ImportError:
        raise ImportError(
            "meshio is required for saving non-.npz mesh files. "
            "Install with: pip install meshio"
        )

    meshio_mesh = meshio.Mesh(
        points=mesh.nodes,
        cells=[("tetra", mesh.elements)],
    )
    meshio.write(str(path), meshio_mesh)


def mesh_info(mesh: TetMesh) -> str:
    """Return a summary string for a tet mesh."""
    bbox_min = mesh.nodes.min(axis=0)
    bbox_max = mesh.nodes.max(axis=0)
    size = bbox_max - bbox_min

    return (
        f"TetMesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements\n"
        f"  BBox: [{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}] — "
        f"[{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]\n"
        f"  Size: {size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}"
    )
