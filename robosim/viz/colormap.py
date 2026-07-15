"""Colormap and element-type visualization utilities.

Provides:
  - stress_to_color : von Mises → jet RGB
  - element_type_color : per-element-type solid color
  - element_type_label : human-readable label string
  - element_type_hud : formatted HUD info string
"""

from __future__ import annotations

import numpy as np

from robosim.physics.fem.elements import ElementType


# ═══════════════════════════════════════════════════════════════
# Element type base colors
# ═══════════════════════════════════════════════════════════════

ELEMENT_TYPE_COLORS: dict[ElementType, np.ndarray] = {
    ElementType.TET4:  np.array([0.35, 0.55, 0.75], dtype=np.float32),   # steel blue
    ElementType.TET10: np.array([0.45, 0.78, 0.45], dtype=np.float32),   # green
    ElementType.HEX8:  np.array([0.88, 0.55, 0.25], dtype=np.float32),   # orange
}

ELEMENT_TYPE_LABELS: dict[ElementType, str] = {
    ElementType.TET4:  "Tet4 (linear tetrahedron, 4 nodes)",
    ElementType.TET10: "Tet10 (quadratic tetrahedron, 10 nodes)",
    ElementType.HEX8:  "Hex8 (trilinear hexahedron, 8 nodes)",
}

ELEMENT_TYPE_SHORT: dict[ElementType, str] = {
    ElementType.TET4:  "Tet4",
    ElementType.TET10: "Tet10",
    ElementType.HEX8:  "Hex8",
}

GAUSS_POINTS_INFO: dict[ElementType, str] = {
    ElementType.TET4:  "1 point",
    ElementType.TET10: "4 points",
    ElementType.HEX8:  "2×2×2 = 8 points",
}


# ═══════════════════════════════════════════════════════════════
# Stress colormaps
# ═══════════════════════════════════════════════════════════════

def stress_to_color(
    vm: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 500.0,
) -> np.ndarray:
    """Map von Mises stress to jet-like RGB colors.

    Parameters
    ----------
    vm : (n,) scalar stress values
    vmin, vmax : color range bounds

    Returns
    -------
    colors : (n, 3) float32 RGB in [0, 1]
    """
    t = np.clip((vm - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(t - 0.75) * 4.0, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(t - 0.50) * 4.0, 0.0, 1.0)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4.0, 0.0, 1.0)
    return np.column_stack([r, g, b]).astype(np.float32)


def gripper_solid_red(nodes_link: np.ndarray) -> np.ndarray:
    """Solid red per-vertex colors for deformable gripper CV imaging.

    Uniform bright red makes the whole finger a stable mask for tip/root
    ridge tracking (no white→red falloff near the palm).

    Parameters
    ----------
    nodes_link : (N, 3) mesh node positions in the finger link frame
                 (only ``N`` is used; positions are unused)

    Returns
    -------
    colors : (N, 3) float32 RGB in [0, 1]
    """
    n = int(np.asarray(nodes_link).shape[0])
    colors = np.zeros((n, 3), dtype=np.float32)
    colors[:, 0] = 1.0
    return colors


def gripper_palm_tip_gradient(nodes_link: np.ndarray) -> np.ndarray:
    """Deprecated alias — solid red (was white→red palm→tip)."""
    return gripper_solid_red(nodes_link)


# ═══════════════════════════════════════════════════════════════
# Element type visualization helpers
# ═══════════════════════════════════════════════════════════════

def element_type_color(etype: ElementType, n_nodes: int) -> np.ndarray:
    """Return a solid per-vertex color array for the given element type.

    Parameters
    ----------
    etype : ElementType
    n_nodes : number of mesh nodes

    Returns
    -------
    colors : (n_nodes, 3) float32 RGB
    """
    base = ELEMENT_TYPE_COLORS.get(etype, np.array([0.6, 0.6, 0.6], dtype=np.float32))
    return np.broadcast_to(base, (n_nodes, 3)).copy()


def element_type_label(etype: ElementType) -> str:
    """Human-readable label for element type."""
    return ELEMENT_TYPE_LABELS.get(etype, str(etype))


def element_type_hud(etype: ElementType, mesh) -> str:
    """Format element type info for HUD overlay.

    Parameters
    ----------
    etype : ElementType
    mesh : FEMesh, TetMesh, or CompositeMesh

    Returns
    -------
    multi-line string with element info
    """
    # CompositeMesh: show per-block info
    if hasattr(mesh, 'blocks'):
        lines = [f"Composite Mesh ({mesh.n_blocks} blocks)"]
        for i, b in enumerate(mesh.blocks):
            short = ELEMENT_TYPE_SHORT.get(b.element_type, "?")
            gauss = GAUSS_POINTS_INFO.get(b.element_type, "?")
            name = b.name or f"block_{i}"
            lines.append(f"  [{name}] {short}: {b.n_elements}e, {b.nodes_per_element}n/e, gauss={gauss}")
        lines.append(f"Total: {mesh.n_elements}e, {mesh.n_nodes}n, {mesh.n_dof} DOFs")
        return "\n".join(lines)

    short = ELEMENT_TYPE_SHORT.get(etype, "?")
    gauss = GAUSS_POINTS_INFO.get(etype, "?")
    npe = mesh.nodes_per_element if hasattr(mesh, "nodes_per_element") else "?"

    lines = [
        f"Element: {short}",
        f"  Nodes/elem: {npe}",
        f"  Gauss: {gauss}",
        f"  Elements: {mesh.n_elements}",
        f"  Total nodes: {mesh.n_nodes}",
        f"  DOFs: {mesh.n_nodes * 3}",
    ]
    return "\n".join(lines)


def composite_element_color(mesh, vm=None, vmin=0.0, vmax=500.0, tint_strength=0.3) -> np.ndarray:
    """Per-vertex color for CompositeMesh, tinted by element type.

    Each node gets a color based on which block(s) contribute to it.
    If `vm` is provided, blends stress colormap with element-type color.
    If `vm` is None, returns pure element-type colors.

    Parameters
    ----------
    mesh : CompositeMesh
    vm : optional (n_nodes,) von Mises stress
    vmin, vmax : stress range for colormap
    tint_strength : blend factor (only used if vm is provided)

    Returns
    -------
    colors : (n_nodes, 3) float32 RGB
    """
    n_nodes = mesh.n_nodes
    color_sum = np.zeros((n_nodes, 3), dtype=np.float64)
    count = np.zeros(n_nodes, dtype=np.float64)

    for block in mesh.blocks:
        base = ELEMENT_TYPE_COLORS.get(
            block.element_type, np.array([0.6, 0.6, 0.6], dtype=np.float32)
        )
        unique_nodes = np.unique(block.elements.ravel())
        color_sum[unique_nodes] += base[np.newaxis, :]
        count[unique_nodes] += 1.0

    count[count == 0] = 1.0
    base_colors = (color_sum / count[:, np.newaxis]).astype(np.float32)

    if vm is None:
        return base_colors

    # Blend with stress colormap
    jet = stress_to_color(vm, vmin, vmax)
    blended = (1.0 - tint_strength) * jet + tint_strength * base_colors
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def stress_with_element_tint(
    vm: np.ndarray,
    etype: ElementType,
    vmin: float = 0.0,
    vmax: float = 500.0,
    tint_strength: float = 0.2,
) -> np.ndarray:
    """Stress colormap with a subtle element-type tint.

    Blends jet stress colors with the element type base color.
    At tint_strength=0, pure jet; at 1.0, pure element color.

    Parameters
    ----------
    vm : (n,) scalar stress values
    etype : ElementType for tinting
    vmin, vmax : stress color range
    tint_strength : blend factor [0, 1]

    Returns
    -------
    colors : (n, 3) float32 RGB
    """
    jet = stress_to_color(vm, vmin, vmax)
    base = ELEMENT_TYPE_COLORS.get(etype, np.array([0.6, 0.6, 0.6], dtype=np.float32))
    tinted = (1.0 - tint_strength) * jet + tint_strength * base[np.newaxis, :]
    return np.clip(tinted, 0.0, 1.0).astype(np.float32)
