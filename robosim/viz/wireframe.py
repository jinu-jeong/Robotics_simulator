"""Wireframe rendering utilities for FEM meshes.

Extracts unique edges from surface triangles and provides Taichi GGUI
helpers for line rendering.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class RenderMode(IntEnum):
    """FEM visualization render modes (toggle with number keys 1-4)."""
    SOLID = 1           # filled triangles, solid color
    WIREFRAME = 2       # edges only
    SOLID_WIRE = 3      # filled + edge overlay
    STRESS_WIRE = 4     # stress colormap + edge overlay


# Display names for HUD
RENDER_MODE_NAMES = {
    RenderMode.SOLID:       "Solid",
    RenderMode.WIREFRAME:   "Wireframe",
    RenderMode.SOLID_WIRE:  "Solid + Wire",
    RenderMode.STRESS_WIRE: "Stress + Wire",
}


def extract_edges(surface_tri: np.ndarray) -> np.ndarray:
    """Extract unique undirected edges from surface triangles.

    Parameters
    ----------
    surface_tri : (n_faces, 3) triangle indices

    Returns
    -------
    edges : (n_edges, 2) int64, unique edge pairs (sorted per edge)
    """
    # Each triangle contributes 3 edges
    n_tri = surface_tri.shape[0]
    all_edges = np.zeros((n_tri * 3, 2), dtype=np.int64)

    all_edges[0::3, 0] = surface_tri[:, 0]
    all_edges[0::3, 1] = surface_tri[:, 1]
    all_edges[1::3, 0] = surface_tri[:, 1]
    all_edges[1::3, 1] = surface_tri[:, 2]
    all_edges[2::3, 0] = surface_tri[:, 2]
    all_edges[2::3, 1] = surface_tri[:, 0]

    # Sort each edge (a, b) → (min, max) for dedup
    sorted_edges = np.sort(all_edges, axis=1)

    # Unique edges
    unique = np.unique(sorted_edges, axis=0)
    return unique


def edges_to_line_indices(edges: np.ndarray) -> np.ndarray:
    """Flatten edge pairs to line index array for Taichi scene.lines().

    Parameters
    ----------
    edges : (n_edges, 2) edge pairs

    Returns
    -------
    line_indices : (n_edges * 2,) int32
    """
    return edges.ravel().astype(np.int32)


def handle_mode_events(window, current_mode: RenderMode) -> RenderMode:
    """Check for keyboard events to switch render mode.

    Keys 1-4 switch modes. Returns the (possibly new) mode.

    Parameters
    ----------
    window : ti.ui.Window
    current_mode : current RenderMode

    Returns
    -------
    new_mode : RenderMode
    """
    import taichi as ti

    for e in window.get_events(ti.ui.PRESS):
        if e.key == '1':
            return RenderMode.SOLID
        elif e.key == '2':
            return RenderMode.WIREFRAME
        elif e.key == '3':
            return RenderMode.SOLID_WIRE
        elif e.key == '4':
            return RenderMode.STRESS_WIRE
    return current_mode
