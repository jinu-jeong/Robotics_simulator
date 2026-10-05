"""Force-vector arrows (ground truth vs. estimated) as 3D geometry.

Visual scale is decoupled from mechanics: a force ``F`` [N] is drawn with
length ``meters_per_newton * |F|`` [m]. When ``meters_per_newton`` is None
the scale is chosen so that the largest arrow in the state has a length of
``auto_max_length_rel * bbox_diagonal``. All arrows in one frame share the
same scale so their relative magnitudes stay comparable.

Anchoring: with ``anchor="tip"`` (default) the arrow *ends* at the force
origin, i.e. a contact force pressing on the surface is drawn as an arrow
arriving at the contact point from outside. With ``anchor="tail"`` the arrow
starts at the origin (natural for reaction forces or displacements).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.primitives import arrow_mesh, to_triangle_soup
from .mesh_renderer import TriangleSoupBuffers
from .state import ForceVector


@dataclass
class ArrowStyle:
    shaft_radius: float
    head_radius: float
    head_length: float
    n_segments: int = 16


class ForceArrowRenderer:
    """Builds and draws arrow meshes for a list of forces."""

    def __init__(self, capacity_triangles: int = 4096) -> None:
        self.buffers = TriangleSoupBuffers(capacity_triangles)
        self.meters_per_newton_used: float = 0.0

    @staticmethod
    def resolve_scale(
        magnitudes: list[float],
        bbox_diagonal: float,
        meters_per_newton: float | None,
        auto_max_length_rel: float,
    ) -> float:
        if meters_per_newton is not None:
            return float(meters_per_newton)
        fmax = max([m for m in magnitudes if m > 0], default=0.0)
        if fmax <= 0.0:
            return 0.0
        return auto_max_length_rel * bbox_diagonal / fmax

    def update(
        self,
        forces: list[tuple[ForceVector, tuple[float, float, float]]],
        bbox_diagonal: float,
        style: ArrowStyle,
        meters_per_newton: float | None,
        auto_max_length_rel: float,
        min_length_rel: float,
        anchor: str = "tip",
    ) -> None:
        """Rebuild the arrow geometry for ``[(force, rgb), ...]``."""
        if anchor not in ("tip", "tail"):
            raise ValueError("anchor must be 'tip' or 'tail'")
        if not forces:
            self.buffers.clear()
            return
        scale = self.resolve_scale([f.magnitude for f, _ in forces], bbox_diagonal, meters_per_newton, auto_max_length_rel)
        self.meters_per_newton_used = scale
        min_len = min_length_rel * bbox_diagonal
        verts, cols = [], []
        for force, rgb in forces:
            mag = force.magnitude
            if mag <= 0.0:
                continue
            length = max(scale * mag, min_len)
            # Keep a visible shaft: never let the head eat the whole arrow.
            head_len = min(style.head_length, 0.28 * length)
            head_rad = min(style.head_radius, 2.2 * style.shaft_radius) * (head_len / max(style.head_length, 1e-12))
            head_rad = max(head_rad, 1.6 * style.shaft_radius)
            shaft_rad = style.shaft_radius
            start = force.origin - length * force.direction if anchor == "tip" else force.origin
            v, f = arrow_mesh(
                start,
                force.direction,
                length,
                shaft_rad,
                head_rad,
                head_len,
                style.n_segments,
            )
            soup = to_triangle_soup(v, f)
            verts.append(soup)
            cols.append(np.tile(np.asarray(rgb, dtype=np.float32), (len(soup), 1)))
        if not verts:
            self.buffers.clear()
            return
        self.buffers.set(np.concatenate(verts), np.concatenate(cols))

    def draw(self, scene) -> None:
        self.buffers.draw(scene, two_sided=False)
