"""Taichi GGUI-based 3D viewer for the robotics simulator.

Uses Metal/Vulkan native rendering for high performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import taichi as ti

from robosim.model.geometry import Geometry, GeometryType
from robosim.math.transforms import Transform


# ── Geometry → triangle mesh generators ──────────────────────────

def _box_triangles(sx: float, sy: float, sz: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate box vertices and triangle indices.

    Returns (8, 3) vertices, (12, 3) faces.
    """
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    verts = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ], dtype=np.float64)
    faces = np.array([
        [0,2,1], [0,3,2],  # -Z
        [4,5,6], [4,6,7],  # +Z
        [0,1,5], [0,5,4],  # -Y
        [2,3,7], [2,7,6],  # +Y
        [0,4,7], [0,7,3],  # -X
        [1,2,6], [1,6,5],  # +X
    ], dtype=np.int32)
    return verts, faces


def _sphere_triangles(radius: float, n_lat: int = 12, n_lon: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Generate UV sphere vertices and triangle indices."""
    verts = []
    # Top pole
    verts.append([0, 0, radius])
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            verts.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ])
    # Bottom pole
    verts.append([0, 0, -radius])
    verts = np.array(verts, dtype=np.float64)

    faces = []
    # Top cap
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + j_next])
    # Middle strips
    for i in range(n_lat - 2):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            base = 1 + i * n_lon
            faces.append([base + j, base + n_lon + j, base + n_lon + j_next])
            faces.append([base + j, base + n_lon + j_next, base + j_next])
    # Bottom cap
    bot = len(verts) - 1
    base = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([base + j, bot, base + j_next])

    return verts, np.array(faces, dtype=np.int32)


def _cylinder_triangles(radius: float, length: float, n_seg: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Generate cylinder vertices and triangle indices (along Z axis, centered)."""
    hz = length / 2
    verts = []
    # Bottom ring
    for i in range(n_seg):
        angle = 2 * np.pi * i / n_seg
        verts.append([radius * np.cos(angle), radius * np.sin(angle), -hz])
    # Top ring
    for i in range(n_seg):
        angle = 2 * np.pi * i / n_seg
        verts.append([radius * np.cos(angle), radius * np.sin(angle), hz])
    # Center bottom, center top
    bot_c = len(verts)
    verts.append([0, 0, -hz])
    top_c = len(verts)
    verts.append([0, 0, hz])
    verts = np.array(verts, dtype=np.float64)

    faces = []
    # Side quads
    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        faces.append([i, i_next, n_seg + i_next])
        faces.append([i, n_seg + i_next, n_seg + i])
    # Bottom cap
    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        faces.append([bot_c, i_next, i])
    # Top cap
    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        faces.append([top_c, n_seg + i, n_seg + i_next])

    return verts, np.array(faces, dtype=np.int32)


def geometry_to_trimesh(geom: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Geometry to (vertices, faces) numpy arrays."""
    if geom.geometry_type == GeometryType.BOX:
        return _box_triangles(*geom.size)
    elif geom.geometry_type == GeometryType.SPHERE:
        return _sphere_triangles(geom.radius)
    elif geom.geometry_type == GeometryType.CYLINDER:
        return _cylinder_triangles(geom.radius, geom.length)
    elif geom.geometry_type == GeometryType.MESH:
        # Use cached mesh data if available (loaded by URDF parser)
        if geom.mesh_loaded:
            return geom.mesh_vertices.copy(), geom.mesh_faces.copy()
        # Fallback: try loading directly
        try:
            import trimesh
            tm = trimesh.load(geom.mesh_path, force="mesh")
            if isinstance(tm, trimesh.Scene):
                tm = tm.dump(concatenate=True)
            if geom.mesh_scale is not None:
                tm.apply_scale(geom.mesh_scale)
            return np.array(tm.vertices, dtype=np.float64), np.array(tm.faces, dtype=np.int32)
        except Exception:
            return _sphere_triangles(0.02)
    else:
        return _sphere_triangles(0.02)


# ── Internal mesh storage ────────────────────────────────────────

@dataclass
class _MeshEntry:
    """A renderable mesh stored in the viewer."""
    base_verts: np.ndarray      # (n, 3) untransformed vertices
    faces: np.ndarray           # (m, 3) triangle indices
    color: np.ndarray           # (3,) RGB float
    opacity: float = 1.0
    # Per-vertex color (overrides solid color)
    per_vertex_color: np.ndarray | None = None
    # Current world-space vertices (updated each frame)
    world_verts: np.ndarray | None = None
    visible: bool = True
    wireframe: bool = False
    # Cached edge index array for wireframe rendering (computed once from faces)
    _cached_edges: np.ndarray | None = field(default=None, repr=False)


@dataclass
class _ParticleEntry:
    """A renderable particle cloud stored in the viewer."""
    positions: np.ndarray                       # (P, 3) world-space
    radius: float = 0.005
    color: np.ndarray = field(default_factory=lambda: np.array([0.35, 0.75, 1.0]))
    per_vertex_color: np.ndarray | None = None  # (P, 3) optional per-particle RGB
    visible: bool = True
    # Private Taichi fields — lazily (re)allocated in _render_frame().
    _ti_pos: "ti.MatrixField | None" = field(default=None, repr=False)
    _ti_col: "ti.MatrixField | None" = field(default=None, repr=False)
    _cap: int = field(default=0, repr=False)


# ── SimViewer ────────────────────────────────────────────────────

class SimViewer:
    """Interactive 3D viewer using Taichi GGUI.

    Replaces PyVista with Metal/Vulkan native rendering.
    """

    def __init__(
        self,
        title: str = "RoboSim Viewer",
        window_size: tuple[int, int] = (1200, 800),
        background: tuple[float, float, float] = (0.15, 0.15, 0.2),
    ):
        self.title = title
        self.window_size = window_size
        self.background = background

        self._meshes: dict[str, _MeshEntry] = {}
        self._particles: dict[str, _ParticleEntry] = {}
        self._window: ti.ui.Window | None = None
        self._canvas = None
        self._scene = None
        self._camera = None

        # Taichi fields (rebuilt when meshes change)
        self._ti_verts: ti.VectorField | None = None
        self._ti_indices: ti.ScalarField | None = None
        self._ti_colors: ti.VectorField | None = None
        # Wireframe line fields
        self._ti_wire_verts: ti.VectorField | None = None
        self._ti_wire_indices: ti.ScalarField | None = None
        self._ti_wire_colors: ti.VectorField | None = None
        self._dirty = True  # needs field rebuild

        self._hud_lines: list[str] = []
        self._initialized = False

        # ── Custom orbit camera state (Z-up) ──
        self._cam_target = np.array([0.0, 0.0, 0.4])
        self._cam_azimuth = 45.0    # degrees, 0 = +X
        self._cam_elevation = 25.0  # degrees above horizon
        self._cam_distance = 3.0
        self._prev_mouse = None     # (x, y) or None
        self._prev_rmouse = None    # for panning
        self._orbit_speed = 0.3     # degrees per pixel
        self._zoom_speed = 0.05
        self._pan_speed = 0.003
        self._camera_locked = False
        # Free (non-orbit) pose used when locking to a palm-mounted camera.
        self._cam_eye: np.ndarray | None = None
        self._cam_lookat: np.ndarray | None = None
        self._cam_up: np.ndarray | None = None

    def lock_camera(
        self,
        target: tuple[float, float, float],
        azimuth: float,
        elevation: float,
        distance: float,
    ) -> None:
        """Fix orbit camera for repeatable CV screenshots."""
        self._cam_target = np.asarray(target, dtype=float)
        self._cam_azimuth = float(azimuth)
        self._cam_elevation = float(elevation)
        self._cam_distance = float(distance)
        self._cam_eye = None
        self._cam_lookat = None
        self._cam_up = None
        self._camera_locked = True
        if self._camera is not None:
            self._apply_camera()

    def set_world_camera(
        self,
        eye: tuple[float, float, float] | np.ndarray,
        lookat: tuple[float, float, float] | np.ndarray,
        up: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
        *,
        lock: bool = True,
    ) -> None:
        """Set an explicit world-space eye / lookat / up (e.g. palm-mounted)."""
        self._cam_eye = np.asarray(eye, dtype=float)
        self._cam_lookat = np.asarray(lookat, dtype=float)
        self._cam_up = np.asarray(up, dtype=float)
        if lock:
            self._camera_locked = True
        if self._camera is not None:
            self._apply_camera()

    def get_framebuffer(self) -> np.ndarray:
        """Return the latest RGBA framebuffer (float32, 0–1)."""
        if self._window is None:
            raise RuntimeError("viewer not initialized")
        return self._window.get_image_buffer_as_numpy()

    def save_screenshot(self, path: str | Path) -> None:
        """Save the current framebuffer to a PNG file."""
        if self._window is None:
            raise RuntimeError("viewer not initialized")
        self._window.save_image(str(path))

    def _cam_position(self) -> np.ndarray:
        """Compute camera world position from orbit parameters."""
        az = np.radians(self._cam_azimuth)
        el = np.radians(self._cam_elevation)
        r = self._cam_distance
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return self._cam_target + np.array([x, y, z])

    def _apply_camera(self):
        """Apply free or orbit camera state to the Taichi camera object."""
        if self._cam_eye is not None and self._cam_lookat is not None:
            up = self._cam_up if self._cam_up is not None else np.array([0.0, 0.0, 1.0])
            self._camera.position(*self._cam_eye)
            self._camera.lookat(*self._cam_lookat)
            self._camera.up(*up)
            return
        pos = self._cam_position()
        self._camera.position(*pos)
        self._camera.lookat(*self._cam_target)
        self._camera.up(0, 0, 1)

    def _handle_camera_input(self):
        """Process mouse/keyboard input for orbit camera (Z-up correct)."""
        if self._camera_locked:
            return
        window = self._window

        # ── Mouse orbit (LMB drag) ──
        curr_mouse = window.get_cursor_pos()
        lmb = window.is_pressed(ti.ui.LMB)
        rmb = window.is_pressed(ti.ui.RMB)

        if lmb:
            if self._prev_mouse is not None:
                dx = (curr_mouse[0] - self._prev_mouse[0]) * self.window_size[0]
                dy = (curr_mouse[1] - self._prev_mouse[1]) * self.window_size[1]
                # Mouse left-right → azimuth (horizontal orbit)
                self._cam_azimuth -= dx * self._orbit_speed
                # Mouse up-down → elevation (vertical orbit)
                self._cam_elevation -= dy * self._orbit_speed
                self._cam_elevation = np.clip(self._cam_elevation, -85.0, 89.0)
            self._prev_mouse = curr_mouse
        else:
            self._prev_mouse = None

        # ── Mouse pan (RMB drag) ──
        if rmb:
            if self._prev_rmouse is not None:
                dx = (curr_mouse[0] - self._prev_rmouse[0]) * self.window_size[0]
                dy = (curr_mouse[1] - self._prev_rmouse[1]) * self.window_size[1]
                az = np.radians(self._cam_azimuth)
                # Screen-right in world (perpendicular to view, horizontal)
                right = np.array([-np.sin(az), np.cos(az), 0.0])
                up = np.array([0.0, 0.0, 1.0])
                self._cam_target -= right * dx * self._pan_speed * self._cam_distance
                self._cam_target += up * dy * self._pan_speed * self._cam_distance
            self._prev_rmouse = curr_mouse
        else:
            self._prev_rmouse = None

        # ── Keyboard zoom (W/S) ──
        if window.is_pressed('w'):
            self._cam_distance *= (1.0 - self._zoom_speed)
        if window.is_pressed('s'):
            self._cam_distance *= (1.0 + self._zoom_speed)
        self._cam_distance = np.clip(self._cam_distance, 0.2, 20.0)

        # ── Keyboard pan (A/D/Q/E) ──
        az = np.radians(self._cam_azimuth)
        right = np.array([-np.sin(az), np.cos(az), 0.0])
        if window.is_pressed('a'):
            self._cam_target -= right * 0.02
        if window.is_pressed('d'):
            self._cam_target += right * 0.02
        if window.is_pressed('e'):
            self._cam_target[2] += 0.02
        if window.is_pressed('q'):
            self._cam_target[2] -= 0.02

    def initialize(self):
        """Create the Taichi window and set up the scene."""
        self._window = ti.ui.Window(self.title, self.window_size, vsync=True)
        self._canvas = self._window.get_canvas()
        self._scene = self._window.get_scene()
        self._camera = ti.ui.Camera()
        self._apply_camera()

        # Ground plane
        gv = np.array([
            [-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0],
        ], dtype=np.float64)
        gf = np.array([[0,1,2], [0,2,3]], dtype=np.int32)
        self._meshes["__ground__"] = _MeshEntry(
            base_verts=gv, faces=gf,
            color=np.array([0.45, 0.45, 0.45]),
            world_verts=gv.copy(),
        )

        # Axis lines as thin boxes
        for axis_name, direction, color in [
            ("__x_axis__", [1, 0, 0], [0.9, 0.2, 0.2]),
            ("__y_axis__", [0, 1, 0], [0.2, 0.9, 0.2]),
            ("__z_axis__", [0, 0, 1], [0.2, 0.2, 0.9]),
        ]:
            d = np.array(direction, dtype=np.float64)
            # Thin cylinder along axis
            verts, faces = _cylinder_triangles(0.005, 2.0, n_seg=6)
            # Rotate to align with axis (default is Z)
            if d[0] > 0:  # X
                verts = verts[:, [2, 1, 0]]  # swap Z↔X
                verts[:, 0] += 1.0  # center
            elif d[1] > 0:  # Y
                verts = verts[:, [0, 2, 1]]  # swap Z↔Y
                verts[:, 1] += 1.0
            else:  # Z
                verts[:, 2] += 0.75
            self._meshes[axis_name] = _MeshEntry(
                base_verts=verts, faces=faces,
                color=np.array(color),
                world_verts=verts.copy(),
            )

        self._dirty = True
        self._initialized = True

    # ── Mesh management ──

    def add_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: np.ndarray = np.array([0.6, 0.6, 0.6]),
        opacity: float = 1.0,
        per_vertex_color: np.ndarray | None = None,
    ):
        """Add a triangle mesh to the scene."""
        self._meshes[name] = _MeshEntry(
            base_verts=vertices.copy(),
            faces=faces.copy(),
            color=color[:3].copy(),
            opacity=opacity,
            per_vertex_color=per_vertex_color,
            world_verts=vertices.copy(),
        )
        self._dirty = True

    def update_mesh_vertices(self, name: str, vertices: np.ndarray):
        """Update vertex positions of an existing mesh."""
        entry = self._meshes[name]
        entry.world_verts = vertices

    def update_mesh_color(self, name: str, per_vertex_color: np.ndarray):
        """Update per-vertex color of an existing mesh."""
        self._meshes[name].per_vertex_color = per_vertex_color

    # ── Particle cloud API ────────────────────────────────────────

    def add_particles(
        self,
        name: str,
        positions: np.ndarray,
        radius: float = 0.005,
        color: np.ndarray = np.array([0.35, 0.75, 1.0]),
        per_vertex_color: np.ndarray | None = None,
    ):
        """Register a particle cloud (e.g. MPM material points).

        Subsequent frames call :meth:`update_particles` with new positions
        and/or colours. Radius and colour-mode stay stable across frames.
        """
        self._particles[name] = _ParticleEntry(
            positions=np.ascontiguousarray(positions, dtype=np.float32),
            radius=float(radius),
            color=np.asarray(color, dtype=np.float32)[:3].copy(),
            per_vertex_color=(
                None if per_vertex_color is None
                else np.ascontiguousarray(per_vertex_color, dtype=np.float32)
            ),
        )

    def update_particles(
        self,
        name: str,
        positions: np.ndarray,
        per_vertex_color: np.ndarray | None = None,
    ):
        """Swap in new positions (and optional per-particle colours)."""
        entry = self._particles[name]
        entry.positions = np.ascontiguousarray(positions, dtype=np.float32)
        if per_vertex_color is not None:
            entry.per_vertex_color = np.ascontiguousarray(
                per_vertex_color, dtype=np.float32
            )

    def set_particles_visible(self, name: str, visible: bool):
        if name in self._particles:
            self._particles[name].visible = visible

    def set_mesh_visible(self, name: str, visible: bool):
        """Show or hide a mesh by name."""
        if name in self._meshes:
            self._meshes[name].visible = visible

    def set_meshes_visible_by_prefix(self, prefix: str, visible: bool):
        """Show or hide all meshes whose name starts with *prefix*."""
        for name, entry in self._meshes.items():
            if name.startswith(prefix):
                entry.visible = visible

    def set_meshes_wireframe_by_prefix(self, prefix: str, wireframe: bool):
        """Enable or disable wireframe rendering for meshes by prefix."""
        for name, entry in self._meshes.items():
            if name.startswith(prefix):
                entry.wireframe = wireframe

    def add_robot_meshes(
        self,
        robot_name: str,
        link_names: list[str],
        geometries: list[tuple[Geometry, Transform]],
        colors: list[np.ndarray | None],
    ):
        """Add visual meshes for a robot's links."""
        for link_name, (geom, origin), color in zip(link_names, geometries, colors):
            verts, faces = geometry_to_trimesh(geom)

            # Apply visual origin transform
            mat = origin.to_matrix()
            R, t = mat[:3, :3], mat[:3, 3]
            verts = (R @ verts.T).T + t

            rgba = color if color is not None else np.array([0.6, 0.6, 0.6, 1.0])
            actor_name = f"{robot_name}/{link_name}"
            self._meshes[actor_name] = _MeshEntry(
                base_verts=verts,
                faces=faces,
                color=rgba[:3],
                opacity=float(rgba[3]) if len(rgba) > 3 else 1.0,
                world_verts=verts.copy(),
            )

        self._dirty = True

    def update_link_transforms(
        self,
        robot_name: str,
        link_names: list[str],
        transforms: list[Transform],
        geometries: list[tuple[Geometry, Transform]],
        colors: list[np.ndarray | None],
    ):
        """Update the transforms of all robot link meshes."""
        for link_name, world_T, (geom, origin), color in zip(
            link_names, transforms, geometries, colors
        ):
            actor_name = f"{robot_name}/{link_name}"
            entry = self._meshes.get(actor_name)
            if entry is None:
                continue

            # Use cached base_verts (already has visual origin baked in)
            # — avoids re-calling geometry_to_trimesh every frame
            verts = entry.base_verts

            # Apply world transform
            mat_w = world_T.to_matrix()
            R_w, t_w = mat_w[:3, :3], mat_w[:3, 3]
            entry.world_verts = (R_w @ verts.T).T + t_w

    # ── Text overlay ──

    def add_text(self, text: str, **kwargs):
        """Set HUD text (displayed next frame)."""
        self._hud_lines = text.split("\n")

    # ── Rendering ──

    def _rebuild_fields(self):
        """Combine all meshes into single Taichi fields for rendering."""
        # Count totals
        total_verts = 0
        total_idx = 0
        for entry in self._meshes.values():
            total_verts += entry.world_verts.shape[0]
            total_idx += entry.faces.shape[0] * 3

        if total_verts == 0:
            return

        self._ti_verts = ti.Vector.field(3, dtype=ti.f32, shape=total_verts)
        self._ti_indices = ti.field(dtype=ti.i32, shape=total_idx)
        self._ti_colors = ti.Vector.field(3, dtype=ti.f32, shape=total_verts)
        self._dirty = False

    def _upload_fields(self):
        """Upload current vertex positions + colors to Taichi fields."""
        solid_verts = []
        solid_colors = []
        solid_indices = []
        s_offset = 0

        wire_verts = []
        wire_colors = []
        wire_indices = []
        w_offset = 0

        for entry in self._meshes.values():
            if not entry.visible:
                continue
            nv = entry.world_verts.shape[0]

            if entry.per_vertex_color is not None:
                colors = entry.per_vertex_color[:nv]
            else:
                colors = np.broadcast_to(entry.color, (nv, 3)).copy()

            if entry.wireframe:
                wire_verts.append(entry.world_verts)
                wire_colors.append(colors)
                # Extract unique edges from triangle faces (cached after first call)
                if entry._cached_edges is None:
                    edges = set()
                    for f in entry.faces:
                        for i in range(3):
                            e = (min(int(f[i]), int(f[(i+1) % 3])),
                                 max(int(f[i]), int(f[(i+1) % 3])))
                            edges.add(e)
                    # Store as flat 1D array: [e0_a, e0_b, e1_a, e1_b, ...]
                    entry._cached_edges = np.array(sorted(edges), dtype=np.int32).ravel()
                wire_indices.append(entry._cached_edges + w_offset)
                w_offset += nv
            else:
                solid_verts.append(entry.world_verts)
                solid_colors.append(colors)
                solid_indices.append(entry.faces.ravel() + s_offset)
                s_offset += nv

        # ── Solid meshes ──
        if solid_verts:
            verts_np = np.concatenate(solid_verts).astype(np.float32)
            colors_np = np.concatenate(solid_colors).astype(np.float32)
            indices_np = np.concatenate(solid_indices).astype(np.int32)

            if self._ti_verts is None or self._ti_verts.shape[0] != verts_np.shape[0]:
                self._ti_verts = ti.Vector.field(3, dtype=ti.f32, shape=verts_np.shape[0])
                self._ti_colors = ti.Vector.field(3, dtype=ti.f32, shape=verts_np.shape[0])
            if self._ti_indices is None or self._ti_indices.shape[0] != indices_np.shape[0]:
                self._ti_indices = ti.field(dtype=ti.i32, shape=indices_np.shape[0])

            self._ti_verts.from_numpy(verts_np)
            self._ti_colors.from_numpy(colors_np)
            self._ti_indices.from_numpy(indices_np)
        else:
            self._ti_verts = None

        # ── Wireframe lines ──
        if wire_verts:
            wv_np = np.concatenate(wire_verts).astype(np.float32)
            wc_np = np.concatenate(wire_colors).astype(np.float32)
            wi_np = np.concatenate(wire_indices).astype(np.int32)

            if self._ti_wire_verts is None or self._ti_wire_verts.shape[0] != wv_np.shape[0]:
                self._ti_wire_verts = ti.Vector.field(3, dtype=ti.f32, shape=wv_np.shape[0])
                self._ti_wire_colors = ti.Vector.field(3, dtype=ti.f32, shape=wv_np.shape[0])
            if self._ti_wire_indices is None or self._ti_wire_indices.shape[0] != wi_np.shape[0]:
                self._ti_wire_indices = ti.field(dtype=ti.i32, shape=wi_np.shape[0])

            self._ti_wire_verts.from_numpy(wv_np)
            self._ti_wire_colors.from_numpy(wc_np)
            self._ti_wire_indices.from_numpy(wi_np)
        else:
            self._ti_wire_verts = None

    def _render_frame(self):
        """Render one frame."""
        self._handle_camera_input()
        self._apply_camera()
        self._scene.set_camera(self._camera)

        self._scene.ambient_light((0.7, 0.7, 0.7))
        self._scene.point_light(pos=(2.0, -2.0, 3.0), color=(0.8, 0.8, 0.8))
        self._scene.point_light(pos=(-1.0, 2.0, 2.0), color=(0.4, 0.4, 0.4))

        self._upload_fields()

        if self._ti_verts is not None:
            self._scene.mesh(
                self._ti_verts, self._ti_indices,
                per_vertex_color=self._ti_colors, two_sided=True,
            )

        if self._ti_wire_verts is not None:
            self._scene.lines(
                self._ti_wire_verts, width=2.0,
                indices=self._ti_wire_indices,
                per_vertex_color=self._ti_wire_colors,
            )

        # Particle clouds — one scene.particles() call per registered entry.
        for entry in self._particles.values():
            if not entry.visible or entry.positions.size == 0:
                continue
            P = entry.positions.shape[0]
            if entry._ti_pos is None or entry._cap < P:
                entry._cap = max(P, 1)
                entry._ti_pos = ti.Vector.field(3, dtype=ti.f32, shape=entry._cap)
                entry._ti_col = ti.Vector.field(3, dtype=ti.f32, shape=entry._cap)
            pos_pad = np.zeros((entry._cap, 3), dtype=np.float32)
            pos_pad[:P] = entry.positions
            entry._ti_pos.from_numpy(pos_pad)
            col_pad = np.zeros((entry._cap, 3), dtype=np.float32)
            if entry.per_vertex_color is not None:
                col_pad[:P] = entry.per_vertex_color
            else:
                col_pad[:P] = entry.color[None, :]
            entry._ti_col.from_numpy(col_pad)
            self._scene.particles(
                entry._ti_pos,
                radius=entry.radius,
                per_vertex_color=entry._ti_col,
            )

        self._canvas.set_background_color(self.background)
        self._canvas.scene(self._scene)

        # HUD overlay
        if self._hud_lines:
            gui = self._window.get_gui()
            n_lines = len(self._hud_lines)
            with gui.sub_window("Info", x=0.01, y=0.01, width=0.4, height=0.03 + 0.03 * n_lines):
                for line in self._hud_lines:
                    gui.text(line)

    # ── Main loop ──

    def add_callback(self, callback: Callable, interval: int = 16, max_steps: int = 10_000_000):
        """Register a step callback (called every frame in the main loop).

        Parameters
        ----------
        callback : called with (step_count: int) each frame
        interval : ignored (Taichi runs as fast as possible with vsync)
        max_steps : stop after this many frames
        """
        self._step_callback = callback
        self._max_steps = max_steps

    def show(self):
        """Run the main rendering loop (blocking)."""
        step = 0
        while self._window.running and step < getattr(self, "_max_steps", 10_000_000):
            if hasattr(self, "_step_callback") and self._step_callback is not None:
                self._step_callback(step)

            self._render_frame()
            self._window.show()
            step += 1

    def update(self):
        """Compatibility stub (not needed for Taichi)."""
        pass

    def close(self):
        """Close the viewer."""
        if self._window is not None:
            self._window.destroy()
