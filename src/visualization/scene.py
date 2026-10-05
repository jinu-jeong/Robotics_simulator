"""Scene composition: ``VisualizationState`` + ``ViewOptions`` -> Taichi draw calls.

:class:`SceneRenderer` owns all GPU buffers and is the only place where the
mapping "physical quantity -> pixels" is decided (colors, amplification,
arrow scaling, marker sizes). It is used by the interactive
:class:`~src.visualization.taichi_viewer.TaichiViewer` and can also render
head-less for tests / screenshots.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import numpy as np

from ..geometry.mesh_utils import feature_edges, unique_edges, vertex_normals
from ..geometry.primitives import to_triangle_soup
from .colormap import colormap
from .force_renderer import ArrowStyle, ForceArrowRenderer
from .mesh_renderer import IndexedMeshBuffers, LineBuffers, PointBuffers, TriangleSoupBuffers, segments_from_edges
from .state import ForceVector, VisualizationState

MODES = ("original", "deformed", "overlay")
COLOR_MODES = ("displacement", "scalar", "solid")


@dataclass
class ViewOptions:
    """Runtime visualization switches. None of these affect physics."""

    mode: str = "overlay"
    amplification: float = 10.0
    show_surface: bool = True
    show_wireframe: bool = True
    show_tet_edges: bool = False
    show_nodes: bool = False
    show_fixed_nodes: bool = True
    show_contact: bool = True
    show_gt_force: bool = True
    show_estimated_force: bool = True
    show_extra_forces: bool = True
    show_objects: bool = True
    show_observation_camera: bool = True
    show_keypoints: bool = True  # surface markers observed by the camera
    color_mode: str = "displacement"  # displacement | scalar | solid
    use_alt_displacement: bool = False  # e.g. show ROM reconstruction instead of full FEM
    show_gui_panel: bool = True
    force_meters_per_newton: float | None = None  # None -> auto

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ViewOptions":
        d = dict(cfg.get("defaults", {}))
        opts = cls()
        for k, v in d.items():
            if k == "color_by_displacement":
                opts.color_mode = "displacement" if v else "solid"
            elif hasattr(opts, k):
                setattr(opts, k, v)
        opts.amplification = float(cfg.get("deformation", {}).get("amplification", opts.amplification))
        opts.force_meters_per_newton = cfg.get("force_display", {}).get("meters_per_newton", None)
        if opts.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        return opts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def cycle_mode(self, step: int = 1) -> None:
        self.mode = MODES[(MODES.index(self.mode) + step) % len(MODES)]

    def cycle_color_mode(self) -> None:
        self.color_mode = COLOR_MODES[(COLOR_MODES.index(self.color_mode) + 1) % len(COLOR_MODES)]


class SceneRenderer:
    """Converts a state + options into GGUI draw calls each frame."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.colors = {k: tuple(float(x) for x in v) for k, v in cfg["colors"].items()}
        self.sizes = dict(cfg["sizes"])
        self.force_cfg = dict(cfg.get("force_display", {}))
        self.colormap_name = cfg.get("deformation", {}).get("colormap", "viridis")
        self.lighting = cfg.get("lighting", {})

        self.state: VisualizationState | None = None
        self._topology_key: tuple | None = None
        self.mesh: IndexedMeshBuffers | None = None
        self.fixed_points = PointBuffers(256)
        self.contact_points = PointBuffers(16)
        self.keypoints = PointBuffers(64)
        self.camera_marker = PointBuffers(4)
        self.camera_lines = LineBuffers(32)
        self.objects = TriangleSoupBuffers(2048)
        self.forces = ForceArrowRenderer(12288)

        self.bbox_min = np.zeros(3)
        self.bbox_max = np.ones(3)
        self.diag = 1.0
        self.color_range: tuple[float, float] = (0.0, 0.0)
        self._last_signature: tuple | None = None

    # ------------------------------------------------------------ state
    def set_state(self, state: VisualizationState) -> None:
        """Install a new state. Re-allocates GPU buffers only if topology changed."""
        key = state.topology_key()
        if self.mesh is None or key != self._topology_key:
            if state.surface_edges is not None and len(state.surface_edges):
                surface_edges = state.surface_edges
            else:
                surface_edges = unique_edges(state.surface_faces) if len(state.surface_faces) else None
            tet_edges = state.element_edges if state.element_edges is not None and len(state.element_edges) else None
            outline = (
                feature_edges(state.nodes_original, state.surface_faces, float(self.cfg.get("overlay", {}).get("outline_angle_deg", 30.0)))
                if len(state.surface_faces)
                else None
            )
            self.mesh = IndexedMeshBuffers(state.n_nodes, state.surface_faces, surface_edges, tet_edges, outline)
            self._topology_key = key
        self.dense_mesh = len(state.surface_faces) > int(self.cfg.get("overlay", {}).get("dense_faces_threshold", 4000))
        self.state = state
        self.bbox_min, self.bbox_max = state.finger_bounds()
        self.diag = float(max(np.linalg.norm(self.bbox_max - self.bbox_min), 1e-9))

        # Wireframe lines are offset along vertex normals to avoid z-fighting.
        self._normals = vertex_normals(state.nodes_original, state.surface_faces) if len(state.surface_faces) else np.zeros_like(state.nodes_original)
        self._wire_offset = float(self.sizes.get("wire_offset_rel", 0.003)) * self.diag
        self.mesh.set_positions("original", state.nodes_original)
        self.mesh.set_positions("original_wire", state.nodes_original + self._wire_offset * self._normals)

        self._upload_static(state)
        self._last_signature = None  # force dynamic refresh

    def _upload_static(self, state: VisualizationState) -> None:
        # Observation camera marker + frustum
        cam = state.observation_camera
        if cam is not None:
            self.camera_marker.set(cam.position[None, :], color=self.colors["observation_camera"])
            v, e = cam.frustum_lines()
            self.camera_lines.set(segments_from_edges(v, e), color=self.colors["observation_frustum"])
        else:
            self.camera_marker.set(np.zeros((0, 3)))
            self.camera_lines.set(np.zeros((0, 3)))

    # ------------------------------------------------------------ dynamic
    def _display_offset(self, points: np.ndarray, alpha: float, use_alt: bool) -> np.ndarray:
        """Move arbitrary points with the (amplified) displacement of the nearest node."""
        st = self.state
        u = st.displacement_alt if use_alt else st.displacement
        if u is None or alpha == 0.0 or len(points) == 0:
            return np.asarray(points, float)
        pts = np.asarray(points, float).reshape(-1, 3)
        d = np.linalg.norm(pts[:, None, :] - st.nodes_original[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        return pts + alpha * u[nearest]

    def _estimate_offset(self, gt_vector: np.ndarray) -> np.ndarray:
        """Small lateral shift for the estimated-force arrow (perpendicular to F_true)."""
        rel = float(self.force_cfg.get("estimate_offset_rel", 0.0))
        if rel <= 0.0:
            return np.zeros(3)
        d = np.asarray(gt_vector, float)
        n = np.linalg.norm(d)
        if n == 0.0:
            return np.zeros(3)
        d = d / n
        # pick the world axis least aligned with d and make it perpendicular
        axis = np.eye(3)[int(np.argmin(np.abs(d)))]
        perp = axis - (axis @ d) * d
        return rel * self.diag * perp / np.linalg.norm(perp)

    def update(self, options: ViewOptions) -> None:
        """Refresh all buffers that depend on the options (cheap if unchanged)."""
        st = self.state
        if st is None or self.mesh is None:
            return
        alpha = float(options.amplification) if options.mode != "original" else 0.0
        use_alt = bool(options.use_alt_displacement and st.displacement_alt is not None)
        sig = (alpha, use_alt, options.color_mode, options.force_meters_per_newton,
               options.show_gt_force, options.show_estimated_force, options.show_extra_forces)
        if sig == self._last_signature:
            return
        self._last_signature = sig

        # Positions (visual amplification only!)
        display = st.displaced(alpha, use_alt=use_alt)
        self.mesh.set_positions("display", display)
        self.mesh.set_positions("display_wire", display + self._wire_offset * self._normals)

        # Colors
        if options.color_mode == "displacement":
            mag = st.displacement_magnitude(use_alt=use_alt)
            self.color_range = (0.0, float(mag.max()) if mag.size else 0.0)
            self.mesh.set_colors(colormap(mag, self.colormap_name, 0.0, self.color_range[1]))
        elif options.color_mode == "scalar" and st.node_scalar is not None:
            s = st.node_scalar
            self.color_range = (float(s.min()), float(s.max()))
            self.mesh.set_colors(colormap(s, self.colormap_name, *self.color_range))
        else:
            self.color_range = (0.0, 0.0)
            self.mesh.fill_colors(self.colors["deformed_surface"])

        # Fixed nodes follow the display positions: if they move, the BC is wrong.
        if st.fixed_nodes is not None and len(st.fixed_nodes):
            self.fixed_points.set(display[st.fixed_nodes], color=self.colors["fixed_nodes"])
        else:
            self.fixed_points.set(np.zeros((0, 3)))

        # Contact markers
        if st.contact_points:
            pts = []
            for c in st.contact_points:
                if c.node_index is not None:
                    pts.append(display[c.node_index])
                else:
                    pts.append(self._display_offset(c.position[None, :], alpha, use_alt)[0])
            self.contact_points.set(np.array(pts), color=self.colors["contact"])
        else:
            self.contact_points.set(np.zeros((0, 3)))

        # Surface markers (keypoints) riding on the displayed surface
        if st.keypoints is not None and len(st.keypoints):
            kp = self._display_offset(st.keypoints, alpha, use_alt)
            vis_c = np.asarray(self.colors.get("keypoint", [1.0, 0.2, 0.1]), float)
            hid_c = np.asarray(self.colors.get("keypoint_hidden", [0.45, 0.45, 0.45]), float)
            cols = np.where(st.keypoints_visible[:, None], vis_c[None, :], hid_c[None, :])
            self.keypoints.set(kp, colors=cols)
        else:
            self.keypoints.set(np.zeros((0, 3)))

        # Force arrows
        force_list = []
        for role, f in st.all_forces():
            if role == "gt" and not options.show_gt_force:
                continue
            if role == "est" and not options.show_estimated_force:
                continue
            if role == "extra" and not options.show_extra_forces:
                continue
            rgb = f.color if f.color is not None else self.colors[
                {"gt": "gt_force", "est": "estimated_force", "extra": "gt_force"}[role]
            ]
            origin = self._display_offset(f.origin[None, :], alpha, use_alt)[0]
            if role == "est" and st.ground_truth_force is not None:
                # Side-by-side: shift the estimate perpendicular to the ground-truth
                # arrow so both stay visible when they (nearly) coincide.
                origin = origin + self._estimate_offset(st.ground_truth_force.vector)
            force_list.append((ForceVector(origin, f.vector, f.label, f.color), rgb))
        style = ArrowStyle(
            shaft_radius=self.sizes["arrow_shaft_radius_rel"] * self.diag,
            head_radius=self.sizes["arrow_head_radius_rel"] * self.diag,
            head_length=self.sizes["arrow_head_length_rel"] * self.diag,
            n_segments=int(self.sizes.get("arrow_segments", 16)),
        )
        self.forces.update(
            force_list,
            self.diag,
            style,
            options.force_meters_per_newton,
            float(self.force_cfg.get("auto_max_length_rel", 0.5)),
            float(self.force_cfg.get("min_length_rel", 0.02)),
            anchor=str(self.force_cfg.get("anchor", "tip")),
        )

        # Objects (optionally riding on the visually deformed surface)
        verts, cols = [], []
        for obj in st.objects:
            if len(obj.faces) == 0:
                continue
            soup = to_triangle_soup(obj.vertices, obj.faces)
            if obj.attach_to is not None:
                shift = self._display_offset(obj.attach_to[None, :], alpha, use_alt)[0] - obj.attach_to
                soup = soup + shift
            verts.append(soup)
            rgb = obj.color if obj.color is not None else self.colors["object"]
            cols.append(np.tile(np.asarray(rgb, np.float32), (len(soup), 1)))
        if verts:
            self.objects.set(np.concatenate(verts), np.concatenate(cols))
        else:
            self.objects.clear()

    # ------------------------------------------------------------ drawing
    def add_lights(self, scene) -> None:
        amb = self.lighting.get("ambient", [0.4, 0.4, 0.4])
        scene.ambient_light(tuple(float(x) for x in amb))
        center = 0.5 * (self.bbox_min + self.bbox_max)
        for pl in self.lighting.get("point_lights", []):
            pos = center + np.asarray(pl["rel_offset"], float) * self.diag
            scene.point_light(pos=tuple(float(x) for x in pos), color=tuple(float(x) for x in pl["color"]))

    def render(self, scene, options: ViewOptions) -> None:
        """Issue all draw calls for the current frame."""
        if self.state is None or self.mesh is None:
            return
        self.update(options)
        self.add_lights(scene)
        st = self.state
        s = self.sizes
        c = self.colors
        mode = options.mode
        has_u = st.displacement is not None

        show_original_solid = mode == "original" or (mode == "overlay" and not has_u)
        show_deformed = mode in ("deformed", "overlay") and has_u
        show_original_ghost = mode == "overlay" and has_u

        if show_original_solid:
            if options.show_surface:
                self.mesh.draw_surface(scene, "original", color=c["original_surface"])
            if options.show_wireframe:
                self.mesh.draw_surface_edges(scene, "original_wire", c["wireframe"], s["wireframe_width"])
            if options.show_tet_edges:
                self.mesh.draw_tet_edges(scene, "original", c["tet_edges"], s["tet_edge_width"])
            if options.show_nodes:
                self.mesh.draw_nodes(scene, "original", s["node_radius_rel"] * self.diag, c["nodes"])

        if show_original_ghost:
            # Undeformed reference (GGUI has no alpha blending): light wireframe for
            # coarse meshes, feature-edge outline for dense ones ("auto"), or forced.
            style = str(self.cfg.get("overlay", {}).get("ghost_style", "auto"))
            if style == "auto":
                style = "outline" if self.dense_mesh else "wireframe"
            if style == "wireframe":
                self.mesh.draw_surface_edges(scene, "original_wire", c["original_wireframe"], s["wireframe_width"])
            elif style == "outline":
                self.mesh.draw_outline(scene, "original_wire", c["original_wireframe"], s.get("outline_width", 2.0))

        if show_deformed:
            if options.show_surface:
                self.mesh.draw_surface(
                    scene, "display",
                    color=c["deformed_surface"],
                    per_vertex_color=options.color_mode != "solid",
                )
            if options.show_wireframe:
                self.mesh.draw_surface_edges(scene, "display_wire", c["wireframe"], s["wireframe_width"])
            if options.show_tet_edges:
                self.mesh.draw_tet_edges(scene, "display", c["tet_edges"], s["tet_edge_width"])
            if options.show_nodes:
                self.mesh.draw_nodes(scene, "display", s["node_radius_rel"] * self.diag, c["nodes"])

        if options.show_fixed_nodes:
            self.fixed_points.draw(scene, s["fixed_node_radius_rel"] * self.diag)
        if options.show_contact:
            self.contact_points.draw(scene, s["contact_radius_rel"] * self.diag)
        if options.show_keypoints:
            self.keypoints.draw(scene, s.get("keypoint_radius_rel", 0.006) * self.diag)
        if options.show_gt_force or options.show_estimated_force or options.show_extra_forces:
            self.forces.draw(scene)
        if options.show_objects:
            self.objects.draw(scene, two_sided=False)
        if options.show_observation_camera and st.observation_camera is not None:
            self.camera_marker.draw(scene, s["camera_marker_size_rel"] * self.diag * 0.35)
            self.camera_lines.draw(scene, s["frustum_line_width"])

    # ------------------------------------------------------------ text
    def summary_lines(self, options: ViewOptions) -> list[str]:
        """Human-readable status (mm / N) for the GUI panel or console."""
        st = self.state
        if st is None:
            return ["<no state>"]
        self.update(options)  # make sure color range / arrow scale are current
        lines = [
            f"mode: {options.mode}   alpha (visual): x{options.amplification:g}",
            f"nodes: {st.n_nodes}   faces: {len(st.surface_faces)}"
            + (f"   element edges: {len(st.element_edges)}" if st.element_edges is not None else ""),
        ]
        if st.displacement is not None:
            use_alt = options.use_alt_displacement and st.displacement_alt is not None
            tag = st.displacement_alt_name if use_alt else "u"
            lines.append(f"max |{tag}|: {1e3 * st.displacement_magnitude(use_alt).max():.4f} mm")
        if st.fixed_nodes is not None:
            lines.append(f"fixed nodes: {len(st.fixed_nodes)}")
        for cp in st.contact_points:
            p = 1e3 * cp.position
            lines.append(f"{cp.label}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm" + (f"  node {cp.node_index}" if cp.node_index is not None else ""))
        gt, est = st.ground_truth_force, st.estimated_force
        if gt is not None:
            v = gt.vector
            lines.append(f"F_true: |F|={gt.magnitude:.4f} N  ({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})")
        if est is not None:
            v = est.vector
            lines.append(f"F_est : |F|={est.magnitude:.4f} N  ({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})")
        if gt is not None and est is not None and gt.magnitude > 0:
            rel = np.linalg.norm(est.vector - gt.vector) / gt.magnitude
            lines.append(f"rel. force error: {100 * rel:.2f} %")
        if self.forces.meters_per_newton_used > 0:
            lines.append(f"arrow scale: {1e3 * self.forces.meters_per_newton_used:.1f} mm/N (visual)")
        if options.color_mode == "displacement":
            lines.append(f"color: |u| 0 .. {1e3 * self.color_range[1]:.4f} mm ({self.colormap_name})")
        elif options.color_mode == "scalar":
            lines.append(f"color: {st.node_scalar_name} {self.color_range[0]:.3g} .. {self.color_range[1]:.3g}")
        else:
            lines.append("color: solid")
        cam = st.observation_camera
        if cam is not None:
            p = 1e3 * cam.position
            lines.append(f"obs cam '{cam.name}': ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm, fov {cam.fov_y_deg:g} deg, {cam.image_width}x{cam.image_height}")
        for k, v in st.info.items():
            lines.append(f"{k}: {v}")
        return lines
