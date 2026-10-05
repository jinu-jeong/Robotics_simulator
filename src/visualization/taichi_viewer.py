"""Interactive Taichi (GGUI) 3D viewer – the project's main debugging interface.

Usage
-----
>>> viewer = TaichiViewer()                 # loads configs/viewer.yaml
>>> viewer.set_state(state)                 # a VisualizationState
>>> viewer.run()                            # blocks until the window closes

For scripted use (dataset inspector, live solver loops) call
``viewer.step()`` yourself, or pass ``on_frame`` to :meth:`run`. Register
extra keys with :meth:`register_key`.

Controls (also printed with ``H``)
----------------------------------
Mouse
  LMB drag ............ orbit          RMB drag ......... pan
  MMB drag / Ctrl+LMB . zoom           Shift+LMB ........ pan
Keyboard
  1 / 2 / 3 ........... original / deformed / overlay      Tab: cycle
  Up / Down ........... deformation amplification up/down
  - / = or Z / X ...... zoom out / in       Arrow-orbit: I J K L
  W wireframe  T tet edges  N nodes  B fixed nodes  C contact
  G ground-truth force  E estimated force  F both  O objects  V observation cam
  M cycle color mode   A toggle alt displacement (e.g. ROM)   P GUI panel
  R reset camera   S screenshot   H help   Esc quit

Head-less rendering (tests, CI, figures) is supported via ``show_window=False``
and :meth:`save_screenshot`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import taichi as ti

from ..utils.config import deep_update, load_config
from .camera import OrbitCamera
from .scene import MODES, SceneRenderer, ViewOptions
from .state import VisualizationState

_TAICHI_INITIALIZED: dict[str, Any] = {"arch": None}


def init_taichi(backend: str = "auto", **kwargs) -> str:
    """Initialize Taichi once with the requested backend.

    ``backend`` in {auto, cpu, cuda, vulkan, metal, gpu}. ``auto`` tries
    cuda -> vulkan -> metal -> cpu and returns the name that succeeded.
    Subsequent calls are no-ops (Taichi can only be initialized once per
    process in a meaningful way for GGUI).
    """
    if _TAICHI_INITIALIZED["arch"] is not None:
        return _TAICHI_INITIALIZED["arch"]
    archs = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "metal": ti.metal, "gpu": ti.gpu}
    order = ["cuda", "vulkan", "metal", "cpu"] if backend == "auto" else [backend]
    last_err: Exception | None = None
    for name in order:
        if name not in archs:
            raise ValueError(f"unknown backend {name!r}; choose from {sorted(archs) + ['auto']}")
        try:
            if name != "cpu" and not ti.lang.misc.is_arch_supported(archs[name]):
                continue
            ti.init(arch=archs[name], log_level=ti.WARN, **kwargs)
            _TAICHI_INITIALIZED["arch"] = name
            return name
        except Exception as e:  # pragma: no cover - depends on machine
            last_err = e
            continue
    raise RuntimeError(f"Could not initialize any Taichi backend from {order}: {last_err}")


class TaichiViewer:
    """Window + input handling around :class:`SceneRenderer`."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        config_name: str = "viewer",
        show_window: bool = True,
        title: str | None = None,
    ) -> None:
        cfg = load_config(config_name)
        self.cfg = deep_update(cfg, config)
        self.backend = init_taichi(self.cfg.get("backend", "auto"))

        w = self.cfg["window"]
        self.res = (int(w["width"]), int(w["height"]))
        self.window = ti.ui.Window(
            title or w.get("title", "Taichi Viewer"),
            self.res,
            vsync=bool(w.get("vsync", True)),
            show_window=show_window,
        )
        self.show_window = show_window
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color(tuple(float(x) for x in self.cfg["background_color"]))
        self.scene = self.window.get_scene()
        self.ti_camera = ti.ui.Camera()

        dc = self.cfg["developer_camera"]
        self.orbit = OrbitCamera(
            yaw_deg=float(dc.get("yaw_deg", -55.0)),
            pitch_deg=float(dc.get("pitch_deg", 22.0)),
            fov_deg=float(dc.get("fov_deg", 40.0)),
        )
        self._dc = dc

        self.options = ViewOptions.from_config(self.cfg)
        self.amp_steps = [float(x) for x in self.cfg.get("deformation", {}).get("amplification_steps", [1, 2, 5, 10, 20, 50, 100])]
        self.renderer = SceneRenderer(self.cfg)
        self.state: VisualizationState | None = None

        self._key_callbacks: dict[str, tuple[Callable[["TaichiViewer"], None], str]] = {}
        self._last_cursor: tuple[float, float] | None = None
        self._panel_rect = (0.0, 0.0, 0.0, 0.0)  # x, y, w, h in [0,1] window coordinates
        self._frame = 0
        self._fps_t0 = time.time()
        self._fps = 0.0
        self.screenshot_dir = Path("results/etc/figures")
        self._overlay_field = None  # picture-in-picture buffer (observation-camera image)
        self._overlay_rect: tuple[int, int, int, int] | None = None
        self._install_default_keys()

    # ================================================================ picture-in-picture
    def set_overlay_image(
        self,
        image: np.ndarray | None,
        corner: str = "top-right",
        max_width_frac: float = 0.34,
        margin_px: int = 10,
        border_px: int = 2,
        border_color=(1.0, 0.55, 0.10),
    ) -> None:
        """Show an (h, w, 3) image (float [0,1] or uint8) as a picture-in-picture inset.

        Used to display what the *observation camera* sees next to the 3D
        debug view (brief §15). Pass ``None`` to remove the inset.
        """
        if image is None:
            self._overlay_field = None
            self._overlay_rect = None
            return
        img = np.asarray(image)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 2 or img.shape[-1] == 1:
            img = np.repeat(img.reshape(img.shape[0], img.shape[1], 1), 3, axis=2)
        h, w = img.shape[:2]
        W, H = self.res
        # integer up/down-scaling (nearest) to at most max_width_frac of the window width
        target_w = max(1, int(max_width_frac * W))
        scale = target_w / w
        k = max(1, int(np.floor(scale))) if scale >= 1 else 1
        if scale >= 1:
            img = np.repeat(np.repeat(img, k, axis=0), k, axis=1)
        else:
            step = int(np.ceil(1 / scale))
            img = img[::step, ::step]
        h, w = img.shape[:2]
        bg = np.asarray(self.cfg["background_color"], np.float32)
        full = np.empty((W, H, 3), np.float32)
        full[:] = bg
        x0 = W - w - margin_px if "right" in corner else margin_px
        y_top = margin_px if "top" in corner else H - h - margin_px
        # GGUI image fields are indexed (x, y) with y pointing up
        y0 = H - y_top - h
        bx0, by0 = max(x0 - border_px, 0), max(y0 - border_px, 0)
        full[bx0 : min(x0 + w + border_px, W), by0 : min(y0 + h + border_px, H)] = np.asarray(border_color, np.float32)
        full[x0 : x0 + w, y0 : y0 + h] = np.transpose(img[::-1], (1, 0, 2))
        if self._overlay_field is None:
            self._overlay_field = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
        self._overlay_field.from_numpy(full)
        self._overlay_rect = (x0, y_top, w, h)

    # ================================================================ state
    def set_state(self, state: VisualizationState, refit_camera: bool | None = None) -> None:
        """Install a state; camera is re-fitted on the first state by default."""
        first = self.state is None
        self.state = state
        self.renderer.set_state(state)
        if first and self.renderer.dense_mesh and self.cfg.get("overlay", {}).get("auto_disable_wireframe", True):
            if self.options.show_wireframe:
                self.options.show_wireframe = False
                print(f"[viewer] dense mesh ({len(state.surface_faces)} faces): surface wireframe disabled (press W to enable)")
        if refit_camera or (refit_camera is None and first):
            self.fit_camera()

    def fit_camera(self) -> None:
        if self.state is None:
            return
        bmin, bmax = self.state.scene_bounds(include_objects=True, include_camera=False)
        self.orbit.fit(bmin, bmax, float(self._dc.get("distance_rel", 2.4)))

    # ================================================================ keys
    def register_key(self, key: str, callback: Callable[["TaichiViewer"], None], help_text: str = "") -> None:
        """Bind ``key`` (Taichi key name, e.g. ``"Left"``, ``"n"``) to a callback."""
        self._key_callbacks[key] = (callback, help_text)

    def _install_default_keys(self) -> None:
        o = self.options
        reg = self.register_key

        def toggle(attr):
            def _f(v):
                setattr(v.options, attr, not getattr(v.options, attr))
            return _f

        def set_mode(m):
            def _f(v):
                v.options.mode = m
            return _f

        reg("1", set_mode("original"), "original geometry")
        reg("2", set_mode("deformed"), "deformed geometry")
        reg("3", set_mode("overlay"), "overlay original + deformed")
        reg(ti.ui.TAB, lambda v: v.options.cycle_mode(), "cycle mode")
        reg(ti.ui.UP, lambda v: v.step_amplification(+1), "amplification up")
        reg(ti.ui.DOWN, lambda v: v.step_amplification(-1), "amplification down")
        reg("w", toggle("show_wireframe"), "toggle surface wireframe")
        reg("t", toggle("show_tet_edges"), "toggle tetrahedral edges")
        reg("n", toggle("show_nodes"), "toggle FEM nodes")
        reg("b", toggle("show_fixed_nodes"), "toggle fixed boundary nodes")
        reg("c", toggle("show_contact"), "toggle contact markers")
        reg("g", toggle("show_gt_force"), "toggle ground-truth force")
        reg("e", toggle("show_estimated_force"), "toggle estimated force")
        reg("f", self._toggle_all_forces, "toggle all force arrows")
        reg("o", toggle("show_objects"), "toggle objects")
        reg("v", toggle("show_observation_camera"), "toggle observation camera")
        reg(";", toggle("show_keypoints"), "toggle surface markers (keypoints)")
        reg("m", lambda v: v.options.cycle_color_mode(), "cycle color mode")
        reg("a", toggle("use_alt_displacement"), "toggle alternative displacement (ROM)")
        reg("p", toggle("show_gui_panel"), "toggle GUI panel")
        reg("r", lambda v: v.orbit.reset(), "reset developer camera")
        reg("s", lambda v: v.save_screenshot(), "save screenshot")
        reg("h", lambda v: print(v.help_text()), "print help")
        reg("-", lambda v: v.orbit.zoom(self._key_zoom()), "zoom out")
        reg("=", lambda v: v.orbit.zoom(1.0 / self._key_zoom()), "zoom in")
        reg("z", lambda v: v.orbit.zoom(self._key_zoom()), "zoom out")
        reg("x", lambda v: v.orbit.zoom(1.0 / self._key_zoom()), "zoom in")
        step = float(self._dc.get("key_orbit_step_deg", 5.0))
        reg("j", lambda v: v.orbit.orbit(-step, 0.0), "orbit left")
        reg("l", lambda v: v.orbit.orbit(+step, 0.0), "orbit right")
        reg("i", lambda v: v.orbit.orbit(0.0, +step), "orbit up")
        reg("k", lambda v: v.orbit.orbit(0.0, -step), "orbit down")
        reg(ti.ui.ESCAPE, lambda v: v.close(), "quit")

    def _key_zoom(self) -> float:
        return float(self._dc.get("key_zoom_factor", 1.15))

    def _toggle_all_forces(self, v: "TaichiViewer") -> None:
        on = not (v.options.show_gt_force or v.options.show_estimated_force)
        v.options.show_gt_force = on
        v.options.show_estimated_force = on
        v.options.show_extra_forces = on

    def step_amplification(self, direction: int) -> None:
        """Move to the next/previous entry of ``amplification_steps``."""
        cur = self.options.amplification
        steps = self.amp_steps
        if direction > 0:
            larger = [s for s in steps if s > cur * (1 + 1e-9)]
            self.options.amplification = larger[0] if larger else cur * 2.0
        else:
            smaller = [s for s in steps if s < cur * (1 - 1e-9)]
            self.options.amplification = smaller[-1] if smaller else max(cur / 2.0, 1e-6)

    def help_text(self) -> str:
        lines = ["Viewer controls:",
                 "  mouse: LMB orbit | RMB / Shift+LMB pan | MMB / Ctrl+LMB zoom"]
        for key, (_, txt) in self._key_callbacks.items():
            lines.append(f"  {key:>6} : {txt}")
        return "\n".join(lines)

    # ================================================================ loop
    def close(self) -> None:
        self.window.running = False

    @property
    def running(self) -> bool:
        return bool(self.window.running)

    def run(self, state: VisualizationState | None = None, on_frame: Callable[["TaichiViewer"], None] | None = None) -> None:
        """Main loop. ``on_frame`` is called once per frame before rendering."""
        if state is not None:
            self.set_state(state)
        print(self.help_text())
        while self.window.running:
            if on_frame is not None:
                on_frame(self)
            self.step()

    def step(self) -> None:
        """Process input, render one frame and present it."""
        self._handle_events()
        self._handle_mouse()
        self._render()
        self._draw_gui()
        self.window.show()
        self._frame += 1
        if self._frame % 30 == 0:
            now = time.time()
            self._fps = 30.0 / max(now - self._fps_t0, 1e-9)
            self._fps_t0 = now

    # ---------------------------------------------------------------- input
    def _handle_events(self) -> None:
        for ev in self.window.get_events(ti.ui.PRESS):
            cb = self._key_callbacks.get(ev.key)
            if cb is not None:
                cb[0](self)

    def _cursor_in_panel(self, x: float, y: float) -> bool:
        px, py, pw, ph = self._panel_rect
        return self.options.show_gui_panel and (px <= x <= px + pw) and (py <= y <= py + ph)

    def _handle_mouse(self) -> None:
        x, y = self.window.get_cursor_pos()
        lmb = self.window.is_pressed(ti.ui.LMB)
        rmb = self.window.is_pressed(ti.ui.RMB)
        mmb = self.window.is_pressed(ti.ui.MMB)
        shift = self.window.is_pressed(ti.ui.SHIFT)
        ctrl = self.window.is_pressed(ti.ui.CTRL)
        if not (lmb or rmb or mmb):
            self._last_cursor = None
            return
        if self._last_cursor is None:
            # Ignore drags that start on the GUI panel.
            self._last_cursor = None if self._cursor_in_panel(x, y) else (x, y)
            return
        dx = x - self._last_cursor[0]
        dy = y - self._last_cursor[1]
        self._last_cursor = (x, y)
        dc = self._dc
        if mmb or (lmb and ctrl):
            self.orbit.zoom(float(np.exp(-dy * float(dc.get("zoom_sensitivity", 3.0)))))
        elif rmb or (lmb and shift):
            k = float(dc.get("pan_sensitivity", 1.2)) * self.orbit.distance
            self.orbit.pan(-dx * k, -dy * k)
        elif lmb:
            k = float(dc.get("orbit_sensitivity", 220.0))
            self.orbit.orbit(-dx * k, -dy * k)

    # ---------------------------------------------------------------- draw
    def _render(self) -> None:
        diag = self.renderer.diag
        self.orbit.apply_to_taichi(
            self.ti_camera,
            z_near=float(self._dc.get("z_near_rel", 0.01)) * diag,
            z_far=float(self._dc.get("z_far_rel", 50.0)) * diag,
        )
        if self._overlay_field is not None:
            self.canvas.set_image(self._overlay_field)  # PIP inset; the 3D scene is drawn on top
        self.scene.set_camera(self.ti_camera)
        self.renderer.render(self.scene, self.options)
        self.canvas.scene(self.scene)

    def _draw_gui(self) -> None:
        if not self.options.show_gui_panel:
            self._panel_rect = (0.0, 0.0, 0.0, 0.0)
            return
        o = self.options
        lines = self.renderer.summary_lines(o)
        n_rows = len(lines) + 20
        h = min(0.97, 0.026 * n_rows)
        self._panel_rect = (0.005, 0.005, 0.30, h)
        gui = self.window.GUI
        with gui.sub_window("Viewer", *self._panel_rect) as g:
            g.text(f"backend: {self.backend}   fps: {self._fps:.0f}")
            for ln in lines:
                g.text(ln)
            g.text("")
            g.text("mode (1/2/3):")
            for m in MODES:
                if g.checkbox(m, o.mode == m) and o.mode != m:
                    o.mode = m
            o.amplification = float(g.slider_float("alpha (visual x)", o.amplification, 1.0, 100.0))
            o.show_surface = g.checkbox("surface", o.show_surface)
            o.show_wireframe = g.checkbox("wireframe (W)", o.show_wireframe)
            o.show_tet_edges = g.checkbox("tet edges (T)", o.show_tet_edges)
            o.show_nodes = g.checkbox("nodes (N)", o.show_nodes)
            o.show_fixed_nodes = g.checkbox("fixed nodes (B)", o.show_fixed_nodes)
            o.show_contact = g.checkbox("contact (C)", o.show_contact)
            o.show_gt_force = g.checkbox("GT force (G)", o.show_gt_force)
            o.show_estimated_force = g.checkbox("estimated force (E)", o.show_estimated_force)
            o.show_objects = g.checkbox("objects (O)", o.show_objects)
            o.show_observation_camera = g.checkbox("observation camera (V)", o.show_observation_camera)
            if self.state is not None and self.state.keypoints is not None:
                o.show_keypoints = g.checkbox("markers (;)", o.show_keypoints)
            if self.state is not None and self.state.displacement_alt is not None:
                o.use_alt_displacement = g.checkbox(f"show {self.state.displacement_alt_name} (A)", o.use_alt_displacement)
            g.text(f"color mode (M): {o.color_mode}")
            if g.button("reset camera (R)"):
                self.orbit.reset()
            if g.button("screenshot (S)"):
                self.save_screenshot()

    # ---------------------------------------------------------------- misc
    def save_screenshot(self, path: str | Path | None = None) -> Path:
        """Render the current frame to a PNG (works head-less)."""
        if path is None:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            path = self.screenshot_dir / f"viewer_{time.strftime('%Y%m%d_%H%M%S')}_{self._frame:06d}.png"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._render()
        self.window.save_image(str(path))
        print(f"[viewer] screenshot saved: {path}")
        return path

    def render_offscreen(self, n_warmup_frames: int = 1) -> np.ndarray:
        """Render and return the frame as an (H, W, 3) float32 RGB array."""
        for _ in range(max(n_warmup_frames, 1)):
            self._render()
        img = self.window.get_image_buffer_as_numpy()
        img = np.asarray(img, dtype=np.float32)
        img = np.transpose(img[..., :3], (1, 0, 2))[::-1]  # (W,H,C) -> (H,W,C), flip y
        return img

    def destroy(self) -> None:
        self.window.destroy()
