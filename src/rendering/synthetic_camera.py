"""Synthetic observation camera: renders what the *research* camera sees.

Strictly separate from the developer (orbit) camera of the Taichi viewer
(see README / brief §15): this renderer owns a hidden GGUI window at the
sensor resolution, sets the GGUI camera from an :class:`ObservationCamera`
(same position / look-at / vertical FOV, principal point at the image centre)
and returns the frame as a NumPy image. Consequently the pinhole model
``ObservationCamera.project`` and the rendered pixels agree to sub-pixel
accuracy – verified by ``scripts/run_synthetic_camera_check.py`` and
``tests/test_rendering.py``.

Pixel convention: continuous image coordinates ``(u, v)`` with ``u`` to the
right, ``v`` down, origin at the top-left corner of the top-left pixel; pixel
``(row j, col i)`` covers ``u ∈ [i, i+1)``, ``v ∈ [j, j+1)``. The centre of
that pixel is therefore ``(i + 0.5, j + 0.5)``.

Rendering is deliberately simple (flat colours, Lambert-ish GGUI shading, no
textures): the goal is a *controlled* vision problem with adjustable
nuisance factors – background colour, light position, image noise – not
photorealism. Photorealistic rendering can replace this class later without
touching the dataset format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..visualization.camera import ObservationCamera
from ..visualization.state import ObjectGeometry


@dataclass
class RenderAppearance:
    """Nuisance parameters of one rendering (all colours RGB in [0, 1])."""

    finger_color: tuple[float, float, float] = (0.75, 0.75, 0.78)
    background_color: tuple[float, float, float] = (0.10, 0.11, 0.13)
    ambient: tuple[float, float, float] = (0.35, 0.35, 0.35)
    light_position: np.ndarray | None = None  # world [m]; None -> behind the camera
    light_color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    marker_color: tuple[float, float, float] = (1.0, 0.2, 0.1)
    marker_radius: float = 0.0012  # [m]
    marker_style: str = "disc"  # disc (flat dot on the surface, like a printed marker) | sphere
    object_color: tuple[float, float, float] = (0.30, 0.75, 0.85)
    image_noise_std: float = 0.0  # Gaussian noise on [0, 1] intensities
    grayscale: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


class SyntheticCameraRenderer:
    """Off-screen GGUI renderer for :class:`ObservationCamera` views.

    One instance per resolution; call :meth:`render` for every sample. The
    finger topology is uploaded once with :meth:`set_mesh`.
    """

    def __init__(self, image_width: int, image_height: int, backend: str = "auto") -> None:
        import taichi as ti

        from ..visualization.mesh_renderer import IndexedMeshBuffers, PointBuffers, TriangleSoupBuffers
        from ..visualization.taichi_viewer import init_taichi

        self.backend = init_taichi(backend)
        self.width, self.height = int(image_width), int(image_height)
        self.window = ti.ui.Window("synthetic camera", (self.width, self.height), show_window=False, vsync=False)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.ti_camera = ti.ui.Camera()
        self._mesh: IndexedMeshBuffers | None = None
        # fixed_draw: constant draw sizes every frame, see TriangleSoupBuffers.
        self._points = PointBuffers(64, fixed_draw=True)
        self._discs = TriangleSoupBuffers(64 * 16, fixed_draw=True)
        self._objects = TriangleSoupBuffers(2048, fixed_draw=True)
        self._IndexedMeshBuffers = IndexedMeshBuffers
        self._n_nodes = 0

    # ------------------------------------------------------------ setup
    def set_mesh(self, n_nodes: int, surface_faces: np.ndarray) -> None:
        self._mesh = self._IndexedMeshBuffers(n_nodes, surface_faces, position_sets=("display",))
        self._n_nodes = int(n_nodes)

    def _apply_camera(self, camera: ObservationCamera, scene_diag: float) -> None:
        if (camera.image_width, camera.image_height) != (self.width, self.height):
            raise ValueError(f"camera resolution {camera.image_width}x{camera.image_height} != renderer {self.width}x{self.height}")
        c = self.ti_camera
        c.position(*map(float, camera.position))
        c.lookat(*map(float, camera.target))
        R = camera.rotation_world_from_camera
        up = -R[:, 1]  # camera -y is up
        c.up(float(up[0]), float(up[1]), float(up[2]))
        c.fov(float(camera.fov_y_deg))
        c.z_near(float(max(1e-3 * scene_diag, 1e-4)))
        c.z_far(float(50.0 * scene_diag + np.linalg.norm(camera.position - camera.target)))

    # ------------------------------------------------------------ render
    def render(
        self,
        nodes_deformed: np.ndarray,
        camera: ObservationCamera,
        appearance: RenderAppearance | None = None,
        markers_xyz: np.ndarray | None = None,
        objects: list[ObjectGeometry] | None = None,
        rng: np.random.Generator | None = None,
        marker_normals: np.ndarray | None = None,
        vertex_colors: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render one frame -> (H, W, 3) float32 RGB in [0, 1] (or (H, W, 1) if grayscale).

        ``markers_xyz`` are drawn as flat discs of radius ``appearance.marker_radius``
        lying on the surface (needs ``marker_normals``) or as spheres.
        ``vertex_colors`` optional (N, 3) RGB in [0, 1] — displacement colormap etc.
        """
        if self._mesh is None:
            raise RuntimeError("call set_mesh() first")
        ap = appearance or RenderAppearance()
        nodes = np.asarray(nodes_deformed, dtype=float).reshape(-1, 3)
        if nodes.shape[0] != self._n_nodes:
            raise ValueError("node count does not match set_mesh()")
        diag = float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0)))

        self._mesh.set_positions("display", nodes)
        self._apply_camera(camera, diag)
        self.canvas.set_background_color(tuple(float(x) for x in ap.background_color))
        scene = self.scene
        scene.set_camera(self.ti_camera)
        scene.ambient_light(tuple(float(x) for x in ap.ambient))
        lp = ap.light_position
        if lp is None:
            lp = camera.position + 0.5 * diag * (-camera.rotation_world_from_camera[:, 1])  # above the camera
        scene.point_light(pos=tuple(float(x) for x in lp), color=tuple(float(x) for x in ap.light_color))

        # The draw-call sequence and the submitted vertex counts must be
        # identical every frame (finger, discs, points, objects): GGUI caches
        # renderables per slot and re-creates their GPU buffers when a slot's
        # size changes, which blanks the finger in all following frames on
        # macOS/MoltenVK. Layers are therefore always drawn at full capacity
        # (``fixed_draw``), with empty layers parked outside the frustum.
        if vertex_colors is not None:
            vc = np.asarray(vertex_colors, np.float32).reshape(-1, 3)
            if vc.shape[0] != self._n_nodes:
                raise ValueError("vertex_colors length does not match set_mesh()")
            self._mesh.set_colors(vc)
            self._mesh.draw_surface(scene, "display", per_vertex_color=True)
        else:
            self._mesh.draw_surface(scene, "display", color=ap.finger_color)
        self._discs.clear()
        self._points.clear()
        self._objects.clear()
        if markers_xyz is not None and len(markers_xyz):
            if ap.marker_style == "disc" and marker_normals is not None:
                v = disc_triangles(np.asarray(markers_xyz, float), np.asarray(marker_normals, float), ap.marker_radius)
                self._discs.set(v, np.tile(np.asarray(ap.marker_color, np.float32), (len(v), 1)))
            else:
                self._points.set(np.asarray(markers_xyz, float), color=ap.marker_color)
        if objects:
            verts, cols = [], []
            for obj in objects:
                v = obj.vertices[obj.faces].reshape(-1, 3)
                verts.append(v)
                cols.append(np.tile(np.asarray(obj.color if obj.color is not None else ap.object_color, np.float32), (len(v), 1)))
            self._objects.set(np.concatenate(verts), np.concatenate(cols))
        self._discs.draw(scene, two_sided=True)
        self._points.draw(scene, ap.marker_radius)
        self._objects.draw(scene)
        self.canvas.scene(scene)

        img = self.window.get_image_buffer_as_numpy()  # (W, H, 4), origin bottom-left
        img = np.transpose(np.asarray(img, dtype=np.float32)[..., :3], (1, 0, 2))[::-1]  # -> (H, W, 3), row 0 = top
        img = np.ascontiguousarray(img)
        if ap.grayscale:
            img = (img @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
        if ap.image_noise_std > 0.0:
            rng = rng or np.random.default_rng()
            img = np.clip(img + rng.normal(0.0, ap.image_noise_std, size=img.shape).astype(np.float32), 0.0, 1.0)
        return img

    def destroy(self) -> None:
        self.window.destroy()


def disc_triangles(centers: np.ndarray, normals: np.ndarray, radius: float, n_seg: int = 16, lift_rel: float = 0.05) -> np.ndarray:
    """Triangle soup (3·M·n_seg, 3) of flat discs centred at ``centers`` ⊥ ``normals``.

    Discs are lifted by ``lift_rel · radius`` along the normal to avoid z-fighting
    with the surface they lie on; the lift is far below pixel resolution.
    """
    c = centers.reshape(-1, 3)
    n = normals.reshape(-1, 3)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    a = np.where(np.abs(n[:, [0]]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    ang = np.linspace(0.0, 2 * np.pi, n_seg + 1)
    rim = c[:, None, :] + lift_rel * radius * n[:, None, :] + radius * (np.cos(ang)[None, :, None] * t1[:, None, :] + np.sin(ang)[None, :, None] * t2[:, None, :])
    ctr = c[:, None, :] + lift_rel * radius * n[:, None, :]
    tri = np.stack([np.repeat(ctr, n_seg, axis=1), rim[:, :-1], rim[:, 1:]], axis=2)  # (M, n_seg, 3, 3)
    return tri.reshape(-1, 3)


# ---------------------------------------------------------------- camera sampling
def nominal_camera(geometry, cfg: Mapping[str, Any], image_width: int, image_height: int, name: str = "obs_cam") -> ObservationCamera:
    """Camera from ``configs/vision.yaml: camera`` (pose relative to the finger)."""
    L = geometry.length
    target = geometry.point_from_relative(cfg.get("target_rel", [0.55, 0.5, 0.5]))
    pos = target + L * np.asarray(cfg.get("position_offset_rel", [0.05, -1.1, 0.55]), float)
    return ObservationCamera(position=pos, target=target, up=np.array(cfg.get("up", [0.0, 0.0, 1.0]), float),
                             fov_y_deg=float(cfg.get("fov_y_deg", 40.0)), image_width=image_width, image_height=image_height,
                             near=0.1 * L, far=1.5 * L, name=name)


def camera_ring(
    geometry,
    cfg: Mapping[str, Any],
    n_views: int,
    image_width: int,
    image_height: int,
) -> list[ObservationCamera]:
    """``n_views`` calibrated cameras equally spaced in azimuth about the look-at.

    Elevation is that of ``nominal_camera`` so the top face stays in view.
    Pose is exact (no jitter): this is the SNR / view-count experiment, not
    Test D camera-pose noise.
    """
    n_views = int(n_views)
    if n_views < 1:
        raise ValueError("n_views must be >= 1")
    nom = nominal_camera(geometry, cfg, image_width, image_height)
    offset = nom.position - nom.target
    up = nom.up / max(float(np.linalg.norm(nom.up)), 1e-15)
    cams: list[ObservationCamera] = []
    for i in range(n_views):
        ang = 2.0 * np.pi * i / n_views
        c, s = float(np.cos(ang)), float(np.sin(ang))
        v_rot = offset * c + np.cross(up, offset) * s + up * float(np.dot(up, offset)) * (1.0 - c)
        cams.append(ObservationCamera(
            position=nom.target + v_rot, target=nom.target, up=nom.up,
            fov_y_deg=nom.fov_y_deg, image_width=image_width, image_height=image_height,
            near=nom.near, far=nom.far, name=f"snr_{i}",
        ))
    return cams


def jitter_camera(cam: ObservationCamera, geometry, jitter: Mapping[str, Any], rng: np.random.Generator, name: str | None = None) -> ObservationCamera:
    """Random perturbation of pose and FOV (uniform in ±amplitude; lengths in units of L)."""
    L = geometry.length
    dp = L * float(jitter.get("position_rel", 0.0)) * rng.uniform(-1, 1, 3)
    dt = L * float(jitter.get("target_rel", 0.0)) * rng.uniform(-1, 1, 3)
    df = float(jitter.get("fov_deg", 0.0)) * rng.uniform(-1, 1)
    return ObservationCamera(position=cam.position + dp, target=cam.target + dt, up=cam.up, fov_y_deg=cam.fov_y_deg + df,
                             image_width=cam.image_width, image_height=cam.image_height, near=cam.near, far=cam.far,
                             name=name or cam.name)
