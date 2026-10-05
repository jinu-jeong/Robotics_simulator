"""Camera models for visualization.

Two logically separate cameras exist in this project:

* :class:`ObservationCamera` – the *research* camera. It has calibrated
  intrinsics/extrinsics, a fixed or controlled pose, and later produces the
  synthetic images consumed by the vision model. In the Taichi viewer it is
  drawn as a marker + frustum so the developer can see what it observes.

* :class:`OrbitCamera` – the *developer* camera used to look at the 3D scene
  in the Taichi viewer. Free orbit / pan / zoom around a target point. It has
  no research meaning and never appears in datasets.

Conventions
-----------
* World frame: right-handed, ``+z`` up (see ``configs/fem.yaml``).
* Camera frame (OpenCV convention): ``+x`` right, ``+y`` down, ``+z`` forward
  (viewing direction). ``R_wc`` maps camera coordinates to world coordinates.
* Angles in the dataclasses are stored in degrees; all lengths in meters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-15:
        raise ValueError("zero-length vector")
    return v / n


def look_at_rotation(position, target, up) -> np.ndarray:
    """Camera-to-world rotation ``R_wc`` (columns = camera axes in world).

    OpenCV convention: z_c = forward, x_c = right, y_c = down.
    """
    fwd = _unit(np.asarray(target, float) - np.asarray(position, float))
    up = np.asarray(up, dtype=float)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-9:  # forward parallel to up: pick another up
        up = np.array([1.0, 0.0, 0.0]) if abs(fwd[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
    right = _unit(right)
    down = np.cross(fwd, right)  # = -true_up, so that (right, down, fwd) is right-handed
    return np.stack([right, down, fwd], axis=1)


@dataclass
class ObservationCamera:
    """Pinhole observation camera (research camera).

    Attributes
    ----------
    position : (3,) camera center in world coordinates [m]
    target : (3,) point the camera looks at [m]
    up : (3,) approximate up vector in world coordinates
    fov_y_deg : vertical field of view [deg]
    image_width, image_height : sensor resolution [px]
    near, far : depth range used only for frustum drawing [m]
    name : label shown in the viewer
    """

    position: np.ndarray = field(default_factory=lambda: np.array([0.05, -0.15, 0.08]))
    target: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.01, 0.005]))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    fov_y_deg: float = 45.0
    image_width: int = 640
    image_height: int = 480
    near: float = 0.01
    far: float = 0.30
    name: str = "obs_cam"

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.target = np.asarray(self.target, dtype=float).reshape(3)
        self.up = np.asarray(self.up, dtype=float).reshape(3)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ObservationCamera":
        return cls(
            position=np.array(cfg["position"], dtype=float),
            target=np.array(cfg["target"], dtype=float),
            up=np.array(cfg.get("up", [0.0, 0.0, 1.0]), dtype=float),
            fov_y_deg=float(cfg.get("fov_y_deg", 45.0)),
            image_width=int(cfg.get("image_width", 640)),
            image_height=int(cfg.get("image_height", 480)),
            near=float(cfg.get("near", 0.01)),
            far=float(cfg.get("far", 0.3)),
            name=str(cfg.get("name", "obs_cam")),
        )

    # ------------------------------------------------------------ geometry
    @property
    def aspect(self) -> float:
        return self.image_width / self.image_height

    @property
    def forward(self) -> np.ndarray:
        return _unit(self.target - self.position)

    @property
    def rotation_world_from_camera(self) -> np.ndarray:
        return look_at_rotation(self.position, self.target, self.up)

    def intrinsics(self) -> np.ndarray:
        """3x3 pinhole matrix K (pixels), principal point at the image center."""
        fy = 0.5 * self.image_height / np.tan(np.deg2rad(self.fov_y_deg) / 2.0)
        fx = fy  # square pixels
        return np.array([[fx, 0.0, self.image_width / 2.0], [0.0, fy, self.image_height / 2.0], [0.0, 0.0, 1.0]])

    def extrinsics(self) -> np.ndarray:
        """4x4 world-to-camera transform ``T_cw``."""
        R_wc = self.rotation_world_from_camera
        T = np.eye(4)
        T[:3, :3] = R_wc.T
        T[:3, 3] = -R_wc.T @ self.position
        return T

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N,3) world points to (N,2) pixel coordinates and depths."""
        P = np.asarray(points_world, dtype=float).reshape(-1, 3)
        T = self.extrinsics()
        Pc = (T[:3, :3] @ P.T).T + T[:3, 3]
        K = self.intrinsics()
        z = Pc[:, 2]
        uv = (K @ Pc.T).T
        uv = uv[:, :2] / np.where(np.abs(z) < 1e-12, 1e-12, z)[:, None]
        return uv, z

    def frustum_corners(self, depth: float) -> np.ndarray:
        """Four world-space corners of the image plane at ``depth`` [m]."""
        h = depth * np.tan(np.deg2rad(self.fov_y_deg) / 2.0)
        w = h * self.aspect
        local = np.array([[-w, -h, depth], [w, -h, depth], [w, h, depth], [-w, h, depth]])  # camera frame
        return (self.rotation_world_from_camera @ local.T).T + self.position

    def frustum_lines(self, depth: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Line segments (vertices (V,3), edges (E,2)) of the view frustum.

        Includes: 4 rays from the camera center to the far plane corners, the
        far rectangle, the near rectangle, and an "up" tick on the far top
        edge so the image orientation is visible.
        """
        far = self.far if depth is None else depth
        near = min(self.near, 0.5 * far)
        cf = self.frustum_corners(far)
        cn = self.frustum_corners(near)
        top_mid = 0.5 * (cf[2] + cf[3])
        R = self.rotation_world_from_camera
        up_tick = top_mid - R[:, 1] * 0.15 * np.linalg.norm(cf[2] - cf[1])  # camera -y is up
        verts = np.concatenate([self.position[None, :], cf, cn, top_mid[None, :], up_tick[None, :]], axis=0)
        edges = [[0, 1], [0, 2], [0, 3], [0, 4]]
        edges += [[1, 2], [2, 3], [3, 4], [4, 1]]
        edges += [[5, 6], [6, 7], [7, 8], [8, 5]]
        edges += [[9, 10]]
        return verts, np.array(edges, dtype=np.int64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": self.position.tolist(),
            "target": self.target.tolist(),
            "up": self.up.tolist(),
            "fov_y_deg": self.fov_y_deg,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "near": self.near,
            "far": self.far,
        }


@dataclass
class OrbitCamera:
    """Developer camera: orbits around ``target`` with spherical coordinates.

    ``yaw`` rotates about the world ``+z`` axis (0 deg = looking from +x),
    ``pitch`` is the elevation above the xy-plane. Pure state + math; the
    Taichi viewer converts this to a ``ti.ui.Camera`` every frame.
    """

    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    distance: float = 0.3
    yaw_deg: float = -55.0
    pitch_deg: float = 22.0
    fov_deg: float = 40.0
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    min_distance: float = 1e-4
    max_distance: float = 1e4

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=float).reshape(3)
        self.up = _unit(self.up)
        self._home = (self.target.copy(), self.distance, self.yaw_deg, self.pitch_deg)

    # ------------------------------------------------------------- basis
    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (right, cam_up, view_dir) unit vectors in world frame."""
        yaw = np.deg2rad(self.yaw_deg)
        pitch = np.deg2rad(np.clip(self.pitch_deg, -89.0, 89.0))
        # Direction from target to camera, in a frame where `up` is z.
        offset_local = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        if np.allclose(self.up, [0, 0, 1]):
            offset = offset_local
        else:  # generic up vector: rotate the local frame
            z = self.up
            x = _unit(np.cross([0.0, 1.0, 0.0], z)) if abs(z[1]) < 0.9 else _unit(np.cross([1.0, 0.0, 0.0], z))
            y = np.cross(z, x)
            offset = offset_local[0] * x + offset_local[1] * y + offset_local[2] * z
        view_dir = -offset
        right = _unit(np.cross(view_dir, self.up))
        cam_up = _unit(np.cross(right, view_dir))
        return right, cam_up, view_dir

    @property
    def position(self) -> np.ndarray:
        _, _, view_dir = self._basis()
        return self.target - self.distance * view_dir

    @property
    def view_direction(self) -> np.ndarray:
        return self._basis()[2]

    # ---------------------------------------------------------- controls
    def orbit(self, d_yaw_deg: float, d_pitch_deg: float) -> None:
        self.yaw_deg = (self.yaw_deg + d_yaw_deg + 180.0) % 360.0 - 180.0
        self.pitch_deg = float(np.clip(self.pitch_deg + d_pitch_deg, -89.0, 89.0))

    def pan(self, dx: float, dy: float) -> None:
        """Move the target in the camera's image plane. ``dx, dy`` in meters."""
        right, cam_up, _ = self._basis()
        self.target = self.target + dx * right + dy * cam_up

    def zoom(self, factor: float) -> None:
        """Multiply the distance by ``factor`` (<1 zooms in)."""
        self.distance = float(np.clip(self.distance * factor, self.min_distance, self.max_distance))

    def fit(self, bbox_min, bbox_max, distance_rel: float = 2.4) -> None:
        """Center on a bounding box and set the distance from its diagonal."""
        bbox_min = np.asarray(bbox_min, dtype=float)
        bbox_max = np.asarray(bbox_max, dtype=float)
        self.target = 0.5 * (bbox_min + bbox_max)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        self.distance = max(distance_rel * diag, self.min_distance)
        self._home = (self.target.copy(), self.distance, self.yaw_deg, self.pitch_deg)

    def reset(self) -> None:
        self.target, self.distance, self.yaw_deg, self.pitch_deg = (
            self._home[0].copy(),
            self._home[1],
            self._home[2],
            self._home[3],
        )

    def apply_to_taichi(self, ti_camera, z_near: float, z_far: float) -> None:
        """Push the current pose into a ``ti.ui.Camera``."""
        p = self.position
        t = self.target
        _, cam_up, _ = self._basis()
        ti_camera.position(float(p[0]), float(p[1]), float(p[2]))
        ti_camera.lookat(float(t[0]), float(t[1]), float(t[2]))
        ti_camera.up(float(cam_up[0]), float(cam_up[1]), float(cam_up[2]))
        ti_camera.fov(float(self.fov_deg))
        ti_camera.z_near(float(z_near))
        ti_camera.z_far(float(z_far))
