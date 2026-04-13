"""Orbit camera controller for the Taichi GGUI viewer.

Extends Taichi's built-in camera with configurable presets,
smooth transitions, and trackball orbit controls.

Usage::

    camera = OrbitCamera(target=[0, 0, 0.5], distance=2.5)
    camera.apply(viewer._camera)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraPreset:
    """Named camera position."""

    name: str
    azimuth: float    # degrees, 0 = +X, 90 = +Y
    elevation: float  # degrees, 0 = horizontal, 90 = top-down
    distance: float
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))


# Common presets
PRESET_FRONT = CameraPreset("Front", azimuth=90, elevation=20, distance=2.5)
PRESET_SIDE = CameraPreset("Side", azimuth=0, elevation=20, distance=2.5)
PRESET_TOP = CameraPreset("Top", azimuth=90, elevation=85, distance=3.0)
PRESET_ISO = CameraPreset("Isometric", azimuth=45, elevation=35, distance=2.5)
PRESET_CLOSE = CameraPreset("Close-up", azimuth=45, elevation=25, distance=1.2,
                             target=np.array([0, 0, 0.3]))


class OrbitCamera:
    """Orbit camera with azimuth/elevation/distance control.

    The camera orbits around a target point on a sphere defined by
    (azimuth, elevation, distance).
    """

    def __init__(
        self,
        target: np.ndarray | None = None,
        azimuth: float = 45.0,
        elevation: float = 30.0,
        distance: float = 2.5,
        up: np.ndarray | None = None,
    ):
        self.target = target if target is not None else np.array([0.0, 0.0, 0.4])
        self.azimuth = azimuth    # degrees
        self.elevation = elevation  # degrees
        self.distance = distance
        self.up = up if up is not None else np.array([0.0, 0.0, 1.0])

        # Limits
        self.min_distance = 0.3
        self.max_distance = 10.0
        self.min_elevation = -85.0
        self.max_elevation = 89.0

        # Smooth interpolation
        self._target_az = azimuth
        self._target_el = elevation
        self._target_dist = distance
        self._target_pos = self.target.copy()
        self._smooth_factor = 0.15  # 0 = instant, 1 = no movement

    @property
    def position(self) -> np.ndarray:
        """Camera position in world coordinates."""
        az = np.radians(self.azimuth)
        el = np.radians(self.elevation)
        r = self.distance

        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        return self.target + np.array([x, y, z])

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        """Rotate the camera around the target."""
        self._target_az += d_azimuth
        self._target_el = np.clip(
            self._target_el + d_elevation,
            self.min_elevation, self.max_elevation,
        )

    def zoom(self, factor: float) -> None:
        """Zoom in/out. factor > 1 zooms out, < 1 zooms in."""
        self._target_dist = np.clip(
            self._target_dist * factor,
            self.min_distance, self.max_distance,
        )

    def pan(self, dx: float, dy: float) -> None:
        """Pan the camera target in screen-space directions."""
        az = np.radians(self.azimuth)
        # Screen-right in world
        right = np.array([-np.sin(az), np.cos(az), 0.0])
        up = self.up
        self._target_pos += dx * right * self.distance * 0.001
        self._target_pos += dy * up * self.distance * 0.001

    def set_preset(self, preset: CameraPreset, instant: bool = False) -> None:
        """Switch to a named preset."""
        self._target_az = preset.azimuth
        self._target_el = preset.elevation
        self._target_dist = preset.distance
        self._target_pos = preset.target.copy()

        if instant:
            self.azimuth = self._target_az
            self.elevation = self._target_el
            self.distance = self._target_dist
            self.target = self._target_pos.copy()

    def update(self, smooth: bool = True) -> None:
        """Smooth-interpolate towards target state.

        Call once per frame before apply().
        """
        if smooth and self._smooth_factor > 0:
            alpha = 1.0 - self._smooth_factor
            self.azimuth += alpha * (self._target_az - self.azimuth)
            self.elevation += alpha * (self._target_el - self.elevation)
            self.distance += alpha * (self._target_dist - self.distance)
            self.target += alpha * (self._target_pos - self.target)
        else:
            self.azimuth = self._target_az
            self.elevation = self._target_el
            self.distance = self._target_dist
            self.target = self._target_pos.copy()

    def apply(self, ti_camera) -> None:
        """Apply current state to a Taichi camera object.

        Parameters
        ----------
        ti_camera : ti.ui.Camera instance
        """
        pos = self.position
        ti_camera.position(*pos)
        ti_camera.lookat(*self.target)
        ti_camera.up(*self.up)

    def look_at(self, target: np.ndarray) -> None:
        """Smoothly move the camera to look at a new point."""
        self._target_pos = target.copy()

    def set_distance(self, distance: float) -> None:
        """Set orbit distance."""
        self._target_dist = np.clip(distance, self.min_distance, self.max_distance)

    def info(self) -> str:
        """Return camera state as a string."""
        pos = self.position
        return (
            f"Camera: az={self.azimuth:.1f} el={self.elevation:.1f} "
            f"dist={self.distance:.2f} "
            f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
            f"target=({self.target[0]:.2f}, {self.target[1]:.2f}, {self.target[2]:.2f})"
        )
