"""Save viewer frames + solid-red CV deflection for grasp validation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from robosim.scene.finger_probe import elastic_force_from_centerline
from robosim.viz.finger_cv import GradientDeflectionEstimator

# Palm-mounted “antenna” camera (offsets in palm_link frame).
# Palm +X = finger length, +Y = left/right, +Z = out of palm face (up).
_PALM_CAM_HEIGHT_Z = 0.42          # camera sits this far above palm along +Z [m]
_PALM_CAM_EYE_X = 0.10             # shift eye toward finger mid-span [m]
_PALM_CAM_LOOK_X = 0.105           # look at mid-finger (tip ≈0.025 + 0.080) [m]
_PALM_LINK = "palm_link"


class FingerCaptureSession:
    """Capture PNG frames from the Taichi viewer and estimate CV deflection."""

    def __init__(
        self,
        out_dir: Path | str,
        *,
        every: int = 5,
        clean_hud: bool = True,
        cam_height_z: float = _PALM_CAM_HEIGHT_Z,
        cam_eye_x: float = _PALM_CAM_EYE_X,
        cam_look_x: float = _PALM_CAM_LOOK_X,
    ):
        self.out_dir = Path(out_dir)
        self.every = max(1, every)
        self.clean_hud = clean_hud
        self.cam_height_z = cam_height_z
        self.cam_eye_x = cam_eye_x
        self.cam_look_x = cam_look_x
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._est = GradientDeflectionEstimator()
        self._finger_bodies: dict = {}
        self._frame_idx = 0
        self._close_captures = 0
        self._near_ref_img: np.ndarray | None = None
        self._latest = {
            "left_deflection_cv_mm": 0.0,
            "right_deflection_cv_mm": 0.0,
            "left_fn_cv_n": 0.0,
            "right_fn_cv_n": 0.0,
        }
        self._manifest_rows: list[dict] = []
        self._profile_rows: list[dict] = []

    def set_finger_bodies(self, bodies: dict) -> None:
        """Map link name → CraigBamptonBody for CV u(s) → K_r force."""
        self._finger_bodies = dict(bodies)

    @property
    def latest(self) -> dict[str, float]:
        return dict(self._latest)

    def lock_viewer_camera(self, viewer) -> None:
        """Disable orbit controls; pose is updated each frame from palm FK."""
        viewer._camera_locked = True

    def update_palm_camera(self, viewer, scene) -> bool:
        """Place the camera above ``palm_link``, looking down at both fingers.

        Returns True if a palm pose was found and applied.
        """
        for rh in scene._robot_handles.values():
            model = rh._model
            try:
                palm_idx = model.link_index(_PALM_LINK)
            except Exception:
                continue
            fk = model.forward_kinematics()
            T = fk[palm_idx]
            R = T.rotation
            p = T.translation

            # Eye: antenna above palm, shifted slightly over the finger span.
            eye_local = np.array([self.cam_eye_x, 0.0, self.cam_height_z])
            look_local = np.array([self.cam_look_x, 0.0, 0.0])
            eye = p + R @ eye_local
            lookat = p + R @ look_local
            # Image "up" = palm +X so fingers run vertically and both left/right show.
            up = R @ np.array([1.0, 0.0, 0.0])
            # If palm +X is nearly parallel to view direction, fall back to world Z.
            view = lookat - eye
            view_n = np.linalg.norm(view)
            if view_n > 1e-9 and abs(float(np.dot(up, view / view_n))) > 0.95:
                up = np.array([0.0, 0.0, 1.0])
            viewer.set_world_camera(eye, lookat, up, lock=True)
            return True
        return False

    def process_after_render(
        self,
        viewer,
        t: float,
        phase: str | None,
        sim_probe: dict,
    ) -> None:
        """Grab framebuffer, optionally save PNG, update CV estimates."""
        if self._frame_idx % self.every != 0:
            self._frame_idx += 1
            return

        img = viewer.get_framebuffer()
        rel = self.out_dir / f"frame_{self._frame_idx:05d}_t{t:.3f}.png"
        rgb = (np.clip(img[..., :3], 0.0, 1.0) * 255).astype(np.uint8)
        try:
            from PIL import Image
            Image.fromarray(rgb).save(rel)
            estimate_img = np.array(Image.open(rel))
        except ImportError:
            try:
                import imageio.v3 as iio
                iio.imwrite(rel, rgb)
                estimate_img = iio.imread(rel)
            except ImportError:
                viewer.save_screenshot(rel)
                estimate_img = img

        ph = phase or ""
        lf = sim_probe.get("left_finger", {})
        rf = sim_probe.get("right_finger", {})
        max_defl_m = max(
            abs(lf.get("deflection_m", 0.0)),
            abs(rf.get("deflection_m", 0.0)),
        )

        # Reference after fingers have mostly closed, before big elastic bend.
        if ph == "CLOSE":
            self._close_captures += 1
            if (
                not self._est.ready
                and self._close_captures >= 8
                and max_defl_m < 1e-3
            ):
                self._est.try_set_reference(estimate_img)
        else:
            self._close_captures = 0
            if (
                ph == "NEAR"
                and max_defl_m < 1e-5
                and not self._est.ready
            ):
                self._near_ref_img = np.array(estimate_img, copy=True)
            elif (
                self._near_ref_img is not None
                and not self._est.ready
                and ph in ("CLOSE", "LIFT")
            ):
                if self._est.try_set_reference(self._near_ref_img):
                    self._near_ref_img = None

        if self._est.ready:
            fields = self._est.estimate_fields(estimate_img)
            left_cv, right_cv = fields.tip_mm
        else:
            fields = None
            left_cv, right_cv = 0.0, 0.0

        # Force from full CV outer-edge field u(s) → CB K_r (not tip→rebuild).
        left_body = self._finger_bodies.get("left_finger")
        right_body = self._finger_bodies.get("right_finger")
        left_fn = 0.0
        right_fn = 0.0
        if fields is not None and fields.left is not None and left_body is not None:
            left_fn = elastic_force_from_centerline(
                left_body,
                "left_finger",
                fields.left.s,
                fields.left.u_mm * 1e-3,
                contact_s=fields.left.contact_s,
            )
        if fields is not None and fields.right is not None and right_body is not None:
            right_fn = elastic_force_from_centerline(
                right_body,
                "right_finger",
                fields.right.s,
                fields.right.u_mm * 1e-3,
                contact_s=fields.right.contact_s,
            )

        left_cs = (
            fields.left.contact_s
            if fields is not None and fields.left is not None else float("nan")
        )
        right_cs = (
            fields.right.contact_s
            if fields is not None and fields.right is not None else float("nan")
        )

        self._latest = {
            "left_deflection_cv_mm": left_cv,
            "right_deflection_cv_mm": right_cv,
            "left_fn_cv_n": left_fn,
            "right_fn_cv_n": right_fn,
            "left_contact_s": left_cs,
            "right_contact_s": right_cs,
        }

        self._manifest_rows.append({
            "frame": self._frame_idx,
            "t": t,
            "phase": ph,
            "file": rel.name,
            "left_deflection_sim_mm": lf.get("deflection_m", 0.0) * 1e3,
            "right_deflection_sim_mm": rf.get("deflection_m", 0.0) * 1e3,
            "left_deflection_cv_mm": left_cv,
            "right_deflection_cv_mm": right_cv,
            "left_fn_sim_n": lf.get("fn_n", 0.0),
            "right_fn_sim_n": rf.get("fn_n", 0.0),
            "left_fn_cv_n": left_fn,
            "right_fn_cv_n": right_fn,
            "left_contact_s": left_cs,
            "right_contact_s": right_cs,
            "cv_ref_ready": int(self._est.ready),
        })

        # Per-station deflection: sim nodal X vs CV centerline (same s→X map).
        for side, probe_key, field in (
            ("left", "left_finger", None if fields is None else fields.left),
            ("right", "right_finger", None if fields is None else fields.right),
        ):
            pr = sim_probe.get(probe_key, {})
            x_sim = pr.get("profile_x_m")
            u_sim = pr.get("profile_u_mm")
            if x_sim is None or u_sim is None or len(x_sim) == 0:
                continue
            x0, x1 = float(x_sim[0]), float(x_sim[-1])
            span = max(x1 - x0, 1e-9)
            if field is not None:
                x_cv = x0 + field.s * span
                u_cv = field.u_mm
            else:
                x_cv = u_cv = None
            for xv, uv in zip(x_sim, u_sim):
                self._profile_rows.append({
                    "frame": self._frame_idx,
                    "t": t,
                    "phase": ph,
                    "finger": side,
                    "source": "sim",
                    "s": float((xv - x0) / span),
                    "x_m": float(xv),
                    "u_mm": float(uv),
                })
            if x_cv is not None:
                for xv, uv in zip(x_cv, u_cv):
                    self._profile_rows.append({
                        "frame": self._frame_idx,
                        "t": t,
                        "phase": ph,
                        "finger": side,
                        "source": "cv",
                        "s": float((xv - x0) / span),
                        "x_m": float(xv),
                        "u_mm": float(uv),
                    })

        self._frame_idx += 1

    def write_manifest(self) -> Path:
        path = self.out_dir / "frames_manifest.csv"
        if not self._manifest_rows:
            path.write_text(
                "frame,t,phase,file,left_deflection_sim_mm,right_deflection_sim_mm,"
                "left_deflection_cv_mm,right_deflection_cv_mm,"
                "left_fn_sim_n,right_fn_sim_n,left_fn_cv_n,right_fn_cv_n,"
                "left_contact_s,right_contact_s,cv_ref_ready\n"
            )
            return path
        keys = list(self._manifest_rows[0].keys())
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._manifest_rows)
        return path

    def write_profiles(self) -> Path:
        """Write sim/CV deflection vs X stations (one row per sample)."""
        path = self.out_dir / "deflection_profiles.csv"
        keys = ["frame", "t", "phase", "finger", "source", "s", "x_m", "u_mm"]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._profile_rows)
        return path
