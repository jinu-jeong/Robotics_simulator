"""Per-finger force measurement used by the grasp controller.

* ``gt``       – Stage A: the true contact force (oracle).
* ``inverse``  – displacement-space LS on the FEM field (known contact).
* ``markers``  – Stage B: project virtual markers through a camera *fixed
  in the finger frame* (pixel noise) → geometric ``q`` → Kalman → ROM force.
* ``world``    – Stage D: same pipeline, but the camera is *fixed in the
  world*. Marker positions are mapped by the known EE pose ``T_ee``
  (robot FK) before projection, so the image Jacobian tracks the moving
  gripper. Default ``q_filter`` is Kalman (reset each episode).
* ``nn``       – Stage E: the *same* world camera, but **no markers**. The
  grasp scene is rendered marker-free (child process, see ``nn_vision.py``)
  and a ResNet-18 (``scripts/train_image_to_q.py`` on
  ``generate_grasp_nn_dataset.py`` data) predicts ``q`` directly from RGB.
  Everything after ``q`` (Kalman, contact search, ROM force) is shared with
  ``world``.

With ``unknown_contact: true`` (default for vision modes), force comes from
residual search on the inner face given ``q̂`` — contact is *not* taken from
``configs/grasp.yaml`` once locked. Until then, ``prelock_known_force`` (default
true) reports ROM force at the config contact so the controller is not blind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contact.inverse_force import InverseForceSolver
from ..evaluation.unknown_contact import UnknownContactLocalizer
from ..rendering.markers import SurfaceMarkers, make_marker_grid
from ..rendering.synthetic_camera import nominal_camera
from ..rom.pod import compute_pod
from ..rom.reduced_mechanics import ReducedModel
from ..vision.features import image_jacobian, solve_q_from_jacobian
from ..vision.marker_model import GeometricMarkerModel
from ..vision.temporal import KalmanQ
from ..visualization.camera import ObservationCamera
from .mechanics import GraspMechanics
from .world import transform_vectors


def default_world_camera(cfg: dict | None = None) -> ObservationCamera:
    """Desk camera looking at the grasp workspace from −y / +z."""
    c = dict(cfg or {})
    return ObservationCamera(
        position=np.asarray(c.get("position", [0.26, -0.20, 0.15]), float),
        target=np.asarray(c.get("target", [0.28, -0.01, 0.02]), float),
        up=np.asarray(c.get("up", [0.0, 0.0, 1.0]), float),
        fov_y_deg=float(c.get("fov_y_deg", 42.0)),
        image_width=int(c.get("image_width", 320)),
        image_height=int(c.get("image_height", 240)),
        near=float(c.get("near", 0.06)),
        far=float(c.get("far", 0.60)),
        name=str(c.get("name", "world_cam")),
    )


def observation_marker_image(camera: ObservationCamera, uv: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Diagnostic (H, W, 3) image of projected markers (no second GGUI window)."""
    h, w = int(camera.image_height), int(camera.image_width)
    img = np.full((h, w, 3), 0.10, np.float32)
    uv = np.asarray(uv, float).reshape(-1, 2)
    visible = np.asarray(visible, bool).reshape(-1)
    r = 2
    for (u, v), ok in zip(uv, visible):
        i, j = int(round(u)), int(round(v))
        if i < 0 or j < 0 or i >= w or j >= h:
            continue
        color = (0.25, 0.95, 0.35) if ok else (0.90, 0.25, 0.20)
        img[max(0, j - r):j + r + 1, max(0, i - r):i + r + 1] = color
    return img


def _in_image(camera: ObservationCamera, uv: np.ndarray, z: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
    return (
        (z > 0)
        & (uv[:, 0] >= margin_px) & (uv[:, 0] <= camera.image_width - margin_px)
        & (uv[:, 1] >= margin_px) & (uv[:, 1] <= camera.image_height - margin_px)
    )


def _inner_face_pod_snapshots(mech_or_model, *, n_x: int = 6, n_z: int = 3) -> np.ndarray:
    """Displacement columns for POD spanning the grasp (inner) face.

    Accepts a :class:`GraspMechanics` or a bare ``FingerFEMModel`` (the latter
    lets the POD basis come from a *reference* material while the ROM uses the
    actual one, see ``pod_reference_material``).
    """
    model = getattr(mech_or_model, "model", mech_or_model)
    g = model.geometry
    cols = []
    for x in np.linspace(0.55, 0.95, n_x):
        for z in np.linspace(0.25, 0.75, n_z):
            p = g.point_from_relative([float(x), 1.0, float(z)])
            for F in (1.0, 2.5):
                res, _ = model.solve_normal_contact(p, float(F))
                cols.append(res.u.reshape(-1))
    return np.stack(cols, axis=1)


def reference_pod_model(fem_cfg: dict, material_override: dict):
    """FEM model with the same geometry / mesh but a reference (E, ν) for the POD basis."""
    from ..fem.finger_model import FingerFEMModel

    cfg = dict(fem_cfg)
    cfg["material"] = {**dict(fem_cfg.get("material", {})), **dict(material_override)}
    model = FingerFEMModel.from_config(cfg)
    model.restrict_contact_surface("side_pos_y")
    return model


@dataclass
class GraspEstimator:
    mode: str
    mech: GraspMechanics
    inverse: InverseForceSolver | None = None
    markers: SurfaceMarkers | None = None
    geo: GeometricMarkerModel | None = None
    camera: object = None
    world_camera: ObservationCamera | None = None
    pixel_noise: float = 0.0
    rng: np.random.Generator | None = None
    last_visible: np.ndarray | None = None
    last_uv: np.ndarray | None = None
    last_q: np.ndarray | None = None
    last_contact_local: np.ndarray | None = None
    last_contact_err_m: float | None = None
    q_filter: str = "kalman"
    kf: KalmanQ | None = None
    unknown_contact: bool = False
    localizer: UnknownContactLocalizer | None = None
    # Robust unknown-contact gate (pixel noise makes mid-force search unstable).
    q_min_norm: float = 1.0e-2
    max_residual_rel: float = 0.22
    lock_residual_rel: float = 0.18
    lock_min_force: float = 1.0
    force_clip: float = 8.0
    lock_votes: int = 5
    lock_radius_m: float = 1.5e-2
    q_lock_norm: float = 1.2e-2
    vote_clear_bad: int = 4
    lock_pick: str = "last"  # which vote becomes the locked contact: last | medoid
    # Until residual-search locks a contact, use ROM force at config contact.
    prelock_known_force: bool = True
    # Self-weight: the observed q contains the known sag Φᵀu_g(R); subtract it
    # (model-based gravity compensation) before force / contact search.
    gravity_compensation: bool = True
    # Stage E (mode "nn"): marker-free world-camera frames → ResNet-18 → q.
    nn_cfg: dict | None = None
    nn: object | None = None  # nn_vision.NNStateEstimator, created lazily
    nn_every: int = 3  # camera tick every k control steps (100 Hz / 3 ≈ 33 Hz)
    scene_object_center: np.ndarray | None = None
    scene_object_R: np.ndarray | None = None
    last_image: np.ndarray | None = None
    _nn_step: int = 0
    _last_force: float = 0.0
    _locked_contact: object | None = None
    _contact_votes: list | None = None
    _bad_vote_streak: int = 0
    # Set by the controller: contact-lock votes are only collected once the
    # preload is reached, so the lock does not depend on the force ramp rate.
    lock_votes_open: bool = True

    @classmethod
    def build(cls, mech: GraspMechanics, cfg: dict, seed: int = 0) -> "GraspEstimator":
        mode = str(cfg.get("mode", "gt")).lower()
        rng = np.random.default_rng(seed)
        inv = InverseForceSolver(mech.model.fem, mech.model.contact)
        markers = geo = cam = world_cam = None
        localizer = None
        want_vision = mode in ("markers", "world", "nn") or bool(cfg.get("prepare_vision", False))
        unknown = bool(cfg.get("unknown_contact", want_vision))
        if want_vision:
            mk_face = str(cfg.get("marker_face", "top"))
            markers = make_marker_grid(
                mech.model.mesh, mech.model.geometry, mk_face,
                nx=int(cfg.get("marker_nx", 8)), ny=int(cfg.get("marker_ny", 3)),
                margin_rel=0.08,
            )
            if unknown:
                # Φ from a reference material (fixed q convention across E, ν); K from the actual one.
                pod_src = mech
                ref_mat = cfg.get("pod_reference_material")
                if ref_mat and cfg.get("_fem_cfg") is not None:
                    pod_src = reference_pod_model(cfg["_fem_cfg"], ref_mat)
                U = _inner_face_pod_snapshots(
                    pod_src,
                    n_x=int(cfg.get("pod_n_x", 6)),
                    n_z=int(cfg.get("pod_n_z", 3)),
                )
            else:
                cols = []
                for F in (0.4, 1.2, 2.4, 3.6):
                    res, _ = mech.model.solve_normal_contact(mech.contact_local, F)
                    cols.append(res.u.reshape(-1))
                U = np.stack(cols, axis=1)
            r_keep = int(cfg.get("q_modes", 8))
            basis = compute_pod(U)
            basis = basis.truncate(min(r_keep, basis.r))
            rom = ReducedModel.from_fem(mech.model.fem, basis, mech.model.contact)
            geo = GeometricMarkerModel(basis, markers, mech.model.mesh, ridge=1e-4, use_noisy=False)
            geo._rom = rom  # type: ignore[attr-defined]
            if unknown:
                face_ids = mech.model.surface_face_ids("side_pos_y")
                uc = cfg.get("unknown_contact_cfg") or {}
                bary = uc.get("barycentric") or [
                    [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1 / 3, 1 / 3, 1 / 3],
                    [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
                ]
                localizer = UnknownContactLocalizer.from_rom(
                    rom, face_ids,
                    barycentric=bary,
                    face_stride=int(uc.get("face_stride", 2)),
                    min_influence=float(uc.get("min_influence", 1e-8)),
                    nonneg=True,
                    r_loc=int(uc.get("r_loc", 4)),
                    refine=bool(uc.get("refine", True)),
                    refine_n=int(uc.get("refine_n", 7)),
                    weight=str(uc.get("weight", "leading")),
                )
            cam = nominal_camera(
                mech.model.geometry, cfg.get("camera", {}),
                int(cfg.get("camera", {}).get("image_width", 320)),
                int(cfg.get("camera", {}).get("image_height", 240)),
            )
            world_cam = default_world_camera(cfg.get("world_camera", {}))
        q_filter = str(cfg.get("q_filter", "kalman")).lower()
        kf = None
        if geo is not None and q_filter == "kalman":
            kf = KalmanQ.create(
                geo.r,
                process=float(cfg.get("kalman_process", 3e-6)),
                measure=float(cfg.get("kalman_measure", 1e-4)),
            )
        return cls(
            mode, mech, inverse=inv, markers=markers, geo=geo, camera=cam,
            world_camera=world_cam, pixel_noise=float(cfg.get("pixel_noise_std", 0.0)), rng=rng,
            q_filter=q_filter, kf=kf, unknown_contact=bool(unknown and localizer is not None),
            localizer=localizer,
            q_min_norm=float(cfg.get("q_min_norm", 8.0e-3)),
            max_residual_rel=float(cfg.get("max_residual_rel", 0.32)),
            lock_residual_rel=float(cfg.get("lock_residual_rel", 0.28)),
            lock_min_force=float(cfg.get("lock_min_force", 0.6)),
            force_clip=float(cfg.get("force_clip", 8.0)),
            lock_votes=int(cfg.get("lock_votes", 5)),
            lock_radius_m=float(cfg.get("lock_radius_m", 1.5e-2)),
            q_lock_norm=float(cfg.get("q_lock_norm", 1.2e-2)),
            vote_clear_bad=int(cfg.get("vote_clear_bad", 4)),
            lock_pick=str(cfg.get("lock_pick", "last")),
            prelock_known_force=bool(cfg.get("prelock_known_force", True)),
            gravity_compensation=bool(cfg.get("gravity_compensation", True)),
            nn_cfg=dict(cfg.get("nn") or {}),
            nn_every=max(1, int((cfg.get("nn") or {}).get("every", 3))),
        )

    def reset_filter(self) -> None:
        """Clear temporal state (call on grasp reset / new episode)."""
        if self.kf is not None:
            self.kf.reset()
        self.last_q = None
        self.last_contact_local = None
        self.last_contact_err_m = None
        self._locked_contact = None
        self._contact_votes = []
        self._bad_vote_streak = 0
        self._nn_step = 0
        self._last_force = 0.0
        self.last_image = None

    # ------------------------------------------------------------ Stage E helpers
    @property
    def basis_fingerprint(self) -> str | None:
        if self.geo is None:
            return None
        from .nn_vision import basis_fingerprint

        return basis_fingerprint(self.geo.basis.Phi)

    def set_scene(self, object_center, object_R=None) -> None:
        """Object pose the marker-free renderer should draw (set by the arm sim each step)."""
        self.scene_object_center = np.asarray(object_center, float).reshape(3)
        self.scene_object_R = None if object_R is None else np.asarray(object_R, float).reshape(3, 3)

    def ensure_nn(self, *, remote: bool = True) -> None:
        """Load the ResNet checkpoint and start the marker-free renderer (once)."""
        if self.nn is not None:
            return
        if self.world_camera is None or self.geo is None:
            raise RuntimeError("nn estimator needs the vision setup (mode=world/nn or prepare_vision at build)")
        from .nn_vision import GraspSceneAppearance, NNStateEstimator

        c = dict(self.nn_cfg or {})
        ckpt = c.get("checkpoint", "results/etc/checkpoints/grasp_nn/resnet18_q_sim_raw.pt")
        self.nn = NNStateEstimator.from_checkpoint(
            ckpt, self.mech, self.world_camera,
            width=int(c.get("image_width", 240)), height=int(c.get("image_height", 180)),
            appearance=GraspSceneAppearance.from_dict(c["scene"]) if "scene" in c else None,
            remote=bool(c.get("remote", remote)),
            expected_basis_fingerprint=self.basis_fingerprint,
        )

    def close(self) -> None:
        if self.nn is not None:
            self.nn.close()
            self.nn = None

    def _q_from_nn(self, u: np.ndarray, T_ee: np.ndarray, opening: float) -> np.ndarray | None:
        """Marker-free frame → q on camera ticks; ``None`` between ticks (hold last filtered q)."""
        self.ensure_nn()
        self._nn_step += 1
        if (self._nn_step - 1) % self.nn_every != 0 and self.last_q is not None:
            return None
        if self.scene_object_center is None:
            c = self.mech.contact_local
            obj_center = T_ee[:3, :3] @ np.array([c[0], 0.0, c[2]]) + T_ee[:3, 3]
            obj_R = T_ee[:3, :3]
        else:
            obj_center, obj_R = self.scene_object_center, self.scene_object_R
        q = self.nn.predict_q(self.mech, u, opening, T_ee, obj_center, obj_R)
        self.last_image = self.nn.last_image
        return q

    def _filter_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1)
        if self.kf is not None:
            q = self.kf.update(q)
        self.last_q = q
        return q

    def _force_at_contact(self, q: np.ndarray, contact) -> float:
        est = self.geo._rom.estimate_force(q, contact, method="displacement", mode="normal")
        return float(np.clip(est.magnitude, 0.0, self.force_clip))

    def _try_lock_contact(self, contact) -> None:
        """Lock only after several nearby accepted localizations agree."""
        if self._contact_votes is None:
            self._contact_votes = []
        self._contact_votes.append(contact)
        need = max(1, int(self.lock_votes))
        self._contact_votes = self._contact_votes[-need:]
        if len(self._contact_votes) < need:
            return
        pos = np.stack([c.position for c in self._contact_votes], axis=0)
        if float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))) > self.lock_radius_m:
            return
        if self.lock_pick == "last":
            self._locked_contact = self._contact_votes[-1]
        else:  # medoid: the vote closest to the others (robust to one outlier in the window)
            d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1).sum(axis=1)
            self._locked_contact = self._contact_votes[int(np.argmin(d))]

    def _force_from_q(self, q: np.ndarray) -> float:
        """ROM force; optionally localize contact on the inner face from ``q``."""
        q = np.asarray(q, float).reshape(-1)
        if not (self.unknown_contact and self.localizer is not None):
            self.last_contact_local = self.mech.contact_local.copy()
            self.last_contact_err_m = 0.0
            return self._force_at_contact(q, self.mech.contact)

        qn = float(np.linalg.norm(q))
        if qn < self.q_min_norm:
            if self._locked_contact is None:
                if self.prelock_known_force and qn >= 0.5 * self.q_min_norm:
                    self.last_contact_local = self.mech.contact_local.copy()
                    self.last_contact_err_m = 0.0
                    return self._force_at_contact(q, self.mech.contact)
                self.last_contact_local = None
                self.last_contact_err_m = None
                return 0.0
            contact = self._locked_contact
            mag = self._force_at_contact(q, contact)
            self.last_contact_local = np.asarray(contact.position, float).copy()
            self.last_contact_err_m = float(np.linalg.norm(contact.position - self.mech.contact_local))
            return mag

        loc = self.localizer.localize(q)
        ok = (
            loc.residual_rel <= self.max_residual_rel
            and self.lock_min_force * 0.25 <= loc.magnitude <= self.force_clip
        )
        lock_ok = (
            ok
            and qn >= self.q_lock_norm
            and loc.residual_rel <= self.lock_residual_rel
            and loc.magnitude >= self.lock_min_force
            and loc.magnitude <= 0.75 * self.force_clip
        )
        if self._locked_contact is not None or not self.lock_votes_open:
            pass
        elif lock_ok:
            self._bad_vote_streak = 0
            self._try_lock_contact(loc.contact)
        else:
            # Don't wipe the vote streak on a single noisy frame.
            self._bad_vote_streak += 1
            if self._bad_vote_streak >= max(1, int(self.vote_clear_bad)):
                self._contact_votes = []
                self._bad_vote_streak = 0

        contact = self._locked_contact
        if contact is None:
            # Still voting on location; optionally drive PI with config-contact force.
            if self.prelock_known_force:
                self.last_contact_local = self.mech.contact_local.copy()
                self.last_contact_err_m = 0.0
                return self._force_at_contact(q, self.mech.contact)
            self.last_contact_local = None if not ok else np.asarray(loc.contact.position, float).copy()
            self.last_contact_err_m = (
                None if not ok
                else float(np.linalg.norm(loc.contact.position - self.mech.contact_local))
            )
            return 0.0

        mag = self._force_at_contact(q, contact)
        self.last_contact_local = np.asarray(contact.position, float).copy()
        self.last_contact_err_m = float(np.linalg.norm(contact.position - self.mech.contact_local))
        return mag

    def _q_from_world(self, u: np.ndarray, T_ee: np.ndarray, opening: float) -> np.ndarray:
        """Project left-finger markers through the world camera and recover ``q``."""
        mesh = self.mech.model.mesh
        world = self.mech.world
        cam = self.world_camera
        p_def = world.left_nodes(self.markers.positions(mesh, u), opening, T_ee=T_ee)
        p0 = world.left_nodes(self.markers.positions(mesh, None), opening, T_ee=T_ee)
        dp = np.einsum("ij,mjk->mik", T_ee[:3, :3], self.geo._dp)
        uv, z = cam.project(p_def)
        if self.pixel_noise > 0.0:
            uv = uv + self.rng.normal(0.0, self.pixel_noise, size=uv.shape)
        n_w = transform_vectors(T_ee, self.markers.normals(mesh, u))
        facing = np.einsum("mj,mj->m", n_w, cam.position[None, :] - p_def) > 0.0
        vis = facing & _in_image(cam, uv, z)
        self.last_uv, self.last_visible = uv, vis
        J = image_jacobian(cam, p0, dp)
        # Self-weight compensation in *pixel* space: the sag u_g is a known load whose
        # marker motion is subtracted before the LS fit (u_g is largely outside span Φ,
        # so subtracting Φᵀu_g after the fit would not remove it).
        u_g = self.gravity_u(T_ee[:3, :3])
        if np.any(u_g):
            p_ref = world.left_nodes(self.markers.positions(mesh, u_g), opening, T_ee=T_ee)
        else:
            p_ref = p0
        uv_ref, _ = cam.project(p_ref)
        return solve_q_from_jacobian(J, uv - uv_ref, vis, ridge=self.geo.ridge)

    def measure(
        self,
        u: np.ndarray,
        lam_true: float,
        *,
        T_ee: np.ndarray | None = None,
        opening: float | None = None,
    ) -> float:
        R = None if T_ee is None else np.asarray(T_ee, float)[:3, :3]
        if self.mode == "gt":
            return float(lam_true)
        if self.mode == "inverse":
            u_c = u - self.gravity_u(R)
            est = self.inverse.estimate(u_c, self.mech.contact, method="displacement", mode="normal")
            return float(est.magnitude)
        if self.mode == "markers":
            uv, _ = self.camera.project(self.markers.positions(self.mech.model.mesh, u))
            if self.pixel_noise > 0.0:
                uv = uv + self.rng.normal(0.0, self.pixel_noise, size=uv.shape)
            vis = self.markers.visibility(self.camera, self.mech.model.mesh, u)
            self.last_uv, self.last_visible = uv, vis
            u_g = self.gravity_u(R)
            if np.any(u_g):  # remove the known sag's marker motion in pixel space
                mesh = self.mech.model.mesh
                uv_g, _ = self.camera.project(self.markers.positions(mesh, u_g))
                uv0, _ = self.camera.project(self.markers.positions(mesh, None))
                uv = uv - (uv_g - uv0)
            q = self._filter_q(self.geo.predict_one(self.camera, uv, vis))
            return self._force_from_q(q)
        if self.mode == "world":
            if self.world_camera is None or self.markers is None:
                raise RuntimeError("world estimator needs prepare_vision / mode=world at build")
            T = np.eye(4) if T_ee is None else np.asarray(T_ee, float)
            g = self.mech.object_width if opening is None else float(opening)
            q = self._filter_q(self._q_from_world(u, T, g))  # sag removed in pixel space
            return self._force_from_q(q)
        if self.mode == "nn":
            if self.world_camera is None or self.geo is None:
                raise RuntimeError("nn estimator needs prepare_vision / mode=nn at build")
            T = np.eye(4) if T_ee is None else np.asarray(T_ee, float)
            g = self.mech.object_width if opening is None else float(opening)
            q_raw = self._q_from_nn(u, T, g)
            if q_raw is None:  # between camera ticks: hold the last estimate (no duplicate votes)
                return self._last_force
            # No q-space correction here: the network is trained on q_sim = Φᵀu_contact, so
            # its q̂ carries no sag term (subtracting Φᵀu_g would *add* a bias — verified
            # in the arm loop: 4 % → 67 % force error). The sag lives in the image only.
            self._last_force = self._force_from_q(self._filter_q(q_raw))
            return self._last_force
        raise ValueError(f"unknown estimator {self.mode!r}")

    # ------------------------------------------------------------ self-weight compensation
    def gravity_u(self, R_world_from_local=None) -> np.ndarray:
        """Known self-weight sag (N, 3) in the finger frame; zero if off / disabled."""
        if not self.gravity_compensation or not self.mech.has_gravity:
            return np.zeros((self.mech.model.mesh.n_nodes, 3))
        return self.mech.gravity_displacement(R_world_from_local)

    def gravity_q(self, R_world_from_local=None) -> np.ndarray:
        """``Φᵀ u_g(R)`` – the part of the observed q explained by self-weight."""
        if self.geo is None or not self.gravity_compensation or not self.mech.has_gravity:
            r = self.geo.basis.r if self.geo is not None else 0
            return np.zeros(r)
        return np.asarray(self.geo.basis.project(self.gravity_u(R_world_from_local)), float)
