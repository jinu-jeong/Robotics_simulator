"""Marker-free world-camera vision for the grasp demo (Stage E).

The world camera of Stage D sees the *grasp scene*: two compliant fingers,
the grasped object and the table, moving with the arm flange ``T_ee``. This
module renders that scene **without markers** and turns the RGB frame into the
reduced coordinates ``q`` of the left finger with the ResNet-18 of
``src/vision/resnet_q.py``. Force and contact then follow from the unchanged
ROM path in :class:`~src.grasp.estimator.GraspEstimator`.

Two consumers share the exact same scene composition and renderer so there is
no training / deployment rendering gap:

* ``scripts/generate_grasp_nn_dataset.py`` – in-process renderer, many states
* ``GraspEstimator`` (mode ``nn``) – :class:`RemoteGraspRenderer`, one frame
  per camera tick. The renderer runs in a **separate process** because the
  interactive Taichi viewer already owns the only GGUI window this process may
  have (a second window segfaults).
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geometry.primitives import box_mesh
from ..visualization.camera import ObservationCamera
from ..visualization.state import ObjectGeometry
from .world import transform_vectors

ObjectTuple = tuple[np.ndarray, np.ndarray, tuple[float, float, float]]  # (vertices, faces, rgb)


# ---------------------------------------------------------------- camera / basis helpers
def nn_camera(world_camera: ObservationCamera, width: int, height: int) -> ObservationCamera:
    """Copy of the world camera at the network's input resolution (same pose / FOV)."""
    return ObservationCamera(
        position=world_camera.position.copy(), target=world_camera.target.copy(), up=world_camera.up.copy(),
        fov_y_deg=float(world_camera.fov_y_deg), image_width=int(width), image_height=int(height),
        near=float(world_camera.near), far=float(world_camera.far), name=f"{world_camera.name}_nn",
    )


def basis_fingerprint(Phi: np.ndarray) -> str:
    """Short hash of a POD basis so a checkpoint can be matched to the estimator's ``q`` convention."""
    return hashlib.sha1(np.ascontiguousarray(np.round(np.asarray(Phi, float), 6)).tobytes()).hexdigest()[:12]


# ---------------------------------------------------------------- scene composition
@dataclass
class GraspSceneAppearance:
    """Fixed colours of the grasp scene (identical for training and deployment)."""

    finger_color: tuple[float, float, float] = (0.75, 0.75, 0.78)
    object_color: tuple[float, float, float] = (0.85, 0.45, 0.18)
    table_color: tuple[float, float, float] = (0.35, 0.35, 0.38)
    background_color: tuple[float, float, float] = (0.10, 0.11, 0.13)
    ambient: tuple[float, float, float] = (0.35, 0.35, 0.35)
    light_color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    light_offset: tuple[float, float, float] = (0.0, 0.0, 0.15)  # relative to the camera position [m]
    image_noise_std: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "GraspSceneAppearance":
        d = dict(d or {})
        kw = {}
        for k in cls.__dataclass_fields__:
            if k in d:
                v = d[k]
                kw[k] = tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else float(v)
        return cls(**kw)


def two_finger_faces(mesh) -> np.ndarray:
    """Surface triangles of left + right finger (right finger nodes offset by N)."""
    return np.concatenate([mesh.surface_faces, mesh.surface_faces + mesh.n_nodes], axis=0)


def compose_grasp_scene(
    mech,
    u_local: np.ndarray,
    opening: float,
    T_ee: np.ndarray,
    object_center: np.ndarray,
    object_R: np.ndarray | None,
    *,
    table_center: np.ndarray,
    table_size: np.ndarray,
    appearance: GraspSceneAppearance,
) -> tuple[np.ndarray, list[ObjectTuple]]:
    """World-frame deformed nodes of both fingers plus object / table triangles.

    Mirrors :func:`src.grasp.scene.build_grasp_state` (both fingers carry the
    same load, the right one mirrored).
    """
    mesh, world = mech.model.mesh, mech.world
    u_local = np.asarray(u_local, float).reshape(-1, 3)
    nodes_L = world.left_nodes(mesh.nodes, opening, T_ee=T_ee) + world.left_disp(u_local, T_ee)
    nodes_R = world.right_nodes(mesh.nodes, opening, T_ee=T_ee) + world.right_disp(u_local, T_ee)
    nodes = np.concatenate([nodes_L, nodes_R], axis=0)
    R = np.eye(3) if object_R is None else np.asarray(object_R, float)
    lam = mech.force_from_opening(opening, np.asarray(T_ee, float)[:3, :3])
    ov, of = box_mesh(np.zeros(3), mech.object_size_at(lam))
    ov = ov @ R.T + np.asarray(object_center, float)
    tv, tf = box_mesh(np.asarray(table_center, float), np.asarray(table_size, float))
    return nodes, [(ov, of, appearance.object_color), (tv, tf, appearance.table_color)]


def left_markers_world(markers, mesh, u_local: np.ndarray, opening: float, T_ee: np.ndarray, world) -> tuple[np.ndarray, np.ndarray]:
    """Marker positions / normals of the deformed left finger in world coordinates."""
    p = world.left_nodes(markers.positions(mesh, u_local), opening, T_ee=T_ee)
    n = transform_vectors(T_ee, markers.normals(mesh, u_local))
    return p, n


# ---------------------------------------------------------------- renderers
class GraspSceneRenderer:
    """In-process off-screen renderer for the composed grasp scene (needs a GGUI window)."""

    def __init__(self, width: int, height: int, n_nodes_total: int, faces: np.ndarray, appearance: GraspSceneAppearance) -> None:
        from ..rendering.synthetic_camera import RenderAppearance, SyntheticCameraRenderer

        self._RenderAppearance = RenderAppearance
        self.r = SyntheticCameraRenderer(int(width), int(height))
        self.r.set_mesh(int(n_nodes_total), np.asarray(faces))
        self.ap = appearance

    def render(
        self,
        nodes: np.ndarray,
        camera: ObservationCamera,
        objects: list[ObjectTuple],
        *,
        markers_xyz: np.ndarray | None = None,
        marker_normals: np.ndarray | None = None,
        marker_radius: float = 0.0012,
        light_position: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        ap = self._RenderAppearance(
            finger_color=self.ap.finger_color, background_color=self.ap.background_color, ambient=self.ap.ambient,
            light_position=(camera.position + np.asarray(self.ap.light_offset, float)) if light_position is None else np.asarray(light_position, float),
            light_color=self.ap.light_color, marker_color=(1.0, 0.2, 0.1), marker_radius=float(marker_radius),
            image_noise_std=float(self.ap.image_noise_std),
        )
        objs = [ObjectGeometry(v, f, color=tuple(c)) for v, f, c in objects]
        img = self.r.render(np.asarray(nodes, float), camera, ap, markers_xyz=markers_xyz, marker_normals=marker_normals,
                            objects=objs, rng=rng)
        return np.round(255.0 * img).astype(np.uint8)

    def close(self) -> None:
        self.r.destroy()


def _remote_worker(conn, width, height, n_nodes_total, faces, ap_dict) -> None:  # pragma: no cover - subprocess
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        rend = GraspSceneRenderer(width, height, n_nodes_total, faces, GraspSceneAppearance.from_dict(ap_dict))
        conn.send(("ready", None))
    except Exception as e:  # noqa: BLE001
        conn.send(("error", repr(e)))
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        try:
            nodes, cam_dict, objects, seed = msg
            cam = ObservationCamera.from_config(cam_dict)
            img = rend.render(nodes, cam, objects, rng=np.random.default_rng(seed))
            conn.send(("ok", img))
        except Exception as e:  # noqa: BLE001
            conn.send(("error", repr(e)))
    rend.close()


class RemoteGraspRenderer:
    """Same rendering as :class:`GraspSceneRenderer`, executed in a child process."""

    def __init__(self, width: int, height: int, n_nodes_total: int, faces: np.ndarray, appearance: GraspSceneAppearance) -> None:
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(
            target=_remote_worker,
            args=(child, int(width), int(height), int(n_nodes_total), np.asarray(faces), appearance.to_dict()),
            daemon=True,
        )
        self._proc.start()
        status, payload = self._conn.recv()
        if status != "ready":
            raise RuntimeError(f"remote grasp renderer failed to start: {payload}")

    def render(self, nodes: np.ndarray, camera: ObservationCamera, objects: list[ObjectTuple], *, seed: int = 0) -> np.ndarray:
        cam_dict = camera.to_dict()
        objs = [(np.asarray(v, np.float32), np.asarray(f, np.int32), tuple(float(x) for x in c)) for v, f, c in objects]
        self._conn.send((np.asarray(nodes, np.float32), cam_dict, objs, int(seed)))
        status, payload = self._conn.recv()
        if status != "ok":
            raise RuntimeError(f"remote grasp renderer error: {payload}")
        return payload

    def close(self) -> None:
        try:
            self._conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()


# ---------------------------------------------------------------- image -> q
@dataclass
class NNStateEstimator:
    """Marker-free frame → ResNet-18 → ``q`` for the grasp estimator."""

    net: Any
    camera: ObservationCamera
    renderer: Any  # GraspSceneRenderer | RemoteGraspRenderer
    appearance: GraspSceneAppearance
    table_center: np.ndarray
    table_size: np.ndarray
    last_image: np.ndarray | None = None
    n_frames: int = 0
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        mech,
        world_camera: ObservationCamera,
        *,
        width: int,
        height: int,
        appearance: GraspSceneAppearance | None = None,
        table_center=(0.28, 0.0, -0.010),
        table_size=(0.42, 0.24, 0.020),
        remote: bool = True,
        expected_basis_fingerprint: str | None = None,
    ) -> "NNStateEstimator":
        from ..vision.resnet_q import ResNetQModel

        net = ResNetQModel.load(checkpoint)
        hw = tuple(int(x) for x in net.image_hw)
        if hw != (int(height), int(width)):
            raise ValueError(f"checkpoint expects {hw[1]}x{hw[0]} images, config asks {width}x{height}")
        fp = net.meta.get("basis_fingerprint")
        if expected_basis_fingerprint is not None and fp is not None and fp != expected_basis_fingerprint:
            raise ValueError(
                f"checkpoint q-basis {fp} != estimator basis {expected_basis_fingerprint} "
                "(regenerate scripts/generate_grasp_nn_dataset.py with the current grasp config)"
            )
        ap = appearance or GraspSceneAppearance.from_dict(net.meta.get("scene_appearance"))
        mesh = mech.model.mesh
        faces = two_finger_faces(mesh)
        R = RemoteGraspRenderer if remote else GraspSceneRenderer
        renderer = R(width, height, 2 * mesh.n_nodes, faces, ap)
        return cls(
            net=net, camera=nn_camera(world_camera, width, height), renderer=renderer, appearance=ap,
            table_center=np.asarray(table_center, float), table_size=np.asarray(table_size, float),
            meta={"checkpoint": str(checkpoint), "basis_fingerprint": fp},
        )

    def frame(self, mech, u_local: np.ndarray, opening: float, T_ee: np.ndarray, object_center, object_R) -> np.ndarray:
        nodes, objs = compose_grasp_scene(
            mech, u_local, opening, T_ee, object_center, object_R,
            table_center=self.table_center, table_size=self.table_size, appearance=self.appearance,
        )
        if isinstance(self.renderer, RemoteGraspRenderer):
            img = self.renderer.render(nodes, self.camera, objs, seed=self.n_frames)
        else:
            img = self.renderer.render(nodes, self.camera, objs, rng=np.random.default_rng(self.n_frames))
        self.n_frames += 1
        self.last_image = img
        return img

    def predict_q(self, mech, u_local: np.ndarray, opening: float, T_ee: np.ndarray, object_center, object_R) -> np.ndarray:
        img = self.frame(mech, u_local, opening, T_ee, object_center, object_R)
        return np.asarray(self.net.predict(img)[0], float)

    def close(self) -> None:
        self.renderer.close()
