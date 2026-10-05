"""Assemble a two-finger grasp (and optional arm) into a ``VisualizationState``."""

from __future__ import annotations

import numpy as np

from ..geometry.primitives import box_mesh, cylinder_mesh, sphere_mesh
from ..visualization.camera import ObservationCamera
from ..visualization.state import ContactPoint, ForceVector, ObjectGeometry, VisualizationState
from .arm_sim import ArmGraspSim
from .sim import GraspSim
from .world import transform_vectors


def arm_link_objects(sim: ArmGraspSim, *, link_radius: float = 0.016) -> list[ObjectGeometry]:
    """Display-only cylinders / joint balls for the 7-DoF arm."""
    frames = sim.arm.frames(sim.q)
    origins = [T[:3, 3] for T in frames]
    pairs = [(0, 1), (1, 2), (2, 4), (4, 7)]  # pedestal, upper, forearm, flange
    colors = [
        (0.28, 0.30, 0.34),
        (0.45, 0.50, 0.58),
        (0.38, 0.55, 0.72),
        (0.55, 0.42, 0.32),
    ]
    radii = [link_radius * 1.25, link_radius, link_radius * 0.92, link_radius * 0.72]
    objs: list[ObjectGeometry] = []
    for (i, j), rgb, r in zip(pairs, colors, radii):
        v, f = cylinder_mesh(origins[i], origins[j], r, n_seg=12)
        objs.append(ObjectGeometry(v, f, name=f"link_{i}_{j}", color=rgb))
    for k, idx in enumerate((1, 2, 4, 7)):
        v, f = sphere_mesh(origins[idx], link_radius * (1.15 if idx < 7 else 0.85), n_lat=7, n_lon=10)
        objs.append(ObjectGeometry(v, f, name=f"joint_{idx}", color=(0.72, 0.62, 0.28)))
    return objs


def build_grasp_state(
    sim: GraspSim,
    *,
    show_markers: bool = True,
    T_ee: np.ndarray | None = None,
    object_center: np.ndarray | None = None,
    object_R: np.ndarray | None = None,
    table_center: np.ndarray | None = None,
    table_size: np.ndarray | None = None,
    extra_objects: list[ObjectGeometry] | None = None,
    info_extra: dict | None = None,
) -> VisualizationState:
    mech, st = sim.mech, sim.state
    mesh = mech.model.mesh
    world = mech.world
    R = None if T_ee is None else np.asarray(T_ee, float)[:3, :3]
    lam = mech.force_from_opening(st.opening, R)
    u_loc = mech.displacement(lam, R)
    nL = mesh.n_nodes
    lift = 0.0 if T_ee is not None else st.lift

    nodes_L = world.left_nodes(mesh.nodes, st.opening, lift, T_ee=T_ee)
    nodes_R = world.right_nodes(mesh.nodes, st.opening, lift, T_ee=T_ee)
    nodes = np.concatenate([nodes_L, nodes_R], axis=0)
    faces = np.concatenate([mesh.surface_faces, mesh.surface_faces + nL], axis=0)
    ee = np.concatenate([mesh.element_edges, mesh.element_edges + nL], axis=0)
    se = np.concatenate([mesh.surface_edges, mesh.surface_edges + nL], axis=0)
    u = np.concatenate([world.left_disp(u_loc, T_ee), world.right_disp(u_loc, T_ee)], axis=0)
    fixed = mesh.nodes[:, 0] < 1e-9
    fixed_idx = np.concatenate([np.nonzero(fixed)[0], np.nonzero(fixed)[0] + nL])

    est = sim.estimator
    cL = world.left_contact_world(mech.contact_local, st.opening, lift, T_ee=T_ee)
    cR = world.right_contact_world(mech.contact_local, st.opening, lift, T_ee=T_ee)
    c_est_local = est.last_contact_local if est.last_contact_local is not None else mech.contact_local
    cL_est = world.left_contact_world(c_est_local, st.opening, lift, T_ee=T_ee)
    FL = np.array([0.0, -lam, 0.0])
    FR = np.array([0.0, +lam, 0.0])
    FE = np.array([0.0, -st.lam_meas, 0.0])
    nL_w = np.array([0.0, 1.0, 0.0])
    nR_w = np.array([0.0, -1.0, 0.0])
    if T_ee is not None:
        FL, FR, FE = transform_vectors(T_ee, np.stack([FL, FR, FE]))
        nL_w, nR_w = transform_vectors(T_ee, np.stack([nL_w, nR_w]))

    if object_center is None:
        object_center = np.array([mech.contact_local[0], 0.0, st.obj_z])
    R = np.eye(3) if object_R is None else np.asarray(object_R, float)
    ov, of = box_mesh(np.zeros(3), mech.object_size_at(lam))
    ov = ov @ R.T + np.asarray(object_center, float)
    if table_center is None:
        table_center = np.array([0.5 * mech.model.geometry.length, 0.0, -0.004])
    if table_size is None:
        table_size = np.array([0.18, 0.12, 0.008])
    objects = [
        ObjectGeometry(ov, of, name="object", color=tuple(float(x) for x in (0.85, 0.45, 0.18))),
        ObjectGeometry.box(table_center, table_size, name="table", color=(0.35, 0.35, 0.38)),
    ]
    if extra_objects:
        objects.extend(extra_objects)

    cam = None
    keypoints = vis = None
    if est.mode in ("world", "nn") and est.world_camera is not None:
        cam = est.world_camera
        if est.mode == "world" and show_markers and est.markers is not None:
            p0 = est.markers.positions(mesh, None)
            keypoints = world.left_nodes(p0, st.opening, lift, T_ee=T_ee)
            vis = est.last_visible
            if vis is None:
                n_w = world.left_disp(est.markers.normals(mesh, u_loc), T_ee)
                uv, z = cam.project(keypoints)
                facing = np.einsum("mj,mj->m", n_w, cam.position[None, :] - keypoints) > 0.0
                vis = facing & (z > 0)
    elif est.camera is not None:
        cam_pos = world.left_nodes(est.camera.position.reshape(1, 3), st.opening, lift, T_ee=T_ee)[0]
        cam_tgt = world.left_nodes(est.camera.target.reshape(1, 3), st.opening, lift, T_ee=T_ee)[0]
        up = est.camera.up if T_ee is None else T_ee[:3, :3] @ est.camera.up
        cam = ObservationCamera(
            position=cam_pos, target=cam_tgt, up=up,
            fov_y_deg=est.camera.fov_y_deg, image_width=est.camera.image_width,
            image_height=est.camera.image_height, name=est.camera.name,
        )
        if show_markers and est.markers is not None:
            p0 = est.markers.positions(mesh, None)
            keypoints = world.left_nodes(p0, st.opening, lift, T_ee=T_ee)
            vis = est.markers.visibility(est.camera, mesh, u_loc)

    hold_need = mech.hold_force
    uc = "unknown contact" if est.unknown_contact else "known contact"
    info = {
        "phase": st.phase,
        "controller": f"{est.mode} ({uc})   target {sim.force_target:.2f} N",
        "force": f"true {lam:.3f} N, meas {st.lam_meas:.3f} N   (hold ≥ {hold_need:.2f} N)",
        "opening / lift": f"{1e3 * st.opening:.1f} mm / {1e3 * st.lift:.1f} mm",
        "object z": f"{1e3 * st.obj_z:.1f} mm   held={st.held}   {'SUCCESS' if sim.success() else st.phase}",
        "weight": f"{mech.weight:.2f} N   μ={mech.mu:g}   m={mech.mass:.3f} kg",
    }
    if st.dwell_left > 0.0:
        info["dwell"] = f"{st.dwell_left:.2f} s left (averaging q at preload)"
    if est.unknown_contact and getattr(est, "_locked_contact", None) is None and st.phase == "squeeze":
        info["vision"] = f"creep→{sim.preload_force:.1f} N then lock contact"
    if est.unknown_contact and est.last_contact_err_m is not None:
        info["contact err"] = f"{1e3 * est.last_contact_err_m:.1f} mm (est vs GT)"
    if est.mode == "nn":
        nn = getattr(est, "nn", None)
        cam_txt = f"{nn.camera.image_width}x{nn.camera.image_height}, {nn.n_frames} frames" if nn is not None else "loading…"
        info["vision"] = f"marker-free RGB → ResNet-18 → q  ({cam_txt})"
    if info_extra:
        info = {**info, **info_extra}
    contacts = [
        ContactPoint(cL, normal=nL_w),
        ContactPoint(cR, normal=nR_w),
    ]
    if est.unknown_contact and est.last_contact_local is not None:
        contacts.append(ContactPoint(cL_est, normal=nL_w))
    return VisualizationState(
        nodes_original=nodes,
        surface_faces=faces,
        element_edges=ee,
        surface_edges=se,
        displacement=u,
        fixed_nodes=fixed_idx,
        contact_points=contacts,
        ground_truth_force=ForceVector(cL, FL, label="F_true L"),
        estimated_force=ForceVector(cL_est, FE, label="F_meas") if est.mode not in ("gt",) else None,
        extra_forces=[ForceVector(cR, FR, label="F_true R")],
        objects=objects,
        observation_camera=cam,
        keypoints=keypoints,
        keypoints_visible=vis,
        info=info,
    )


def build_arm_grasp_state(sim: ArmGraspSim, *, show_markers: bool = True) -> VisualizationState:
    table_c = np.array([0.28, 0.0, sim.table_z - 0.010])
    table_s = np.array([0.42, 0.24, 0.020])
    info = {
        "arm phase": f"{sim.phase}   t={sim.t:.2f} s   IK ‖e‖={1e3 * sim.ik_err:.1f} mm-eq",
        "EE z / obj z": (
            f"{1e3 * sim.T_ee[2, 3]:.1f} / {1e3 * sim.obj_z:.1f} mm   "
            f"held={sim.held()}   {'SUCCESS' if sim.success() else sim.phase}"
        ),
        "q [rad]": "  ".join(f"{qi:+.2f}" for qi in sim.q),
        "phase": sim.phase,
        "object z": f"{1e3 * sim.obj_z:.1f} mm   held={sim.held()}   {'SUCCESS' if sim.success() else sim.phase}",
    }
    return build_grasp_state(
        sim.grasp,
        show_markers=show_markers,
        T_ee=sim.T_ee,
        object_center=sim.object_center(),
        object_R=sim.object_R,
        table_center=table_c,
        table_size=table_s,
        extra_objects=arm_link_objects(sim),
        info_extra=info,
    )
