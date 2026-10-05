#!/usr/bin/env python
"""Stage E – paired marker-on / marker-off dataset of the ARM GRASP scene (world camera).

Same idea as ``generate_markerfree_dataset.py`` but for the closed-loop demo:
the state is the Stage D grasp scene (two fingers on the flange ``T_ee``, the
object between them, the table) seen by the world camera of
``configs/grasp.yaml``. Contact location on the inner face, force, flange pose
(including the lift) and light are sampled; every state is rendered twice
from the identical scene (marker discs on / off).

Labels
  q_sim     Φ_rᵀ u in the grasp estimator's POD basis (what the ROM consumes)
  q_marker  Stage D marker path (markers → world camera projection + pixel
            noise → Jacobian LS) on the same state

Run:
    python scripts/generate_grasp_nn_dataset.py                 # configs/grasp_nn.yaml
    python scripts/generate_grasp_nn_dataset.py --limit 100     # quick test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grasp.arm import make_T  # noqa: E402
from src.grasp.factory import build_arm_grasp_sim  # noqa: E402
from src.grasp.nn_vision import (  # noqa: E402
    GraspSceneAppearance,
    GraspSceneRenderer,
    basis_fingerprint,
    compose_grasp_scene,
    left_markers_world,
    nn_camera,
    two_finger_faces,
)
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.vision.paired_dataset import PairedVisionDataset  # noqa: E402


def _rot_small(rng: np.random.Generator, deg: float) -> np.ndarray:
    a = np.deg2rad(rng.uniform(-deg, deg, 3))
    cx, cy, cz = np.cos(a)
    sx, sy, sz = np.sin(a)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="grasp_nn")
    p.add_argument("--output", default=None)
    p.add_argument("--limit", type=int, default=None, help="cap the number of paired samples (debug)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    gcfg = load_config(cfg.get("grasp_config", "grasp"))
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    rng = np.random.default_rng(seed)
    out_path = Path(args.output or cfg["output"])

    # Stage D machinery: mechanics, estimator POD basis, markers, world camera, arm poses.
    sim = build_arm_grasp_sim(gcfg, estimator_mode="world")
    est, mech = sim.estimator, sim.mech
    mesh, g, world = mech.model.mesh, mech.model.geometry, mech.world
    basis = est.geo.basis
    W, H = int(cfg["image"]["width"]), int(cfg["image"]["height"])
    cam = nn_camera(est.world_camera, W, H)
    ap = GraspSceneAppearance.from_dict(cfg.get("scene"))
    table_c = np.array([0.28, 0.0, sim.table_z - 0.010])
    table_s = np.array([0.42, 0.24, 0.020])
    marker_radius = 0.12 * g.height
    px_noise = float(gcfg.get("estimator", {}).get("pixel_noise_std", 0.0))

    # ---- sample plan
    s_cfg = cfg["samples"]
    xr = s_cfg.get("contact_x_rel", [0.55, 0.97, 13])
    zr = s_cfg.get("contact_z_rel", [0.15, 0.85, 7])
    fr = s_cfg.get("forces_N", [0.15, 3.5, 10])
    xs, zs = np.linspace(xr[0], xr[1], int(xr[2])), np.linspace(zr[0], zr[1], int(zr[2]))
    forces = np.linspace(fr[0], fr[1], int(fr[2]))
    repeats = int(s_cfg.get("pose_repeats", 2))
    n_free = int(s_cfg.get("n_free", 300))
    jit = s_cfg.get("pose_jitter", {})
    xy_j, lift_max = 1e-3 * float(jit.get("xy_mm", 4.0)), float(jit.get("lift_max_m", 0.04))
    rot_j, light_j = float(jit.get("rot_deg", 2.0)), float(jit.get("light_jitter_m", 0.05))

    # unit-load solves per contact (u is linear in the force)
    contacts = []
    t0 = time.time()
    for x in xs:
        for z in zs:
            pt = g.point_from_relative([float(x), 1.0, float(z)])
            res, c = mech.model.solve_normal_contact(pt, 1.0)
            contacts.append((c.position.copy(), c.normal.copy(), res.u.copy()))
    print(f"{len(contacts)} inner-face contacts solved ({time.time() - t0:.1f} s); "
          f"{len(forces)} forces x {repeats} poses + {n_free} free frames")

    plan: list[tuple[int, float, str]] = []
    for ci in range(len(contacts)):
        for F in forces:
            for _ in range(repeats):
                plan.append((ci, float(F), "grasp"))
    for _ in range(n_free):
        plan.append((int(rng.integers(len(contacts))), 0.0, "free"))
    rng.shuffle(plan)
    if args.limit is not None:
        plan = plan[: int(args.limit)]
    S = len(plan)

    T_g, T_pre = sim.T_grasp, sim.T_pre
    pre_off = T_g[:3, :3].T @ (T_pre[:3, 3] - T_g[:3, 3])  # pregrasp offset in the EE frame
    opening_start = float(gcfg["gripper"]["opening_start"])

    renderer = GraspSceneRenderer(W, H, 2 * mesh.n_nodes, two_finger_faces(mesh), ap)
    M = len(est.markers)
    img_marker = np.zeros((S, H, W, 3), np.uint8)
    img_raw = np.zeros((S, H, W, 3), np.uint8)
    marker_px = np.zeros((S, M, 2), np.float32)
    marker_px_clean = np.zeros((S, M, 2), np.float32)
    marker_vis = np.zeros((S, M), bool)
    q_sim = np.zeros((S, basis.r))
    q_marker = np.zeros((S, basis.r))
    force_mag = np.zeros(S)
    force_vec = np.zeros((S, 3))
    c_pos = np.zeros((S, 3))
    c_nrm = np.zeros((S, 3))
    K_all, T_all, cpos, cfov = np.zeros((S, 3, 3)), np.zeros((S, 4, 4)), np.zeros((S, 3)), np.zeros(S)
    appearance: list[dict] = []
    T_ee_all = np.zeros((S, 4, 4))

    print(f"rendering {S} paired frames ({W}x{H}) through the world camera")
    t0 = time.time()
    for s, (ci, F, kind) in enumerate(plan):
        c_position, c_normal, u_unit = contacts[ci]
        u_contact = F * u_unit
        if kind == "grasp":
            opening = mech.opening_from_force(F)
            off = np.array([rng.uniform(-xy_j, xy_j), rng.uniform(-xy_j, xy_j), rng.uniform(0.0, lift_max)])
        else:
            # gripper open, flange somewhere on pregrasp -> grasp -> lift
            opening = float(rng.uniform(mech.object_width + 1e-3, opening_start))
            a = rng.uniform(0.0, 1.0)
            off = (T_g[:3, :3] @ (a * pre_off)) if rng.uniform() < 0.5 else np.array([0.0, 0.0, rng.uniform(0.0, lift_max)])
            off = off + np.array([rng.uniform(-xy_j, xy_j), rng.uniform(-xy_j, xy_j), 0.0])
        R = T_g[:3, :3] @ _rot_small(rng, rot_j)
        T_ee = make_T(R, T_g[:3, 3] + off)
        # Rendered state = contact deformation + self-weight sag for this flange orientation
        # (fem.gravity). The regression target q_sim stays contact-only, so the network learns
        # to be invariant to the known sag; the marker teacher compensates it in pixel space.
        u = u_contact + mech.gravity_displacement(R)
        # squeezed / lifted: the box sits between the fingers at the sampled contact (rides the flange);
        # free: it rests on the table where the grasp pose would meet it
        if kind == "grasp":
            obj_center = T_ee[:3, :3] @ np.array([c_position[0], 0.0, c_position[2]]) + T_ee[:3, 3]
            obj_R = R
        else:
            obj_center, obj_R = sim._ee_object_center(T_g), np.eye(3)
        nodes, objs = compose_grasp_scene(mech, u, opening, T_ee, obj_center, obj_R, table_center=table_c, table_size=table_s, appearance=ap)
        light = cam.position + np.asarray(ap.light_offset) + rng.uniform(-light_j, light_j, 3)
        mk_p, mk_n = left_markers_world(est.markers, mesh, u, opening, T_ee, world)
        uv, z = cam.project(mk_p)
        facing = np.einsum("mj,mj->m", mk_n, cam.position[None, :] - mk_p) > 0.0
        vis = facing & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= W) & (uv[:, 1] >= 0) & (uv[:, 1] <= H)
        noise_seed = int(rng.integers(0, 2**31 - 1))
        img_marker[s] = renderer.render(nodes, cam, objs, markers_xyz=mk_p[vis], marker_normals=mk_n[vis], marker_radius=marker_radius,
                                        light_position=light, rng=np.random.default_rng(noise_seed))
        img_raw[s] = renderer.render(nodes, cam, objs, light_position=light, rng=np.random.default_rng(noise_seed))
        # Stage D marker teacher on this state (same camera pose, its own resolution-independent LS)
        est.rng = np.random.default_rng(noise_seed + 1)
        q_marker[s] = est._q_from_world(u, T_ee, opening)
        uv_noisy = uv + (rng.normal(0.0, px_noise, uv.shape) if px_noise > 0 else 0.0)
        marker_px_clean[s], marker_px[s], marker_vis[s] = uv, uv_noisy, vis
        q_sim[s] = basis.project(u_contact)
        force_mag[s] = F
        force_vec[s] = -F * c_normal
        c_pos[s], c_nrm[s] = c_position, c_normal
        K_all[s], T_all[s], cpos[s], cfov[s] = cam.intrinsics(), cam.extrinsics(), cam.position, cam.fov_y_deg
        T_ee_all[s] = T_ee
        appearance.append({"kind": kind, "opening": float(opening), "light_position": light.tolist(), "T_ee": T_ee.tolist()})
        if (s + 1) % 200 == 0 or s + 1 == S:
            el = time.time() - t0
            print(f"  {s + 1}/{S}  ({el:.1f} s, {1e3 * el / (s + 1):.1f} ms/pair)")
    renderer.close()

    pds = PairedVisionDataset(
        images_marker=img_marker, images_raw=img_raw,
        marker_px=marker_px, marker_px_clean=marker_px_clean, marker_visible=marker_vis,
        q_sim=q_sim, q_marker=q_marker, force_magnitude=force_mag, force_vector=force_vec,
        contact_position=c_pos, contact_normal=c_nrm,
        camera_K=K_all, camera_T_cw=T_all, camera_position=cpos, camera_fov_y_deg=cfov, fem_index=np.arange(S),
        appearance=appearance,
        meta={
            "config": cfg, "grasp_config": gcfg, "seed": seed, "self_contained": True, "fem_dataset": None, "rom_basis": None,
            "q_modes": int(basis.r), "basis_fingerprint": basis_fingerprint(basis.Phi), "scene_appearance": ap.to_dict(),
            "markers": est.markers.to_dict(), "nominal_camera": cam.to_dict(), "world_camera": est.world_camera.to_dict(),
            "q_marker_teacher": {"estimator": "stage-D world markers (Jacobian LS)", "pixel_noise_std": px_noise},
            "contact_face": "side_pos_y", "n_contacts": len(contacts),
            "self_weight": {"enabled": bool(mech.has_gravity),
                            "gravity_world": None if mech.model.gravity_world is None else [float(v) for v in mech.model.gravity_world],
                            "note": "rendered u = u_contact + u_g(R_ee); q_sim = Phi^T u_contact (target is contact-only)"},
            "pixel_convention": "u right, v down, origin top-left corner; pixel (i,j) centre at (i+0.5, j+0.5)",
        },
    )
    out = pds.save(out_path)
    save_yaml(cfg, out_path.with_suffix(".config.yaml"))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB on disk)")
    for k, v in pds.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
