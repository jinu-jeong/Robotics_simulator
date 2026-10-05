#!/usr/bin/env python
"""Marker-free experiment – generate the PAIRED marker-on / marker-off vision dataset.

For every sample of the FEM contact-sweep dataset and every random observation
camera view, the *identical* state is rendered twice:

  1. marker-on  : deformed finger + marker discs (+ probe)   -> teacher modality
  2. marker-off : same nodes, camera, light, background, and the same image-noise
                  RNG stream; ``markers_xyz=None``            -> deployment modality

Stored per sample (see ``src/vision/paired_dataset.py``):

  * both images,
  * marker pixels (clean + noisy) and visibility,
  * ``q_sim``    = Φ_rᵀ u              (exact simulation state),
  * ``q_marker`` = existing marker → q pipeline on the noisy pixels (teacher),
  * force, contact location / normal, camera K / T_cw, FEM index, appearance.

The existing ``generate_vision_dataset.py`` and its outputs are untouched.

Run:
    python scripts/generate_markerfree_dataset.py                      # configs/markerfree.yaml
    python scripts/generate_markerfree_dataset.py --views 1 --limit 50 # quick test
    python scripts/generate_markerfree_dataset.py --no-probe --output data/processed/markerfree_top_noprobe.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fem.contact_dataset import ContactDataset  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.rendering.markers import make_marker_grid  # noqa: E402
from src.rendering.synthetic_camera import RenderAppearance, SyntheticCameraRenderer, jitter_camera, nominal_camera  # noqa: E402
from src.rom.pod import PODBasis  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402
from src.vision.marker_model import GeometricMarkerModel  # noqa: E402
from src.vision.paired_dataset import PairedVisionDataset  # noqa: E402
from src.visualization.state import ObjectGeometry  # noqa: E402


def build_teacher(basis: PODBasis, markers, mesh, cfg: dict) -> GeometricMarkerModel:
    """Existing marker → q estimator used as the q_marker teacher."""
    name = str(cfg.get("estimator", "iterative")).lower()
    if name == "geometric":
        n_iter, huber = 1, None
    elif name == "iterative":
        n_iter, huber = int(cfg.get("n_iter", 2)), cfg.get("huber_px", 1.5)
    else:
        raise ValueError(f"teacher.estimator must be geometric|iterative, got {name!r}")
    return GeometricMarkerModel(
        basis, markers, mesh,
        ridge=float(cfg.get("geometric_ridge", 1e-4)),
        use_noisy=bool(cfg.get("use_noisy", True)),
        n_iter=n_iter, huber=None if huber is None else float(huber),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="markerfree")
    p.add_argument("--fem-dataset", default=None)
    p.add_argument("--basis", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--views", type=int, default=None, help="camera views per FEM sample (overrides config)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N FEM samples (debug)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-probe", action="store_true", help="do not render the probe sphere at the contact (ablation: no object cue)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.no_probe:
        cfg["appearance"] = {**cfg["appearance"], "show_probe": False}
    fem_path = args.fem_dataset or cfg["fem_dataset"]
    basis_path = args.basis or cfg["rom_basis"]
    out_path = Path(args.output or cfg["output"])
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    rng = np.random.default_rng(seed)

    ds = ContactDataset.load(fem_path)
    model = FingerFEMModel.from_config(ds.meta["fem_config"])
    g, mesh = model.geometry, model.mesh
    basis = PODBasis.load(basis_path).truncate(int(cfg.get("q_modes", 8)))
    n_fem = len(ds) if args.limit is None else min(args.limit, len(ds))
    n_views = int(args.views if args.views is not None else cfg["camera"].get("n_views_per_sample", 1))

    im_cfg, cam_cfg, mk_cfg, ap_cfg = cfg["image"], cfg["camera"], cfg["markers"], cfg["appearance"]
    W, H = int(im_cfg["width"]), int(im_cfg["height"])
    C = 1 if im_cfg.get("grayscale", False) else 3
    markers = make_marker_grid(mesh, g, mk_cfg.get("face", "top"), int(mk_cfg["nx"]), int(mk_cfg["ny"]), float(mk_cfg.get("margin_rel", 0.05)))
    M = len(markers)
    cam0 = nominal_camera(g, cam_cfg, W, H)
    renderer = SyntheticCameraRenderer(W, H)
    renderer.set_mesh(mesh.n_nodes, mesh.surface_faces)
    teacher = build_teacher(basis, markers, mesh, cfg.get("teacher", {}))
    center = g.point_from_relative([0.5, 0.5, 0.5])
    backgrounds = [tuple(float(x) for x in b) for b in ap_cfg.get("background_colors", [[0.1, 0.11, 0.13]])]
    light0 = center + g.length * np.asarray(ap_cfg.get("light_offset_rel", [0.5, -1.0, 1.5]), float)
    light_jit = g.length * float(ap_cfg.get("light_jitter_rel", 0.0))
    px_noise = float(mk_cfg.get("pixel_noise_std", 0.0))
    probe_r = 0.4 * g.height

    S = n_fem * n_views
    img_marker = np.zeros((S, H, W, C), np.uint8)
    img_raw = np.zeros((S, H, W, C), np.uint8)
    marker_px = np.zeros((S, M, 2), np.float32)
    marker_px_clean = np.zeros((S, M, 2), np.float32)
    marker_vis = np.zeros((S, M), bool)
    q_sim = np.zeros((S, basis.r))
    q_marker = np.zeros((S, basis.r))
    K_all, T_all, cpos, cfov = np.zeros((S, 3, 3)), np.zeros((S, 4, 4)), np.zeros((S, 3)), np.zeros(S)
    fem_index = np.zeros(S, np.int64)
    appearance, cameras = [], []

    print(f"paired rendering: {n_fem} FEM samples x {n_views} views = {S} pairs ({W}x{H}x{C}), "
          f"{M} markers, q in R^{basis.r}, teacher={cfg.get('teacher', {}).get('estimator', 'iterative')}")
    t0 = time.time()
    s = 0
    for i in range(n_fem):
        u = ds.displacement(i)
        nodes_def = mesh.nodes + u
        pos = markers.positions(mesh, u)
        nrm = markers.normals(mesh, u)
        qi = basis.project(u)
        u_c = u[np.argmin(np.linalg.norm(mesh.nodes - ds.contact_position[i], axis=1))]
        objs = [ObjectGeometry.sphere(ds.contact_position[i] + u_c + probe_r * ds.contact_normal[i], probe_r)] if ap_cfg.get("show_probe", True) else None
        for _ in range(n_views):
            cam = jitter_camera(cam0, g, cam_cfg.get("jitter", {}), rng, name=f"obs_cam_{s}")
            ap = RenderAppearance(
                finger_color=tuple(float(x) for x in ap_cfg.get("finger_color", [0.75, 0.75, 0.78])),
                background_color=backgrounds[int(rng.integers(len(backgrounds)))],
                ambient=tuple(float(x) for x in ap_cfg.get("ambient", [0.35, 0.35, 0.35])),
                light_position=light0 + light_jit * rng.uniform(-1, 1, 3),
                light_color=tuple(float(x) for x in ap_cfg.get("light_color", [0.8, 0.8, 0.8])),
                marker_color=tuple(float(x) for x in mk_cfg.get("color", [1.0, 0.2, 0.1])),
                marker_radius=float(mk_cfg.get("radius_rel", 0.12)) * g.height,
                image_noise_std=float(im_cfg.get("noise_std", 0.0)),
                grayscale=bool(im_cfg.get("grayscale", False)),
            )
            vis = markers.visibility(cam, mesh, u)
            # identical noise realisation for both renders: one seed, two fresh streams
            noise_seed = int(rng.integers(0, 2**31 - 1))
            im_on = renderer.render(nodes_def, cam, ap, markers_xyz=pos[vis], marker_normals=nrm[vis], objects=objs,
                                    rng=np.random.default_rng(noise_seed))
            im_off = renderer.render(nodes_def, cam, ap, markers_xyz=None, objects=objs,
                                     rng=np.random.default_rng(noise_seed))
            uv, _ = cam.project(pos)
            uv_noisy = uv + (rng.normal(0.0, px_noise, uv.shape) if px_noise > 0 else 0.0)
            img_marker[s] = np.round(255.0 * im_on).astype(np.uint8)
            img_raw[s] = np.round(255.0 * im_off).astype(np.uint8)
            marker_px_clean[s] = uv
            marker_px[s] = uv_noisy
            marker_vis[s] = vis
            q_sim[s] = qi
            q_marker[s] = teacher.predict_one(cam, uv_noisy if teacher.use_noisy else uv, vis)
            K_all[s], T_all[s], cpos[s], cfov[s] = cam.intrinsics(), cam.extrinsics(), cam.position, cam.fov_y_deg
            fem_index[s] = i
            appearance.append(ap.to_dict())
            cameras.append(cam.to_dict())
            s += 1
        if (i + 1) % 100 == 0 or i + 1 == n_fem:
            el = time.time() - t0
            print(f"  {i + 1}/{n_fem} FEM samples  ({s} pairs, {el:.1f} s, {1e3 * el / max(s, 1):.1f} ms/pair)")
    renderer.destroy()

    pds = PairedVisionDataset(
        images_marker=img_marker, images_raw=img_raw,
        marker_px=marker_px, marker_px_clean=marker_px_clean, marker_visible=marker_vis,
        q_sim=q_sim, q_marker=q_marker,
        force_magnitude=ds.force_magnitude[fem_index], force_vector=ds.force_vector[fem_index],
        contact_position=ds.contact_position[fem_index], contact_normal=ds.contact_normal[fem_index],
        camera_K=K_all, camera_T_cw=T_all, camera_position=cpos, camera_fov_y_deg=cfov, fem_index=fem_index,
        appearance=appearance,
        meta={"config": cfg, "seed": seed, "fem_dataset": str(fem_path), "rom_basis": str(basis_path), "q_modes": int(basis.r),
              "markers": markers.to_dict(), "nominal_camera": cam0.to_dict(), "cameras": cameras,
              "q_marker_teacher": dict(cfg.get("teacher", {})),
              "pixel_convention": "u right, v down, origin top-left corner; pixel (i,j) centre at (i+0.5, j+0.5)",
              "geometry": ds.meta.get("geometry", {})},
    )
    out = pds.save(out_path)
    save_yaml(cfg, out_path.with_suffix(".config.yaml"))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB on disk)")
    for k, v in pds.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
