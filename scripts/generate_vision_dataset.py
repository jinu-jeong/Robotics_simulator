#!/usr/bin/env python
"""Milestone 5 – generate the synthetic vision dataset (images + marker keypoints + mechanics labels).

For every sample of the FEM contact-sweep dataset (Milestone 2) and every
random observation-camera view:

  1. jitter the camera pose / FOV, pick a background colour, jitter the light,
  2. render the deformed finger (+ marker discs, + probe sphere) off-screen,
  3. project the markers with the *same* pinhole model (verified by
     ``run_synthetic_camera_check.py``), add optional pixel noise,
  4. store the image, marker pixels/visibility, q = Φ_rᵀu, force, contact and
     camera parameters (see ``src/vision/dataset.py``).

Run:
    python scripts/generate_vision_dataset.py                       # configs/vision.yaml
    python scripts/generate_vision_dataset.py --views 2 --limit 100 # quick test
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
from src.vision.dataset import VisionDataset  # noqa: E402
from src.visualization.state import ObjectGeometry  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="vision")
    p.add_argument("--fem-dataset", default=None)
    p.add_argument("--basis", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--views", type=int, default=None, help="camera views per FEM sample (overrides config)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N FEM samples (debug)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
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
    center = g.point_from_relative([0.5, 0.5, 0.5])
    backgrounds = [tuple(float(x) for x in b) for b in ap_cfg.get("background_colors", [[0.1, 0.11, 0.13]])]
    light0 = center + g.length * np.asarray(ap_cfg.get("light_offset_rel", [0.5, -1.0, 1.5]), float)
    light_jit = g.length * float(ap_cfg.get("light_jitter_rel", 0.0))
    px_noise = float(mk_cfg.get("pixel_noise_std", 0.0))
    probe_r = 0.4 * g.height

    S = n_fem * n_views
    images = np.zeros((S, H, W, C), np.uint8)
    marker_px = np.zeros((S, M, 2), np.float32)
    marker_px_clean = np.zeros((S, M, 2), np.float32)
    marker_vis = np.zeros((S, M), bool)
    q = np.zeros((S, basis.r))
    K_all, T_all, cpos, cfov = np.zeros((S, 3, 3)), np.zeros((S, 4, 4)), np.zeros((S, 3)), np.zeros(S)
    fem_index = np.zeros(S, np.int64)
    appearance, cameras = [], []

    print(f"rendering {n_fem} FEM samples x {n_views} views = {S} images ({W}x{H}x{C}), {M} markers, q in R^{basis.r}")
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
            img = renderer.render(nodes_def, cam, ap, markers_xyz=pos[vis], marker_normals=nrm[vis], objects=objs, rng=rng)
            uv, _ = cam.project(pos)
            images[s] = np.round(255.0 * img).astype(np.uint8)
            marker_px_clean[s] = uv
            marker_px[s] = uv + (rng.normal(0.0, px_noise, uv.shape) if px_noise > 0 else 0.0)
            marker_vis[s] = vis
            q[s] = qi
            K_all[s], T_all[s], cpos[s], cfov[s] = cam.intrinsics(), cam.extrinsics(), cam.position, cam.fov_y_deg
            fem_index[s] = i
            appearance.append(ap.to_dict())
            cameras.append(cam.to_dict())
            s += 1
        if (i + 1) % 100 == 0 or i + 1 == n_fem:
            el = time.time() - t0
            print(f"  {i + 1}/{n_fem} FEM samples  ({s} images, {el:.1f} s, {1e3 * el / s:.1f} ms/image)")
    renderer.destroy()

    vds = VisionDataset(
        images=images, marker_px=marker_px, marker_px_clean=marker_px_clean, marker_visible=marker_vis, q=q,
        force_magnitude=ds.force_magnitude[fem_index], force_vector=ds.force_vector[fem_index],
        contact_position=ds.contact_position[fem_index], contact_normal=ds.contact_normal[fem_index],
        camera_K=K_all, camera_T_cw=T_all, camera_position=cpos, camera_fov_y_deg=cfov, fem_index=fem_index,
        appearance=appearance,
        meta={"config": cfg, "seed": seed, "fem_dataset": str(fem_path), "rom_basis": str(basis_path), "q_modes": int(basis.r),
              "markers": markers.to_dict(), "nominal_camera": cam0.to_dict(), "cameras": cameras,
              "pixel_convention": "u right, v down, origin top-left corner; pixel (i,j) centre at (i+0.5, j+0.5)",
              "geometry": ds.meta.get("geometry", {})},
    )
    out = vds.save(out_path)
    save_yaml(cfg, out_path.with_suffix(".config.yaml"))
    summ = vds.summary()
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB on disk)")
    for k, v in summ.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
