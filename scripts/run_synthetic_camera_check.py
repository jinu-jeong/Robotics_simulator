#!/usr/bin/env python
"""Milestone 5 – verify the synthetic observation camera against the pinhole model.

Renders one FEM sample with red marker spheres from the observation camera,
detects the marker blobs in the rendered image and compares their centroids
with ``ObservationCamera.project`` of the marker 3D positions. Agreement to a
fraction of a pixel proves that intrinsics/extrinsics used for the labels are
the ones the renderer actually used (GGUI fov / look-at / image orientation).

Also produced: the marker displacement field in the image (undeformed ->
deformed, i.e. the visual signal a vision model sees), a strip of renderings
over increasing force, and a 3D debug-viewer screenshot showing the
observation camera frustum and the markers (visible = red, hidden = grey).

Run:
    python scripts/run_synthetic_camera_check.py                     # sample 899 of the contact-sweep dataset
    python scripts/run_synthetic_camera_check.py --sample 300 --width 640 --height 480 --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fem.contact_dataset import ContactDataset  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.rendering.markers import make_marker_grid  # noqa: E402
from src.rendering.synthetic_camera import RenderAppearance, SyntheticCameraRenderer, nominal_camera  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402
from src.visualization.state import ObjectGeometry  # noqa: E402


def detect_red_blobs(img: np.ndarray) -> np.ndarray:
    """Centroids (u, v) of red-dominant blobs in an (H, W, 3) float image, pixel-centre convention."""
    from scipy import ndimage

    mask = (img[..., 0] > 0.55) & (img[..., 1] < 0.45) & (img[..., 2] < 0.45)
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((0, 2))
    cm = np.array(ndimage.center_of_mass(mask, lab, range(1, n + 1)))  # (row, col)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    cm = cm[sizes >= 3]
    return np.stack([cm[:, 1] + 0.5, cm[:, 0] + 0.5], axis=1)  # (u, v) at pixel centres


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default="data/processed/contact_sweep_top.npz")
    p.add_argument("--config", default="vision")
    p.add_argument("--sample", type=int, default=899)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--out", default="results/etc/synthetic_camera")
    args = p.parse_args()

    cfg = load_config(args.config)
    ds = ContactDataset.load(args.dataset)
    model = FingerFEMModel.from_config(ds.meta["fem_config"])
    g, mesh = model.geometry, model.mesh
    run_dir = make_run_dir(args.out)
    i = args.sample % len(ds)
    u = ds.displacement(i)

    mk = cfg["markers"]
    markers = make_marker_grid(mesh, g, mk.get("face", "top"), int(mk["nx"]), int(mk["ny"]), float(mk.get("margin_rel", 0.05)))
    cam = nominal_camera(g, cfg["camera"], args.width, args.height)
    ap = RenderAppearance(marker_radius=float(mk.get("radius_rel", 0.12)) * g.height, marker_color=(1.0, 0.1, 0.05),
                          finger_color=(0.7, 0.7, 0.72), background_color=(0.1, 0.11, 0.13))
    renderer = SyntheticCameraRenderer(args.width, args.height)
    renderer.set_mesh(mesh.n_nodes, mesh.surface_faces)

    # ------------------------------------------------------------------ 1) pixel consistency
    pos_def = markers.positions(mesh, u)
    vis = markers.visibility(cam, mesh, u, margin_px=4.0)
    uv, depth = cam.project(pos_def)
    nrm_def = markers.normals(mesh, u)
    img = renderer.render(mesh.nodes + u, cam, ap, markers_xyz=pos_def[vis], marker_normals=nrm_def[vis])
    blobs = detect_red_blobs(img)
    d = np.linalg.norm(uv[vis][:, None, :] - blobs[None, :, :], axis=2)
    nearest = d.min(axis=1) if len(blobs) else np.full(vis.sum(), np.nan)
    print(f"sample {i}: F = {ds.force_magnitude[i]:.3f} N, max|u| = {1e3 * ds.max_displacement[i]:.3f} mm")
    print(f"markers: {len(markers)} total, {int(vis.sum())} visible; blobs detected: {len(blobs)}")
    print(f"projection vs rendered blob centroid: mean {nearest.mean():.3f} px, max {nearest.max():.3f} px  (image {args.width}x{args.height})")
    ok = bool(len(blobs) == vis.sum() and nearest.max() < 1.0)
    print("PASS" if ok else "FAIL", "– renderer and pinhole model agree to < 1 px" if ok else "– mismatch between renderer and pinhole model")

    # ------------------------------------------------------------------ 2) marker motion in the image
    pos_0 = markers.positions(mesh, None)
    uv0, _ = cam.project(pos_0)
    flow = uv - uv0
    print(f"marker image displacement (undeformed -> deformed): mean {np.linalg.norm(flow[vis], axis=1).mean():.2f} px, "
          f"max {np.linalg.norm(flow[vis], axis=1).max():.2f} px")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    ax[0].imshow(img, extent=(0, args.width, args.height, 0))
    ax[0].plot(uv[vis, 0], uv[vis, 1], "+", color="cyan", ms=10, mew=1.2, label="pinhole projection (visible)")
    ax[0].plot(uv[~vis, 0], uv[~vis, 1], "x", color="grey", ms=6, label="projection (hidden)")
    if len(blobs):
        ax[0].plot(blobs[:, 0], blobs[:, 1], "o", mfc="none", mec="yellow", ms=9, label="rendered blob centroid")
    ax[0].set(title=f"observation camera {args.width}x{args.height}: |proj − blob| mean {nearest.mean():.2f} px, max {nearest.max():.2f} px",
              xlim=(0, args.width), ylim=(args.height, 0))
    ax[0].legend(loc="lower right", fontsize=8)
    vis0 = markers.visibility(cam, mesh, None)
    img_flat = renderer.render(mesh.nodes, cam, ap, markers_xyz=pos_0[vis0], marker_normals=markers.normals(mesh, None)[vis0])
    ax[1].imshow(0.5 * img_flat + 0.5 * img, extent=(0, args.width, args.height, 0))
    ax[1].quiver(uv0[vis, 0], uv0[vis, 1], flow[vis, 0], flow[vis, 1], angles="xy", scale_units="xy", scale=1, color="cyan", width=0.004)
    ax[1].plot(uv0[vis, 0], uv0[vis, 1], ".", color="white", ms=4)
    ax[1].set(title=f"marker image motion, F = {ds.force_magnitude[i]:.2f} N (blend of undeformed and deformed renderings)",
              xlim=(0, args.width), ylim=(args.height, 0))
    fig.tight_layout()
    fig.savefig(run_dir / "camera_check.png", dpi=120)

    # strip over force levels at the same contact
    same = np.nonzero(np.all(np.isclose(ds.contact_position, ds.contact_position[i]), axis=1))[0]
    pick = same[np.linspace(0, len(same) - 1, min(4, len(same))).astype(int)]
    fig, ax = plt.subplots(1, len(pick), figsize=(4.2 * len(pick), 3.4))
    for a, j in zip(np.atleast_1d(ax), pick):
        uj = ds.displacement(j)
        pj = markers.positions(mesh, uj)
        vj = markers.visibility(cam, mesh, uj)
        r = 0.4 * g.height
        u_contact = uj[np.argmin(np.linalg.norm(mesh.nodes - ds.contact_position[j], axis=1))]
        probe = ObjectGeometry.sphere(ds.contact_position[j] + u_contact + ds.contact_normal[j] * r, r, color=(0.3, 0.75, 0.85))
        im_j = renderer.render(mesh.nodes + uj, cam, ap, markers_xyz=pj[vj], marker_normals=markers.normals(mesh, uj)[vj], objects=[probe])
        a.imshow(im_j, extent=(0, args.width, args.height, 0))
        a.set(title=f"F = {ds.force_magnitude[j]:.2f} N, max|u| = {1e3 * ds.max_displacement[j]:.2f} mm", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(run_dir / "force_strip.png", dpi=110)
    print(f"figures -> {run_dir}")

    save_json({"sample": i, "image": [args.width, args.height], "n_markers": len(markers), "n_visible": int(vis.sum()),
               "n_blobs": len(blobs), "px_error_mean": float(nearest.mean()), "px_error_max": float(nearest.max()), "pass": ok,
               "camera": cam.to_dict(), "flow_px_mean": float(np.linalg.norm(flow[vis], axis=1).mean())}, run_dir / "metrics.json")

    # ------------------------------------------------------------------ 3) debug viewer: frustum + markers
    from src.visualization.taichi_viewer import TaichiViewer

    state = ds.to_visualization_state(i, extra_info={"markers": f"{int(vis.sum())}/{len(markers)} visible from '{cam.name}'",
                                                     "camera check": f"|proj − blob| max {nearest.max():.2f} px ({'PASS' if ok else 'FAIL'})"})
    state.observation_camera = cam
    state.keypoints = pos_0
    state.keypoints_visible = vis
    viewer = TaichiViewer(show_window=not args.headless, title="synthetic camera check – debug view")
    viewer.options.amplification = 1.0
    viewer.set_state(state)
    viewer.save_screenshot(run_dir / "debug_view.png")
    if not args.headless:
        viewer.run()
    renderer.destroy()


if __name__ == "__main__":
    main()
