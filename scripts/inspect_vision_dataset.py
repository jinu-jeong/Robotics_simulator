#!/usr/bin/env python
"""Milestone 5 – inspect the synthetic vision dataset.

Two views of the same data:

* Figures (always written): a grid of rendered observation images with the
  stored marker labels overlaid (visible = cyan, hidden = grey), and the
  "signal strength" plot – marker image motion [px] versus force for every
  sample, together with the marker pixel-noise level.
* Interactive 3D debug view (unless ``--headless``): the FEM deformation, the
  observation camera frustum of the current sample and the markers in 3D,
  with the *rendered observation image* as a picture-in-picture inset – so the
  research camera and the developer camera are visibly separate (brief §15).

Keys (in addition to the viewer defaults):
    Left / Right    previous / next sample (±10 with Shift)     Space   random sample
    Home / End      first / last sample                          u       toggle marker overlay in the inset
    y               toggle noisy / clean marker labels in the inset

Run:
    python scripts/inspect_vision_dataset.py
    python scripts/inspect_vision_dataset.py data/processed/vision_top.npz --sample 1799 --headless
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
from src.rendering.markers import SurfaceMarkers  # noqa: E402
from src.utils.io import make_run_dir, save_json  # noqa: E402
from src.vision.dataset import VisionDataset  # noqa: E402


def paint_markers(img: np.ndarray, px: np.ndarray, visible: np.ndarray, size: int = 1) -> np.ndarray:
    """Draw small crosses at marker pixels (cyan = visible, grey = hidden) into a float RGB image copy."""
    out = np.array(img, dtype=np.float32, copy=True)
    H, W = out.shape[:2]
    for (u, v), vis in zip(px, visible):
        i, j = int(np.floor(u)), int(np.floor(v))
        if not (0 <= i < W and 0 <= j < H):
            continue
        c = (0.0, 1.0, 1.0) if vis else (0.6, 0.6, 0.6)
        out[j, max(i - size, 0) : i + size + 1] = c
        out[max(j - size, 0) : j + size + 1, i] = c
    return out


def undeformed_projection(vds: VisionDataset, markers: SurfaceMarkers, mesh) -> np.ndarray:
    """(S, M, 2) projections of the undeformed markers with each sample's camera."""
    p0 = markers.positions(mesh, None)
    out = np.zeros_like(vds.marker_px_clean)
    for s in range(len(vds)):
        out[s] = vds.camera(s).project(p0)[0]
    return out


def make_figures(vds: VisionDataset, markers: SurfaceMarkers, mesh, run_dir: Path, n_grid: int, rng: np.random.Generator) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = len(vds)
    pick = np.sort(rng.choice(S, size=min(n_grid, S), replace=False))
    ncol = 4
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow), constrained_layout=True)
    for a in np.ravel(axes):
        a.set(xticks=[], yticks=[])
    for a, s in zip(np.ravel(axes), pick):
        H, W = vds.image_shape[:2]
        a.imshow(vds.image_float(s), extent=(0, W, H, 0), cmap="gray" if vds.image_shape[2] == 1 else None)
        vis = vds.marker_visible[s]
        a.plot(vds.marker_px[s, vis, 0], vds.marker_px[s, vis, 1], "+", color="cyan", ms=5, mew=0.8)
        a.plot(vds.marker_px[s, ~vis, 0], vds.marker_px[s, ~vis, 1], "x", color="grey", ms=4, mew=0.8)
        c = vds.contact_position[s]
        a.set_title(f"#{s}  F={vds.force_magnitude[s]:.2f} N  x_c={1e3 * c[0]:.0f} mm  fov={vds.camera_fov_y_deg[s]:.1f}°", fontsize=8)
        a.set(xlim=(0, W), ylim=(H, 0))
    fig.suptitle(f"synthetic observations ({vds.image_shape[1]}x{vds.image_shape[0]}), marker labels: cyan = visible, grey = hidden", fontsize=10)
    fig.savefig(run_dir / "samples_grid.png", dpi=110)

    # signal strength: marker image motion vs force
    uv0 = undeformed_projection(vds, markers, mesh)
    flow = np.linalg.norm(vds.marker_px_clean - uv0, axis=2)  # (S, M)
    vis = vds.marker_visible
    flow_mean = np.array([flow[s, vis[s]].mean() if vis[s].any() else np.nan for s in range(S)])
    flow_max = np.array([flow[s, vis[s]].max() if vis[s].any() else np.nan for s in range(S)])
    px_noise = float(vds.meta.get("config", {}).get("markers", {}).get("pixel_noise_std", 0.0))
    noise_rms = float(np.sqrt(np.mean((vds.marker_px - vds.marker_px_clean) ** 2)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    xc = 1e3 * vds.contact_position[:, 0]
    sc = ax[0].scatter(vds.force_magnitude, flow_mean, c=xc, s=6, cmap="viridis")
    ax[0].scatter(vds.force_magnitude, flow_max, c=xc, s=6, cmap="viridis", marker="^", alpha=0.35)
    if px_noise > 0:
        ax[0].axhline(px_noise, color="r", ls="--", lw=1, label=f"marker pixel noise σ = {px_noise:g} px")
        ax[0].legend(fontsize=8)
    ax[0].set(xlabel="force magnitude [N]", ylabel="marker image motion [px]",
              title="visual signal: mean (•) / max (▲) marker motion vs force", yscale="log")
    ax[0].grid(alpha=0.3, which="both")
    plt.colorbar(sc, ax=ax[0], label="contact x [mm]")
    ax[1].hist(flow[vis].ravel(), bins=60, color="C0", alpha=0.8)
    ax[1].set(xlabel="per-marker image motion [px]", ylabel="count", title=f"all visible markers (noise RMS {noise_rms:.2f} px)", yscale="log")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "signal_vs_force.png", dpi=120)

    # camera pose spread
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(1e3 * vds.camera_position[:, 0], 1e3 * vds.camera_position[:, 1], s=4, c=vds.camera_fov_y_deg, cmap="plasma")
    ax.set(xlabel="camera x [mm]", ylabel="camera y [mm]", title="observation camera positions (colour = FOV)", aspect="equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "camera_poses.png", dpi=120)

    stats = {
        "n_samples": S, "flow_px_mean_over_samples": float(np.nanmean(flow_mean)), "flow_px_max": float(np.nanmax(flow_max)),
        "flow_px_at_min_force_mean": float(np.nanmean(flow_mean[vds.force_magnitude <= vds.force_magnitude.min() + 1e-9])),
        "marker_pixel_noise_rms": noise_rms, "visible_fraction": float(vis.mean()),
        "fraction_samples_with_mean_flow_below_noise": float(np.mean(flow_mean < px_noise)) if px_noise > 0 else 0.0,
    }
    return stats


class VisionInspector:
    def __init__(self, vds: VisionDataset, fem: ContactDataset, markers: SurfaceMarkers, mesh, viewer, start: int) -> None:
        self.vds, self.fem, self.markers, self.mesh, self.v = vds, fem, markers, mesh, viewer
        self.i = start % len(vds)
        self.overlay = True
        self.noisy = True
        self.rng = np.random.default_rng(0)
        self.p0 = markers.positions(mesh, None)
        self._install_keys()
        self.apply(first=True)

    def _install_keys(self) -> None:
        import taichi as ti

        v = self.v
        shift = lambda: v.window.is_pressed(ti.ui.SHIFT)  # noqa: E731
        v.register_key(ti.ui.RIGHT, lambda _: self.step(10 if shift() else 1), "next sample (+10 with Shift)")
        v.register_key(ti.ui.LEFT, lambda _: self.step(-10 if shift() else -1), "previous sample (-10 with Shift)")
        v.register_key("Home", lambda _: self.goto(0), "first sample")
        v.register_key("End", lambda _: self.goto(len(self.vds) - 1), "last sample")
        v.register_key(ti.ui.SPACE, lambda _: self.goto(int(self.rng.integers(len(self.vds)))), "random sample")
        v.register_key("u", lambda _: self.toggle("overlay"), "toggle marker overlay in the inset")
        v.register_key("y", lambda _: self.toggle("noisy"), "toggle noisy / clean marker labels in the inset")

    def toggle(self, name: str) -> None:
        setattr(self, name, not getattr(self, name))
        self.apply()

    def step(self, d: int) -> None:
        self.goto(self.i + d)

    def goto(self, i: int) -> None:
        self.i = int(i) % len(self.vds)
        self.apply()

    def apply(self, first: bool = False) -> None:
        s, vds = self.i, self.vds
        fi = int(vds.fem_index[s])
        cam = vds.camera(s)
        vis = vds.marker_visible[s]
        px = vds.marker_px[s] if self.noisy else vds.marker_px_clean[s]
        flow = np.linalg.norm(vds.marker_px_clean[s] - cam.project(self.p0)[0], axis=1)
        ap = vds.appearance[s]
        info = {
            "vision sample": f"{s} / {len(vds) - 1}   (FEM sample {fi})",
            "q (first 4)": "  ".join(f"{x:+.2e}" for x in vds.q[s, :4]),
            "markers visible": f"{int(vis.sum())} / {len(vis)}",
            "marker motion": f"mean {flow[vis].mean():.1f} px, max {flow[vis].max():.1f} px" if vis.any() else "n/a",
            "camera": f"fov {cam.fov_y_deg:.1f} deg, pos ({1e3 * cam.position[0]:.0f}, {1e3 * cam.position[1]:.0f}, {1e3 * cam.position[2]:.0f}) mm",
            "appearance": f"bg {tuple(round(x, 2) for x in ap['background_color'])}, noise {ap['image_noise_std']:g}",
            "inset": f"{'noisy' if self.noisy else 'clean'} labels (Y), overlay {'on' if self.overlay else 'off'} (U)",
        }
        state = self.fem.to_visualization_state(fi, extra_info=info)
        state.observation_camera = cam
        state.keypoints = self.p0
        state.keypoints_visible = vis
        self.v.set_state(state, refit_camera=first)
        img = vds.image_float(s)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=2)
        self.v.set_overlay_image(paint_markers(img, px, vis) if self.overlay else img)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default="data/processed/vision_top.npz")
    p.add_argument("--fem-dataset", default=None, help="defaults to the path stored in the dataset meta")
    p.add_argument("--sample", type=int, default=-1)
    p.add_argument("--grid", type=int, default=12, help="number of images in the sample grid figure")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--out", default="results/etc/vision_dataset")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    vds = VisionDataset.load(args.dataset)
    fem = ContactDataset.load(args.fem_dataset or vds.meta["fem_dataset"])
    model = FingerFEMModel.from_config(fem.meta["fem_config"])
    markers = SurfaceMarkers.from_dict(vds.meta["markers"])
    run_dir = make_run_dir(args.out)
    print("dataset:", {k: v for k, v in vds.summary().items()})

    stats = make_figures(vds, markers, model.mesh, run_dir, args.grid, np.random.default_rng(args.seed))
    print("visual signal statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")
    save_json({"summary": vds.summary(), "signal": stats}, run_dir / "metrics.json")
    print(f"figures -> {run_dir}")

    from src.visualization.taichi_viewer import TaichiViewer

    viewer = TaichiViewer(show_window=not args.headless, title="vision dataset – debug view (inset = observation camera)")
    viewer.options.amplification = 1.0
    insp = VisionInspector(vds, fem, markers, model.mesh, viewer, args.sample)
    viewer.save_screenshot(run_dir / "debug_view.png")
    if not args.headless:
        viewer.run()
    else:
        insp.goto(len(vds) // 3)
        viewer.save_screenshot(run_dir / "debug_view_2.png")
        viewer.destroy()


if __name__ == "__main__":
    main()
