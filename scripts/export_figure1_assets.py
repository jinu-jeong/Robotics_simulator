#!/usr/bin/env python
"""Gather paper Figure-1 concept / algorithm-schematic panels into ``results/figure1/``.

Figure 1 is the overall pipeline diagram (you draw the boxes); these PNGs are
the visual building blocks to drop into that schematic:

    Camera → (markers | marker-free NN) → q → ROM → contact → force

Sources live under ``results/etc/`` (legacy milestone dumps). Re-run after
moving or regenerating those dumps.

    python scripts/export_figure1_assets.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.figure_res import MPL_DPI  # noqa: E402

ETC = ROOT / "results" / "etc"
OUT = ROOT / "results" / "figure1"

# (dest name, preferred source relative to results/etc/, …fallbacks)
# Prefer *lateral* (inner-face / −y) panels so Fig.1 matches Stages D/E grasp.
# Order: schematic → world grasp → paired / finger → analysis → path detail.
# Dropped: marker_projection, force_recovery, self_weight alone, pipeline_gt/vision_q.
PANELS: list[tuple[str, list[str]]] = [
    ("04_finger_scene_overlay.png", ["figures/lateral/01_scene_overlay_lateral.png", "figures/milestone0a_overlay.png"]),
    ("05_fem_convergence.png", ["figures/lateral/03_fem_convergence_lateral.png", "figures/milestone1_convergence.png"]),
    ("06_pod_modes.png", ["figures/lateral/06_pod_modes_lateral.png", "rom/20260907_041350/pod_modes.png"]),
    ("07_rom_vs_full.png", ["figures/lateral/07_rom_vs_full_lateral.png", "rom/20260907_041350/rom_analysis.png"]),
    ("08_two_paths_to_q.png", ["markerfree/eval_20260912_001424/summary_bars.png"]),
]

README = """Paper Figure 1 – 9 panels (schematic first)

Shared resolution: src/utils/figure_res.py
  viewer 12800×7200 · matplotlib dpi 1120
  regenerate all: python scripts/regenerate_paper_figures.py

01  pipeline schematic (method overview)           ← FIRST
02  world grasp scene (arm + gripper + object)
    early approach_ee, photo only
    regenerate: python scripts/render_figure1_grasp_scene.py
03  paired marker-on vs marker-free (|u| colormap)
04  finger close-up + F_true / F_est / self-weight
05  FEM mesh convergence vs beam theory
06  POD modes Φ
07  ROM vs full (tip |δ| vs rank; λ(x) at fixed tip δ*)
08  held-out: sim / marker / NN → same ROM
09  information paths — same state, 3 routes to q/λ
    A privileged uv | B marker RGB+LS | C marker-free ResNet

Dropped as redundant:
  marker_projection (overlap 03/04)
  force_recovery (no signal beyond 07)
  self_weight alone (already in 04)
  old pipeline_gt / pipeline_vision_q (overlap 04)

Regenerate lateral:  python scripts/render_figure1_lateral.py
Grasp scene (02):    python scripts/render_figure1_grasp_scene.py
Lateral FEM (05):    python scripts/run_cantilever_verification.py --config cantilever_lateral --no-view --fine
Then export:         python scripts/export_figure1_assets.py
"""


def _detect_red_marker_centroids(img: "np.ndarray", *, min_area: int = 2, max_area: int = 60) -> "np.ndarray":
    """Simple colour-threshold blob centres for the rendered red marker discs.

    The orange grasp object also triggers the red mask — large components are
    rejected by ``max_area`` so only marker-sized blobs remain.
    """
    import numpy as np

    r, g, b = img[:, :, 0].astype(np.int16), img[:, :, 1].astype(np.int16), img[:, :, 2].astype(np.int16)
    mask = (r > 120) & (r > g + 25) & (r > b + 25)
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    cents = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if not (min_area <= len(cells) <= max_area):
            continue
        yy = np.mean([c[0] for c in cells])
        xx = np.mean([c[1] for c in cells])
        cents.append((xx, yy))  # (u, v) image coords
    return np.asarray(cents, float).reshape(-1, 2)


def _pick_info_sample(d):
    """Loaded grasp frame + nearest free (zero-force) frame for Δuv illustration."""
    import numpy as np

    F = np.asarray(d.force_magnitude, float)
    loaded = np.where(F > 0.8)[0]
    if loaded.size == 0:
        loaded = np.where(F > 0.1)[0]
    idx = int(loaded[int(np.argmin(np.abs(F[loaded] - 1.5)))]) if loaded.size else 0
    free = np.where(F < 0.05)[0]
    if free.size == 0:
        return idx, idx
    cam = np.asarray(d.camera_position, float)
    j = int(free[int(np.argmin(np.linalg.norm(cam[free] - cam[idx], axis=1)))])
    return idx, j


def _force_from_q(rom, q, contact_xyz):
    c = rom.mapping.locate(contact_xyz)
    return float(rom.estimate_force(q, c, method="displacement", mode="normal").magnitude)


def _q_bar(ax, q, q_ref=None, title="q", color="tab:blue"):
    import numpy as np

    q = np.asarray(q, float).reshape(-1)
    x = np.arange(1, len(q) + 1)
    ax.bar(x, q, color=color, alpha=0.85, width=0.7, label="estimate")
    if q_ref is not None:
        q_ref = np.asarray(q_ref, float).reshape(-1)
        ax.plot(x, q_ref, "ko-", ms=3.5, lw=1.0, label="q_sim")
        ax.legend(fontsize=7, loc="best")
    ax.set_xticks(x)
    ax.set_xlabel("mode", fontsize=8)
    ax.set_ylabel("q", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    ax.tick_params(labelsize=7)


def _force_callout(ax, lam_hat, lam_true, subtitle="ROM force (known contact)"):
    ax.axis("off")
    ax.set_title(subtitle, fontsize=9)
    ax.text(0.5, 0.62, f"λ̂ = {lam_hat:.2f} N", ha="center", va="center", fontsize=14, fontweight="bold", color="tab:green")
    ax.text(0.5, 0.28, f"λ  = {lam_true:.2f} N  (GT)", ha="center", va="center", fontsize=11, color="0.35")
    err = abs(lam_hat - lam_true) / max(abs(lam_true), 1e-9)
    ax.text(0.5, 0.05, f"rel. err {100 * err:.1f} %", ha="center", va="center", fontsize=9, color="0.4")


def _information_path_panels() -> list:
    """Three paths: privileged uv numbers | detect markers from RGB | marker-free NN.

    Uses real world-camera frames from ``grasp_nn_top.npz`` and the Stage-E ResNet.
    """
    sys.path.insert(0, str(ROOT))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from src.grasp.factory import build_arm_grasp_sim
        from src.utils.config import load_config
        from src.vision.paired_dataset import PairedVisionDataset
        from src.vision.resnet_q import ResNetQModel
    except Exception as e:  # noqa: BLE001
        print(f"  skip information-path panels: {e}")
        return []

    ds_path = ROOT / "data/processed/grasp_nn_top.npz"
    ckpt = ROOT / "results/etc/checkpoints/grasp_nn/resnet18_q_sim_raw.pt"
    if not ds_path.is_file():
        print("  MISSING information paths (no grasp_nn_top.npz)")
        return []

    d = PairedVisionDataset.load(ds_path)
    idx, j_free = _pick_info_sample(d)
    F_true = float(d.force_magnitude[idx])
    contact = np.asarray(d.contact_position[idx], float)
    img_m = np.asarray(d.images_marker[idx])
    img_r = np.asarray(d.images_raw[idx])
    px_clean = np.asarray(d.marker_px_clean[idx], float)
    px_noisy = np.asarray(d.marker_px[idx], float)
    vis = np.asarray(d.marker_visible[idx], bool)
    px_free = np.asarray(d.marker_px_clean[j_free], float)
    q_sim = np.asarray(d.q_sim[idx], float)
    q_marker = np.asarray(d.q_marker[idx], float)

    # ROM for known-contact force (same Φ / K as the live demo)
    gcfg = load_config("grasp")
    sim = build_arm_grasp_sim(gcfg, estimator_mode="world")
    rom = sim.estimator.geo._rom
    lam_sim = _force_from_q(rom, q_sim, contact)
    lam_marker = _force_from_q(rom, q_marker, contact)

    q_nn = None
    lam_nn = None
    if ckpt.is_file():
        net = ResNetQModel.load(ckpt)
        q_nn = net.predict(img_r[None])[0]
        lam_nn = _force_from_q(rom, q_nn, contact)
    sim.estimator.close()

    detected = _detect_red_marker_centroids(img_m)
    # match detected blobs to clean projections for Δuv (nearest neighbour)
    duv_src = px_free[vis] if vis.any() else px_free
    duv_dst = px_noisy[vis] if vis.any() else px_noisy
    if detected.size and duv_src.size:
        # use projected clean vs free for arrows (pipeline observation model); overlay detected centres
        pass

    written: list = []

    # ---- 17a privileged: numbers only -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), gridspec_kw={"width_ratios": [1.15, 1.0, 0.7]})
    ax = axes[0]
    ax.axis("off")
    ax.set_title("(A) privileged marker pixels (given)", fontsize=10)
    lines = ["marker  u [px]   v [px]"]
    show = np.where(vis)[0][:8]
    for i in show:
        lines.append(f"  {i:2d}    {px_clean[i, 0]:6.1f}  {px_clean[i, 1]:6.1f}")
    if int(vis.sum()) > 8:
        lines.append(f"  …  ({int(vis.sum())} visible markers)")
    ax.text(0.05, 0.92, "\n".join(lines), transform=ax.transAxes, va="top", family="monospace", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="0.7"))
    ax.text(0.05, 0.08, "no camera image — coordinates are inputs", transform=ax.transAxes, fontsize=8, color="0.45")
    _q_bar(axes[1], q_sim, title="q = Φᵀu  (exact / from LS on uv)", color="0.35")
    _force_callout(axes[2], lam_sim, F_true, subtitle="ROM → force")
    fig.suptitle("Path A — marker positions known a priori (sim / MoCap). Force only; contact can be known or searched.", fontsize=10)
    fig.tight_layout()
    plt.close(fig)  # path A used only inside combined 17

    # ---- 17b detect markers from RGB --------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.0), gridspec_kw={"width_ratios": [1.0, 0.85, 1.0, 0.95, 0.7]})
    ax = axes[0]
    ax.imshow(img_m)
    ax.set_title("① world-camera RGB\n(markers painted)", fontsize=9)
    ax.axis("off")

    ax = axes[1]
    r, g, b = img_m[:, :, 0].astype(np.int16), img_m[:, :, 1].astype(np.int16), img_m[:, :, 2].astype(np.int16)
    mask = (r > 120) & (r > g + 25) & (r > b + 25)
    ax.imshow(mask, cmap="gray")
    ax.set_title("② red-blob threshold\n(R ≫ G,B)", fontsize=9)
    ax.axis("off")

    ax = axes[2]
    ax.imshow(img_m)
    if detected.size:
        ax.scatter(detected[:, 0], detected[:, 1], s=28, facecolors="none", edgecolors="lime", linewidths=1.2, label="detected")
    ax.scatter(px_clean[vis, 0], px_clean[vis, 1], s=8, c="cyan", marker="+", label="projected GT")
    # Δuv: free → loaded (pipeline observation: rest pixels vs observed)
    for a, bpt in zip(duv_src, duv_dst):
        ax.annotate("", xy=(bpt[0], bpt[1]), xytext=(a[0], a[1]),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=0.9, mutation_scale=8))
    ax.set_title(f"③ centroids + Δuv\n({len(detected)} blobs / {int(vis.sum())} GT)", fontsize=9)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.7)
    ax.axis("off")

    _q_bar(axes[3], q_marker, q_ref=q_sim, title="④ LS → q̂ (image Jacobian)", color="tab:blue")
    _force_callout(axes[4], lam_marker, F_true, subtitle="⑤ ROM → force")
    fig.suptitle("Path B — markers drawn on the finger; extract pixels from the camera image, then same q → ROM as A.", fontsize=10)
    fig.tight_layout()
    plt.close(fig)  # path B used only inside combined 17

    # ---- 17c marker-free NN -----------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.0), gridspec_kw={"width_ratios": [1.0, 0.55, 1.0, 0.7]})
    ax = axes[0]
    ax.imshow(img_r)
    ax.set_title("① marker-free RGB\n(same camera, no dots)", fontsize=9)
    ax.axis("off")
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    ax.add_patch(FancyBboxPatch((0.12, 0.18), 0.76, 0.64, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#dbe9ff", ec="0.3", lw=1.2))
    ax.text(0.5, 0.55, "ResNet-18", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.32, "240×180 → q ∈ ℝ⁸", ha="center", va="center", fontsize=8, color="0.35")
    ax.set_title("② network", fontsize=9)
    if q_nn is not None:
        _q_bar(axes[2], q_nn, q_ref=q_sim, title="③ output q̂  (vs q_sim)", color="tab:orange")
        _force_callout(axes[3], lam_nn, F_true, subtitle="④ ROM → force")
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "checkpoint missing", ha="center")
        axes[3].axis("off")
    fig.suptitle("Path C — no markers at deployment; ResNet reads silhouette / shading → q̂ → same ROM.", fontsize=10)
    fig.tight_layout()
    plt.close(fig)  # path C used only inside combined 17

    # ---- 17 combined strip ------------------------------------------------------------
    fig = plt.figure(figsize=(14.5, 9.2))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 5, figure=fig, height_ratios=[1, 1.05, 1], hspace=0.42, wspace=0.35)

    # row A
    ax = fig.add_subplot(gs[0, 0:2])
    ax.axis("off")
    ax.set_title("A · privileged uv (numbers only)", fontsize=10, loc="left")
    lines = ["marker   u      v"] + [f"  {i:2d}   {px_clean[i,0]:5.1f}  {px_clean[i,1]:5.1f}" for i in show]
    if int(vis.sum()) > 8:
        lines.append(f"  … ({int(vis.sum())} markers)")
    ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", family="monospace", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="0.7"))
    ax = fig.add_subplot(gs[0, 2:4])
    _q_bar(ax, q_sim, title="q from known uv", color="0.35")
    ax = fig.add_subplot(gs[0, 4])
    _force_callout(ax, lam_sim, F_true)

    # row B
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(img_m); ax.axis("off"); ax.set_title("B · camera RGB", fontsize=9)
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(mask, cmap="gray"); ax.axis("off"); ax.set_title("red threshold", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(img_m)
    if detected.size:
        ax.scatter(detected[:, 0], detected[:, 1], s=22, facecolors="none", edgecolors="lime", linewidths=1.1)
    for a, bpt in zip(duv_src, duv_dst):
        ax.annotate("", xy=(bpt[0], bpt[1]), xytext=(a[0], a[1]),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=0.8, mutation_scale=7))
    ax.axis("off"); ax.set_title(f"blobs + Δuv ({len(detected)})", fontsize=9)
    ax = fig.add_subplot(gs[1, 3])
    _q_bar(ax, q_marker, q_ref=q_sim, title="LS → q̂", color="tab:blue")
    ax = fig.add_subplot(gs[1, 4])
    _force_callout(ax, lam_marker, F_true)

    # row C
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(img_r); ax.axis("off"); ax.set_title("C · marker-free RGB", fontsize=9)
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.1, 0.22), 0.8, 0.56, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#dbe9ff", ec="0.3"))
    ax.text(0.5, 0.5, "ResNet-18\n→ q ∈ ℝ⁸", ha="center", va="center", fontsize=10)
    ax.set_title("network", fontsize=9)
    ax = fig.add_subplot(gs[2, 2:4])
    if q_nn is not None:
        _q_bar(ax, q_nn, q_ref=q_sim, title="network output q̂", color="tab:orange")
    ax = fig.add_subplot(gs[2, 4])
    if lam_nn is not None:
        _force_callout(ax, lam_nn, F_true)

    fig.suptitle(
        f"What each path receives  (same state: F={F_true:.2f} N, world camera {img_r.shape[1]}×{img_r.shape[0]})",
        fontsize=11,
    )
    dest = OUT / "09_information_paths.png"
    fig.savefig(dest, dpi=MPL_DPI, bbox_inches="tight")
    plt.close(fig)
    written.append(dest)
    print(f"  {dest.name}  (combined A/B/C)")

    return written


def _grasp_scene_panel() -> Path | None:
    """Fig.1 / 01: early approach_ee photo only (no HUD). Prefer live re-render."""
    try:
        import importlib.util

        path = ROOT / "scripts" / "render_figure1_grasp_scene.py"
        spec = importlib.util.spec_from_file_location("render_figure1_grasp_scene", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod.render(OUT / "02_grasp_scene.png")
    except Exception as e:  # noqa: BLE001
        print(f"  live render 02 failed ({e}); falling back to figure2 stage shots")
    # Fallback: early figure2 stages only (still may include HUD).
    for rel in (
        "figure2/world/stage_approach_ee.png",
        "figure2/world/stage_reach.png",
        "figure2/nn/stage_approach_ee.png",
    ):
        src = ROOT / "results" / rel
        if src.is_file():
            dst = OUT / "02_grasp_scene.png"
            shutil.copy2(src, dst)
            print(f"  {dst.name}  ←  {rel}  (fallback)")
            return dst
    print("  MISSING 02_grasp_scene.png (run scripts/render_figure1_grasp_scene.py)")
    return None


def _schematic_panel() -> Path | None:
    """Draft block diagram of the method (boxes to redraw in the final Figure 1)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except Exception as e:  # noqa: BLE001
        print(f"  skip schematic panel: {e}")
        return None

    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(-0.05, 4.6); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=9.5, ec="0.25"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, text=None, style="-|>", ls="-", color="0.2"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14, lw=1.3, color=color, linestyle=ls))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.16, text, ha="center", va="bottom", fontsize=8.5, color=color)

    vis, phys, ctrl = "#dbe9ff", "#e6f4e1", "#fff2cc"
    box(0.2, 1.7, 1.7, 1.2, "fixed RGB camera\n(world frame)", vis)
    box(2.6, 3.0, 2.7, 1.1, "markers (privileged, sim only)\npixels → LS with image Jacobian", vis, 8.6)
    box(2.6, 0.5, 2.7, 1.1, "marker-free RGB → ResNet-18\n(trained on paired frames)", vis, 8.6)
    box(6.0, 1.7, 1.3, 1.2, "reduced\ndeformation\nq ∈ ℝ⁸\nq = Φᵀu", "#eeeeee", 9)
    box(8.0, 1.7, 2.2, 1.2, "linear FEM + Galerkin ROM\nK_r(E, ν) = Φᵀ K(E, ν) Φ\nresidual search → contact c\nleast squares → force λ", phys, 8.6)
    box(10.7, 1.7, 1.15, 1.2, "grasp / lift\ncontroller\n(PI on λ̂)", ctrl, 8.8)
    arrow(1.9, 2.55, 2.6, 3.4); arrow(1.9, 2.05, 2.6, 1.2)
    arrow(5.3, 3.5, 6.0, 2.6, "q̂ (D)"); arrow(5.3, 1.0, 6.0, 1.8, "q̂ (E)")
    arrow(7.3, 2.3, 8.0, 2.3, "Kalman"); arrow(10.2, 2.3, 10.7, 2.3, "ĉ, λ̂")
    ax.add_patch(FancyArrowPatch((11.3, 1.7), (11.3, 0.12), arrowstyle="-", lw=1.1, color="0.35"))
    ax.add_patch(FancyArrowPatch((11.3, 0.12), (1.0, 0.12), arrowstyle="-", lw=1.1, color="0.35"))
    ax.add_patch(FancyArrowPatch((1.0, 0.12), (1.0, 1.7), arrowstyle="-|>", mutation_scale=14, lw=1.1, color="0.35"))
    ax.text(7.6, 0.18, "gripper opening / arm pose  →  new deformation seen by the camera", ha="center", va="bottom", fontsize=8.3, color="0.35")
    ax.text(4.0, 4.35, "frozen after training: Φ, marker geometry, ResNet", ha="center", fontsize=8.8, color="tab:blue")
    ax.text(9.1, 3.25, "material change (E, ν): rebuild K only – no retraining", ha="center", fontsize=8.8, color="tab:green")
    ax.text(3.95, 2.3, "distillation\ntarget", ha="center", va="center", fontsize=8, color="0.35", style="italic")
    ax.add_patch(FancyArrowPatch((3.95, 3.0), (3.95, 1.6), arrowstyle="-|>", mutation_scale=12, lw=1.0, color="0.45", linestyle="--"))
    fig.tight_layout()
    dest = OUT / "01_pipeline_schematic.png"
    fig.savefig(dest, dpi=MPL_DPI)
    plt.close(fig)
    print(f"  {dest.name}  (generated)")
    return dest


def _copy_panel(name: str, candidates: list[str]) -> Path | None:
    for rel in candidates:
        src = ETC / rel
        if src.is_file():
            dst = OUT / name
            shutil.copy2(src, dst)
            print(f"  {name}  ←  etc/{rel}")
            return dst
    print(f"  MISSING {name}  (tried {candidates})")
    return None


def _paired_panel() -> Path | None:
    """Marker-on / marker-off pair — same state, |u| colormap like Fig.1 / 01."""
    sys.path.insert(0, str(ROOT))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from src.grasp.factory import build_arm_grasp_sim
        from src.grasp.nn_vision import (
            GraspSceneAppearance,
            compose_grasp_scene,
            left_markers_world,
            two_finger_faces,
        )
        from src.rendering.synthetic_camera import RenderAppearance, SyntheticCameraRenderer
        from src.rom.pod import PODBasis
        from src.utils.config import load_config
        from src.visualization.camera import ObservationCamera
        from src.visualization.colormap import colormap
        from src.visualization.state import ObjectGeometry
        from src.vision.paired_dataset import PairedVisionDataset
    except Exception as e:  # noqa: BLE001
        print(f"  skip paired panel: {e}")
        return None

    path = ROOT / "data/processed/grasp_nn_top.npz"
    if not path.is_file():
        print("  MISSING 03_paired_marker_on_off.png (no grasp_nn_top.npz)")
        return None

    d = PairedVisionDataset.load(path)
    F = d.force_magnitude
    idx = int(np.argmin(np.abs(F - np.median(F[F > 0.5])))) if np.any(F > 0.5) else 0
    ap_s = d.appearance[idx]
    T_ee = np.asarray(ap_s["T_ee"], float)
    opening = float(ap_s["opening"])
    light = np.asarray(ap_s["light_position"], float)
    R = T_ee[:3, :3]
    F_n = float(F[idx])
    c_pos = np.asarray(d.contact_position[idx], float)
    q = np.asarray(d.q_sim[idx], float)

    gcfg = load_config("grasp")
    sim = build_arm_grasp_sim(gcfg, estimator_mode="world")
    mech, est = sim.grasp.mech, sim.estimator
    mesh, world = mech.model.mesh, mech.world
    basis: PODBasis = est.geo.basis
    u_contact = basis.reconstruct(q).reshape(-1, 3)
    u = u_contact + mech.gravity_displacement(R)
    obj_center = T_ee[:3, :3] @ np.array([c_pos[0], 0.0, c_pos[2]]) + T_ee[:3, 3]
    scene_ap = GraspSceneAppearance.from_dict(d.meta.get("scene_appearance"))
    table_c = np.array([0.28, 0.0, sim.table_z - 0.010])
    table_s = np.array([0.42, 0.24, 0.020])
    nodes, objs = compose_grasp_scene(
        mech, u, opening, T_ee, obj_center, R,
        table_center=table_c, table_size=table_s, appearance=scene_ap,
    )
    # Per-node |u| in world (same colormap as Taichi viewer / Fig.01).
    u_L = world.left_disp(u, T_ee)
    u_R = world.right_disp(u, T_ee)
    mag = np.linalg.norm(np.concatenate([u_L, u_R], axis=0), axis=1)
    vcol = colormap(mag, "viridis")

    markers = est.markers
    if markers is None:
        raise RuntimeError("estimator has no markers")
    mk_p, mk_n = left_markers_world(markers, mesh, u, opening, T_ee, world)

    # High-res paper pair (same aspect as world camera / NN frames).
    W, H = 1920, 1440
    wcfg = dict(d.meta.get("world_camera") or gcfg.get("estimator", {}).get("world_camera", {}))
    wcfg["image_width"], wcfg["image_height"] = W, H
    cam = ObservationCamera.from_config(wcfg)

    rend = SyntheticCameraRenderer(W, H)
    rend.set_mesh(2 * mesh.n_nodes, two_finger_faces(mesh))
    rap = RenderAppearance(
        finger_color=scene_ap.finger_color,
        background_color=scene_ap.background_color,
        ambient=scene_ap.ambient,
        light_position=light,
        light_color=scene_ap.light_color,
        marker_color=(1.0, 0.2, 0.1),
        marker_radius=0.0012,
        object_color=scene_ap.object_color,
        image_noise_std=0.0,
    )
    obj_geoms = [ObjectGeometry(v, f, color=tuple(c)) for v, f, c in objs]

    uv, z = cam.project(mk_p)
    facing = np.einsum("mj,mj->m", mk_n, cam.position[None, :] - mk_p) > 0.0
    vis = facing & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= W) & (uv[:, 1] >= 0) & (uv[:, 1] <= H)

    img_m = rend.render(
        nodes, cam, rap, markers_xyz=mk_p[vis], marker_normals=mk_n[vis],
        objects=obj_geoms, vertex_colors=vcol,
    )
    img_r = rend.render(nodes, cam, rap, objects=obj_geoms, vertex_colors=vcol)
    try:
        rend.destroy()
    except Exception:  # noqa: BLE001
        pass
    sim.estimator.close()

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].imshow(np.clip(img_m, 0, 1))
    ax[0].set_title("markers (privileged / teacher)")
    ax[0].axis("off")
    ax[1].imshow(np.clip(img_r, 0, 1))
    ax[1].set_title("marker-free RGB (deployment)")
    ax[1].axis("off")
    fig.suptitle(f"Paired views of identical state  (F={F_n:.2f} N, grasp_scene, |u| colormap)", fontsize=10)
    fig.tight_layout()
        dest = OUT / "03_paired_marker_on_off.png"
    fig.savefig(dest, dpi=MPL_DPI)
    plt.close(fig)
    print(f"  {dest.name}  ←  live render[{idx}]  {W}x{H}  |u|-viridis")
    return dest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"exporting Figure-1 panels → {OUT}")
    n = 0
    for name, cands in PANELS:
        if _copy_panel(name, cands) is not None:
            n += 1
    if _paired_panel() is not None:
        n += 1
    if _grasp_scene_panel() is not None:
        n += 1
    if _schematic_panel() is not None:
        n += 1
    for p in _information_path_panels():
        n += 1
    (OUT / "README.txt").write_text(README)
    print(f"done: {n} panels + README.txt")


if __name__ == "__main__":
    main()
