#!/usr/bin/env python
"""Regenerate Figure-1 *scene* panels with lateral (grasp) loading.

Early milestone dumps pressed the top face (−z). Stages D/E press the inner
face (−y). Figure 1 should match that — one story, no vertical/lateral caveat.

Writes ``results/etc/figures/lateral/``. Then run
``python scripts/export_figure1_assets.py`` to copy into ``results/figure1/``.

    python scripts/render_figure1_lateral.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.grasp.estimator import _inner_face_pod_snapshots  # noqa: E402
from src.rendering.markers import make_marker_grid  # noqa: E402
from src.rom.pod import compute_pod  # noqa: E402
from src.rom.reduced_mechanics import ReducedModel  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.figure_res import MPL_DPI, viewer_window_config  # noqa: E402
from src.visualization.camera import ObservationCamera  # noqa: E402
from src.visualization.taichi_viewer import TaichiViewer  # noqa: E402

OUT = ROOT / "results" / "etc" / "figures" / "lateral"
FORCE_N = 2.0
CONTACT_REL = [0.85, 1.0, 0.50]
# Viewer defaults: green = GT contact, magenta = estimated contact.
_COLOR_F_TRUE = (0.10, 0.90, 0.20)
_COLOR_F_EST = (0.95, 0.20, 0.85)
_COLOR_SELF_WEIGHT = (0.85, 0.78, 0.22)


def _self_weight_arrows(model: FingerFEMModel, *, n_x: int = 9, n_y: int = 1):
    """Distributed ρg arrows *under* the beam (not on the top — avoids marker clutter).

    Magnitudes are visual so shafts stay longer than the arrow heads when drawn
    next to a ~2 N contact force (auto-scaled together).
    """
    from src.visualization.state import ForceVector

    if not model.has_gravity:
        return []
    g = model.geometry
    gvec = np.asarray(model.gravity_world, float).reshape(3)
    gnorm = float(np.linalg.norm(gvec))
    if gnorm <= 0.0:
        return []
    direction = gvec / gnorm  # −z
    # Tip well below the bottom face so the whole shaft hangs under the beam.
    tip_gap = 2.2 * g.height
    xs = np.linspace(0.12, 0.90, n_x) * g.length
    ys = np.linspace(0.50, 0.50, n_y) * g.width if n_y == 1 else np.linspace(0.30, 0.70, n_y) * g.width
    # ~28 % of contact-arrow length — short vs F_true, but shaft stays visible.
    dF = 0.28 * float(FORCE_N)
    pts = [np.array([x, y, -tip_gap], float) for x in xs for y in ys]
    return [
        ForceVector(p, dF * direction, label="self-weight", color=_COLOR_SELF_WEIGHT)
        for p in pts
    ]


def _crop_whitespace(path: Path, *, pad: int = 12, thresh: float = 0.97) -> None:
    """Trim near-white margins so the finger fills the frame (PIL, preserves px)."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None  # paper screenshots can exceed PIL's default ~89 MP
    except Exception:  # noqa: BLE001
        return
    img = Image.open(path)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        return
    rgb = arr[..., :3]
    content = np.any(rgb < thresh, axis=2)
    if not np.any(content):
        return
    ys, xs = np.where(content)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, rgb.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, rgb.shape[1])
    img.crop((x0, y0, x1, y1)).save(path)


def _burn_force_legend(path: Path, entries: list[tuple[str, tuple[float, float, float]]]) -> None:
    """Draw a small color legend on a screenshot (GUI panel is off for fig1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    img = plt.imread(path)
    h, w = img.shape[:2]
    # Keep native pixel size: figsize inches = px / dpi.
    fig, ax = plt.subplots(figsize=(w / MPL_DPI, h / MPL_DPI), dpi=MPL_DPI)
    ax.imshow(img)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    # Light-card legend for white-background paper figures.
    box = FancyBboxPatch(
        (0.015, 0.02), 0.42, 0.045 + 0.055 * len(entries),
        transform=ax.transAxes, boxstyle="round,pad=0.01",
        facecolor=(1.0, 1.0, 1.0, 0.88), edgecolor=(0.25, 0.25, 0.25, 0.55),
        linewidth=0.8, zorder=10,
    )
    ax.add_patch(box)
    y0 = 0.02 + 0.055 * len(entries) - 0.01
    for i, (label, rgb) in enumerate(entries):
        y = y0 - 0.055 * i
        ax.plot([0.03, 0.07], [y, y], color=rgb, lw=3.5, solid_capstyle="round",
                transform=ax.transAxes, zorder=11)
        ax.text(0.085, y, label, color="0.1", fontsize=8, va="center", ha="left",
                transform=ax.transAxes, zorder=11)
    fig.savefig(path, dpi=MPL_DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def _obs_camera(geom, *, w: int = 640, h: int = 480) -> ObservationCamera:
    target = 0.5 * geom.size
    # Close enough that the frustum sits next to the finger instead of
    # swallowing the frame (the old 1.4 L standoff did that from yaw=8).
    pos = target + np.array([0.10 * geom.length, -0.65 * geom.length, 0.40 * geom.length])
    return ObservationCamera(
        position=pos, target=target, fov_y_deg=38.0,
        image_width=w, image_height=h,
        near=0.08 * geom.length, far=0.95 * geom.length, name="fig1_cam",
    )


def _frame_bounds(viewer: TaichiViewer, *, include_camera: bool) -> tuple[np.ndarray, np.ndarray]:
    """Tight bbox of the finger content (amplified mesh + ghost + extras).

    Observation-camera position is *not* included — it sits far away and would
    leave a huge empty margin. The frustum can still be drawn; cropping trims it.
    """
    del include_camera  # kept for call-site compatibility
    st = viewer.state
    assert st is not None
    alpha = float(viewer.options.amplification)
    display = st.displaced(alpha)
    bmin = np.minimum(st.nodes_original.min(axis=0), display.min(axis=0))
    bmax = np.maximum(st.nodes_original.max(axis=0), display.max(axis=0))
    if st.extra_forces:
        tips = np.asarray([f.origin for f in st.extra_forces], float).reshape(-1, 3)
        tips_d = viewer.renderer._display_offset(tips, alpha, False)
        pad = np.array([0.0, 0.0, 0.03 * float(viewer.renderer.diag)])
        bmin = np.minimum(bmin, tips_d.min(axis=0) - pad)
        bmax = np.maximum(bmax, tips_d.max(axis=0))
    # Contact / estimated force arrows also stick out of the mesh.
    for f in st.all_forces():
        if f[0] == "extra":
            continue
        origin = viewer.renderer._display_offset(f[1].origin[None, :], alpha, False)[0]
        # tip-anchored: shaft extends opposite to force direction
        extent = 0.35 * float(viewer.renderer.diag)  # generous; crop cleans up
        tip = origin
        tail = origin - extent * f[1].direction
        pts = np.vstack([tip, tail])
        bmin = np.minimum(bmin, pts.min(axis=0))
        bmax = np.maximum(bmax, pts.max(axis=0))
    return bmin, bmax


def _shot(
    viewer: TaichiViewer,
    path: Path,
    *,
    yaw: float = -55.0,
    pitch: float = 22.0,
    include_camera: bool = False,
    legend: list[tuple[str, tuple[float, float, float]]] | None = None,
) -> Path:
    """Screenshot from the default 3/4 view (finger fills the frame).

    Looking from the tip (yaw≈0) foreshortens the 10 cm finger into a postage
    stamp and makes the observation-camera frustum dominate the frame — that
    was the hex-mesh Fig.1 regression. yaw=-55 / pitch=22 matches the viewer
    config and the pre-hex panels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    st = viewer.state
    if st is not None:
        bmin, bmax = _frame_bounds(viewer, include_camera=include_camera)
        # Tight framing on the amplified geometry (not the undeformed finger alone).
        viewer.orbit.fit(bmin, bmax, 1.02)
    viewer.orbit.yaw_deg = float(yaw)
    viewer.orbit.pitch_deg = float(pitch)
    viewer.orbit.fov_deg = 32.0
    viewer.options.show_wireframe = True
    viewer.options.show_gui_panel = False
    viewer.options.show_observation_camera = bool(include_camera)
    viewer.save_screenshot(path)
    _crop_whitespace(path)
    if legend:
        _burn_force_legend(path, legend)
    return path


def _configure_viewer(viewer: TaichiViewer) -> None:
    viewer.options.amplification = 10.0  # same visual alpha as the interactive viewer
    viewer.options.color_mode = "displacement"
    viewer.options.show_keypoints = True
    viewer.options.mode = "overlay"  # undeformed ghost + deformed solid
    viewer.options.show_wireframe = True
    viewer.options.show_gui_panel = False
    # Paper-style: white background, black undeformed outline.
    viewer.cfg["background_color"] = [1.0, 1.0, 1.0]
    viewer.canvas.set_background_color((1.0, 1.0, 1.0))
    viewer.renderer.colors["original_wireframe"] = (0.08, 0.08, 0.08)
    # Soft lighting so white bg does not wash the mesh out.
    viewer.renderer.lighting["ambient"] = [0.55, 0.55, 0.55]


def _plot_pod_modes(path: Path, basis, nodes, faces, geom) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    r_show = min(4, basis.r)
    fig = plt.figure(figsize=(11.0, 2.9))
    amp = 0.012
    for k in range(r_show):
        ax = fig.add_subplot(1, r_show, k + 1, projection="3d")
        phi = basis.Phi[:, k].reshape(-1, 3)
        tip = np.argmax(np.linalg.norm(phi, axis=1))
        if phi[tip, 1] > 0:
            phi = -phi
        scale = amp / max(float(np.linalg.norm(phi, axis=1).max()), 1e-12)
        xyz = nodes + scale * phi
        tris = xyz[faces]
        poly = Poly3DCollection(tris, alpha=0.88, linewidths=0.12, edgecolor="0.4")
        poly.set_array(np.linalg.norm(phi, axis=1)[faces].mean(axis=1))
        poly.set_cmap("viridis")
        ax.add_collection3d(poly)
        ax.set_xlim(0, geom.length)
        ax.set_ylim(0, geom.width)
        ax.set_zlim(0, geom.height)
        ax.set_box_aspect((geom.length, geom.width, geom.height))
        ax.view_init(elev=24, azim=-58)
        ax.set_title(f"mode {k + 1}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    fig.suptitle("POD modes from inner-face contact snapshots (same Φ as Stages D/E)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=MPL_DPI)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gcfg = load_config("grasp")
    fem_cfg = dict(gcfg["fem"])
    fem_cfg["loading"] = {
        "contact_position_rel": CONTACT_REL,
        "force_magnitude": FORCE_N,
        "force_direction": [0.0, -1.0, 0.0],
    }

    t0 = time.time()
    model = FingerFEMModel.from_config(fem_cfg)
    model.restrict_contact_surface("side_pos_y")
    point = model.geometry.point_from_relative(CONTACT_REL)
    result, contact = model.solve_normal_contact(point, FORCE_N)
    F_true = -FORCE_N * np.asarray(contact.normal, float)
    # Self-weight (fem.gravity in grasp.yaml) by superposition; upright finger → −z sag.
    u_g = model.gravity_displacement()
    u_contact = result.u.copy()
    result.u = u_contact + u_g
    print(f"lateral FEM: n={contact.normal}, |u_contact|_max={1e3 * np.linalg.norm(u_contact, axis=1).max():.3f} mm, "
          f"self-weight sag |u_g|_max={1e3 * np.linalg.norm(u_g, axis=1).max():.4f} mm "
          f"({'on' if model.has_gravity else 'off'}) ({time.time() - t0:.1f}s)")

    cam = _obs_camera(model.geometry)
    F_est = 0.92 * F_true + 0.05 * FORCE_N * np.array([0.25, 0.0, 0.2])
    markers = make_marker_grid(
        model.mesh, model.geometry, face="top",
        nx=int(gcfg.get("estimator", {}).get("marker_nx", 8)),
        ny=int(gcfg.get("estimator", {}).get("marker_ny", 3)),
    )
    kp = markers.positions(model.mesh)

    # ---- 3D viewer panels (GGUI) -----------------------------------------------------
    # (observation-camera RGB panel dropped — overlaps Fig.1 / 02–03)
    viewer = TaichiViewer(show_window=False, title="fig1 lateral", config=viewer_window_config())
    _configure_viewer(viewer)
    # Small heads + thin shafts so self-weight arrows show a visible tail (not head-only).
    viewer.renderer.sizes["arrow_shaft_radius_rel"] = 0.004
    viewer.renderer.sizes["arrow_head_radius_rel"] = 0.009
    viewer.renderer.sizes["arrow_head_length_rel"] = 0.018
    viewer.renderer.force_cfg["min_length_rel"] = 0.01
    sw_arrows = _self_weight_arrows(model)
    legend_gt_est_sw = [
        ("F_true — GT contact force", _COLOR_F_TRUE),
        ("F_est — estimated contact force", _COLOR_F_EST),
        ("self-weight (ρ g)", _COLOR_SELF_WEIGHT),
    ]

    # 01 — sole 3D scene panel (camera + GT/est + self-weight). No near-duplicate clones.
    state = model.visualization_state(
        result, contact, F_true, estimated_force=F_est, with_object=True, observation_camera=cam,
        info={"contact": "inner face", "F_true": f"{FORCE_N:.1f} N", "F_est": f"{np.linalg.norm(F_est):.2f} N"},
    )
    state.keypoints = kp
    state.keypoints_visible = markers.visibility(cam, model.mesh, result.u)
    state.extra_forces = list(sw_arrows)
    viewer.set_state(state)
    print(f"  → {_shot(viewer, OUT / '01_scene_overlay_lateral.png', include_camera=True, legend=legend_gt_est_sw).relative_to(ROOT)}")

    # Self-weight-only panel dropped from Fig.1 (ρg already shown on finger overlay).

    # POD / ROM plots (no extra GGUI)
    U = _inner_face_pod_snapshots(model, n_x=6, n_z=3)
    basis = compute_pod(U).truncate(8)
    print(f"  → {_plot_pod_modes(OUT / '06_pod_modes_lateral.png', basis, model.mesh.nodes, model.mesh.surface_faces, model.geometry).relative_to(ROOT)}")

    # 07 — Left: λ = grasp force_target, sweep x/L, tip |δ| vs rank.
    # Right: fixed tip |δ|* (same deformation), sweep x/L∈[0.55,1], λ needed (FEM vs ROM @ grasp r).
    ranks = (1, 2, 4, 8)
    r_force = int(gcfg.get("estimator", {}).get("q_modes", max(ranks)))
    if r_force not in ranks:
        ranks = tuple(sorted(set(ranks) | {r_force}))
    x_grasp = float(gcfg.get("contact", {}).get("position_rel", CONTACT_REL)[0])
    lam_target = float(gcfg.get("control", {}).get("force_target", 2.2))
    F_fixed = float(lam_target)  # left: grasp hold/lift force, vary r only
    roms = {r: ReducedModel.from_fem(model.fem, basis.truncate(r), model.contact) for r in ranks}
    rom = roms[r_force]
    x_grid = np.linspace(0.55, 1.0, 10)
    z_mid = 0.50
    tip = model.mesh.nodes_on_plane(0, model.geometry.length)  # beam tip face, independent of r

    def _tip_mm(u, normal):
        d = np.asarray(normal, float)
        d /= np.linalg.norm(d)
        return 1e3 * abs(float(np.mean(u[tip] @ d)))

    xs, tip_fem = [], []
    tip_rom = {r: [] for r in ranks}
    for x in x_grid:
        p = model.geometry.point_from_relative([float(x), 1.0, z_mid])
        res, c = model.solve_normal_contact(p, F_fixed)
        tip_fem.append(_tip_mm(res.u, c.normal))
        for r, rom_r in roms.items():
            tip_rom[r].append(_tip_mm(rom_r.solve(c, -F_fixed * c.normal), c.normal))
        xs.append(float(x))
    xs = np.asarray(xs)
    tip_fem = np.asarray(tip_fem)
    tip_rom = {r: np.asarray(v) for r, v in tip_rom.items()}
    tip_err = {r: 100 * np.abs(tip_rom[r] - tip_fem) / np.maximum(tip_fem, 1e-30) for r in ranks}

    # Fixed tip deformation δ*: grasp operating point (force_target @ grasp x/L).
    p_g = model.geometry.point_from_relative([x_grasp, 1.0, z_mid])
    res_g, c_g = model.solve_normal_contact(p_g, F_fixed)
    delta_star_mm = _tip_mm(res_g.u, c_g.normal)

    lam_fem, lam_rom, tip_chk = [], [], []
    for x in x_grid:
        p = model.geometry.point_from_relative([float(x), 1.0, z_mid])
        # unit-load FEM → scale λ so tip |δ| = δ*
        res1, c = model.solve_normal_contact(p, 1.0)
        tip1 = _tip_mm(res1.u, c.normal)
        lam = float(delta_star_mm / max(tip1, 1e-30))
        res, c = model.solve_normal_contact(p, lam)
        # ROM: same — unit ROM tip, scale to δ*
        tip1_r = _tip_mm(rom.solve(c, -1.0 * c.normal), c.normal)
        lam_r = float(delta_star_mm / max(tip1_r, 1e-30))
        lam_fem.append(lam)
        lam_rom.append(lam_r)
        tip_chk.append(_tip_mm(res.u, c.normal))
    lam_fem, lam_rom = np.asarray(lam_fem), np.asarray(lam_rom)
    tip_chk = np.asarray(tip_chk)
    rel_f = 100 * np.abs(lam_rom - lam_fem) / np.maximum(lam_fem, 1e-30)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
    h0, = ax[0].plot(xs, tip_fem, "-", color="0.15", lw=2.2, label="FEM")
    rom_handles = [h0]
    rom_labels = ["FEM"]
    cmap = plt.cm.Blues
    for i, r in enumerate(ranks):
        color = cmap(0.35 + 0.55 * i / max(len(ranks) - 1, 1))
        h, = ax[0].plot(xs, tip_rom[r], "o--" if r != r_force else "s-", color=color,
                        ms=6.5 if r == r_force else 5.5, lw=2.0 if r == r_force else 1.2,
                        mfc="none", mec=color, mew=1.4 if r == r_force else 1.3)
        rom_handles.append(h)
        rom_labels.append(f"r={r}" + (" (grasp)" if r == r_force else ""))
    ax[0].axvline(x_grasp, color="0.45", ls=":", lw=1.1, zorder=0)
    ax[0].legend(rom_handles, rom_labels, fontsize=7.5, loc="upper left", ncol=3, columnspacing=0.9)
    ax[0].set(xlabel="contact x / L", ylabel="tip |δ| [mm]  (beam end)",
              title=f"Tip deflection vs rank  (grasp hold force = {F_fixed:g} N)")
    ax[0].grid(True, alpha=0.3)
    y0, y1 = float(tip_fem.min()), float(tip_fem.max())
    ax[0].set_ylim(y0 * 0.92, y1 * 1.08)

    h1, = ax[1].plot(xs, lam_fem, "-", color="C1", lw=2.2)
    h2, = ax[1].plot(xs, lam_rom, "s--", color="C1", ms=6.5, lw=1.4,
                      mfc="none", mec="C1", mew=1.4)
    ax[1].axvline(x_grasp, color="0.45", ls=":", lw=1.1, zorder=0)
    ax[1].legend([h1, h2], ["FEM λ", f"ROM λ (r={r_force})"], fontsize=7.5, loc="upper right")
    ax[1].set(xlabel="contact x / L", ylabel="force λ [N]  for fixed tip deflection",
              title=f"Force for fixed tip deflection  (grasp r = {r_force})")
    ax[1].grid(True, alpha=0.3)
    yf0, yf1 = float(min(lam_fem.min(), lam_rom.min())), float(max(lam_fem.max(), lam_rom.max()))
    ax[1].set_ylim(yf0 * 0.88, yf1 * 1.08)

    fig.suptitle(
        f"Galerkin ROM — left: ranks at grasp hold force; "
        f"right: fixed tip deflection at grasp r={r_force}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    # Same plain-English footnote style under both panels (no in-plot boxes).
    ax[0].text(
        0.5, -0.24,
        f"Grasp operating point: force_target, x/L={x_grasp:g}   ·   "
        + " | ".join(f"r={r}: {np.median(tip_err[r]):.1f}%" for r in ranks),
        transform=ax[0].transAxes, ha="center", va="top", fontsize=7, color="0.25",
    )
    ax[1].text(
        0.5, -0.24,
        f"Grasp operating point: force_target, x/L={x_grasp:g}   ·   "
        f"|λ_ROM−λ_FEM|/λ median {np.median(rel_f):.2f} %",
        transform=ax[1].transAxes, ha="center", va="top", fontsize=7, color="0.25",
    )
    p07 = OUT / "07_rom_vs_full_lateral.png"
    fig.savefig(p07, dpi=MPL_DPI)
    plt.close(fig)
    print(f"  → {p07.relative_to(ROOT)}  (δ*={delta_star_mm:.3f} mm, grasp r={r_force}, "
          f"x/L={x_grasp:g}, med |λ| {np.median(rel_f):.2f}%, tipchk {tip_chk.mean():.3f} mm)")

    # Dropped from Fig.1 export: force-recovery ±gravity bars (identical at this OP),
    # pipeline_gt / Φq 3D clones (overlap finger overlay).

    try:
        viewer.destroy()
    except Exception:  # noqa: BLE001
        pass
    (OUT / "README.txt").write_text(
        "Figure-1 lateral (inner-face) panels — same loading as Stages D/E.\n"
        "Contact: side_pos_y (−y). Force: 2 N. Markers: top face.\n"
        f"Self-weight: {'on (fem.gravity, superposed; q compensated by Φᵀu_g)' if model.has_gravity else 'off'}.\n\n"
        "python scripts/render_figure1_lateral.py\n"
        "python scripts/export_figure1_assets.py\n"
    )
    print(f"done → {OUT}")


if __name__ == "__main__":
    main()
