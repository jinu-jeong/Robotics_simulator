"""Force strip chart for the world-camera grasp inset (no interactive window)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def display_force_estimate(
    lam_meas: np.ndarray,
    *,
    ema: float = 0.18,
    active_eps: float = 0.05,
) -> np.ndarray:
    """Causal EMA of the estimate for plots only — does not change the controller.

    Uses the estimate alone (never the ground truth). While ``lam_meas`` is
    still gated (~0), the display stays at 0.
    """
    ym = np.asarray(lam_meas, float).reshape(-1)
    ema = float(np.clip(ema, 0.0, 1.0))
    out = np.zeros_like(ym)
    smooth = 0.0
    active = False
    for i, m in enumerate(ym):
        if not active:
            if m < active_eps:
                continue
            active = True
            smooth = float(m)
        else:
            smooth = (1.0 - ema) * smooth + ema * float(m)
        out[i] = smooth
    return out


def render_force_strip(
    log: Mapping[str, Any],
    *,
    width: int = 420,
    height: int = 200,
    beautify: bool = True,
    ema: float = 0.18,
) -> np.ndarray:
    """Render GT vs estimate force history to an RGB float image in ``[0, 1]``."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    t = np.asarray(log.get("t", []), float)
    fig = Figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#101214")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#16181c")
    if len(t) > 0:
        yt = np.asarray(log["lam_true"], float)
        ym = np.asarray(log["lam_meas"], float)
        if beautify:
            ym = display_force_estimate(ym, ema=ema)
        ax.plot(t, yt, color="#4C9BE8", lw=2.0, label="GT")
        ax.plot(t, ym, color="#F0A202", lw=2.0, ls="--", label="estimate")
        tgt = float(log["force_target"][-1])
        hold = float(log["hold_force"][-1])
        ax.axhline(tgt, color="#E8E8E8", ls="--", lw=1.0, label="target")
        ax.axhline(hold, color="#E45756", ls=":", lw=1.2, label="hold")
        ymin = float(min(0.0, yt.min(), ym.min(), hold)) - 0.15
        ymax = float(max(yt.max(), ym.max(), tgt, hold)) + 0.35
        if ymax - ymin < 1.0:
            ymax = ymin + 1.0
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(float(t[0]), float(t[-1]) + 1e-6)
    ax.set_xlabel("t [s]", color="0.85", fontsize=8)
    ax.set_ylabel("force [N]", color="0.85", fontsize=8)
    ax.set_title("GT vs estimate", color="0.95", fontsize=9, pad=4)
    ax.tick_params(colors="0.75", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("0.45")
    ax.grid(alpha=0.25, color="0.6")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.35, labelcolor="0.9")
    fig.tight_layout(pad=0.35)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[:, :, :3].astype(np.float32) / 255.0
    fig.clear()
    return buf


def stack_camera_and_force(camera_rgb: np.ndarray, force_rgb: np.ndarray) -> np.ndarray:
    """Stack observation inset above the force strip (same width)."""
    cam = np.asarray(camera_rgb, np.float32)
    frc = np.asarray(force_rgb, np.float32)
    if cam.max() > 1.5:
        cam = cam / 255.0
    if frc.max() > 1.5:
        frc = frc / 255.0
    w = max(cam.shape[1], frc.shape[1])

    def _pad(img: np.ndarray, w: int) -> np.ndarray:
        if img.shape[1] == w:
            return img
        out = np.zeros((img.shape[0], w, 3), np.float32)
        x0 = (w - img.shape[1]) // 2
        out[:, x0 : x0 + img.shape[1]] = img
        return out

    return np.concatenate([_pad(cam, w), _pad(frc, w)], axis=0)
