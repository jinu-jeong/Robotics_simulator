"""Plot finger deflection / normal-force logs for sim vs CV validation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Optional CV columns (filled later by vision pipeline)
_CV_DEFL = ("left_deflection_cv_mm", "right_deflection_cv_mm")
_CV_FN = ("left_fn_cv_n", "right_fn_cv_n")


def _load_log(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty log: {path}")

    out: dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = []
        for row in rows:
            raw = row[key].strip()
            if raw == "":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(raw))
                except ValueError:
                    vals.append(np.nan)
        out[key] = np.asarray(vals, dtype=float)
    out["phase"] = np.array([row["phase"] for row in rows])
    return out


def _has_cv(data: dict[str, np.ndarray]) -> bool:
    if not all(k in data for k in (*_CV_DEFL, *_CV_FN)):
        return False
    vals = np.concatenate([data[k] for k in (*_CV_DEFL, *_CV_FN)])
    return bool(np.nanmax(np.abs(vals)) > 1e-6)


def _phase_spans(t: np.ndarray, phases: np.ndarray) -> list[tuple[str, float, float]]:
    spans: list[tuple[str, float, float]] = []
    if len(t) == 0:
        return spans
    start = 0
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            spans.append((str(phases[start]), float(t[start]), float(t[i - 1])))
            start = i
    spans.append((str(phases[start]), float(t[start]), float(t[-1])))
    return spans


def _y_lim_from(*arrays: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    """Shared axis limits covering all finite values in ``arrays``."""
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return lo - pad * span, hi + pad * span


def plot_finger_force_log(
    csv_path: Path | str,
    out_path: Path | str | None = None,
    *,
    show: bool = False,
) -> Path:
    """Plot sim vs CV validation, grouped by quantity.

    Layout (2×1):
      (1) deflection — left/right sim + CV overlay
      (2) normal force — left/right sim + CV overlay

    Sim = solid, CV = dashed (same color per finger). Returns path to PNG.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for finger force plots "
            "(conda: pip install matplotlib)"
        ) from exc

    csv_path = Path(csv_path)
    if out_path is None:
        out_path = csv_path.with_name("finger_force_log.png")
    out_path = Path(out_path)

    data = _load_log(csv_path)
    t = data["t"]
    spans = _phase_spans(t, data["phase"])
    obj = data.get("object_mode", np.array(["?"]))[0]
    fing = data.get("finger_mode", np.array(["?"]))[0]
    cv_ok = _has_cv(data)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        f"Finger validation — object={obj}  finger={fing}\n"
        f"(sim red solid, CV blue dashed)",
        fontsize=12,
    )

    phase_colors = {
        "HOME": "#eef2ff", "FOLD": "#f0fdf4", "APPROACH": "#fff7ed",
        "NEAR": "#fef9c3", "CLOSE": "#fce7f3", "LIFT": "#e0e7ff",
        "RELEASE": "#f3f4f6",
    }
    # Source color: sim = red, CV = blue. Shade = left / right finger.
    _SIM_L, _SIM_R = "#dc2626", "#9f1239"
    _CV_L, _CV_R = "#2563eb", "#1e3a8a"

    def _shade_phases(ax) -> None:
        for name, t0, t1 in spans:
            ax.axvspan(t0, t1, color=phase_colors.get(name, "#f9fafb"), alpha=0.55)

    # ── (1) deflection: sim + CV ──
    ax = axes[0]
    _shade_phases(ax)
    ax.plot(t, data["left_deflection_mm"], label="left sim", color=_SIM_L, lw=1.4)
    ax.plot(t, data["right_deflection_mm"], label="right sim", color=_SIM_R, lw=1.4)
    if cv_ok:
        ax.plot(t, data[_CV_DEFL[0]], "--", label="left CV", color=_CV_L, lw=1.4)
        ax.plot(t, data[_CV_DEFL[1]], "--", label="right CV", color=_CV_R, lw=1.4)
    ax.set_ylabel("signed deflection [mm]")
    ax.set_title("① Deflection (sim vs CV, + = closing)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    defl_series = [data["left_deflection_mm"], data["right_deflection_mm"]]
    if cv_ok:
        defl_series.extend([data[_CV_DEFL[0]], data[_CV_DEFL[1]]])
    ax.set_ylim(_y_lim_from(*defl_series))

    # ── (2) normal force: sim + CV ──
    ax = axes[1]
    _shade_phases(ax)
    ax.plot(t, data["left_fn_n"], label="left sim", color=_SIM_L, lw=1.4)
    ax.plot(t, data["right_fn_n"], label="right sim", color=_SIM_R, lw=1.4)
    if cv_ok:
        ax.plot(t, data[_CV_FN[0]], "--", label="left CV→f", color=_CV_L, lw=1.4)
        ax.plot(t, data[_CV_FN[1]], "--", label="right CV→f", color=_CV_R, lw=1.4)
    ax.set_ylabel("normal force [N]")
    ax.set_xlabel("time [s]")
    ax.set_title("② Normal force (sim vs CV via u(s)→K_r, contact δ)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fn_series = [data["left_fn_n"], data["right_fn_n"]]
    if cv_ok:
        fn_series.extend([data[_CV_FN[0]], data[_CV_FN[1]]])
    ax.set_ylim(_y_lim_from(*fn_series))

    phase_legend = [
        Patch(facecolor=phase_colors.get(n, "#f9fafb"), alpha=0.55, label=n)
        for n, _, _ in spans
    ]
    if phase_legend:
        fig.legend(
            handles=phase_legend, loc="lower center", ncol=min(7, len(phase_legend)),
            fontsize=7, frameon=False,
        )

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_deflection_profiles(
    csv_path: Path | str,
    out_path: Path | str | None = None,
    *,
    phase: str = "LIFT",
    show: bool = False,
) -> Path:
    """Overlay sim vs CV deflection vs link-X for a phase (mean over frames).

    Layout: left / right finger side by side. Sim = red solid, CV = blue dashed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for deflection profile plots"
        ) from exc

    csv_path = Path(csv_path)
    if out_path is None:
        out_path = csv_path.with_name("deflection_profile.png")
    out_path = Path(out_path)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty profile log: {csv_path}")

    phase_rows = [r for r in rows if r.get("phase") == phase]
    if not phase_rows:
        # Fall back to any frames that have both sources
        phase_rows = rows
        phase_label = "all phases"
    else:
        phase_label = phase

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fig.suptitle(
        f"Deflection vs X — {phase_label} (mean over frames)\n"
        f"sim red solid, CV blue dashed",
        fontsize=12,
    )

    for ax, finger in zip(axes, ("left", "right")):
        sub = [r for r in phase_rows if r.get("finger") == finger]
        for source, color, style, label in (
            ("sim", "#dc2626", "-", f"{finger} sim"),
            ("cv", "#2563eb", "--", f"{finger} CV"),
        ):
            src = [r for r in sub if r.get("source") == source]
            if not src:
                continue
            # Bin by rounded s so frames share stations, then mean u
            buckets: dict[float, list[tuple[float, float]]] = {}
            for r in src:
                s = round(float(r["s"]), 4)
                buckets.setdefault(s, []).append(
                    (float(r["x_m"]), float(r["u_mm"]))
                )
            ss = sorted(buckets)
            xs = np.array([np.mean([p[0] for p in buckets[s]]) for s in ss])
            us = np.array([np.mean([p[1] for p in buckets[s]]) for s in ss])
            ax.plot(xs * 1e3, us, style, color=color, lw=1.6, label=label)

        ax.set_xlabel("link X [mm]")
        ax.set_title(f"{finger} finger")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.axhline(0.0, color="#9ca3af", lw=0.8)

    axes[0].set_ylabel("closing deflection u [mm]")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


_STATIONS = (
    (0.0, "x = 0 (root)"),
    (0.5, "x = 0.5 L (mid)"),
    (1.0, "x = L (tip)"),
)


def _station_series_from_profiles(
    rows: list[dict],
    *,
    finger: str,
    source: str,
    s_target: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate deflection at ``s_target`` vs time from profile CSV rows.

    Returns ``(t, u_mm, phase)`` sorted by time. Missing frames are skipped.
    """
    by_frame: dict[int, list[dict]] = {}
    for r in rows:
        if r.get("finger") != finger or r.get("source") != source:
            continue
        fr = int(float(r["frame"]))
        by_frame.setdefault(fr, []).append(r)
    ts: list[float] = []
    us: list[float] = []
    phases: list[str] = []
    for fr in sorted(by_frame):
        grp = by_frame[fr]
        s = np.asarray([float(r["s"]) for r in grp], dtype=float)
        u = np.asarray([float(r["u_mm"]) for r in grp], dtype=float)
        if s.size == 0:
            continue
        order = np.argsort(s)
        s, u = s[order], u[order]
        # Clamp query to available span (CV often stops short of s=1)
        s_q = float(np.clip(s_target, float(s[0]), float(s[-1])))
        ts.append(float(grp[0]["t"]))
        us.append(float(np.interp(s_q, s, u)))
        phases.append(str(grp[0].get("phase", "")))
    return (
        np.asarray(ts, dtype=float),
        np.asarray(us, dtype=float),
        np.asarray(phases),
    )


def plot_station_trajectories(
    csv_path: Path | str,
    out_path: Path | str | None = None,
    *,
    title: str | None = None,
    show: bool = False,
) -> Path:
    """Time histories of deflection at x=0, 0.5L, L — sim vs CV.

    Layout (3×2): rows = stations (root / mid / tip), columns = left / right.
    Sim = red solid, CV = blue dashed. Phase bands shaded like force log.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for station trajectory plots"
        ) from exc

    csv_path = Path(csv_path)
    if out_path is None:
        out_path = csv_path.with_name("deflection_stations_vs_t.png")
    out_path = Path(out_path)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty profile log: {csv_path}")

    # Shared time axis from any series for phase spans
    t_all = np.asarray(sorted({float(r["t"]) for r in rows}), dtype=float)
    # Reconstruct phase timeline (one phase per unique t from sim-left)
    phase_by_t: dict[float, str] = {}
    for r in rows:
        if r.get("finger") == "left" and r.get("source") == "sim":
            phase_by_t[float(r["t"])] = str(r.get("phase", ""))
    t_phase = np.asarray(sorted(phase_by_t), dtype=float)
    phases = np.asarray([phase_by_t[t] for t in t_phase])
    spans = _phase_spans(t_phase, phases) if t_phase.size else []

    phase_colors = {
        "HOME": "#eef2ff", "FOLD": "#f0fdf4", "APPROACH": "#fff7ed",
        "NEAR": "#fef9c3", "CLOSE": "#fce7f3", "LIFT": "#e0e7ff",
        "RELEASE": "#f3f4f6",
    }
    _SIM, _CV = "#dc2626", "#2563eb"

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig.suptitle(
        title
        or (
            "Deflection vs time at x = 0, 0.5L, L\n"
            "(sim red solid, CV blue dashed)"
        ),
        fontsize=12,
    )

    all_u: list[np.ndarray] = []
    for row_i, (s_tgt, s_label) in enumerate(_STATIONS):
        for col_i, finger in enumerate(("left", "right")):
            ax = axes[row_i, col_i]
            for name, t0, t1 in spans:
                ax.axvspan(
                    t0, t1, color=phase_colors.get(name, "#f9fafb"), alpha=0.55,
                )
            t_s, u_s, _ = _station_series_from_profiles(
                rows, finger=finger, source="sim", s_target=s_tgt,
            )
            t_c, u_c, _ = _station_series_from_profiles(
                rows, finger=finger, source="cv", s_target=s_tgt,
            )
            if t_s.size:
                ax.plot(t_s, u_s, "-", color=_SIM, lw=1.4, label="sim")
                all_u.append(u_s)
            if t_c.size:
                ax.plot(t_c, u_c, "--", color=_CV, lw=1.4, label="CV")
                all_u.append(u_c)
            ax.set_ylabel("u [mm]")
            ax.grid(True, alpha=0.3)
            ax.axhline(0.0, color="#9ca3af", lw=0.8)
            if row_i == 0:
                ax.set_title(f"{finger} — {s_label}")
            else:
                ax.set_title(s_label, fontsize=10)
            if row_i == 0 and col_i == 0:
                ax.legend(loc="upper left", fontsize=8)
            if row_i == 2:
                ax.set_xlabel("time [s]")

    if all_u:
        ymin, ymax = _y_lim_from(*all_u)
        for ax in axes.ravel():
            ax.set_ylim(ymin, ymax)

    phase_legend = [
        Patch(facecolor=phase_colors.get(n, "#f9fafb"), alpha=0.55, label=n)
        for n, _, _ in spans
    ]
    if phase_legend:
        fig.legend(
            handles=phase_legend,
            loc="lower center",
            ncol=min(7, len(phase_legend)),
            fontsize=7,
            frameon=False,
        )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_station_trajectories_grid(
    run_dirs: list[Path | str],
    out_path: Path | str,
    *,
    finger: str = "left",
    show: bool = False,
) -> Path:
    """Compare tip/mid/root time series across several ``grasp_d*`` runs.

    One column per depth, three rows (root / mid / tip). Sim solid, CV dashed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for station trajectory plots"
        ) from exc

    dirs = [Path(d) for d in run_dirs]
    dirs = [d for d in dirs if (d / "frames" / "deflection_profiles.csv").is_file()]
    if not dirs:
        raise ValueError("no run dirs with frames/deflection_profiles.csv")

    n = len(dirs)
    fig, axes = plt.subplots(3, n, figsize=(2.4 * n + 1.5, 8), sharey="row")
    if n == 1:
        axes = axes.reshape(3, 1)
    fig.suptitle(
        f"{finger} finger — u(t) at x=0, 0.5L, L across grasp depths\n"
        f"(sim red solid, CV blue dashed)",
        fontsize=12,
    )
    _SIM, _CV = "#dc2626", "#2563eb"

    for col, d in enumerate(dirs):
        csv_path = d / "frames" / "deflection_profiles.csv"
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        depth_tag = d.name.replace("grasp_", "")
        for row_i, (s_tgt, s_label) in enumerate(_STATIONS):
            ax = axes[row_i, col]
            t_s, u_s, _ = _station_series_from_profiles(
                rows, finger=finger, source="sim", s_target=s_tgt,
            )
            t_c, u_c, _ = _station_series_from_profiles(
                rows, finger=finger, source="cv", s_target=s_tgt,
            )
            if t_s.size:
                ax.plot(t_s, u_s, "-", color=_SIM, lw=1.1, label="sim")
            if t_c.size:
                ax.plot(t_c, u_c, "--", color=_CV, lw=1.1, label="CV")
            ax.grid(True, alpha=0.3)
            ax.axhline(0.0, color="#9ca3af", lw=0.7)
            if row_i == 0:
                ax.set_title(depth_tag, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{s_label}\nu [mm]", fontsize=8)
            if row_i == 2:
                ax.set_xlabel("t [s]", fontsize=8)
            if row_i == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def summarize_grasp_depth_lifts(
    run_dirs: list[Path | str],
    *,
    finger: str = "left",
) -> list[dict[str, float | str]]:
    """LIFT-phase medians of tip deflection / force per ``grasp_d*`` run."""
    rows_out: list[dict[str, float | str]] = []
    tip_key = f"{finger}_deflection_mm"
    tip_cv = f"{finger}_deflection_cv_mm"
    fn_key = f"{finger}_fn_n"
    fn_cv = f"{finger}_fn_cv_n"
    for d in sorted(Path(p) for p in run_dirs):
        log = d / "finger_force_log.csv"
        if not log.is_file():
            continue
        data = list(csv.DictReader(log.open()))
        lift = [r for r in data if r.get("phase") == "LIFT"]
        if not lift:
            continue
        tag = d.name.replace("grasp_", "")
        try:
            depth = float(tag.lstrip("d"))
        except ValueError:
            depth = float("nan")

        def med(key: str) -> float:
            return float(np.median([float(r[key]) for r in lift]))

        ts, tc = med(tip_key), med(tip_cv)
        fs, fc = med(fn_key), med(fn_cv)
        rows_out.append({
            "run": d.name,
            "depth": depth,
            "tip_sim_mm": ts,
            "tip_cv_mm": tc,
            "tip_ratio": tc / ts if abs(ts) > 1e-9 else float("nan"),
            "fn_sim_n": fs,
            "fn_cv_n": fc,
            "fn_ratio": fc / fs if abs(fs) > 1e-12 else float("nan"),
        })
    return rows_out


def plot_grasp_depth_sweep(
    run_dirs: list[Path | str],
    out_path: Path | str | None = None,
    *,
    finger: str = "left",
    show: bool = False,
) -> Path:
    """Two-panel depth sweep: tip deflection and normal force (sim vs CV)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for grasp depth sweep plots"
        ) from exc

    summary = summarize_grasp_depth_lifts(run_dirs, finger=finger)
    if not summary:
        raise ValueError("no grasp_d* runs with LIFT force logs")

    if out_path is None:
        out_path = Path("runs") / "grasp_depth_sweep_lift.png"
    out_path = Path(out_path)

    depth = np.asarray([r["depth"] for r in summary], dtype=float)
    tip_s = np.asarray([r["tip_sim_mm"] for r in summary], dtype=float)
    tip_c = np.asarray([r["tip_cv_mm"] for r in summary], dtype=float)
    fn_s = np.asarray([r["fn_sim_n"] for r in summary], dtype=float)
    fn_c = np.asarray([r["fn_cv_n"] for r in summary], dtype=float)

    _SIM, _CV = "#dc2626", "#2563eb"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle(
        f"Grasp-depth sweep — {finger} finger, LIFT median\n"
        f"(sim red, CV blue; F=2U/|δ_c| at contact station)",
        fontsize=11,
    )

    ax = axes[0]
    ax.plot(depth, tip_s, "o-", color=_SIM, lw=1.6, label="sim")
    ax.plot(depth, tip_c, "s--", color=_CV, lw=1.6, label="CV")
    ax.set_xlabel("grasp depth")
    ax.set_ylabel("tip deflection [mm]")
    ax.set_title("① Tip δ")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(depth, fn_s, "o-", color=_SIM, lw=1.6, label="sim")
    ax.plot(depth, fn_c, "s--", color=_CV, lw=1.6, label="CV")
    ax.set_xlabel("grasp depth")
    ax.set_ylabel("normal force [N]")
    ax.set_title("② Normal force")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)

    # Side CSV next to the plot
    csv_path = out_path.with_suffix(".csv")
    keys = list(summary[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)

    if show:
        plt.show()
    plt.close(fig)
    return out_path

