"""Estimate finger elastic deflection from solid-red viewer frames.

Palm-cam frames in this demo show:
  • finger length along image **X** (palm → tip, left → right)
  • left/right fingers stacked in image **Y** (upper / lower)
  • closing bend = vertical (Y) motion of each ridge

Materials-mechanics path:
  1. Track each finger's **outer** silhouette edge along length s (image X).
     Upper/left → top edge; lower/right → bottom edge.
  2. Intra-frame clamped BC: ``u(s) = y(s) − mean(y near root)`` — palm/root
     relative deflection (no temporal ref; CLOSE→LIFT cam shifts swamp δ).
  3. Contact station from cantilever point-load fit to elastic u(s).
  4. Tip sample of u(s) is logged; full u(s) → min-energy CB → K_r force.

CLOSE hardening:
  • Gate sub-pixel tip/RMS so rigid squeeze is not read as elastic force.
  • Temporally lock upper/lower Y peaks so mid-gap collapse cannot merge ridges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FINGER_LENGTH_M = 0.160
_MAX_DEFLECTION_MM = 25.0
_N_SPLINE = 24
_MAX_RIDGE_JUMP_PX = 25.0
_MAX_REF_JUMP_PX = 20.0
# Rigid-close silhouette tilt is ~1–2 px ≈ 0.3–0.5 mm; gate below this.
_MIN_TIP_MM = 0.85
_MIN_RMS_MM = 0.35
# Peak latch / mid-gap collapse thresholds [px].
_PEAK_SEP_MIN_PX = 80.0
_PEAK_JUMP_REJECT_PX = 120.0
_PEAK_TRACK_BAND_PX = 80.0


def _rgb(img: np.ndarray) -> np.ndarray:
    a = img[..., :3]
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def pure_red_mask(
    img: np.ndarray,
    *,
    r_min: float = 0.85,
    gb_max: float = 0.30,
) -> np.ndarray:
    """Boolean mask of solid-red finger pixels (excludes orange palm)."""
    rgb = _rgb(img)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r >= r_min) & (g <= gb_max) & (b <= gb_max)


def redness_map(img: np.ndarray) -> np.ndarray:
    """Highlight solid-red fingers; orange palm suppressed.

    Kept for debug overlays; tracking uses :func:`pure_red_mask`.
    """
    rgb = _rgb(img)
    soft = np.clip(rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2]), 0.0, 1.0)
    return np.where(pure_red_mask(img), soft, 0.0).astype(np.float32)


def _dominant_run_bounds(col: np.ndarray) -> tuple[float, float] | None:
    """(start, end) of the longest True run in a 1-D boolean column (inclusive)."""
    if col.size < 3 or not np.any(col):
        return None
    padded = np.concatenate([[False], col.astype(bool), [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    if starts.size == 0:
        return None
    lengths = ends - starts + 1
    k = int(np.argmax(lengths))
    if int(lengths[k]) < 5:
        return None
    return float(starts[k]), float(ends[k])


def _dominant_run_mid(col: np.ndarray) -> float | None:
    """Midpoint of the longest True run in a 1-D boolean column."""
    b = _dominant_run_bounds(col)
    if b is None:
        return None
    return float(0.5 * (b[0] + b[1]))


def _ridge_y_in_roi(
    mask: np.ndarray,
    x: int,
    y_lo: int,
    y_hi: int,
    *,
    which: str = "mid",
    y_prev: float | None = None,
    max_jump: float = _MAX_RIDGE_JUMP_PX,
) -> float | None:
    """Finger-band sample in ``[y_lo, y_hi]`` at column ``x``.

    ``which``:
      - ``mid``: band midpoint
      - ``lo``: smaller-Y edge (top of band in image)
      - ``hi``: larger-Y edge (bottom of band in image)

    If ``y_prev`` is set, prefer a hit within ``max_jump`` of it.
    """
    y_lo = int(max(0, y_lo))
    y_hi = int(min(mask.shape[0], y_hi))
    if y_hi - y_lo < 5 or x < 0 or x >= mask.shape[1]:
        return None

    def _pick(abs_lo: int, abs_hi: int) -> float | None:
        b = _dominant_run_bounds(mask[abs_lo:abs_hi, x])
        if b is None:
            return None
        a0, a1 = abs_lo + b[0], abs_lo + b[1]
        if which == "lo":
            return float(a0)
        if which == "hi":
            return float(a1)
        return float(0.5 * (a0 + a1))

    if y_prev is not None:
        lo = max(y_lo, int(round(y_prev - max_jump)))
        hi = min(y_hi, int(round(y_prev + max_jump)) + 1)
        y = _pick(lo, hi)
        if y is not None:
            return y
    return _pick(y_lo, y_hi)


def _remove_rigid_translation(s: np.ndarray, u: np.ndarray, root_frac: float = 0.15) -> np.ndarray:
    """Subtract mean displacement on the root band (clamped-cantilever BC)."""
    m = s <= root_frac
    if not np.any(m):
        m = s <= float(s.min() + 1e-9)
    return u - float(np.mean(u[m]))


def _curvature_abs(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    if s.size < 5:
        return np.zeros_like(u)
    ds = float(np.median(np.diff(s)))
    if ds < 1e-9:
        return np.zeros_like(u)
    ker = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    ker /= ker.sum()
    us = np.convolve(u, ker, mode="same")
    upp = np.zeros_like(us)
    upp[1:-1] = (us[2:] - 2.0 * us[1:-1] + us[:-2]) / (ds * ds)
    upp[0] = upp[-1] = 0.0
    return np.abs(upp)


def _max_adjacent_jump(y: np.ndarray) -> float:
    d = np.abs(np.diff(y[np.isfinite(y)]))
    return float(d.max()) if d.size else 0.0


def _hist_mode_y(
    ys: np.ndarray,
    y_lo: float,
    y_hi: float,
    *,
    bins: int = 48,
) -> float | None:
    """Y-mode of points falling in ``[y_lo, y_hi]``."""
    if y_hi - y_lo < 5.0:
        return None
    m = (ys >= y_lo) & (ys < y_hi)
    if int(m.sum()) < 30:
        return None
    hist, edges = np.histogram(ys[m], bins=bins, range=(y_lo, y_hi))
    if hist.max() < 10:
        return None
    i = int(np.argmax(hist))
    return float(0.5 * (edges[i] + edges[i + 1]))


def _find_finger_peaks(
    ys: np.ndarray,
    y0: int,
    y1: int,
    prev_u: float | None,
    prev_l: float | None,
) -> tuple[float, float] | None:
    """Upper / lower finger Y centers, with optional temporal lock."""
    if prev_u is not None and prev_l is not None and prev_l - prev_u > 60.0:
        mid = 0.5 * (prev_u + prev_l)
        # Prefer modes near previous peaks; split at last mid so squeeze
        # cannot latch both peaks onto the same finger.
        pu = _hist_mode_y(
            ys,
            max(y0, prev_u - _PEAK_TRACK_BAND_PX),
            min(mid - 4.0, prev_u + _PEAK_TRACK_BAND_PX),
        )
        pl = _hist_mode_y(
            ys,
            max(mid + 4.0, prev_l - _PEAK_TRACK_BAND_PX),
            min(y1 + 1, prev_l + _PEAK_TRACK_BAND_PX),
        )
        if pu is None:
            pu = _hist_mode_y(ys, float(y0), mid - 4.0)
        if pl is None:
            pl = _hist_mode_y(ys, mid + 4.0, float(y1 + 1))
        if pu is not None and pl is not None and pl - pu >= 50.0:
            # Reject unphysical peak teleport while mid-gap collapses.
            if (
                abs(pu - prev_u) > _PEAK_JUMP_REJECT_PX
                or abs(pl - prev_l) > _PEAK_JUMP_REJECT_PX
            ):
                return None
            return float(pu), float(pl)

    # Cold start: two global modes with a minimum separation.
    hist, edges = np.histogram(ys, bins=64, range=(y0, y1 + 1))
    order = np.argsort(hist)[::-1]
    peaks: list[float] = []
    for i in order:
        if hist[i] < max(20, int(0.02 * hist.max())):
            break
        cy = 0.5 * (edges[i] + edges[i + 1])
        if all(abs(cy - p) > 40 for p in peaks):
            peaks.append(cy)
        if len(peaks) >= 2:
            break
    if len(peaks) < 2:
        return None
    peaks.sort()
    return float(peaks[0]), float(peaks[1])


@dataclass
class FingerField:
    s: np.ndarray
    u_mm: np.ndarray
    tip_mm: float
    contact_s: float
    px_per_mm: float


@dataclass
class CenterlineEstimate:
    left: FingerField | None
    right: FingerField | None

    @property
    def tip_mm(self) -> tuple[float, float]:
        l = 0.0 if self.left is None else self.left.tip_mm
        r = 0.0 if self.right is None else self.right.tip_mm
        return l, r


class GradientDeflectionEstimator:
    """Outer-edge elastic u(s) — length along image X, bend along image Y.

    Tracks each finger's **outer** silhouette edge (upper finger → top/lo edge,
    lower finger → bottom/hi edge), then subtracts the root-band mean y so
    ``u(s) = y(s) − y(root)`` — palm/root relative deflection.
    """

    def __init__(
        self,
        red_thresh: float = 0.12,  # unused; kept for call-site compat
        s_lo: float = 0.0,
        s_hi: float = 1.0,
        min_mask_pixels: int = 400,
        n_samples: int = _N_SPLINE,
        edge: str = "outer",
    ):
        self.red_thresh = red_thresh
        self.s_lo = s_lo
        self.s_hi = s_hi
        self.min_mask_pixels = min_mask_pixels
        self.n_samples = n_samples
        if edge not in ("outer", "inner", "mid"):
            raise ValueError("edge must be 'outer', 'inner', or 'mid'")
        self.edge = edge
        self._ref_s: np.ndarray | None = None
        self._ref_y_u: np.ndarray | None = None
        self._ref_y_l: np.ndarray | None = None
        self._px_per_mm = 1.0
        self._ready = False
        self._last: CenterlineEstimate = CenterlineEstimate(None, None)
        self._peak_u: float | None = None
        self._peak_l: float | None = None

    def _edge_which(self, side: str) -> str:
        """Map finger side → band edge selector for ``_ridge_y_in_roi``."""
        # Upper = left finger, lower = right. Image Y increases downward.
        # outer: away from grasp midgap; inner: toward midgap.
        if self.edge == "mid":
            return "mid"
        if side == "upper":
            return "lo" if self.edge == "outer" else "hi"
        return "hi" if self.edge == "outer" else "lo"

    def _sample_centerlines(
        self, img: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        """Return (s, y_upper, y_lower, px_per_mm). ``s=0`` root, ``s=1`` tip."""
        mask = pure_red_mask(img)
        if int(mask.sum()) < self.min_mask_pixels:
            return None
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        # Two Y-modes = upper / lower finger (temporally locked once known).
        found = _find_finger_peaks(ys, y0, y1, self._peak_u, self._peak_l)
        if found is None:
            return None
        peaks = [found[0], found[1]]
        mid = int(0.5 * (peaks[0] + peaks[1]))
        # Keep a hard floor on mid-gap so outer-edge ROIs do not collapse.
        if peaks[1] - peaks[0] < _PEAK_SEP_MIN_PX:
            return None
        y0 = int(max(y0, peaks[0] - 50))
        y1 = int(min(y1, peaks[1] + 50))
        gap = max(8, abs(y1 - y0) // 25)
        up_hi = mid - gap
        lo_lo = mid + gap

        which_u = self._edge_which("upper")
        which_l = self._edge_which("lower")

        # Columns where both finger bands exist (pure red only → no palm).
        w_full = max(x1 - x0, 1)
        step = max(1, w_full // 80)
        x_ok: list[int] = []
        for x in range(x0, x1 + 1, step):
            if (
                _ridge_y_in_roi(mask, x, y0, up_hi, which=which_u) is not None
                and _ridge_y_in_roi(mask, x, lo_lo, y1 + 1, which=which_l) is not None
            ):
                x_ok.append(x)
        if len(x_ok) < 6:
            return None
        x0, x1 = int(min(x_ok)), int(max(x_ok))
        w = max(x1 - x0, 1)

        # Dense track root → tip with continuity (seed at first good column).
        s = np.linspace(self.s_lo, self.s_hi, self.n_samples)
        yu = np.full(self.n_samples, np.nan)
        yl = np.full(self.n_samples, np.nan)
        y_u_prev: float | None = None
        y_l_prev: float | None = None
        for i, frac in enumerate(s):
            x = int(np.clip(round(x0 + frac * w), x0, x1))
            yu[i] = _ridge_y_in_roi(
                mask, x, y0, up_hi, which=which_u, y_prev=y_u_prev,
            )
            yl[i] = _ridge_y_in_roi(
                mask, x, lo_lo, y1 + 1, which=which_l, y_prev=y_l_prev,
            )
            if np.isfinite(yu[i]):
                y_u_prev = float(yu[i])
            if np.isfinite(yl[i]):
                y_l_prev = float(yl[i])

        ok = np.isfinite(yu) & np.isfinite(yl)
        if int(ok.sum()) < max(8, self.n_samples // 2):
            return None
        for arr in (yu, yl):
            good = np.isfinite(arr)
            if good.all():
                continue
            arr[~good] = np.interp(s[~good], s[good], arr[good])

        # Reject discontinuous tracks (palm bleed / edge flip).
        if (
            _max_adjacent_jump(yu) > _MAX_REF_JUMP_PX * 1.5
            or _max_adjacent_jump(yl) > _MAX_REF_JUMP_PX * 1.5
        ):
            return None

        px_span = float(w) * (self.s_hi - self.s_lo)
        span_m = (self.s_hi - self.s_lo) * _FINGER_LENGTH_M
        px_per_mm = float(px_span / max(span_m * 1e3, 1e-6))
        # Commit peak lock only after a clean sample (reject happens above).
        self._peak_u = float(peaks[0])
        self._peak_l = float(peaks[1])
        return s, yu, yl, max(px_per_mm, 1e-6)

    def _field_from_centerline(
        self,
        s: np.ndarray,
        y: np.ndarray,
        px_per_mm: float,
        *,
        closing_sign: float,
    ) -> FingerField:
        # Clamped root: remove mean image-Y on the proximal band only.
        u_px = _remove_rigid_translation(s, y.astype(float))
        u_mm = np.clip(
            closing_sign * u_px / max(px_per_mm, 1e-6),
            -_MAX_DEFLECTION_MM,
            _MAX_DEFLECTION_MM,
        )

        tip = float(u_mm[-1])
        rms = float(np.sqrt(np.mean(u_mm * u_mm)))
        # Kill rigid-close silhouette tilt (≈1–2 px) before real contact.
        if abs(tip) < _MIN_TIP_MM and rms < _MIN_RMS_MM:
            z = np.zeros_like(u_mm)
            return FingerField(
                s=s.copy(),
                u_mm=z,
                tip_mm=0.0,
                contact_s=float("nan"),
                px_per_mm=float(px_per_mm),
            )

        from robosim.scene.finger_probe import contact_s_from_defl_profile

        sc = contact_s_from_defl_profile(s, u_mm)
        if sc is None:
            sc = float(s[int(np.argmax(np.abs(u_mm)))])
        # Proximal-only fits with tiny tip are almost always empty-close noise.
        if sc < 0.35 and abs(tip) < 1.5 * _MIN_TIP_MM:
            z = np.zeros_like(u_mm)
            return FingerField(
                s=s.copy(),
                u_mm=z,
                tip_mm=0.0,
                contact_s=float("nan"),
                px_per_mm=float(px_per_mm),
            )
        return FingerField(
            s=s.copy(),
            u_mm=u_mm.copy(),
            tip_mm=tip,
            contact_s=float(sc),
            px_per_mm=float(px_per_mm),
        )

    def try_set_reference(self, img: np.ndarray) -> bool:
        sampled = self._sample_centerlines(img)
        if sampled is None:
            return False
        s, yu, yl, px = sampled
        if (
            _max_adjacent_jump(yu) > _MAX_REF_JUMP_PX
            or _max_adjacent_jump(yl) > _MAX_REF_JUMP_PX
        ):
            return False
        self._ref_s = s
        self._ref_y_u = yu.copy()
        self._ref_y_l = yl.copy()
        self._px_per_mm = px
        self._ready = True
        return True

    @property
    def ready(self) -> bool:
        return self._ready

    def estimate_fields(self, img: np.ndarray) -> CenterlineEstimate:
        if not self.ready:
            return self._last
        sampled = self._sample_centerlines(img)
        if sampled is None:
            # Narrow-gap / peak-jump: keep last good elastic fields.
            return self._last
        s, yu, yl, px = sampled
        px_use = max(px, self._px_per_mm)

        # Upper → left, lower → right. Palm-cam Y↓: contact bends tips
        # *outward* in the image (upper tip ↑, lower tip ↓) while sim
        # tip_deflection_y_m is reported +closing; flip so CV tip sign
        # matches the probe convention for overlay plots.
        left = self._field_from_centerline(s, yu, px_use, closing_sign=-1.0)
        right = self._field_from_centerline(s, yl, px_use, closing_sign=+1.0)
        # Flat tip-to-root ridge after a large bend ⇒ lost outer edge; hold.
        if (
            self._last.left is not None
            and abs(self._last.left.tip_mm) > 2.0
            and abs(left.tip_mm) < 0.5
            and abs(float(yu[-1] - yu[0])) < 4.0
        ):
            left = self._last.left
        if (
            self._last.right is not None
            and abs(self._last.right.tip_mm) > 2.0
            and abs(right.tip_mm) < 0.5
            and abs(float(yl[-1] - yl[0])) < 4.0
        ):
            right = self._last.right
        est = CenterlineEstimate(left=left, right=right)
        self._last = est
        return est

    def estimate(self, img: np.ndarray) -> tuple[float, float]:
        return self.estimate_fields(img).tip_mm
