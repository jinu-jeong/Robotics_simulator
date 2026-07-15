"""Finger deflection / normal-force probes for grasp logging and CV validation."""

from __future__ import annotations

import numpy as np

# Legacy tip-band length (fallback when no contact forces / s_c unknown).
TIP_LEN_M = 0.050
_CONTACT_BAND_FRAC = 0.05  # ± fraction of length around s_c for δ_c

_FINGER_LINKS = ("left_finger", "right_finger")


def tip_band_indices(ref_nodes: np.ndarray, tip_len: float = TIP_LEN_M) -> np.ndarray:
    """Node indices in the prong tip band (+X end in link frame)."""
    x_tip = ref_nodes[:, 0].max() - tip_len
    return np.where(ref_nodes[:, 0] >= x_tip)[0]


def node_arc_s(ref_nodes: np.ndarray) -> np.ndarray:
    """Root→tip arc fraction s ∈ [0, 1] from link-frame X."""
    x = np.asarray(ref_nodes, dtype=float)[:, 0]
    span = max(float(x.max() - x.min()), 1e-9)
    return (x - float(x.min())) / span


def station_band_indices(
    ref_nodes: np.ndarray,
    s_c: float,
    half_band: float = _CONTACT_BAND_FRAC,
) -> np.ndarray:
    """Nodes whose arc fraction lies within ``±half_band`` of ``s_c``."""
    s = node_arc_s(ref_nodes)
    sc = float(np.clip(s_c, 0.0, 1.0))
    band = np.where(np.abs(s - sc) <= half_band)[0]
    if band.size == 0:
        return np.asarray([int(np.argmin(np.abs(s - sc)))], dtype=int)
    return band


def contact_s_from_nodal_forces(
    ref_nodes: np.ndarray,
    f_ext: np.ndarray | None,
    *,
    min_force: float = 1e-6,
) -> float | None:
    """Force-magnitude-weighted contact station s_c ∈ [0, 1], or None."""
    if f_ext is None:
        return None
    f = np.asarray(f_ext, dtype=float).reshape(-1, 3)
    if f.shape[0] != ref_nodes.shape[0]:
        return None
    w = np.linalg.norm(f, axis=1)
    mask = w > min_force
    if not np.any(mask):
        return None
    s = node_arc_s(ref_nodes)
    return float(np.average(s[mask], weights=w[mask]))


def _beam_point_load_shape(s: np.ndarray, a: float) -> np.ndarray:
    """Cantilever unit shape for a concentrated load at station ``a`` (L=EI=F=1).

    ``u = x²(3a−x)/6`` for ``x≤a``, ``a²(3x−a)/6`` for ``x≥a``.
    """
    s = np.asarray(s, dtype=float)
    a = float(np.clip(a, 0.05, 1.0))
    out = np.empty_like(s)
    m = s <= a
    out[m] = (s[m] ** 2) * (3.0 * a - s[m]) / 6.0
    out[~m] = (a ** 2) * (3.0 * s[~m] - a) / 6.0
    return out


def contact_s_from_defl_profile(
    x_m: np.ndarray,
    u_mm: np.ndarray,
    *,
    s_lo: float = 0.20,
    n_a: int = 41,
) -> float | None:
    """Contact station by fitting a cantilever point-load shape to ``u(s)``.

    Pixel κ is too noisy on CV ridges; a 1-parameter beam fit recovers the
    load station that best matches the observed closing deflection.
    """
    x = np.asarray(x_m, dtype=float).ravel()
    u = np.asarray(u_mm, dtype=float).ravel()
    if x.size < 5 or u.size != x.size:
        return None
    span = max(float(x.max() - x.min()), 1e-9)
    s = (x - float(x.min())) / span
    target = u - float(u[0])
    if float(np.max(np.abs(target))) < 1e-6:
        return None
    best_err = float("inf")
    best_a = 1.0
    for a in np.linspace(max(s_lo, 0.2), 1.0, int(n_a)):
        phi = _beam_point_load_shape(s, float(a))
        den = float(np.dot(phi, phi)) + 1e-18
        scale = float(np.dot(phi, target)) / den
        err = float(np.sum((target - scale * phi) ** 2))
        if err < best_err:
            best_err = err
            best_a = float(a)
    return best_a

def free_contact_node_indices(body) -> np.ndarray:
    """Finger nodes allowed to contact the grasped object (exclude palm anchors)."""
    n = int(body.mesh.nodes.shape[0])
    fixed = getattr(body, "fixed_nodes", None)
    if fixed is None or len(fixed) == 0:
        s = node_arc_s(body.mesh.nodes)
        return np.where(s >= 0.25)[0]
    fixed = np.asarray(fixed, dtype=int)
    return np.setdiff1d(np.arange(n, dtype=int), fixed)


def tip_deflection_max_m(
    body,
    R_link: np.ndarray,
    t_link: np.ndarray,
    tip_len: float = TIP_LEN_M,
) -> float:
    """Max |Δx| on tip-band nodes in the finger link frame [m]."""
    ref = body.mesh.nodes
    tip = tip_band_indices(ref, tip_len)
    if len(tip) == 0:
        return 0.0
    x_local = (body.x - t_link) @ R_link
    delta = x_local[tip] - ref[tip]
    return float(np.linalg.norm(delta, axis=1).max())


def tip_deflection_y_m(
    body,
    R_link: np.ndarray,
    t_link: np.ndarray,
    link_name: str,
    tip_len: float = TIP_LEN_M,
) -> float:
    """Mean tip-band displacement along link Y (closing direction) [m].

    Positive = closing-side bend (same sign convention as CV). Left /
    right flip link-Y so both fingers report + when bending toward the
    object.
    """
    ref = body.mesh.nodes
    tip = tip_band_indices(ref, tip_len)
    if len(tip) == 0:
        return 0.0
    x_local = (body.x - t_link) @ R_link
    dy = x_local[tip, 1] - ref[tip, 1]
    # Empirically: closing contact drives left tip +Y / right tip −Y in
    # link frame; flip right so both read positive when closing.
    sign = 1.0 if "left" in link_name.lower() else -1.0
    return float(sign * dy.mean())


def deflection_profile_closing_mm(
    body,
    R_link: np.ndarray,
    t_link: np.ndarray,
    link_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Closing-signed deflection vs link-frame X, averaged per X-station.

    Returns
    -------
    x_m : (M,) unique reference X stations [m], root → tip
    u_mm : (M,) mean closing-signed Y deflection at each station [mm]
    """
    ref = body.mesh.nodes
    x_local = (body.x - t_link) @ R_link
    dy = x_local[:, 1] - ref[:, 1]
    sign = 1.0 if "left" in link_name.lower() else -1.0
    u_mm = sign * dy * 1e3
    x = ref[:, 0]
    # Round to suppress float duplicates from hex mesh
    x_key = np.round(x, decimals=6)
    xs = np.unique(x_key)
    us = np.empty(xs.size, dtype=float)
    for i, xv in enumerate(xs):
        us[i] = float(np.mean(u_mm[x_key == xv]))
    order = np.argsort(xs)
    return xs[order], us[order]


def interpolate_tip_bend_u(
    body,
    link_name: str,
    tip_defl_m: float,
    tip_len: float = TIP_LEN_M,
) -> np.ndarray:
    """Nodal ``u`` from CV tip closing bend via min-energy CB modes [m].

    Prescribes mean tip-band |u_y| = ``tip_defl_m`` (closing sign), then
    solves the reduced static problem

        min  ½ qᵀ K_r q   s.t.   B q = δ

    so the displacement lives in the CB subspace and uses ``K_r`` directly
    (no scalar ``K_eff``). Falls back to a tip-band-only Y field if the
    reduced basis is not ready.
    """
    ref = body.mesh.nodes
    n = int(ref.shape[0])
    u = np.zeros((n, 3), dtype=float)
    if abs(tip_defl_m) < 1e-12:
        return u

    tip = tip_band_indices(ref, tip_len)
    if tip.size == 0:
        return u

    sign = 1.0 if "left" in link_name.lower() else -1.0
    delta = sign * float(tip_defl_m)

    K_r = getattr(body, "_K_r", None)
    Phi = getattr(body, "_Phi_CB", None)
    if K_r is None or Phi is None:
        u[tip, 1] = delta
        return u

    # B: mean tip Y from reduced coords, B q = (1/n_tip) Σ_i Φ[tip_i, y] · q
    tip_y_dofs = tip * 3 + 1
    B = Phi[tip_y_dofs, :].mean(axis=0)  # (n_r,)
    b2 = float(B @ B)
    if b2 < 1e-18:
        u[tip, 1] = delta
        return u

    # q = K_r⁺ Bᵀ (B K_r⁺ Bᵀ)⁻¹ δ   (min-energy for linear constraint)
    try:
        K_inv_B = np.linalg.solve(K_r, B)
    except np.linalg.LinAlgError:
        K_inv_B = np.linalg.lstsq(K_r, B, rcond=None)[0]
    denom = float(B @ K_inv_B)
    if abs(denom) < 1e-18:
        u[tip, 1] = delta
        return u
    q = K_inv_B * (delta / denom)
    return (Phi @ q).reshape(n, 3)


def elastic_normal_force_from_u(
    body,
    u: np.ndarray,
    tip_len: float = TIP_LEN_M,
    contact_s: float | None = None,
) -> float:
    """Equivalent contact elastic force from CB strain energy on nodal ``u`` [N].

    ``U = ½ qᵀ K_r q``, then ``F = 2U / |δ_c|`` where ``δ_c`` is mean |u_Y|
    on the **contact station band** around ``contact_s`` (fallback: tip band).
    """
    K_r = getattr(body, "_K_r", None)
    C_q = getattr(body, "_C_q", None)
    Phi = getattr(body, "_Phi_CB", None)
    free_dofs = getattr(body, "_free_dofs", None)
    if K_r is None or C_q is None or Phi is None or free_dofs is None:
        return 0.0

    ref = body.mesh.nodes
    if contact_s is not None and np.isfinite(contact_s):
        band = station_band_indices(ref, float(contact_s))
    else:
        band = tip_band_indices(ref, tip_len)
    if band.size == 0:
        return 0.0

    u_flat = np.asarray(u, dtype=float).reshape(-1)
    q_r = C_q @ u_flat[free_dofs]
    U = 0.5 * float(q_r @ (K_r @ q_r))
    if U < 1e-12:
        return 0.0

    u_modal = (Phi @ q_r).reshape(-1, 3)
    delta = float(np.abs(u_modal[band, 1]).mean())
    if delta < 1e-7:
        delta = float(np.linalg.norm(u_modal[band], axis=1).max())
    if delta < 1e-7:
        return 0.0
    return 2.0 * U / delta


def _minenergy_q_from_centerline(
    body,
    link_name: str,
    s: np.ndarray,
    u_closing_m: np.ndarray,
) -> np.ndarray | None:
    """Reduced coords ``q`` matching measured closing ``u(s)`` at min energy."""
    s = np.asarray(s, dtype=float).ravel()
    u_closing_m = np.asarray(u_closing_m, dtype=float).ravel()
    if s.size < 2 or u_closing_m.size != s.size:
        return None
    if float(np.max(np.abs(u_closing_m))) < 1e-12:
        return None

    K_r = getattr(body, "_K_r", None)
    Phi = getattr(body, "_Phi_CB", None)
    if K_r is None or Phi is None:
        return None

    ref = body.mesh.nodes
    sign = 1.0 if "left" in link_name.lower() else -1.0
    delta = sign * u_closing_m
    x = ref[:, 0]
    span = max(float(x.max() - x.min()), 1e-9)
    s_node = (x - float(x.min())) / span

    n_con = int(min(8, s.size))
    idxs = np.unique(np.linspace(0, s.size - 1, n_con, dtype=int))
    idxs = np.asarray([i for i in idxs if s[i] >= 0.15], dtype=int)
    if idxs.size < 2:
        idxs = np.asarray([s.size // 2, s.size - 1], dtype=int)

    rows = []
    rhs = []
    half_band = 0.04
    for i in idxs:
        band = np.where(np.abs(s_node - float(s[i])) <= half_band)[0]
        if band.size == 0:
            j = int(np.argmin(np.abs(s_node - float(s[i]))))
            band = np.asarray([j], dtype=int)
        y_dofs = band * 3 + 1
        rows.append(Phi[y_dofs, :].mean(axis=0))
        rhs.append(float(delta[i]))
    B = np.asarray(rows, dtype=float)
    d = np.asarray(rhs, dtype=float)
    try:
        K_inv_BT = np.linalg.solve(K_r, B.T)
    except np.linalg.LinAlgError:
        K_inv_BT = np.linalg.lstsq(K_r, B.T, rcond=None)[0]
    G = B @ K_inv_BT
    try:
        lam = np.linalg.solve(G, d)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(G, d, rcond=None)[0]
    return K_inv_BT @ lam


def interpolate_centerline_u(
    body,
    link_name: str,
    s: np.ndarray,
    u_closing_m: np.ndarray,
) -> np.ndarray:
    """Map closing-signed centerline ``u(s)`` onto nodal link-frame ``u`` [m].

    Uses **min-energy CB modes** constrained to the measured stations
    (same idea as :func:`interpolate_tip_bend_u`, but with the full
    profile). Direct nodal painting of image ``u(s)`` is avoided — that
    injects out-of-subspace content and inflates ``U``.
    """
    ref = body.mesh.nodes
    n = int(ref.shape[0])
    out = np.zeros((n, 3), dtype=float)
    s = np.asarray(s, dtype=float).ravel()
    u_closing_m = np.asarray(u_closing_m, dtype=float).ravel()
    if s.size < 2 or u_closing_m.size != s.size:
        return out

    q = _minenergy_q_from_centerline(body, link_name, s, u_closing_m)
    Phi = getattr(body, "_Phi_CB", None)
    if q is not None and Phi is not None:
        return (Phi @ q).reshape(n, 3)

    # Fallback: paint Y from interpolated closing field
    if float(np.max(np.abs(u_closing_m))) < 1e-12:
        return out
    sign = 1.0 if "left" in link_name.lower() else -1.0
    x = ref[:, 0]
    span = max(float(x.max() - x.min()), 1e-9)
    s_node = (x - float(x.min())) / span
    out[:, 1] = sign * np.interp(s_node, s, u_closing_m)
    return out


def elastic_force_from_centerline(
    body,
    link_name: str,
    s: np.ndarray,
    u_closing_m: np.ndarray,
    contact_s: float | None = None,
    tip_len: float = TIP_LEN_M,
) -> float:
    """Project CV elastic centerline into CB subspace and return contact force [N].

    Fits min-energy ``q`` to measured ``u(s)``, then
    ``U = ½ qᵀ K_r q`` and ``F = 2U / |δ_c|`` at the **contact station**
    ``contact_s`` (CV curvature / peak). Falls back to tip band if ``contact_s``
    is missing.
    """
    K_r = getattr(body, "_K_r", None)
    Phi = getattr(body, "_Phi_CB", None)
    q_r = _minenergy_q_from_centerline(body, link_name, s, u_closing_m)
    if q_r is None or K_r is None or Phi is None:
        u = interpolate_centerline_u(body, link_name, s, u_closing_m)
        return elastic_normal_force_from_u(
            body, u, tip_len=tip_len, contact_s=contact_s,
        )

    U = 0.5 * float(q_r @ (K_r @ q_r))
    if U < 1e-12:
        return 0.0

    u_modal = (Phi @ q_r).reshape(-1, 3)
    ref = body.mesh.nodes
    if contact_s is not None and np.isfinite(contact_s):
        band = station_band_indices(ref, float(contact_s))
    else:
        band = tip_band_indices(ref, tip_len)
    if band.size == 0:
        return 0.0
    delta = float(np.abs(u_modal[band, 1]).mean())
    if delta < 1e-7:
        delta = float(np.linalg.norm(u_modal[band], axis=1).max())
    if delta < 1e-7:
        return 0.0
    return 2.0 * U / delta


def elastic_normal_force_from_cv_defl(
    body,
    link_name: str,
    tip_defl_m: float,
    tip_len: float = TIP_LEN_M,
) -> float:
    """CV tip deflection → interpolated nodal ``u`` → ``K_r`` elastic force [N]."""
    if abs(tip_defl_m) < 1e-9:
        return 0.0
    u = interpolate_tip_bend_u(body, link_name, tip_defl_m, tip_len=tip_len)
    return elastic_normal_force_from_u(body, u, tip_len=tip_len)


def elastic_normal_force_n(
    body,
    R_link: np.ndarray,
    t_link: np.ndarray,
    link_name: str,
    tip_len: float = TIP_LEN_M,
    f_ext: np.ndarray | None = None,
    contact_s: float | None = None,
) -> tuple[float, float | None]:
    """Sim elastic contact force [N] and contact station ``s_c`` (or None).

    ``δ`` conjugate uses the same cantilever point-load fit as CV (shared
    sensor model). Force-weighted ``f_ext`` / held ``contact_s`` remain
    fallbacks when the deflection fit is unavailable.
    """
    ref = body.mesh.nodes
    x_local = (body.x - t_link) @ R_link
    u = x_local - ref
    sign = 1.0 if "left" in link_name.lower() else -1.0
    sc = contact_s_from_defl_profile(ref[:, 0], sign * u[:, 1] * 1e3)
    if sc is None:
        sc = contact_s_from_nodal_forces(ref, f_ext)
    if sc is None and contact_s is not None and np.isfinite(contact_s):
        sc = float(contact_s)
    fn = elastic_normal_force_from_u(
        body, u, tip_len=tip_len, contact_s=sc,
    )
    return fn, sc


def normal_force_from_f_ext(f_ext: np.ndarray) -> float:
    """Total contact force magnitude on a deformable finger from penalty f_ext [N]."""
    if f_ext is None or np.linalg.norm(f_ext) < 1e-12:
        return 0.0
    return float(np.linalg.norm(f_ext.reshape(-1, 3).sum(axis=0)))


def normal_force_from_wrench(w: np.ndarray, link_name: str) -> float:
    """Normal grasp force from a cached 6-DOF link wrench [N].

    ``w[3:]`` is force in the link frame; closing axis is ±Y (left/right).
    """
    if w is None or np.linalg.norm(w[3:]) < 1e-12:
        return 0.0
    push_sign = -1.0 if "left" in link_name.lower() else 1.0
    return abs(float(w[3] * push_sign))


def empty_probe() -> dict[str, dict[str, float]]:
    return {
        name: {
            "deflection_m": 0.0,
            "deflection_y_m": 0.0,
            "fn_n": 0.0,
            "contact_s": float("nan"),
        }
        for name in _FINGER_LINKS
    }
