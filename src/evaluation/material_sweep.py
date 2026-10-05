"""Material generalisation of the q → mechanics stage (paper Fig. 5).

Everything *before* ``q`` is frozen: the POD basis ``Φ`` (fixed q convention),
the marker geometry / image Jacobian and the marker-free ResNet. Only the
material (E, ν) of the linear, homogeneous, isotropic finger changes, which
means only the stiffness ``K`` – and hence the reduced ``K_r = ΦᵀKΦ`` and the
localiser influence fields – is rebuilt. No retraining, no new basis.

For each material we compare, on held-out inner-face contacts:

* ``oracle``  q = Φᵀu(E, ν)             (what the vision stage should return)
* ``marker``  Stage-D marker pixels → q  (world camera, pixel noise)
* ``nn``      marker-free frame → ResNet → q  (optional, needs a checkpoint)

each pushed through (a) the **updated** ROM ``K_r(E, ν)`` and (b) the **stale**
reference ROM ``K_r(E₀, ν₀)``. For E-only changes (b) is wrong by exactly
``E₀/E − 1``; (a) should stay at the reference accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..contact.contact_mapping import SurfaceContact
from ..fem.finger_model import FingerFEMModel
from ..rom.pod import PODBasis
from ..rom.reduced_mechanics import ReducedModel
from .unknown_contact import UnknownContactLocalizer


# ---------------------------------------------------------------- materials
def material_variants(base: dict, e_factors, nu_values) -> list[dict]:
    """E sweep at the base ν plus a ν sweep at the base E (reference included once)."""
    E0, nu0 = float(base["youngs_modulus"]), float(base["poisson_ratio"])
    out: list[dict] = []
    seen = set()
    for f in e_factors:
        key = (round(float(f), 6), round(nu0, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": f"E x{float(f):g}", "E_factor": float(f), "youngs_modulus": E0 * float(f), "poisson_ratio": nu0, "sweep": "E"})
    for nu in nu_values:
        key = (1.0, round(float(nu), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": f"nu {float(nu):g}", "E_factor": 1.0, "youngs_modulus": E0, "poisson_ratio": float(nu), "sweep": "nu"})
    return out


def build_material_model(fem_cfg: dict, material: dict, contact_face: str = "side_pos_y") -> FingerFEMModel:
    cfg = dict(fem_cfg)
    cfg["material"] = {**dict(fem_cfg.get("material", {})),
                       "youngs_modulus": float(material["youngs_modulus"]),
                       "poisson_ratio": float(material["poisson_ratio"])}
    model = FingerFEMModel.from_config(cfg)
    model.restrict_contact_surface(contact_face)
    return model


# ---------------------------------------------------------------- one material case
@dataclass
class MaterialCase:
    material: dict
    model: FingerFEMModel
    rom: ReducedModel              # fixed Φ, K from this material
    rom_stale: ReducedModel        # fixed Φ, K from the reference material
    localizer: UnknownContactLocalizer
    localizer_stale: UnknownContactLocalizer
    contacts: list[SurfaceContact] = field(default_factory=list)
    u_unit: list[np.ndarray] = field(default_factory=list)  # (N, 3) per contact for 1 N

    @classmethod
    def build(
        cls,
        fem_cfg: dict,
        material: dict,
        basis: PODBasis,
        rom_reference: ReducedModel,
        contact_points: np.ndarray,
        *,
        localizer_cfg: dict | None = None,
        contact_face: str = "side_pos_y",
    ) -> "MaterialCase":
        model = build_material_model(fem_cfg, material, contact_face)
        rom = ReducedModel.from_fem(model.fem, basis, model.contact)
        uc = dict(localizer_cfg or {})
        face_ids = model.surface_face_ids(contact_face)
        bary = uc.get("barycentric") or [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3],
            [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        ]
        mk = lambda r: UnknownContactLocalizer.from_rom(  # noqa: E731
            r, face_ids, barycentric=bary,
            face_stride=int(uc.get("face_stride", 2)), min_influence=float(uc.get("min_influence", 1e-8)),
            nonneg=True, r_loc=int(uc.get("r_loc", 4)), refine=bool(uc.get("refine", True)),
            refine_n=int(uc.get("refine_n", 7)), weight=str(uc.get("weight", "leading")),
        )
        case = cls(material=dict(material), model=model, rom=rom, rom_stale=rom_reference,
                   localizer=mk(rom), localizer_stale=mk(rom_reference))
        for p in np.asarray(contact_points, float).reshape(-1, 3):
            res, c = model.solve_normal_contact(p, 1.0)
            case.contacts.append(c)
            case.u_unit.append(res.u.copy())
        return case

    # -------------------------------------------------------------- physics from q
    def force_known(self, q: np.ndarray, contact: SurfaceContact, *, stale: bool = False) -> float:
        rom = self.rom_stale if stale else self.rom
        return float(rom.estimate_force(q, contact, method="displacement", mode="normal").magnitude)

    def force_unknown(self, q: np.ndarray, *, stale: bool = False):
        loc = (self.localizer_stale if stale else self.localizer).localize(q)
        return float(loc.magnitude), np.asarray(loc.contact.position, float), float(loc.residual_rel)


# ---------------------------------------------------------------- evaluation
def held_out_inner_face_points(geometry, x_rel, z_rel) -> np.ndarray:
    pts = [geometry.point_from_relative([float(x), 1.0, float(z)]) for x in x_rel for z in z_rel]
    return np.asarray(pts, float)


def projection_residual(basis: PODBasis, u: np.ndarray) -> float:
    """‖u − ΦΦᵀu‖ / ‖u‖ – how well the *fixed* basis spans this material's field."""
    v = np.asarray(u, float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-300:
        return 0.0
    return float(np.linalg.norm(v - basis.Phi @ (basis.Phi.T @ v)) / n)


def evaluate_case(
    case: MaterialCase,
    basis: PODBasis,
    forces: np.ndarray,
    q_sources: dict[str, Callable[[np.ndarray, SurfaceContact, float], np.ndarray | None]],
    *,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every q source × {updated, stale} K on all (contact, force) samples.

    ``q_sources`` maps a name to ``f(u_true, contact, force) -> q`` (``None`` = skip sample).
    Returns per-sample arrays (for plotting) and medians.
    """
    forces = np.asarray(forces, float).reshape(-1)
    rows: dict[str, dict[str, list]] = {
        name: {"F": [], "F_known": [], "F_known_stale": [], "F_unk": [], "F_unk_stale": [],
               "c_err_mm": [], "c_err_stale_mm": [], "q_norm": [], "q_rel": []}
        for name in q_sources
    }
    proj = []
    for ci, (c, u1) in enumerate(zip(case.contacts, case.u_unit)):
        for F in forces:
            u = F * u1
            q_true = basis.project(u)
            proj.append(projection_residual(basis, u))
            for name, src in q_sources.items():
                q = src(u, c, float(F))
                if q is None:
                    continue
                q = np.asarray(q, float).reshape(-1)
                r = rows[name]
                r["F"].append(float(F))
                r["q_norm"].append(float(np.linalg.norm(q)))
                r["q_rel"].append(float(np.linalg.norm(q - q_true) / max(np.linalg.norm(q_true), 1e-30)))
                r["F_known"].append(case.force_known(q, c))
                r["F_known_stale"].append(case.force_known(q, c, stale=True))
                m, p, _ = case.force_unknown(q)
                ms, ps, _ = case.force_unknown(q, stale=True)
                r["F_unk"].append(m)
                r["F_unk_stale"].append(ms)
                r["c_err_mm"].append(1e3 * float(np.linalg.norm(p - c.position)))
                r["c_err_stale_mm"].append(1e3 * float(np.linalg.norm(ps - c.position)))
        if log is not None and (ci + 1) % max(1, len(case.contacts) // 3) == 0:
            log(f"    contact {ci + 1}/{len(case.contacts)}")

    def _med(a):
        a = np.asarray(a, float)
        return float(np.median(a)) if a.size else float("nan")

    summary: dict[str, Any] = {"material": case.material, "proj_residual_median": _med(proj), "n_contacts": len(case.contacts),
                               "forces": forces.tolist(), "sources": {}}
    for name, r in rows.items():
        F = np.asarray(r["F"], float)
        if F.size == 0:
            continue
        rel = lambda a: np.abs(np.asarray(a, float) - F) / F  # noqa: E731
        summary["sources"][name] = {
            "n": int(F.size),
            "q_rel_median": _med(r["q_rel"]),
            "force_known_rel_median": _med(rel(r["F_known"])),
            "force_known_rel_median_staleK": _med(rel(r["F_known_stale"])),
            "force_unknown_rel_median": _med(rel(r["F_unk"])),
            "force_unknown_rel_median_staleK": _med(rel(r["F_unk_stale"])),
            "contact_err_mm_median": _med(r["c_err_mm"]),
            "contact_err_mm_median_staleK": _med(r["c_err_stale_mm"]),
            "q_norm_median": _med(r["q_norm"]),
            "samples": {k: [float(x) for x in v] for k, v in r.items()},
        }
    return summary
