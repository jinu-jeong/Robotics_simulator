"""Fig. 5 – material generalisation: fixed Φ, only K → K_r rebuilt."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.material_sweep import (
    MaterialCase,
    build_material_model,
    evaluate_case,
    held_out_inner_face_points,
    material_variants,
    projection_residual,
)
from src.grasp.estimator import GraspEstimator, _inner_face_pod_snapshots
from src.grasp.mechanics import GraspMechanics
from src.rom.pod import compute_pod
from src.rom.reduced_mechanics import ReducedModel

FEM_CFG = {
    "geometry": {"length": 0.1, "width": 0.02, "height": 0.01},
    "mesh": {"nx": 12, "ny": 3, "nz": 3},
    "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
    "boundary_conditions": {"fixed_surface": "root"},
}


@pytest.fixture(scope="module")
def reference():
    model = build_material_model(FEM_CFG, FEM_CFG["material"])
    U = _inner_face_pod_snapshots(model, n_x=3, n_z=2)
    basis = compute_pod(U).truncate(4)
    rom = ReducedModel.from_fem(model.fem, basis, model.contact)
    pts = held_out_inner_face_points(model.geometry, [0.7, 0.9], [0.5])
    return model, basis, rom, pts


def test_material_variants_dedup():
    v = material_variants({"youngs_modulus": 1.0, "poisson_ratio": 0.4}, [0.5, 1.0, 2.0], [0.4, 0.3])
    names = [m["name"] for m in v]
    assert names == ["E x0.5", "E x1", "E x2", "nu 0.3"]  # reference ν not duplicated
    assert all(m["poisson_ratio"] == 0.4 for m in v if m["sweep"] == "E")


def test_updated_k_recovers_force_stale_k_scales_with_e(reference):
    model, basis, rom_ref, pts = reference
    for f in (0.5, 2.0):
        mat = {"youngs_modulus": FEM_CFG["material"]["youngs_modulus"] * f, "poisson_ratio": 0.4, "name": f"E x{f}", "E_factor": f, "sweep": "E"}
        case = MaterialCase.build(FEM_CFG, mat, basis, rom_ref, pts, localizer_cfg={"face_stride": 1, "refine": True})
        # E-only change: the field just scales, so the reference Φ spans it exactly as well as at E0
        res, _ = model.solve_normal_contact(pts[0], 1.0)
        assert projection_residual(basis, case.u_unit[0]) == pytest.approx(projection_residual(basis, res.u), rel=1e-6)
        summ = evaluate_case(case, basis, np.array([1.0, 2.0]), {"oracle": lambda u, c, F: basis.project(u)})
        s = summ["sources"]["oracle"]
        assert s["force_known_rel_median"] < 0.02
        # stale reference K is wrong by exactly E0/E - 1 (linear elasticity)
        assert s["force_known_rel_median_staleK"] == pytest.approx(abs(1.0 / f - 1.0), rel=0.05)
        assert s["contact_err_mm_median"] < 3.0


def test_nu_change_needs_only_k_rebuild(reference):
    model, basis, rom_ref, pts = reference
    mat = {"youngs_modulus": FEM_CFG["material"]["youngs_modulus"], "poisson_ratio": 0.3, "name": "nu 0.3", "E_factor": 1.0, "sweep": "nu"}
    case = MaterialCase.build(FEM_CFG, mat, basis, rom_ref, pts)
    summ = evaluate_case(case, basis, np.array([1.5]), {"oracle": lambda u, c, F: basis.project(u)})
    s = summ["sources"]["oracle"]
    assert summ["proj_residual_median"] < 0.10           # reference Φ (r=4, coarse mesh) still spans the ν-changed field
    assert s["force_known_rel_median"] < 0.08             # updated K_r keeps the force within the ROM's held-out accuracy
    # ν only changes the shear / local-indentation part of the response (a few % of the force), which is
    # the same order as the Galerkin error at a held-out contact, so stale vs updated K is not ordered
    # robustly here (unlike the E sweep, where the stale error is exactly |E0/E − 1|).
    assert s["force_known_rel_median_staleK"] < 0.08


def test_estimator_pod_reference_material_pins_basis():
    def _build(E_factor: float, pin: bool) -> GraspEstimator:
        cfg = dict(FEM_CFG)
        cfg["material"] = {**FEM_CFG["material"], "youngs_modulus": FEM_CFG["material"]["youngs_modulus"] * E_factor}
        model = build_material_model(cfg, cfg["material"])
        mech = GraspMechanics.build(model, [0.85, 1.0, 0.5], [0.028, 0.024, 0.01], mass=0.12, mu=0.45)
        est_cfg = {"mode": "world", "q_modes": 4, "pod_n_x": 3, "pod_n_z": 2, "_fem_cfg": cfg}
        if pin:
            est_cfg["pod_reference_material"] = FEM_CFG["material"]
        return GraspEstimator.build(mech, est_cfg)

    ref = _build(1.0, pin=False)
    soft_pinned = _build(0.5, pin=True)
    assert soft_pinned.basis_fingerprint == ref.basis_fingerprint  # q convention shared with the reference
    # the pinned estimator still uses the *soft* stiffness: K_r halves
    Kr_ref, Kr_soft = ref.geo._rom.K_r, soft_pinned.geo._rom.K_r
    assert np.allclose(Kr_soft, 0.5 * Kr_ref, rtol=1e-6, atol=1e-9 * np.abs(Kr_ref).max())
