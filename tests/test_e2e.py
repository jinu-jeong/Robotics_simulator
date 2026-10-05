"""Milestone 7 tests: unknown-contact localizer and end-to-end packaging."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import force_metrics, relative_errors, summarize
from src.evaluation.unknown_contact import UnknownContactLocalizer
from src.fem.finger_model import FingerFEMModel
from src.rom.pod import compute_pod
from src.vision.pipeline import build_rom


@pytest.fixture(scope="module")
def small_rom():
    model = FingerFEMModel.from_config({
        "geometry": {"length": 0.1, "width": 0.012, "height": 0.01},
        "mesh": {"nx": 12, "ny": 3, "nz": 2},
        "material": {"youngs_modulus": 5.0e7, "poisson_ratio": 0.4},
        "boundary_conditions": {"fixed_surface": "root"},
    })
    model.restrict_contact_surface("top")
    contacts = [model.geometry.point_from_relative([x, 0.5, 1.0]) for x in (0.45, 0.6, 0.75, 0.9)]
    cols, meta = [], []
    for cpos in contacts:
        for F in (1.0, 2.0):
            res, c = model.solve_normal_contact(cpos, F)
            cols.append(res.u.reshape(-1))
            meta.append({"contact": c, "F": F, "u": res.u})
    basis = compute_pod(np.stack(cols, axis=1))
    basis = basis.truncate(min(4, basis.r))
    rom = build_rom(model, basis, "top")
    return model, rom, basis, meta


def test_metrics_helpers():
    hat = np.array([1.0, 2.0, 3.1])
    true = np.array([1.0, 2.0, 3.0])
    rel = relative_errors(hat, true)
    assert np.allclose(rel[:2], 0.0) and rel[2] == pytest.approx(0.1 / 3.0)
    s = summarize(rel)
    assert s["mean"] == pytest.approx(rel.mean())
    fm = force_metrics(hat, true, np.tile([0, 0, -1.0], (3, 1)) * hat[:, None], np.tile([0, 0, -1.0], (3, 1)) * true[:, None])
    assert fm["force_mae_N"] == pytest.approx(np.mean(np.abs(hat - true)))
    assert fm["direction_cos_mean"] == pytest.approx(1.0)


def test_unknown_contact_recovers_from_oracle_q(small_rom):
    model, rom, basis, meta = small_rom
    loc = UnknownContactLocalizer.from_rom(
        rom, model.surface_face_ids("top"),
        barycentric=[[1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        face_stride=1,
    )
    assert len(loc) > 10
    sample = meta[-1]  # tip-ish, 2 N
    q = basis.project(sample["u"])
    out = loc.localize(q)
    pos_err = np.linalg.norm(out.contact.position - sample["contact"].position)
    assert pos_err < 0.008, f"localization {1e3 * pos_err:.1f} mm too large"
    assert abs(out.magnitude - sample["F"]) / sample["F"] < 0.1
    assert out.residual_rel < 0.05
    loc2 = UnknownContactLocalizer.from_rom(
        rom, model.surface_face_ids("top"),
        barycentric=[[1 / 3, 1 / 3, 1 / 3]],
        face_stride=1, r_loc=3, refine=True, weight="leading",
    )
    out2 = loc2.localize(q)
    pos2 = np.linalg.norm(out2.contact.position - sample["contact"].position)
    assert pos2 < 0.010
    assert abs(out2.magnitude - sample["F"]) / sample["F"] < 0.12


def test_unknown_contact_rejects_empty_influence(small_rom):
    model, rom, basis, meta = small_rom
    loc = UnknownContactLocalizer.from_rom(rom, model.surface_face_ids("top")[:3], face_stride=1, min_influence=1e9)
    with pytest.raises(RuntimeError):
        loc.localize(basis.project(meta[0]["u"]))


def test_build_e2e_pipeline_smoke():
    """Smoke-test packaging against the real processed datasets when present."""
    from pathlib import Path

    if not Path("data/processed/vision_top.npz").exists():
        pytest.skip("vision dataset not generated")
    from src.evaluation.e2e import build_e2e_pipeline
    from src.utils.config import load_config

    cfg = load_config("e2e")
    cfg["unknown_contact"] = {**cfg.get("unknown_contact", {}), "face_stride": 5, "barycentric": [[1 / 3, 1 / 3, 1 / 3]]}
    pipe, split = build_e2e_pipeline(cfg, estimator="ridge")
    idx = np.nonzero(split.test)[0]
    # prefer mid/high loads so relative-error sanity isn't dominated by the noise floor
    idx = idx[pipe.vds.force_magnitude[idx] > 1.0][:8]
    known = pipe.evaluate(idx, known_contact=True)
    assert known["n"] == len(idx) and known["force_rel_median"] < 0.35
    assert known["time_median_ms"] < 50
    unk = pipe.evaluate(idx, known_contact=False)
    assert unk["contact_err_mm_median"] < 40
    pred = pipe.predict_one(int(idx[0]), known_contact=True)
    assert pred.u_hat.shape[1] == 3 and pred.force_magnitude >= 0
