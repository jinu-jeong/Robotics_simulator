"""Milestone 3 – inverse force recovery from displacement fields (known contact)."""

import numpy as np
import pytest

from src.contact.contact_mapping import contact_from_face
from src.contact.inverse_force import (
    InverseForceSolver,
    add_displacement_noise,
    direction_error_deg,
    force_error_metrics,
)
from src.contact.observation import (
    face_observation,
    full_observation,
    observation_from_name,
    surface_observation,
)
from src.fem.finger_model import FingerFEMModel
from src.utils.config import load_config


@pytest.fixture(scope="module")
def model():
    m = FingerFEMModel.from_config(load_config("fem"), nx=16, ny=4, nz=4)
    m.restrict_contact_surface("top")
    return m


@pytest.fixture(scope="module")
def sample(model):
    g = model.geometry
    res, c = model.solve_normal_contact(g.point_from_relative([0.8, 0.7, 1.0]), 1.7)
    return res, c


# ------------------------------------------------------------------ observation operators
def test_observation_operators(model):
    part, mesh = model.partition, model.mesh
    full = full_observation(part)
    surf = surface_observation(mesh, part)
    top = face_observation(mesh, part, model.geometry, "top")
    assert full.n_obs == part.n_free
    assert 0 < top.n_obs < surf.n_obs < full.n_obs
    assert np.all(np.isin(surf.dofs, part.free_dofs))
    topz = observation_from_name("top:z", mesh, part, model.geometry)
    assert topz.n_obs == top.n_obs // 3
    assert np.all(topz.dofs % 3 == 2)
    u = np.arange(part.n_dof, dtype=float).reshape(-1, 3)
    assert np.array_equal(topz.apply(u), u.reshape(-1)[topz.dofs])
    with pytest.raises(KeyError):
        face_observation(mesh, part, model.geometry, "nowhere")


def test_contact_from_face_roundtrip(model, sample):
    _, c = sample
    c2 = contact_from_face(model.mesh, c.face_index, c.barycentric)
    assert np.allclose(c2.position, c.position)
    assert np.allclose(c2.normal, c.normal)
    assert np.array_equal(c2.node_indices, c.node_indices)


def test_vector_operator_consistent_with_scalar_operator(model, sample):
    _, c = sample
    d = np.array([0.3, -0.4, -0.866])
    d /= np.linalg.norm(d)
    B = model.contact.operator(c, d).toarray().ravel()
    Bxyz = model.contact.vector_operator(c).toarray()
    assert np.allclose(Bxyz @ d, B)
    assert np.allclose(Bxyz.sum(axis=0), 1.0)  # unit force distributed with weights summing to one


# ------------------------------------------------------------------ exact recovery
@pytest.mark.parametrize("method", ["force", "displacement"])
def test_exact_recovery_normal_mode(model, sample, method):
    res, c = sample
    inv = InverseForceSolver(model.fem, model.contact)
    est = inv.estimate(res.u, c, method=method)
    assert est.magnitude == pytest.approx(1.7, rel=1e-8)
    assert np.allclose(est.force_vector, -1.7 * c.normal, atol=1e-8)
    assert not est.clipped
    assert est.residual_rel < 1e-6


def test_exact_recovery_vector_mode_and_direction(model):
    g = model.geometry
    F = np.array([0.4, -0.2, -1.5])  # oblique force: recover all three components
    res, c = model.solve_point_force(g.point_from_relative([0.6, 0.4, 1.0]), F)
    inv = InverseForceSolver(model.fem, model.contact)
    for method in ("force", "displacement"):
        est = inv.estimate(res.u, c, method=method, mode="vector")
        assert np.allclose(est.force_vector, F, rtol=1e-7, atol=1e-9)
        assert direction_error_deg(F, est.force_vector)[0] < 1e-5


def test_partial_observation_recovers_exactly(model, sample):
    res, c = sample
    for name in ("surface", "top", "top:z", "side_pos_y:xz"):
        obs = observation_from_name(name, model.mesh, model.partition, model.geometry)
        inv = InverseForceSolver(model.fem, model.contact, obs)
        assert inv.estimate(res.u, c).magnitude == pytest.approx(1.7, rel=1e-8), name
    # the force-space residual is only defined on the full field
    inv = InverseForceSolver(model.fem, model.contact, surface_observation(model.mesh, model.partition))
    with pytest.raises(ValueError):
        inv.estimate(res.u, c, method="force")


def test_linearity_and_nonnegativity(model, sample):
    res, c = sample
    inv = InverseForceSolver(model.fem, model.contact)
    assert inv.estimate(2.0 * res.u, c).magnitude == pytest.approx(3.4, rel=1e-8)
    # a pulled (not pressed) finger violates λ >= 0 -> clipped to zero
    est = inv.estimate(-res.u, c)
    assert est.magnitude == 0.0 and est.clipped
    assert inv.estimate(-res.u, c, nonneg=False).magnitude == pytest.approx(-1.7, rel=1e-8)


def test_influence_field_cache_and_forward_model(model, sample):
    res, c = sample
    inv = InverseForceSolver(model.fem, model.contact)
    G1 = inv.influence_field(c)
    G2 = inv.influence_field(c)
    assert G1 is G2
    u_pred = inv.predicted_displacement(c, -1.7 * c.normal)
    assert np.allclose(u_pred, res.u, atol=1e-12)


# ------------------------------------------------------------------ noise
def test_displacement_space_is_far_more_noise_robust(model, sample):
    res, c = sample
    inv = InverseForceSolver(model.fem, model.contact)
    rng = np.random.default_rng(1)
    sigma = 1e-3 * res.max_displacement()  # 0.1 % of the peak deformation
    err = {"force": [], "displacement": []}
    for _ in range(5):
        un = add_displacement_noise(res.u, sigma, rng, model.partition.free_dofs)
        for m in err:
            err[m].append(abs(inv.estimate(un, c, method=m, nonneg=False).magnitude - 1.7) / 1.7)
    e_force, e_disp = np.mean(err["force"]), np.mean(err["displacement"])
    assert e_disp < 1e-2
    assert e_force > 10 * e_disp


def test_noise_leaves_fixed_dofs_untouched(model, sample):
    res, _ = sample
    rng = np.random.default_rng(0)
    un = add_displacement_noise(res.u, 1e-4, rng, model.partition.free_dofs)
    assert un.shape == res.u.shape
    fixed = model.partition.fixed_dofs
    assert np.all(un.reshape(-1)[fixed] == res.u.reshape(-1)[fixed])
    assert np.std(un - res.u) == pytest.approx(1e-4 * np.sqrt(model.partition.n_free / model.partition.n_dof), rel=0.1)


# ------------------------------------------------------------------ systematic model errors
def test_mesh_mismatch_gives_bounded_constant_overestimate(model):
    """Truth from a 2x nested finer mesh, inverted with the coarse (stiffer) model."""
    from src.geometry.mesh_utils import coincident_node_map

    cfg = load_config("fem")
    fine = FingerFEMModel.from_config(cfg, nx=32, ny=8, nz=8)
    fine.restrict_contact_surface("top")
    node_map = coincident_node_map(model.mesh.nodes, fine.mesh.nodes)
    inv = InverseForceSolver(model.fem, model.contact)
    ratios = []
    for rel in [(0.5, 0.5), (0.9, 0.5), (0.7, 0.2)]:
        pnt = model.geometry.point_from_relative([*rel, 1.0])
        res_f, _ = fine.solve_normal_contact(pnt, 1.0)
        res_c, c = model.solve_normal_contact(pnt, 1.0)
        u_truth = res_f.u[node_map]
        lam = inv.estimate(u_truth, c).magnitude
        softness = res_f.max_displacement() / res_c.max_displacement()
        assert softness > 1.0  # finer T4 mesh is softer
        assert 1.0 < lam < 1.6  # over-estimate, bounded by the stiffness ratio
        assert lam == pytest.approx(softness, rel=0.05)  # the bias *is* the discretisation stiffness error
        ratios.append(lam)
    assert np.ptp(ratios) < 0.1  # nearly independent of the contact location


def test_contact_location_error_sensitivity(model):
    g = model.geometry
    res, c_true = model.solve_normal_contact(g.point_from_relative([0.8, 0.5, 1.0]), 1.0)
    inv = InverseForceSolver(model.fem, model.contact)

    def est(dx=0.0, dy=0.0):
        q = c_true.position + np.array([dx, dy, 0.0])
        return inv.estimate(res.u, model.contact.locate(q)).magnitude

    # Beam-theory sensitivity of the deflection to the load position a:  d ln δ / da = 2/a − 1/(3L − a)
    a, L = c_true.position[0], g.length
    slope_theory = 2.0 / a - 1.0 / (3 * L - a)  # 1/m
    dx = 6e-3
    slope_fem = -(est(dx=+dx) - est(dx=-dx)) / (2 * dx)
    assert est(dx=+dx) < 1.0 < est(dx=-dx)  # contact assumed closer to the tip -> softer model -> smaller force
    assert slope_fem == pytest.approx(slope_theory, rel=0.3)
    assert abs(est(dy=3e-3) - 1.0) < 0.02  # lateral error is nearly irrelevant (bending dominates)


def test_youngs_modulus_error_scales_estimate_linearly(model):
    import copy

    g = model.geometry
    res, c = model.solve_normal_contact(g.point_from_relative([0.9, 0.5, 1.0]), 1.0)
    cfg = copy.deepcopy(load_config("fem"))
    cfg["material"]["youngs_modulus"] = 1.1 * float(cfg["material"]["youngs_modulus"])
    stiffer = FingerFEMModel.from_config(cfg, nx=16, ny=4, nz=4)
    stiffer.restrict_contact_surface("top")
    lam = InverseForceSolver(stiffer.fem, stiffer.contact).estimate(res.u, c).magnitude
    assert lam == pytest.approx(1.1, rel=1e-6)


# ------------------------------------------------------------------ metrics
def test_force_error_metrics():
    t = np.array([0.5, 1.0, 2.0, 3.0])
    m = force_error_metrics(t, t)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["fit_slope"] == pytest.approx(1.0) and m["r2"] == pytest.approx(1.0)
    m = force_error_metrics(t, 1.1 * t)
    assert m["mean_rel_error"] == pytest.approx(0.1)
    assert m["fit_slope"] == pytest.approx(1.1)
    assert m["max_abs_error"] == pytest.approx(0.3)
    assert direction_error_deg([0, 0, -1], [0, 1, -1])[0] == pytest.approx(45.0)
