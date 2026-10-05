"""Tests for the config-driven FingerFEMModel (point contact through B(c))."""

import numpy as np
import pytest

from src.fem.beam_theory import CantileverReference
from src.fem.finger_model import FingerFEMModel
from src.utils.config import load_config


@pytest.fixture(scope="module")
def model():
    return FingerFEMModel.from_config(load_config("fem"), nx=20, ny=4, nz=4)


def test_build_from_config(model):
    n_corner = 21 * 5 * 5
    assert model.mesh.n_nodes == n_corner + len(model.mesh.element_edges)  # corners + one mid node per edge
    assert model.mesh.n_hexes == 16 * 4 * 4 and model.mesh.n_tets == 6 * 4 * 4 * 4  # tip 20 % Kuhn-split
    assert model.partition.n_fixed == 3 * (25 + 4 * 5 + 4 * 5)  # root corners + root mid-edge nodes
    assert model.fem._lu is not None
    # root face excluded from contact candidates
    assert not np.any(np.all(np.abs(model.mesh.nodes[model.contact.faces][:, :, 0]) < 1e-9, axis=1))


def test_point_force_bends_towards_load_and_balances(model):
    g = model.geometry
    res, c = model.solve_point_force([0.9 * g.length, 0.5 * g.width, g.height], [0, 0, -1.0])
    assert np.allclose(c.position[2], g.height)
    assert np.allclose(res.applied_resultant(), [0, 0, -1.0])
    assert res.equilibrium_residual() < 1e-9
    assert np.allclose(res.u[res.fixed_nodes], 0.0)
    tip = res.u[model.mesh.nodes_on_plane(0, g.length), 2].mean()
    assert tip < 0
    # displacement grows monotonically along the beam (mid-plane nodes)
    mid = model.mesh.nodes_on_plane(1, 0.5 * g.width)
    xs = model.mesh.nodes[mid, 0]
    uz = -res.u[mid, 2]
    order = np.argsort(xs)
    # average per x-station then check monotone increase
    stations = np.unique(xs)
    means = np.array([uz[xs == s].mean() for s in stations])
    assert np.all(np.diff(means) > -1e-12)
    # within the expected T4 stiffness band of beam theory (load at a = 0.9 L)
    a = 0.9 * g.length
    EI = model.material.E * g.width * g.height**3 / 12
    delta_tip_beam = 1.0 * a**2 * (3 * g.length - a) / (6 * EI)
    assert 0.55 * delta_tip_beam < -tip < 1.05 * delta_tip_beam


def test_normal_contact_matches_point_force(model):
    g = model.geometry
    p = [0.7 * g.length, 0.3 * g.width, g.height]
    r1, c1 = model.solve_normal_contact(p, 1.5)
    r2, c2 = model.solve_point_force(p, 1.5 * -c1.normal)
    assert np.allclose(r1.u, r2.u)
    assert np.allclose(c1.position, c2.position)


def test_side_contact_deflects_sideways(model):
    g = model.geometry
    res, c = model.solve_point_force([0.5 * g.length, g.width, 0.5 * g.height], [0, -1.0, 0])
    assert np.allclose(c.normal, [0, 1, 0])
    tip = res.u[model.mesh.nodes_on_plane(0, g.length)]
    assert tip[:, 1].mean() < 0
    assert abs(tip[:, 1].mean()) > 5 * abs(tip[:, 2].mean())


def test_visualization_state_from_model(model):
    g = model.geometry
    res, c = model.solve_point_force([0.9 * g.length, 0.5 * g.width, g.height], [0, 0, -1.0])
    st = model.visualization_state(res, c, [0, 0, -1.0], estimated_force=[0, 0, -0.9])
    assert st.ground_truth_force.magnitude == pytest.approx(1.0)
    assert st.estimated_force.magnitude == pytest.approx(0.9)
    assert len(st.objects) == 1 and st.objects[0].attach_to is not None
    assert st.node_scalar is not None and st.node_scalar.max() > 0
    assert st.contact_points[0].node_index in c.node_indices
