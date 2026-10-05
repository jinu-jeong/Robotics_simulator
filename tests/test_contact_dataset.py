"""Dataset generation + round-trip tests (small mesh, few samples)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_fem_dataset import generate  # noqa: E402
from src.fem.contact_dataset import ContactDataset  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DS_CFG = {
    "seed": 3,
    "store_dtype": "float64",
    "contact": {"surface": "top", "sampling": "grid", "x_rel": {"min": 0.4, "max": 0.9, "n": 3}, "y_rel": {"min": 0.25, "max": 0.75, "n": 2}},
    "force": {"sampling": "grid", "magnitude": {"min": 0.5, "max": 1.0, "n": 2}},
}


@pytest.fixture(scope="module")
def generated():
    fem_cfg = load_config("fem")
    model = FingerFEMModel.from_config(fem_cfg, nx=16, ny=4, nz=3)
    return generate(DS_CFG, fem_cfg, model, verbose=False), model


def test_shapes_and_labels(generated):
    ds, model = generated
    assert len(ds) == 3 * 2 * 2
    assert ds.U.shape == (12, model.mesh.n_nodes, 3)
    g = model.geometry
    assert np.allclose(ds.contact_position[:, 2], g.height)  # on the top surface
    assert np.allclose(ds.contact_normal, [0, 0, 1])
    assert np.allclose(ds.force_vector, -ds.force_magnitude[:, None] * ds.contact_normal)
    assert np.allclose(ds.max_displacement, np.linalg.norm(ds.U, axis=2).max(axis=1))
    assert np.all(ds.U[:, ds.fixed_nodes, :] == 0.0)
    # face index refers to the global surface_faces array and contains the contact point
    for i in range(len(ds)):
        tri = ds.nodes[ds.surface_faces[ds.contact_face[i]]]
        assert np.allclose(ds.contact_bary[i] @ tri, ds.contact_position[i])


def test_linearity_across_force_levels(generated):
    ds, _ = generated
    # samples are ordered (contact-major, force-minor): pairs (0.5 N, 1.0 N) share a contact
    for k in range(0, len(ds), 2):
        assert np.allclose(ds.contact_position[k], ds.contact_position[k + 1])
        assert np.allclose(ds.U[k + 1], 2.0 * ds.U[k], rtol=1e-9, atol=1e-15)


def test_deflection_grows_with_contact_distance_from_root(generated):
    ds, _ = generated
    one_newton = ds.force_magnitude == 1.0
    x = ds.contact_position[one_newton, 0]
    umax = ds.max_displacement[one_newton]
    for xa in np.unique(x):
        for xb in np.unique(x):
            if xb > xa:
                assert umax[x == xb].mean() > umax[x == xa].mean()


def test_roundtrip_npz(generated, tmp_path):
    ds, _ = generated
    p = ds.save(tmp_path / "ds.npz")
    ds2 = ContactDataset.load(p)
    assert len(ds2) == len(ds)
    for name in ("nodes", "tets", "hexes", "surface_faces", "surface_edges", "element_edges", "fixed_nodes", "U", "contact_position", "contact_normal",
                 "contact_face", "contact_bary", "force_magnitude", "force_vector", "max_displacement"):
        assert np.array_equal(getattr(ds, name), getattr(ds2, name)), name
    assert ds2.meta["material"]["youngs_modulus"] == ds.meta["material"]["youngs_modulus"]
    assert ds2.meta["dataset_config"]["contact"]["surface"] == "top"


def test_snapshot_matrix_and_subset(generated):
    ds, _ = generated
    X = ds.snapshot_matrix()
    assert X.shape == (ds.n_dofs, len(ds))
    assert np.allclose(X[:, 3], ds.U[3].reshape(-1))
    sub = ds.subset([1, 5, 7])
    assert len(sub) == 3 and np.allclose(sub.U[1], ds.U[5])


def test_sample_to_visualization_state(generated):
    ds, _ = generated
    st = ds.to_visualization_state(5, estimated_force=[0, 0, -0.4])
    assert st.n_nodes == ds.n_nodes
    assert st.ground_truth_force.magnitude == pytest.approx(ds.force_magnitude[5])
    assert st.estimated_force is not None
    assert st.contact_points[0].node_index in ds.surface_faces[ds.contact_face[5]]
    assert "sample" in st.info


def test_random_sampling_and_side_surface():
    fem_cfg = load_config("fem")
    model = FingerFEMModel.from_config(fem_cfg, nx=10, ny=2, nz=2)
    cfg = {
        "seed": 1, "store_dtype": "float32",
        "contact": {"surface": "side_pos_y", "sampling": "random", "n_random": 4,
                    "x_rel": {"min": 0.3, "max": 0.9}, "y_rel": {"min": 0.2, "max": 0.8}},
        "force": {"sampling": "random", "magnitude": {"min": 0.1, "max": 2.0}, "n_random": 2},
    }
    ds = generate(cfg, fem_cfg, model, verbose=False)
    assert len(ds) == 8 and ds.U.dtype == np.float32
    assert np.allclose(ds.contact_position[:, 1], model.geometry.width)
    assert np.allclose(ds.contact_normal, [0, 1, 0])
    assert np.all((ds.force_magnitude >= 0.1) & (ds.force_magnitude <= 2.0))
    # side load -> dominant -y deflection
    assert np.all(ds.U[:, :, 1].min(axis=1) < 0)
