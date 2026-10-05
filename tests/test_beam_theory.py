"""Point-load / eccentric-load beam theory and the FEM's agreement with it."""

import numpy as np
import pytest

from src.fem.beam_theory import CantileverReference, PointLoadCantilever, rectangular_torsion_constant, section_fit
from src.fem.finger_model import FingerFEMModel
from src.utils.config import load_config

L, W, H, E, NU = 0.1, 0.02, 0.01, 5e7, 0.4


def test_torsion_constant_matches_tabulated_values():
    # Roark / Timoshenko-Goodier: J = beta a b^3, beta(1) = 0.1406, beta(2) = 0.229
    assert rectangular_torsion_constant(1.0, 1.0) == pytest.approx(0.1406, rel=5e-3)
    assert rectangular_torsion_constant(2.0, 1.0) / (2.0 * 1.0**3) == pytest.approx(0.229, rel=5e-3)
    assert rectangular_torsion_constant(1.0, 2.0) == rectangular_torsion_constant(2.0, 1.0)


def test_tip_load_reduces_to_cantilever_reference():
    ref = CantileverReference(L, W, H, E, NU, 1.0)
    pl = PointLoadCantilever(L, W, H, E, NU, force_z=-1.0, a=L, e=0.0)
    assert -pl.tip_deflection(shear=False) == pytest.approx(ref.tip_deflection_euler_bernoulli, rel=1e-12)
    assert -pl.tip_deflection(shear=True) == pytest.approx(ref.tip_deflection_timoshenko, rel=1e-12)
    assert pl.tip_twist() == 0.0
    # Deflection curve beyond the load point is a straight line (zero curvature).
    x = np.linspace(0.6, 1.0, 5) * L
    pl2 = PointLoadCantilever(L, W, H, E, NU, force_z=-1.0, a=0.5 * L)
    w = pl2.bending_deflection(x, shear=False)
    assert np.allclose(np.diff(w, 2), 0.0, atol=1e-15)


def test_lateral_axis_uses_width_as_bending_thickness():
    ref_z = CantileverReference(L, W, H, E, NU, 1.0)
    ref_y = CantileverReference(L, W, H, E, NU, 1.0, axis="y")
    assert ref_y.second_moment == pytest.approx(H * W**3 / 12.0)
    # W = 2H → I_y = 4 I_z → lateral bending deflection is a quarter of the top-face one
    assert ref_y.tip_deflection_euler_bernoulli == pytest.approx(0.25 * ref_z.tip_deflection_euler_bernoulli)
    # shear term is identical (same A, κ, G) so its fraction is larger for the stiffer direction
    assert ref_y.tip_deflection_timoshenko - ref_y.tip_deflection_euler_bernoulli == pytest.approx(
        ref_z.tip_deflection_timoshenko - ref_z.tip_deflection_euler_bernoulli)
    assert ref_y.summary()["slenderness_L_over_H"] == pytest.approx(L / W)
    with pytest.raises(ValueError):
        CantileverReference(L, W, H, E, NU, 1.0, axis="x")


def test_fem_lateral_tip_load_converges_to_lateral_beam_theory():
    """Lateral (−y) end-face load, as in the grasp: FEM slightly below theory, no y–z coupling."""
    from src.fem.boundary import clamp_nodes
    from src.fem.loads import total_force_on_plane
    from src.fem.material import LinearElasticMaterial
    from src.fem.solver import LinearFEM
    from src.geometry.finger import FingerGeometry, make_rectangular_finger_mesh, root_fixed_nodes

    geom = FingerGeometry(L, W, H)
    mat = LinearElasticMaterial(E, NU)

    def tip_solve(res):
        mesh = make_rectangular_finger_mesh(geom, *res)
        fem = LinearFEM(mesh, mat, clamp_nodes(mesh.n_nodes, root_fixed_nodes(mesh)))
        tip = mesh.nodes_on_plane(0, L)
        f_y = total_force_on_plane(mesh, 0, L, np.array([0.0, -1.0, 0.0]))
        f_z = total_force_on_plane(mesh, 0, L, np.array([0.0, 0.0, -1.0]))
        U_y, U_z = fem.solve(f_y).u, fem.solve(f_z).u
        return U_y[tip].mean(axis=0), U_z[tip].mean(axis=0), f_y @ U_z.reshape(-1), f_z @ U_y.reshape(-1)

    u_y, u_z, w_yz, w_zy = tip_solve((20, 4, 2))
    ratio_y = -u_y[1] / CantileverReference(L, W, H, E, NU, 1.0, axis="y").tip_deflection_timoshenko
    ratio_z = -u_z[2] / CantileverReference(L, W, H, E, NU, 1.0, axis="z").tip_deflection_timoshenko
    assert 0.97 < ratio_y < 1.0
    assert 0.95 < ratio_z < 1.0
    # Hex-dominant mesh: the y–z coupling of the old all-Kuhn T4 mesh (13 % at 30×6×4) is gone;
    # what remains is reciprocal (Maxwell–Betti: f_y·u_z == f_z·u_y) and tiny (only the tip is Kuhn-split).
    assert w_yz == pytest.approx(w_zy, rel=1e-8)
    assert abs(u_y[2]) < 1e-3 * abs(u_y[1])


def test_eccentric_load_twists_with_correct_sign():
    # Pressing down (F_z < 0) on the +y side must push that side further down.
    pl = PointLoadCantilever(L, W, H, E, NU, force_z=-1.0, a=0.9 * L, e=0.007)
    nodes = np.array([[L, 0.0, H], [L, W, H]])
    u = pl.displacement_field(nodes)
    assert u[1, 2] < u[0, 2] < 0.0
    assert pl.tip_twist() < 0.0
    assert PointLoadCantilever(L, W, H, E, NU, -1.0, 0.9 * L, -0.007).tip_twist() == pytest.approx(-pl.tip_twist())


def test_section_fit_recovers_kinematic_field():
    pl = PointLoadCantilever(L, W, H, E, NU, force_z=-1.0, a=0.7 * L, e=0.005)
    xs = np.linspace(0, L, 11)
    ys = np.linspace(0, W, 4)
    zs = np.linspace(0, H, 3)
    nodes = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    u = pl.displacement_field(nodes)
    x_fit, w_fit, t_fit = section_fit(nodes, u, W, H)
    assert np.allclose(x_fit, xs)
    assert np.allclose(w_fit, pl.bending_deflection(xs), rtol=1e-10, atol=1e-15)
    assert np.allclose(t_fit, pl.twist(xs), rtol=1e-10, atol=1e-15)


@pytest.fixture(scope="module")
def coarse_model():
    cfg = load_config("fem")
    m = FingerFEMModel.from_config(cfg, nx=30, ny=4, nz=6)
    m.restrict_contact_surface("top")
    return m


def test_fem_bending_and_torsion_follow_beam_theory(coarse_model):
    m, g = coarse_model, coarse_model.geometry
    F = 1.0
    res, c = m.solve_normal_contact(g.point_from_relative([0.9, 0.85, 1.0]), F)
    res0, _ = m.solve_normal_contact(g.point_from_relative([0.9, 0.5, 1.0]), F)
    th = PointLoadCantilever(g.length, g.width, g.height, m.material.E, m.material.nu,
                             force_z=-F, a=float(c.position[0]), e=float(c.position[1] - 0.5 * g.width))
    xs, w_fem, t_fem = section_fit(m.mesh.nodes, res.u, g.width, g.height)
    t_bias = section_fit(m.mesh.nodes, res0.u, g.width, g.height)[2]
    w_th, t_th = th.bending_deflection(xs), th.twist(xs)

    # T4 elements are over-stiff: FEM magnitude below theory, same sign, within 40 % on this coarse mesh.
    assert w_fem[-1] < 0 and w_th[-1] < 0
    assert 0.6 < w_fem[-1] / w_th[-1] < 1.0
    t_deb = t_fem[-1] - t_bias[-1]
    assert t_deb < 0 and t_th[-1] < 0
    assert 0.6 < t_deb / t_th[-1] < 1.0
    # Shape: normalised deflection curve close to theory everywhere.
    assert np.max(np.abs(w_fem / w_fem[-1] - w_th / w_th[-1])) < 0.05
    # Twist grows linearly up to the load and stays constant beyond it (within the free-end zone).
    beyond = xs > th.a + 1e-9
    if beyond.any():
        assert np.ptp(t_fem[beyond]) < 0.1 * abs(t_deb)
