"""Stage-5 parity: C++ contact narrow-phase kernels match the Python
reference. Run each pair (Python via env-flag override, C++ via default).
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cpp_contact():
    try:
        from robosim import _cpp
        assert hasattr(_cpp, "contact")
    except (ImportError, AssertionError) as e:
        pytest.skip(f"robosim._cpp.contact not built: {e}")
    return _cpp.contact


def _reload_sdf_with_env(disable_cpp: bool):
    """Reload sdf.py with the C++-disable env var set/unset so the
    module-level ``_USE_CPP_CONTACT`` flag is recomputed."""
    if disable_cpp:
        os.environ["ROBOSIM_NO_CPP_CONTACT"] = "1"
    else:
        os.environ.pop("ROBOSIM_NO_CPP_CONTACT", None)
    from robosim.physics.contact import sdf
    return importlib.reload(sdf)


def _close(a, b, atol=1e-12):
    np.testing.assert_allclose(a, b, atol=atol)


def _assert_cp_eq(a, b, atol=1e-10):
    if a is None and b is None:
        return
    assert (a is None) == (b is None), f"contact existence mismatch: {a} vs {b}"
    _close(a.point_a, b.point_a, atol=atol)
    _close(a.point_b, b.point_b, atol=atol)
    _close(a.normal,  b.normal,  atol=atol)
    assert abs(a.penetration - b.penetration) < atol


def test_sphere_ground_parity(cpp_contact):
    sdf_py = _reload_sdf_with_env(disable_cpp=True)
    sdf_cp = _reload_sdf_with_env(disable_cpp=False)
    try:
        for c, r in [
            (np.array([0.0, 0.0, 0.4]), 0.5),
            (np.array([0.0, 0.0, 0.6]), 0.5),    # no contact
            (np.array([1.0, -2.0, 0.0]), 0.5),
        ]:
            cp_py = sdf_py.sphere_ground(c, r, 0.0, np.array([0., 0., 1.]))
            cp_cp = sdf_cp.sphere_ground(c, r, 0.0, np.array([0., 0., 1.]))
            _assert_cp_eq(cp_py, cp_cp)
    finally:
        _reload_sdf_with_env(disable_cpp=False)


def test_box_ground_parity(cpp_contact):
    sdf_py = _reload_sdf_with_env(disable_cpp=True)
    sdf_cp = _reload_sdf_with_env(disable_cpp=False)
    try:
        # Slightly rotated box partially below ground.
        theta = 0.2
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0, 0, 1.0]])
        c = np.array([0.5, 0.0, 0.03])
        h = np.array([0.1, 0.1, 0.1])
        py = sdf_py.box_ground(c, R, h, 0.0)
        cp = sdf_cp.box_ground(c, R, h, 0.0)
        assert len(py) == len(cp)
        for a, b in zip(py, cp):
            _assert_cp_eq(a, b)
    finally:
        _reload_sdf_with_env(disable_cpp=False)


def test_box_box_parity(cpp_contact):
    sdf_py = _reload_sdf_with_env(disable_cpp=True)
    sdf_cp = _reload_sdf_with_env(disable_cpp=False)
    try:
        # Two overlapping boxes (one slightly rotated)
        ca = np.array([0.0, 0.0, 0.0]); ha = np.array([0.1, 0.1, 0.1])
        cb = np.array([0.18, 0.02, 0.0]); hb = np.array([0.1, 0.1, 0.1])
        Ra = np.eye(3)
        theta = 0.3
        Rb = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta),  np.cos(theta), 0],
                       [0, 0, 1.0]])
        py = sdf_py.box_box(ca, Ra, ha, cb, Rb, hb)
        cp = sdf_cp.box_box(ca, Ra, ha, cb, Rb, hb)
        assert len(py) == len(cp)
        for a, b in zip(py, cp):
            _assert_cp_eq(a, b, atol=1e-9)
    finally:
        _reload_sdf_with_env(disable_cpp=False)


def test_box_box_separated(cpp_contact):
    sdf_py = _reload_sdf_with_env(disable_cpp=True)
    sdf_cp = _reload_sdf_with_env(disable_cpp=False)
    try:
        ca = np.array([0.0, 0.0, 0.0]); ha = np.array([0.1, 0.1, 0.1])
        cb = np.array([1.0, 0.0, 0.0]); hb = np.array([0.1, 0.1, 0.1])
        py = sdf_py.box_box(ca, np.eye(3), ha, cb, np.eye(3), hb)
        cp = sdf_cp.box_box(ca, np.eye(3), ha, cb, np.eye(3), hb)
        assert py == [] and cp == []
    finally:
        _reload_sdf_with_env(disable_cpp=False)


def test_box_sphere_parity(cpp_contact):
    sdf_py = _reload_sdf_with_env(disable_cpp=True)
    sdf_cp = _reload_sdf_with_env(disable_cpp=False)
    try:
        bc = np.array([0.0, 0.0, 0.0]); bh = np.array([0.1, 0.1, 0.1])
        theta = 0.4
        Rb = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta),  np.cos(theta), 0],
                       [0, 0, 1.0]])
        for sc, sr in [
            (np.array([0.15, 0.02, 0.0]), 0.05),    # touching
            (np.array([0.5, 0.0, 0.0]), 0.05),      # no contact
            (np.array([0.0, 0.0, 0.0]), 0.02),      # inside
        ]:
            a = sdf_py.box_sphere(bc, Rb, bh, sc, sr)
            b = sdf_cp.box_sphere(bc, Rb, bh, sc, sr)
            _assert_cp_eq(a, b)
    finally:
        _reload_sdf_with_env(disable_cpp=False)


def test_sphere_sphere_parity(cpp_contact):
    sdf_py = _reload_sdf_with_env(disable_cpp=True)
    sdf_cp = _reload_sdf_with_env(disable_cpp=False)
    try:
        for c1, r1, c2, r2 in [
            (np.array([0., 0., 0.]), 0.5, np.array([0.8, 0., 0.]), 0.5),
            (np.array([0., 0., 0.]), 0.5, np.array([2.0, 0., 0.]), 0.5),
        ]:
            a = sdf_py.sphere_sphere(c1, r1, c2, r2)
            b = sdf_cp.sphere_sphere(c1, r1, c2, r2)
            _assert_cp_eq(a, b)
    finally:
        _reload_sdf_with_env(disable_cpp=False)
