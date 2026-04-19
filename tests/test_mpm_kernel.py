"""Sanity tests for the quadratic B-spline MPM kernel.

Verifies partition-of-unity, zero-sum gradients, linear-field reproduction,
and the tensor-product assembly.
"""

from __future__ import annotations

import numpy as np
import pytest

from robosim.physics.mpm.grid import Grid
from robosim.physics.mpm.kernel import (
    quadratic_bspline_weights,
    tensor_product_weights,
)
from robosim.physics.mpm.particles import Particles


RNG = np.random.default_rng(seed=0)


def _random_particles(n: int, lower=(-0.1, -0.1, -0.1), upper=(0.4, 0.4, 0.4)):
    lo = np.array(lower)
    hi = np.array(upper)
    return lo + (hi - lo) * RNG.random((n, 3))


def test_partition_of_unity_per_axis():
    x_p = _random_particles(32)
    _, w, _ = quadratic_bspline_weights(x_p, dx=0.05, origin=np.array([-0.2, -0.2, -0.2]))
    # Sum of 3 weights along each axis must equal 1.
    s = w.sum(axis=-1)                # (P, 3)
    np.testing.assert_allclose(s, 1.0, atol=1e-12)


def test_gradient_sum_zero_per_axis():
    x_p = _random_particles(32)
    _, _, dw = quadratic_bspline_weights(x_p, dx=0.05, origin=np.array([-0.2, -0.2, -0.2]))
    s = dw.sum(axis=-1)               # (P, 3)
    np.testing.assert_allclose(s, 0.0, atol=1e-12)


def test_tensor_product_partition_of_unity():
    x_p = _random_particles(16)
    _, w, dw = quadratic_bspline_weights(x_p, dx=0.05, origin=np.array([-0.2, -0.2, -0.2]))
    W, gradW = tensor_product_weights(w, dw)
    # 27-node weights sum to 1.
    np.testing.assert_allclose(W.sum(axis=(1, 2, 3)), 1.0, atol=1e-12)
    # And gradients sum to zero (vector result, sum over stencil).
    np.testing.assert_allclose(gradW.sum(axis=(1, 2, 3)), 0.0, atol=1e-12)


def test_reproduces_position_exactly():
    """Quadratic B-spline should reproduce linear fields exactly.

    In particular, Σ W_ijk * x_node_ijk must equal the particle position.
    """
    dx = 0.05
    origin = np.array([-0.2, -0.2, -0.2])
    x_p = _random_particles(8)
    base, w, dw = quadratic_bspline_weights(x_p, dx=dx, origin=origin)
    W, _ = tensor_product_weights(w, dw)

    for p in range(x_p.shape[0]):
        acc = np.zeros(3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    node_ijk = base[p] + np.array([i, j, k])
                    x_node = origin + node_ijk * dx
                    acc += W[p, i, j, k] * x_node
        np.testing.assert_allclose(acc, x_p[p], atol=1e-12)


def test_reproduces_linear_field_exactly():
    """Gather of a linear field f(x) = a·x + b reproduces the analytic value."""
    dx = 0.05
    origin = np.array([-0.2, -0.2, -0.2])
    a = np.array([1.3, -0.7, 2.1])
    b = 0.42
    x_p = _random_particles(8)
    base, w, dw = quadratic_bspline_weights(x_p, dx=dx, origin=origin)
    W, _ = tensor_product_weights(w, dw)

    for p in range(x_p.shape[0]):
        val = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    node_ijk = base[p] + np.array([i, j, k])
                    x_node = origin + node_ijk * dx
                    val += W[p, i, j, k] * (a @ x_node + b)
        np.testing.assert_allclose(val, a @ x_p[p] + b, atol=1e-12)


def test_gradient_recovers_linear_coefficient():
    """Gather of ∇(a·x + b) via the tensor gradient recovers ``a``."""
    dx = 0.05
    origin = np.array([-0.2, -0.2, -0.2])
    a = np.array([1.3, -0.7, 2.1])
    b = 0.42
    x_p = _random_particles(8)
    base, w, dw = quadratic_bspline_weights(x_p, dx=dx, origin=origin)
    W, gradW = tensor_product_weights(w, dw)

    for p in range(x_p.shape[0]):
        grad_acc = np.zeros(3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    node_ijk = base[p] + np.array([i, j, k])
                    x_node = origin + node_ijk * dx
                    f_node = a @ x_node + b
                    grad_acc += f_node * gradW[p, i, j, k]
        np.testing.assert_allclose(grad_acc, a, atol=1e-10)


def test_particles_default_state():
    x = np.zeros((5, 3))
    v = np.zeros((5, 3))
    m = np.ones(5)
    V0 = np.full(5, 1e-6)
    pts = Particles(x=x, v=v, m=m, V0=V0)
    assert pts.n == 5
    assert pts.F.shape == (5, 3, 3)
    assert pts.C.shape == (5, 3, 3)
    # F defaults to identity, C to zero.
    for p in range(5):
        np.testing.assert_array_equal(pts.F[p], np.eye(3))
    np.testing.assert_array_equal(pts.C, 0.0)


def test_grid_from_bounds_covers_region():
    g = Grid.from_bounds(lower=np.zeros(3), upper=np.full(3, 0.2), dx=0.05, pad=2)
    assert g.m.shape == g.shape
    assert g.v.shape == g.shape + (3,)
    # Any point inside [0, 0.2] must have a valid 3-node stencil within bounds.
    pts = np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1]])
    base, _, _ = quadratic_bspline_weights(pts, dx=g.dx, origin=g.origin)
    assert np.all(base >= 0)
    assert np.all(base + 2 < np.array(g.shape))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
