"""Quadratic B-spline interpolation kernel for MPM.

Standard MPM/MLS-MPM kernel with a 3x3x3 stencil per particle. Base node is
chosen so that the stencil is centred on the particle:

    base = floor(x_p / dx - 0.5)

Per-axis weights (i in {0, 1, 2}) using u = x_p/dx - base ∈ [0.5, 1.5]:

    w_0(u) = 0.5 * (1.5 - u)^2
    w_1(u) = 0.75 - (u - 1.0)^2
    w_2(u) = 0.5 * (u - 0.5)^2

The kernel has C^1 continuity, reproduces linear fields exactly, and its
weights sum to 1 with zero-sum gradients per axis.
"""

from __future__ import annotations

import numpy as np


def quadratic_bspline_weights(
    x_p: np.ndarray,
    dx: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-axis quadratic B-spline weights for a batch of particles.

    Parameters
    ----------
    x_p : (P, 3) float64
        Particle positions in world coordinates.
    dx : float
        Uniform grid spacing.
    origin : (3,) float64
        World-space coordinate of grid node index (0, 0, 0).

    Returns
    -------
    base : (P, 3) int64
        Lower-corner node index of the 3x3x3 stencil for each particle.
    w : (P, 3, 3) float64
        Per-axis weights. ``w[p, a, i]`` is the weight along axis ``a`` for
        stencil index ``i ∈ {0, 1, 2}`` (so the tensor-product 3D weight for
        node offset ``(i, j, k)`` is ``w[p, 0, i] * w[p, 1, j] * w[p, 2, k]``).
    dw_dx : (P, 3, 3) float64
        Per-axis weight gradients with respect to world coordinates.
    """
    origin = np.asarray(origin, dtype=np.float64)
    xi = (x_p - origin) / dx                          # (P, 3) grid-space
    base = np.floor(xi - 0.5).astype(np.int64)        # (P, 3)
    u = xi - base                                     # (P, 3) ∈ [0.5, 1.5]

    w0 = 0.5 * (1.5 - u) ** 2
    w1 = 0.75 - (u - 1.0) ** 2
    w2 = 0.5 * (u - 0.5) ** 2

    # d/du of each weight, then divide by dx for d/dx.
    dw0 = -(1.5 - u) / dx
    dw1 = -2.0 * (u - 1.0) / dx
    dw2 = (u - 0.5) / dx

    w = np.stack([w0, w1, w2], axis=-1)               # (P, 3, 3)
    dw_dx = np.stack([dw0, dw1, dw2], axis=-1)        # (P, 3, 3)
    return base, w, dw_dx


def tensor_product_weights(
    w: np.ndarray,
    dw_dx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 3D tensor-product weights and gradients from per-axis weights.

    Parameters
    ----------
    w : (P, 3, 3)
        Per-axis weights from :func:`quadratic_bspline_weights`.
    dw_dx : (P, 3, 3)
        Per-axis weight gradients.

    Returns
    -------
    W : (P, 3, 3, 3)
        ``W[p, i, j, k] = w[p, 0, i] * w[p, 1, j] * w[p, 2, k]``.
    gradW : (P, 3, 3, 3, 3)
        ``gradW[p, i, j, k, a]`` is the derivative of ``W[p, i, j, k]``
        with respect to world coordinate axis ``a``.
    """
    wx, wy, wz = w[:, 0, :], w[:, 1, :], w[:, 2, :]          # (P, 3) each
    dwx, dwy, dwz = dw_dx[:, 0, :], dw_dx[:, 1, :], dw_dx[:, 2, :]

    W = (
        wx[:, :, None, None]
        * wy[:, None, :, None]
        * wz[:, None, None, :]
    )

    gW_x = (
        dwx[:, :, None, None]
        * wy[:, None, :, None]
        * wz[:, None, None, :]
    )
    gW_y = (
        wx[:, :, None, None]
        * dwy[:, None, :, None]
        * wz[:, None, None, :]
    )
    gW_z = (
        wx[:, :, None, None]
        * wy[:, None, :, None]
        * dwz[:, None, None, :]
    )
    gradW = np.stack([gW_x, gW_y, gW_z], axis=-1)            # (P, 3, 3, 3, 3)
    return W, gradW
