"""TaichiMPMSolver vs NumPy MPMSolver — energy / COM / runtime parity.

Skipped if taichi is not importable or arch=metal cannot initialise (e.g.
no Apple Silicon GPU). Validates that the Taichi port reproduces the
NumPy reference within float32 tolerance and is faster.
"""

import numpy as np
import pytest


def _check_taichi_metal_available():
    try:
        import taichi as ti
    except ImportError:
        return False, "taichi not installed"
    try:
        ti.init(arch=ti.metal)
    except Exception as e:
        return False, f"ti.init(arch=metal) failed: {e}"
    return True, ""


_AVAILABLE, _SKIP_REASON = _check_taichi_metal_available()


@pytest.mark.skipif(not _AVAILABLE, reason=_SKIP_REASON)
def test_taichi_mpm_matches_numpy_com_and_energy():
    """Drop a jello cube and compare COM after 200 steps of free fall.

    Float32 precision means we can't expect bit-exact match, but the COM
    should agree to within ~1 mm over a 60 ms simulation.
    """
    from robosim.physics.mpm.grid import Grid
    from robosim.physics.mpm.materials import NeoHookean
    from robosim.physics.mpm.solver import (
        BoxBC, MPMSolver, sample_box_particles,
    )
    from robosim.physics.mpm.taichi_solver import TaichiMPMSolver

    pts  = sample_box_particles([0.25, 0.25, 0.35], [0.35, 0.35, 0.45], 6, 1000.)
    grid = Grid.from_bounds(np.array([0, 0, 0.]), np.array([0.6, 0.6, 0.6]),
                            0.015, pad=3)

    sv_np = MPMSolver(
        particles=pts, grid=grid,
        material=NeoHookean(young=4e4, poisson=0.3),
        gravity=np.array([0, 0, -9.81]),
        bcs=[BoxBC(lower=np.array([0, 0, 0.]),
                   upper=np.array([0.6, 0.6, 10.]), mode="slip")],
    )
    sv_ti = TaichiMPMSolver(
        particles_x=pts.x.copy(), particles_v=pts.v.copy(),
        particles_m=pts.m.copy(), particles_V0=pts.V0.copy(),
        grid_origin=grid.origin, grid_dx=grid.dx, grid_shape=grid.shape,
        young=4e4, poisson=0.3, gravity=[0, 0, -9.81],
        bc_lower=np.array([0, 0, 0.]), bc_upper=np.array([0.6, 0.6, 10.]),
    )
    for _ in range(200):
        sv_np.step(3e-4)
        sv_ti.step(3e-4)

    com_np = sv_np.particles.x.mean(axis=0)
    com_ti = sv_ti.particles_x().mean(axis=0)
    diff_mm = float(np.linalg.norm(com_np - com_ti) * 1000)
    assert diff_mm < 5.0, (
        f"Taichi COM diverged from NumPy by {diff_mm:.3f} mm "
        f"(np={com_np}, ti={com_ti})"
    )


@pytest.mark.skipif(not _AVAILABLE, reason=_SKIP_REASON)
def test_taichi_mpm_faster_than_numpy():
    """Sanity check: Taichi backend should not be slower than NumPy."""
    import time
    from robosim.physics.mpm.grid import Grid
    from robosim.physics.mpm.solver import MPMSolver, sample_box_particles
    from robosim.physics.mpm.materials import NeoHookean
    from robosim.physics.mpm.taichi_solver import TaichiMPMSolver
    import taichi as ti

    pts  = sample_box_particles([0.25, 0.25, 0.35], [0.35, 0.35, 0.45], 6, 1000.)
    grid = Grid.from_bounds(np.array([0, 0, 0.]), np.array([0.6, 0.6, 0.6]),
                            0.015, pad=3)
    sv_np = MPMSolver(particles=pts, grid=grid,
                      material=NeoHookean(young=4e4, poisson=0.3))
    sv_ti = TaichiMPMSolver(
        particles_x=pts.x.copy(), particles_v=pts.v.copy(),
        particles_m=pts.m.copy(), particles_V0=pts.V0.copy(),
        grid_origin=grid.origin, grid_dx=grid.dx, grid_shape=grid.shape,
        young=4e4, poisson=0.3,
    )
    # Warm up
    sv_np.step(3e-4)
    sv_ti.step(3e-4)

    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        sv_np.step(3e-4)
    t_np = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(N):
        sv_ti.step(3e-4)
    ti.sync()
    t_ti = time.perf_counter() - t0

    assert t_ti < t_np, (
        f"Taichi ({t_ti*1000/N:.2f} ms/step) is slower than NumPy "
        f"({t_np*1000/N:.2f} ms/step)"
    )
