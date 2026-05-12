"""Stage-0 smoke tests for the C++ extension (``robosim._cpp``).

These verify only the build pipeline — that the wheel ships a working
shared object and that the pybind11 + Eigen wiring is alive. Real
physics goes through Stages 1+ and lands in this file's siblings as
each module gets a numpy-reference parity test.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cpp():
    try:
        from robosim import _cpp
    except ImportError as e:
        pytest.skip(f"robosim._cpp not built: {e}")
    return _cpp


def test_module_loads_and_reports_stage(cpp):
    assert cpp is not None
    assert hasattr(cpp, "__stage__")
    assert cpp.__stage__ == 0


def test_add_scalar_smoke(cpp):
    assert cpp.add(2.0, 3.0) == 5.0
    assert cpp.add(-1.5, 0.5) == -1.0


def test_square_array_numpy_roundtrip(cpp):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = cpp.square_array(x)
    np.testing.assert_allclose(y, x * x)
    # Round-trip preserves dtype + length, doesn't crash on empty.
    assert y.dtype == np.float64
    assert cpp.square_array(np.zeros(0)).shape == (0,)
