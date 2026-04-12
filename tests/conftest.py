"""Shared test fixtures."""

from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
URDF_DIR = EXAMPLES_DIR / "urdf"


@pytest.fixture
def simple_arm_urdf():
    return URDF_DIR / "simple_arm.urdf"
