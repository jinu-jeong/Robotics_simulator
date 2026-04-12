"""Test that FEM bodies rotate correctly under asymmetric contact."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.spatial.transform import Rotation
from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.solver import FEMSolver, DeformableBody
from robosim.physics.contact.detection import GroundPlane
from robosim.physics.contact.solver import ContactSolver


def _make_tilted_cube(tilt_deg=30.0, drop_height=0.3, side=0.2):
    mesh = TetMesh.create_box(
        origin=np.array([-side / 2, -side / 2, drop_height]),
        size=np.array([side, side, side]),
        divisions=(3, 3, 3),
    )
    com = mesh.nodes.mean(axis=0)
    R = Rotation.from_euler('y', tilt_deg, degrees=True).as_matrix()
    mesh.nodes[:] = (R @ (mesh.nodes - com).T).T + com
    return mesh


def _run(mesh, n_steps=400):
    body = DeformableBody(
        name="cube", mesh=mesh,
        material=CorotationalElastic(young=1e6, poisson=0.3),
        density=1000.0,
    )
    fem = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]), damping=0.02)
    fem.initialize(dt=0.001)
    contact = ContactSolver(ground=GroundPlane(height=0.0))
    min_z = 999.0
    for _ in range(n_steps):
        fem.step()
        contact.resolve_fem_contact(body, restitution=0.3, friction_mu=0.5)
        min_z = min(min_z, body.x[:, 2].min())
    return body, min_z


class TestFEMRotation:
    def test_tilted_drop_produces_rotation(self):
        """30-deg tilted cube must show lateral CoM shift."""
        mesh = _make_tilted_cube(tilt_deg=30.0)
        body, _ = _run(mesh)
        dx = abs(body.x.mean(axis=0)[0])
        assert dx > 0.005, f"CoM X shift too small ({dx:.4f}m)"

    def test_flat_drop_no_lateral_drift(self):
        """Flat cube should drop straight down."""
        mesh = TetMesh.create_box(
            origin=np.array([-0.1, -0.1, 0.3]),
            size=np.array([0.2, 0.2, 0.2]),
            divisions=(3, 3, 3),
        )
        body, _ = _run(mesh)
        com = body.x.mean(axis=0)
        assert abs(com[0]) < 0.002, f"Flat drop X drift: {com[0]:.4f}"
        assert abs(com[1]) < 0.002, f"Flat drop Y drift: {com[1]:.4f}"

    def test_no_deep_penetration(self):
        """No node should be below ground."""
        mesh = _make_tilted_cube(tilt_deg=45.0)
        _, min_z = _run(mesh)
        assert min_z >= -1e-10, f"Penetration: {min_z:.6f}"

    def test_no_nan_explosion(self):
        """Mesh must not explode."""
        mesh = _make_tilted_cube(tilt_deg=45.0, drop_height=0.5)
        body, _ = _run(mesh, n_steps=500)
        assert np.all(np.isfinite(body.x)), "NaN in positions"
        assert np.all(np.isfinite(body.v)), "NaN in velocities"
        assert np.abs(body.x).max() < 10.0, "Positions diverged"
