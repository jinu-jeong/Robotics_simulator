"""Verify impulse-based FEM contact: energy decay, rotation, stability."""

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


def _drop(drop_height=0.4, tilt_deg=0.0, n_steps=600, restitution=0.3):
    side = 0.2
    mesh = TetMesh.create_box(
        origin=np.array([-side / 2, -side / 2, drop_height]),
        size=np.array([side, side, side]),
        divisions=(3, 3, 3),
    )
    if tilt_deg:
        com = mesh.nodes.mean(axis=0)
        R = Rotation.from_euler('y', tilt_deg, degrees=True).as_matrix()
        mesh.nodes[:] = (R @ (mesh.nodes - com).T).T + com

    body = DeformableBody(
        name="cube", mesh=mesh,
        material=CorotationalElastic(young=1e6, poisson=0.3),
        density=1000.0,
    )
    fem = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]), damping=0.02)
    fem.initialize(dt=0.001)
    contact = ContactSolver(ground=GroundPlane(height=0.0))

    M_diag = body._M.diagonal()
    mass = M_diag[::3]
    total_mass = mass.sum()

    com_z_history = []
    min_z = 999.0
    for _ in range(n_steps):
        fem.step()
        contact.resolve_fem_contact(body, restitution=restitution, friction_mu=0.5)
        com_z_history.append(float((mass * body.x[:, 2]).sum() / total_mass))
        min_z = min(min_z, body.x[:, 2].min())

    return body, np.array(com_z_history), min_z


class TestImpulseContact:
    def test_energy_decay(self):
        """Bounce height must be less than drop height."""
        _, cz, _ = _drop(drop_height=0.4, restitution=0.3)
        initial = cz[0]
        peak_after = cz[250:].max()
        assert peak_after < initial, (
            f"Energy increased: bounce {peak_after:.4f} > drop {initial:.4f}"
        )

    def test_tilted_rotation(self):
        """Tilted cube must rotate (CoM X shift)."""
        body, _, _ = _drop(tilt_deg=30.0, n_steps=400)
        dx = abs(body.x.mean(axis=0)[0])
        assert dx > 0.005, f"No rotation: dx={dx:.4f}"

    def test_no_penetration(self):
        """No node should ever be below ground after contact resolution."""
        _, _, min_z = _drop(tilt_deg=45.0, drop_height=0.5, n_steps=500)
        assert min_z >= -1e-10, f"Penetration: z_min={min_z:.6f}"

    def test_no_nan(self):
        """Mesh must not explode."""
        body, _, _ = _drop(tilt_deg=45.0, drop_height=0.5, n_steps=500)
        assert np.all(np.isfinite(body.x)), "NaN in positions"
        assert np.all(np.isfinite(body.v)), "NaN in velocities"
