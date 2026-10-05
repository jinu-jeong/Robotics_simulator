#!/usr/bin/env python
"""Milestone 3 – Taichi viewer for inverse force recovery: F_true vs F_est per sample.

For each dataset sample the contact force is re-estimated live from the stored
displacement field (optionally with synthetic Gaussian noise) and both arrows
are drawn at the contact point: green = ground truth, magenta = estimate. The
GUI panel lists F_true, F_est and the relative error; ``A`` toggles between the
(noisy) measured field and the forward model ``u = g(c) λ̂`` of the estimate,
so a bad fit is visible as a mismatch between the two shapes.

Keys (in addition to the standard viewer controls):
    Right / Left          next / previous sample  (Shift: ±10)
    Home / End / Space    first / last / random sample
    Y                     cycle noise level σ  (Shift+Y: backwards)
    D                     toggle method: displacement-space <-> force-space LS
    Q                     toggle mode: normal (scalar λ) <-> vector (3 unknowns)
    U                     cycle observation operator: full -> surface -> top:z

Run:
    python scripts/inspect_inverse_force.py data/processed/contact_sweep_top.npz
    python scripts/inspect_inverse_force.py <file> --start 899 --sigma 1e-5
    python scripts/inspect_inverse_force.py <file> --headless --screenshot /tmp/inv.png --start 899 --sigma 1e-5 --method force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contact.inverse_force import InverseForceSolver, add_displacement_noise, direction_error_deg  # noqa: E402
from src.contact.observation import observation_from_name  # noqa: E402
from src.fem.contact_dataset import ContactDataset  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402

NOISE_LEVELS = [0.0, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4]
OBSERVATIONS = ["full", "surface", "top:z"]


class InverseForceInspector:
    def __init__(self, ds: ContactDataset, model: FingerFEMModel, viewer, start: int = 0,
                 sigma: float = 0.0, method: str = "displacement", mode: str = "normal", seed: int = 0) -> None:
        self.ds, self.model, self.viewer = ds, model, viewer
        self.pos = int(start) % len(ds)
        self.rng = np.random.default_rng(seed)
        self.method, self.mode = method, mode
        self.noise_idx = NOISE_LEVELS.index(sigma) if sigma in NOISE_LEVELS else len(NOISE_LEVELS)
        if self.noise_idx == len(NOISE_LEVELS):
            NOISE_LEVELS.append(sigma)
        self.obs_idx = 0
        self.solvers = {}
        self._install_keys()
        self.show()

    # ----------------------------------------------------------- solver / estimate
    @property
    def sigma(self) -> float:
        return NOISE_LEVELS[self.noise_idx]

    @property
    def observation_name(self) -> str:
        return OBSERVATIONS[self.obs_idx] if self.method == "displacement" else "full"

    def solver(self) -> InverseForceSolver:
        name = self.observation_name
        if name not in self.solvers:
            m = self.model
            self.solvers[name] = InverseForceSolver(m.fem, m.contact, observation_from_name(name, m.mesh, m.partition, m.geometry))
        return self.solvers[name]

    def show(self) -> None:
        i, ds, m = self.pos, self.ds, self.model
        contact = m.contact.contact_from_face(ds.contact_face[i], ds.contact_bary[i])
        u = ds.displacement(i)
        u_meas = add_displacement_noise(u, self.sigma, self.rng, m.partition.free_dofs) if self.sigma > 0 else u
        solver = self.solver()
        est = solver.estimate(u_meas, contact, method=self.method, mode=self.mode)
        F_true, F_est = ds.force_vector[i], est.force_vector
        rel = np.linalg.norm(F_est - F_true) / max(np.linalg.norm(F_true), 1e-300)
        ang = float(direction_error_deg(F_true, F_est)[0]) if est.magnitude > 0 else float("nan")

        info = {
            "method": f"{self.method}-space LS, mode={self.mode}, obs={solver.observation.name} ({solver.observation.n_obs} dofs)",
            "noise σ": f"{self.sigma:.1e} m  ({100 * self.sigma / max(ds.max_displacement[i], 1e-300):.3f} % of max|u|)",
            "λ_true / λ_est": f"{ds.force_magnitude[i]:.4f} / {est.magnitude:.4f} N" + ("  (clipped to 0)" if est.clipped else ""),
            "|F_est - F_true| / |F_true|": f"{100 * rel:.3f} %",
            "direction error": f"{ang:.3f} deg",
            "fit residual": f"{100 * est.residual_rel:.3f} %",
            "keys": "Y noise  D method  Q mode  U observation  A model/measured",
        }
        state = ds.to_visualization_state(i, estimated_force=F_est, extra_info=info)
        state.displacement = u_meas
        state.displacement_alt = solver.predicted_displacement(contact, F_est)
        state.displacement_alt_name = "forward model u = g(c) λ̂"
        self.viewer.set_state(state, refit_camera=None if self.viewer.state is None else False)
        print(f"[inverse] sample {i:4d}  {self.method:>12s}/{self.mode:<6s} obs={solver.observation.name:<8s} σ={self.sigma:.0e}  "
              f"λ_true={ds.force_magnitude[i]:6.3f}  λ_est={est.magnitude:8.4f} N  rel err={100 * rel:8.3f} %  dir err={ang:6.3f}°")

    # ----------------------------------------------------------- keys
    def _install_keys(self) -> None:
        import taichi as ti

        v = self.viewer
        shift = lambda: v.window.is_pressed(ti.ui.SHIFT)  # noqa: E731
        v.register_key(ti.ui.RIGHT, lambda _: self.step(10 if shift() else 1), "next sample (+10 with Shift)")
        v.register_key(ti.ui.LEFT, lambda _: self.step(-10 if shift() else -1), "previous sample (-10 with Shift)")
        v.register_key("Home", lambda _: self.goto(0), "first sample")
        v.register_key("End", lambda _: self.goto(len(self.ds) - 1), "last sample")
        v.register_key(ti.ui.SPACE, lambda _: self.goto(int(self.rng.integers(len(self.ds)))), "random sample")
        v.register_key("y", lambda _: self.cycle_noise(-1 if shift() else 1), "cycle noise level σ")
        v.register_key("d", lambda _: self.toggle_method(), "toggle displacement-/force-space method")
        v.register_key("q", lambda _: self.toggle_mode(), "toggle normal/vector force mode")
        v.register_key("u", lambda _: self.cycle_observation(), "cycle observation operator")

    def step(self, d: int) -> None:
        self.goto(self.pos + d)

    def goto(self, pos: int) -> None:
        self.pos = int(pos) % len(self.ds)
        self.show()

    def cycle_noise(self, d: int) -> None:
        self.noise_idx = (self.noise_idx + d) % len(NOISE_LEVELS)
        self.show()

    def toggle_method(self) -> None:
        self.method = "force" if self.method == "displacement" else "displacement"
        self.show()

    def toggle_mode(self) -> None:
        self.mode = "vector" if self.mode == "normal" else "normal"
        self.show()

    def cycle_observation(self) -> None:
        if self.method != "displacement":
            print("[inverse] the force-space method needs the full field; switch to displacement-space (D) first")
            return
        self.obs_idx = (self.obs_idx + 1) % len(OBSERVATIONS)
        self.show()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default="data/processed/contact_sweep_top.npz")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.0, help="initial noise std [m]")
    p.add_argument("--method", choices=["displacement", "force"], default="displacement")
    p.add_argument("--mode", choices=["normal", "vector"], default="normal")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amplification", type=float, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--screenshot", type=str, default=None)
    args = p.parse_args()

    ds = ContactDataset.load(args.dataset)
    model = FingerFEMModel.from_config(ds.meta["fem_config"])
    if not np.allclose(model.mesh.nodes, ds.nodes):
        raise RuntimeError("rebuilt FEM mesh does not match the dataset mesh")

    from src.visualization.taichi_viewer import TaichiViewer

    viewer = TaichiViewer(show_window=not args.headless, title=f"Inverse force – {Path(args.dataset).name}")
    if args.amplification is not None:
        viewer.options.amplification = args.amplification
    insp = InverseForceInspector(ds, model, viewer, start=args.start, sigma=args.sigma, method=args.method, mode=args.mode, seed=args.seed)
    if args.screenshot:
        viewer.save_screenshot(args.screenshot)
    if not args.headless:
        viewer.run()
    return insp


if __name__ == "__main__":
    main()
