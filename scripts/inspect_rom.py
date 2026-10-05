#!/usr/bin/env python
"""Milestone 4 – Taichi viewer: full FEM deformation vs POD reduced-order reconstruction.

For each dataset sample the viewer holds the full displacement ``u`` and the
ROM reconstruction ``Φ_r Φ_rᵀ u``; ``A`` toggles between them (the panel shows
the relative reconstruction error). The force is estimated from the reduced
coordinates ``q̂ = Φ_rᵀ u`` with the reduced mechanics (magenta arrow) next to
the ground truth (green). A mode browser shows the individual POD modes.

Keys (in addition to the standard viewer controls):
    Right / Left          next / previous sample  (Shift: ±10)
    Home / End / Space    first / last / random sample
    . / ,                 more / fewer modes r
    Y                     cycle noise level σ added to u (Shift+Y: backwards)
    D                     toggle reduced force estimator: displacement-space <-> force-space
    Q                     toggle mode browser (shows POD mode k instead of a sample)
    ] / [                 next / previous mode k (in mode browser)
    A                     show ROM reconstruction instead of full u

Run:
    python scripts/inspect_rom.py                                     # dataset + basis from data/processed/
    python scripts/inspect_rom.py --basis data/processed/rom_basis_top.npz --r 4 --start 899
    python scripts/inspect_rom.py --headless --screenshot /tmp/rom.png --r 2 --start 899
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contact.inverse_force import add_displacement_noise  # noqa: E402
from src.fem.contact_dataset import ContactDataset  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.rom.pod import PODBasis  # noqa: E402
from src.rom.reduced_mechanics import ReducedModel  # noqa: E402

NOISE_LEVELS = [0.0, 1e-6, 1e-5, 1e-4, 3e-4]


class ROMInspector:
    def __init__(self, ds: ContactDataset, model: FingerFEMModel, basis: PODBasis, viewer, start: int = 0, r: int = 4,
                 method: str = "displacement", seed: int = 0) -> None:
        self.ds, self.model, self.basis, self.viewer = ds, model, basis, viewer
        self.rom_full = ReducedModel.from_fem(model.fem, basis, model.contact)
        self.r_list = sorted({1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, basis.r} & set(range(1, basis.r + 1)))
        self.r_idx = self.r_list.index(r) if r in self.r_list else min(range(len(self.r_list)), key=lambda i: abs(self.r_list[i] - r))
        self.pos = int(start) % len(ds)
        self.noise_idx = 0
        self.method = method
        self.mode_browser = False
        self.mode_k = 0
        self.rng = np.random.default_rng(seed)
        self.test_set = set(basis.meta.get("test_indices", []))
        self._install_keys()
        self.show()

    @property
    def r(self) -> int:
        return self.r_list[self.r_idx]

    @property
    def sigma(self) -> float:
        return NOISE_LEVELS[self.noise_idx]

    # ----------------------------------------------------------- display
    def show(self) -> None:
        if self.mode_browser:
            self._show_mode()
        else:
            self._show_sample()

    def _show_sample(self) -> None:
        i, ds = self.pos, self.ds
        rom = self.rom_full.truncate(self.r)
        u = ds.displacement(i)
        u_meas = add_displacement_noise(u, self.sigma, self.rng, self.model.partition.free_dofs) if self.sigma > 0 else u
        q = rom.basis.project(u_meas)
        u_rom = rom.basis.reconstruct(q)
        rec_err = np.linalg.norm(u_rom - u) / max(np.linalg.norm(u), 1e-300)
        contact = self.model.contact.contact_from_face(ds.contact_face[i], ds.contact_bary[i])
        est = rom.estimate_force(q, contact, method=self.method)
        F_true = ds.force_vector[i]
        rel = np.linalg.norm(est.force_vector - F_true) / max(np.linalg.norm(F_true), 1e-300)
        split = "held-out contact" if i in self.test_set else "train contact"
        info = {
            "ROM": f"r = {self.r} modes ({100 * self.basis.energy_fraction(self.r):.4f} % energy), sample is a {split}",
            "reconstruction": f"‖u − Φ_rΦ_rᵀu‖/‖u‖ = {100 * rec_err:.3f} %   (A toggles full <-> ROM)",
            "q̂ (first 4)": "  ".join(f"{v * 1e3:+.3f}" for v in q[:4]) + " (×1e-3)",
            "force": f"{est.method}: λ_true {ds.force_magnitude[i]:.4f} / λ̂ {est.magnitude:.4f} N  (rel err {100 * rel:.3f} %)",
            "noise σ": f"{self.sigma:.0e} m",
            "keys": ". , modes | Y noise | D estimator | Q mode browser",
        }
        state = ds.to_visualization_state(i, estimated_force=est.force_vector, extra_info=info)
        state.displacement = u_meas
        state.displacement_alt = u_rom
        state.displacement_alt_name = f"ROM r={self.r}"
        self.viewer.set_state(state, refit_camera=None if self.viewer.state is None else False)
        print(f"[rom] sample {i:4d} ({split:16s}) r={self.r:3d} σ={self.sigma:.0e}  rec err {100 * rec_err:7.3f} %   "
              f"λ {ds.force_magnitude[i]:6.3f} -> {est.magnitude:8.4f} N ({100 * rel:+8.3f} %)  [{est.method}]")

    def _show_mode(self) -> None:
        from src.visualization.state import VisualizationState

        k = self.mode_k
        phi = self.basis.mode(k)
        scale = 0.05 * self.model.geometry.length / max(np.abs(phi).max(), 1e-300)  # 5 % of L peak, visual only
        s = self.basis.singular_values
        info = {
            "POD mode": f"k = {k + 1} / {self.basis.r}   σ_k/σ_1 = {s[k] / s[0]:.3e}   energy share {100 * s[k] ** 2 / np.sum(s ** 2):.4f} %",
            "display": f"φ_k scaled to {1e3 * 0.05 * self.model.geometry.length:.1f} mm peak (unit-norm mode, visual only)",
            "keys": "] [ mode | Q back to samples",
        }
        state = VisualizationState(
            nodes_original=self.ds.nodes, surface_faces=self.ds.surface_faces, element_edges=self.ds.element_edges, surface_edges=self.ds.surface_edges,
            displacement=scale * phi, fixed_nodes=self.ds.fixed_nodes, info=info,
        )
        self.viewer.set_state(state, refit_camera=False)
        print(f"[rom] mode {k + 1}: σ/σ1 = {s[k] / s[0]:.3e}")

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
        v.register_key(".", lambda _: self.change_r(+1), "more modes")
        v.register_key(",", lambda _: self.change_r(-1), "fewer modes")
        v.register_key("y", lambda _: self.cycle_noise(-1 if shift() else 1), "cycle noise level")
        v.register_key("d", lambda _: self.toggle_method(), "toggle reduced force estimator")
        v.register_key("q", lambda _: self.toggle_browser(), "toggle POD mode browser")
        v.register_key("]", lambda _: self.change_mode(+1), "next POD mode")
        v.register_key("[", lambda _: self.change_mode(-1), "previous POD mode")

    def step(self, d: int) -> None:
        self.goto(self.pos + d)

    def goto(self, pos: int) -> None:
        self.pos = int(pos) % len(self.ds)
        self.mode_browser = False
        self.show()

    def change_r(self, d: int) -> None:
        self.r_idx = int(np.clip(self.r_idx + d, 0, len(self.r_list) - 1))
        self.show()

    def cycle_noise(self, d: int) -> None:
        self.noise_idx = (self.noise_idx + d) % len(NOISE_LEVELS)
        self.show()

    def toggle_method(self) -> None:
        self.method = "force" if self.method == "displacement" else "displacement"
        self.show()

    def toggle_browser(self) -> None:
        self.mode_browser = not self.mode_browser
        self.show()

    def change_mode(self, d: int) -> None:
        self.mode_k = int(np.clip(self.mode_k + d, 0, self.basis.r - 1))
        if self.mode_browser:
            self.show()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", default="data/processed/contact_sweep_top.npz")
    p.add_argument("--basis", default="data/processed/rom_basis_top.npz")
    p.add_argument("--r", type=int, default=4)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--method", choices=["displacement", "force"], default="displacement")
    p.add_argument("--mode", type=int, default=None, help="start in the mode browser at this mode (1-based)")
    p.add_argument("--amplification", type=float, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--screenshot", type=str, default=None)
    p.add_argument("--show-rom", action="store_true", help="start with the ROM reconstruction displayed (A)")
    args = p.parse_args()

    ds = ContactDataset.load(args.dataset)
    basis = PODBasis.load(args.basis)
    if basis.n_dof != ds.n_dofs:
        raise RuntimeError("basis does not match the dataset")
    model = FingerFEMModel.from_config(ds.meta["fem_config"])
    if not np.allclose(model.mesh.nodes, ds.nodes):
        raise RuntimeError("rebuilt FEM mesh does not match the dataset mesh")

    from src.visualization.taichi_viewer import TaichiViewer

    viewer = TaichiViewer(show_window=not args.headless, title=f"ROM – {Path(args.basis).name}")
    if args.amplification is not None:
        viewer.options.amplification = args.amplification
    insp = ROMInspector(ds, model, basis, viewer, start=args.start, r=args.r, method=args.method)
    if args.mode is not None:
        insp.mode_k = int(np.clip(args.mode - 1, 0, basis.r - 1))
        insp.mode_browser = True
        insp.show()
    if args.show_rom:
        viewer.options.use_alt_displacement = True
    if args.screenshot:
        viewer.save_screenshot(args.screenshot)
    if not args.headless:
        viewer.run()
    return insp


if __name__ == "__main__":
    main()
