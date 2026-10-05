#!/usr/bin/env python
"""Milestone 2 – interactive Taichi inspector for a contact-sweep dataset.

Step through samples with the keyboard while the viewer shows the deformed
finger, contact marker, force arrow and probe object. The GUI panel prints
sample index, force magnitude, contact position and max displacement so
suspicious samples are easy to spot.

Keys (in addition to the standard viewer controls):
    Right / Left          next / previous sample
    Shift + Right / Left  jump ±10 samples
    Home / End            first / last sample  (also: 0 -> first)
    Space                 random sample
    U                     cycle sort order: index -> force -> max |u| -> contact x

Run:
    python scripts/inspect_fem_dataset.py data/processed/contact_sweep_top.npz
    python scripts/inspect_fem_dataset.py data/processed/contact_sweep_top.npz --start 42
    python scripts/inspect_fem_dataset.py <file> --headless --screenshot /tmp/s.png --start 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fem.contact_dataset import ContactDataset  # noqa: E402


class DatasetInspector:
    """Holds the navigation state and pushes samples into the viewer."""

    SORTS = ("index", "force", "max_u", "contact_x")

    def __init__(self, ds: ContactDataset, viewer, start: int = 0) -> None:
        self.ds = ds
        self.viewer = viewer
        self.sort_mode = 0
        self.order = np.arange(len(ds))
        self.pos = int(start) % len(ds)
        self.rng = np.random.default_rng(0)
        self._install_keys()
        self.show()

    # ----------------------------------------------------------- state
    @property
    def index(self) -> int:
        return int(self.order[self.pos])

    def _resort(self) -> None:
        cur = self.index
        mode = self.SORTS[self.sort_mode]
        if mode == "index":
            self.order = np.arange(len(self.ds))
        elif mode == "force":
            self.order = np.argsort(self.ds.force_magnitude, kind="stable")
        elif mode == "max_u":
            self.order = np.argsort(self.ds.max_displacement, kind="stable")
        else:
            self.order = np.argsort(self.ds.contact_position[:, 0], kind="stable")
        self.pos = int(np.nonzero(self.order == cur)[0][0])

    def step(self, d: int) -> None:
        self.pos = (self.pos + d) % len(self.ds)
        self.show()

    def show(self) -> None:
        i = self.index
        state = self.ds.to_visualization_state(
            i, extra_info={"order": f"{self.SORTS[self.sort_mode]} ({self.pos + 1}/{len(self.ds)})"}
        )
        # fit the camera only for the very first sample; keep the user's view afterwards
        self.viewer.set_state(state, refit_camera=None if self.viewer.state is None else False)
        f, p, um = self.ds.force_magnitude[i], 1e3 * self.ds.contact_position[i], 1e3 * self.ds.max_displacement[i]
        print(f"[inspect] sample {i:5d}  F = {f:6.3f} N  contact = ({p[0]:6.2f}, {p[1]:6.2f}, {p[2]:6.2f}) mm  max|u| = {um:.4f} mm")

    # ----------------------------------------------------------- keys
    def _install_keys(self) -> None:
        import taichi as ti

        v = self.viewer
        shift = lambda: v.window.is_pressed(ti.ui.SHIFT)  # noqa: E731
        v.register_key(ti.ui.RIGHT, lambda _: self.step(10 if shift() else 1), "next sample (+10 with Shift)")
        v.register_key(ti.ui.LEFT, lambda _: self.step(-10 if shift() else -1), "previous sample (-10 with Shift)")
        v.register_key("Home", lambda _: self._goto(0), "first sample")
        v.register_key("End", lambda _: self._goto(len(self.ds) - 1), "last sample")
        v.register_key("0", lambda _: self._goto(0), "first sample")
        v.register_key(ti.ui.SPACE, lambda _: self._goto(int(self.rng.integers(len(self.ds)))), "random sample")
        v.register_key("u", lambda _: self._cycle_sort(), "cycle sort order")

    def _goto(self, pos: int) -> None:
        self.pos = int(pos) % len(self.ds)
        self.show()

    def _cycle_sort(self) -> None:
        self.sort_mode = (self.sort_mode + 1) % len(self.SORTS)
        self._resort()
        self.show()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=str)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--amplification", type=float, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--screenshot", type=str, default=None)
    args = p.parse_args()

    ds = ContactDataset.load(args.dataset)
    print(f"loaded {args.dataset}")
    for k, v in ds.summary().items():
        print(f"  {k}: {v}")

    from src.visualization.taichi_viewer import TaichiViewer

    viewer = TaichiViewer(show_window=not args.headless, title=f"Dataset inspector – {Path(args.dataset).name}")
    if args.amplification is not None:
        viewer.options.amplification = args.amplification
    insp = DatasetInspector(ds, viewer, start=args.start)
    if args.screenshot:
        viewer.save_screenshot(args.screenshot)
    if not args.headless:
        viewer.run()
    return insp


if __name__ == "__main__":
    main()
