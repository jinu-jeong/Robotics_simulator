#!/usr/bin/env python
"""Regenerate paper figures at the shared high resolution (``src.utils.figure_res``).

Order:
  1. Fig.1 lateral 3D / plots
  2. Fig.1 FEM convergence (cantilever_lateral)
  3. Fig.1 export (incl. 01 grasp scene live render)
  4. Fig.2 world + nn arm grasp (headless)
  5. Fig.5 material sweep (offline plots; skip closed-loop by default for speed)

    python scripts/regenerate_paper_figures.py
    python scripts/regenerate_paper_figures.py --skip-figure5
    python scripts/regenerate_paper_figures.py --with-figure5-closed-loop
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(cmd: list[str]) -> None:
    print("\n===", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-figure2", action="store_true")
    p.add_argument("--skip-figure5", action="store_true")
    p.add_argument("--with-figure5-closed-loop", action="store_true")
    args = p.parse_args()

    _run([PY, "scripts/render_figure1_lateral.py"])
    _run([PY, "scripts/run_cantilever_verification.py", "--config", "cantilever_lateral", "--no-view", "--fine"])
    _run([PY, "scripts/export_figure1_assets.py"])

    if not args.skip_figure2:
        _run([PY, "scripts/run_arm_grasp.py", "--mode", "world", "--headless"])
        _run([PY, "scripts/run_arm_grasp.py", "--mode", "nn", "--headless"])

    if not args.skip_figure5:
        cmd = [PY, "scripts/run_material_sweep.py"]
        if not args.with_figure5_closed_loop:
            cmd.append("--no-closed-loop")
        _run(cmd)

    print("\nDone regenerating paper figures.")


if __name__ == "__main__":
    main()
