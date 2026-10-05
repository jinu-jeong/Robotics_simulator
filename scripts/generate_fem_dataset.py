#!/usr/bin/env python
"""Milestone 2 – generate the contact-position / force-magnitude sweep dataset.

For every (contact point, force magnitude) pair a pressing normal contact is
applied through ``B(c)`` and the linear FEM is solved (the stiffness matrix is
factorised once, so each sample costs milliseconds). Everything is stored in
one self-describing ``.npz`` (see ``src/fem/contact_dataset.py``).

Run:
    python scripts/generate_fem_dataset.py                       # configs/dataset.yaml
    python scripts/generate_fem_dataset.py --config dataset --output data/processed/test.npz
    python scripts/generate_fem_dataset.py --sampling random --n-random 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fem.contact_dataset import ContactDataset  # noqa: E402
from src.fem.finger_model import FingerFEMModel  # noqa: E402
from src.utils.config import load_config, save_yaml  # noqa: E402


def contact_points(cfg, geom, rng) -> np.ndarray:
    """Surface points (S_c, 3) on the requested finger face."""
    c = cfg["contact"]
    surface = c.get("surface", "top")
    if c.get("sampling", "grid") == "grid":
        xr, yr = c["x_rel"], c["y_rel"]
        xs = np.linspace(xr["min"], xr["max"], int(xr["n"]))
        ys = np.linspace(yr["min"], yr["max"], int(yr["n"]))
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        rel = np.stack([X.ravel(), Y.ravel()], axis=1)
    else:
        n = int(c.get("n_random", 300))
        rel = np.stack([
            rng.uniform(c["x_rel"]["min"], c["x_rel"]["max"], n),
            rng.uniform(c["y_rel"]["min"], c["y_rel"]["max"], n),
        ], axis=1)
    L, W, H = geom.length, geom.width, geom.height
    if surface == "top":
        return np.stack([rel[:, 0] * L, rel[:, 1] * W, np.full(len(rel), H)], axis=1)
    if surface == "bottom":
        return np.stack([rel[:, 0] * L, rel[:, 1] * W, np.zeros(len(rel))], axis=1)
    if surface == "side_pos_y":  # second coordinate runs along z
        return np.stack([rel[:, 0] * L, np.full(len(rel), W), rel[:, 1] * H], axis=1)
    if surface == "side_neg_y":
        return np.stack([rel[:, 0] * L, np.zeros(len(rel)), rel[:, 1] * H], axis=1)
    raise KeyError(f"unknown contact surface {surface!r}")


def force_magnitudes(cfg, rng) -> np.ndarray:
    f = cfg["force"]
    m = f["magnitude"]
    if f.get("sampling", "grid") == "grid":
        return np.linspace(m["min"], m["max"], int(m["n"]))
    n = int(f.get("n_random", 1))
    return np.exp(rng.uniform(np.log(m["min"]), np.log(m["max"]), n))


def generate(cfg, fem_cfg, model: FingerFEMModel, verbose: bool = True) -> ContactDataset:
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    pts = contact_points(cfg, model.geometry, rng)
    surface = cfg["contact"].get("surface", "top")
    # restrict candidate faces to the chosen surface so projections never jump to another face
    model.restrict_contact_surface(surface)

    random_force = cfg["force"].get("sampling", "grid") == "random"
    mags_grid = force_magnitudes(cfg, rng)
    S = len(pts) * (len(mags_grid) if not random_force else int(cfg["force"].get("n_random", 1)))
    N = model.mesh.n_nodes
    dtype = np.float32 if cfg.get("store_dtype", "float32") == "float32" else np.float64

    U = np.zeros((S, N, 3), dtype=dtype)
    cpos, cnorm, cbary = np.zeros((S, 3)), np.zeros((S, 3)), np.zeros((S, 3))
    cface = np.zeros(S, dtype=np.int64)
    fmag, fvec, umax = np.zeros(S), np.zeros((S, 3)), np.zeros(S)

    t0 = time.time()
    k = 0
    max_resid = 0.0
    for p in pts:
        mags = force_magnitudes(cfg, rng) if random_force else mags_grid
        for F in mags:
            res, c = model.solve_normal_contact(p, float(F))
            U[k] = res.u.astype(dtype)
            cpos[k], cnorm[k], cbary[k], cface[k] = c.position, c.normal, c.barycentric, c.face_index
            fmag[k], fvec[k], umax[k] = F, -F * c.normal, res.max_displacement()
            max_resid = max(max_resid, res.equilibrium_residual())
            k += 1
            if verbose and (k % 100 == 0 or k == S):
                print(f"  {k}/{S} samples  ({time.time() - t0:.1f}s)  max eq. residual {max_resid:.1e}")

    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geometry": {"length": model.geometry.length, "width": model.geometry.width, "height": model.geometry.height},
        "material": model.material.to_dict(),
        "mesh": dict(model.mesh.meta),
        "contact_surface": surface,
        "load_type": "pressing normal point contact via B(c) (barycentric on surface triangle)",
        "dataset_config": cfg,
        "fem_config": fem_cfg,
        "max_equilibrium_residual": max_resid,
        "units": {"length": "m", "force": "N", "stress": "Pa"},
    }
    return ContactDataset(
        nodes=model.mesh.nodes, tets=model.mesh.tets, surface_faces=model.mesh.surface_faces,
        fixed_nodes=model.partition.fully_fixed_nodes(),
        hexes=model.mesh.hexes, surface_edges=model.mesh.surface_edges, element_edges=model.mesh.element_edges,
        U=U, contact_position=cpos, contact_normal=cnorm, contact_face=cface, contact_bary=cbary,
        force_magnitude=fmag, force_vector=fvec, max_displacement=umax, meta=meta,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="dataset")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--sampling", choices=["grid", "random"], default=None, help="override contact sampling")
    p.add_argument("--n-random", type=int, default=None)
    p.add_argument("--surface", default=None, help="top | bottom | side_pos_y | side_neg_y")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.sampling:
        cfg["contact"]["sampling"] = args.sampling
    if args.n_random is not None:
        cfg["contact"]["n_random"] = args.n_random
    if args.surface:
        cfg["contact"]["surface"] = args.surface
    if args.seed is not None:
        cfg["seed"] = args.seed
    out = Path(args.output or cfg["output"])

    fem_cfg = load_config(cfg.get("fem_config", "fem"))
    t0 = time.time()
    model = FingerFEMModel.from_config(fem_cfg)
    print(f"FEM model: {model.mesh.n_nodes} nodes, {model.mesh.n_dofs} dofs, factorised in {time.time() - t0:.2f}s")

    ds = generate(cfg, fem_cfg, model)
    path = ds.save(out)
    save_yaml(cfg, out.with_suffix(".config.yaml"))
    print(f"saved {path}  ({path.stat().st_size / 1e6:.1f} MB on disk)")
    for k, v in ds.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
