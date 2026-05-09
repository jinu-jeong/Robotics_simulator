# RoboSim

A modular, research-grade robotics physics simulator that puts **rigid-body, finite-element, and material-point** dynamics into a single hybrid framework. Pure Python (NumPy / SciPy) with Taichi GGUI rendering. Built for reproducible robotics research on contact-rich, deformable, and topology-changing scenes.

---

## What's in the framework

| Pillar | Capabilities |
|---|---|
| **Rigid Body Dynamics (RBD)** | Featherstone ABA / RNEA / CRBA, N-DOF articulated chains, free-floating bodies, URDF loading |
| **Finite Element Method (FEM)** | Tet4 / Tet10 / Hex8, Corotational and Neo-Hookean materials, implicit Euler + Newton-Raphson, sparse CSR assembly |
| **Material Point Method (MPM)** | Explicit MLS-MPM (APIC), Neo-Hookean / von-Mises J2 / Drucker-Prager / damaged-elastic materials |
| **Reduced-Order Model** | Craig-Bampton CMS (fixed-interface modes + boundary DOFs) for 10–100× FEM speedup |
| **Contact** | BVH broad phase, SAT box-box, SDF primitives, FEM vertex-face & edge-edge, MPM kinematic colliders |
| **Coupling** | RBD ↔ FEM penalty springs, RBD/CB → MPM one-way kinematic colliders for grasp-and-drop scenarios |
| **Visualization** | Taichi GGUI (Metal / Vulkan), per-element von-Mises stress colouring, ImGui live-tuning |

Three solvers, one API. The same `Scene` object can host an articulated robot, a deformable FEM body, and a granular MPM pile, stepped together from one loop.

---

## Why MPM and FEM together

FEM is the right tool for small-to-moderate elastic deformation on a fixed mesh — it gives a banded stiffness matrix, accurate modal behaviour, and clean reduction (Craig-Bampton). It struggles when deformation is large, plastic, or topology-changing (clay shearing, sand piling, soft tissue tearing).

MPM handles exactly that regime: particles carry mass and history, the background grid carries momentum, and there is no mesh to remesh. Plasticity, fracture, and granular flow are all expressed as return mappings on the per-particle deformation gradient.

In RoboSim the two solvers stay in their lane: a robot grips a rigid (or Craig-Bampton-reduced) box and drops it on an MPM clay pile. The pile sees the box through a kinematic collider every substep — one-way coupling that is fast, stable, and physically reasonable for the asymmetric mass ratios typical of manipulation tasks.

---

## Installation

**Requirements:** Python 3.10+, macOS (Metal) or Linux (Vulkan).

```bash
git clone https://github.com/<your-handle>/Robotics_Simulator.git
cd Robotics_Simulator

conda create -n robosim python=3.11
conda activate robosim
pip install -e .
```

Optional extras:

```bash
pip install -e .[dev]   # pytest, pytest-benchmark, mypy
pip install -e .[fem]   # meshio + tetgen for arbitrary mesh import
pip install taichi       # required for GGUI viewer (rendering only)
```

Core dependencies stay deliberately small (`numpy`, `scipy`, `trimesh`); the physics layer has no GPU/CPP build step.

---

## Quick start

All standalone simulation demos live under `examples/pure_simulation/`.
Reinforcement-learning environments live under `examples/rl/`.

```bash
# Rigid box drop
python examples/pure_simulation/drop_test.py

# Deformable FEM bunny drop
python examples/pure_simulation/drop_test.py --shape bunny --mode fem

# Craig-Bampton reduced bunny (10–100× faster than full FEM)
python examples/pure_simulation/drop_test.py --shape bunny --mode cb

# Robot grasp-and-lift demo (rigid / FEM / CB-reduced deformable box)
python examples/pure_simulation/grasp_scene.py                # rigid (default, penalty + kinematic grip)
python examples/pure_simulation/grasp_scene.py --mode fem
python examples/pure_simulation/grasp_scene.py --mode cb

# MPM jelly cube on a floor
python examples/pure_simulation/mpm_jello_drop.py

# MPM clay tear (damaged Neo-Hookean → fracture)
python examples/pure_simulation/mpm_clay_tear.py

# MPM plasticity (von-Mises J2 + Drucker-Prager sand)
python examples/pure_simulation/mpm_plasticity_demo.py

# Robot grasps a (rigid or CB-reduced) box and drops it on an MPM clay pile
python examples/pure_simulation/mpm_grasp_drop.py                  # rigid box
python examples/pure_simulation/mpm_grasp_drop.py --mode cb        # deformable box
python examples/pure_simulation/mpm_grasp_drop.py --material sand  # Drucker-Prager pile
```

Viewer controls: left-drag — orbit, scroll / `W` / `S` — zoom, `A` / `D` — pan, `ESC` — quit.

---

## Project layout

```
robosim/
├── model/             # Robot definition, URDF parser, geometry, scene factory
├── scene/             # High-level Scene API (objects, handles, runner)
├── physics/
│   ├── rbd/           # Featherstone ABA / RNEA / CRBA, semi-implicit integration
│   ├── fem/           # Mesh, elements, materials, assembly, implicit solver,
│   │                  #   Craig-Bampton reduction
│   ├── mpm/           # MLS-MPM solver, materials, grid, kernel, transfer,
│   │                  #   boundaries / colliders
│   ├── contact/       # BVH, SAT, SDF, vertex-face & edge-edge
│   └── coupling/      # RBD ↔ FEM penalty interface
├── control/           # Joint PD, trajectory phases
├── math/              # SE(3), spatial algebra, interpolation
├── viz/               # Taichi GGUI viewer, stress shading, ImGui overlays
└── io/                # Logger, replay, mesh I/O

examples/
├── pure_simulation/   # 11 runnable physics demos (FEM, MPM, hybrid, URDF, grasp)
└── rl/                # Gymnasium env wrappers + SB3 training scripts
tests/                 # 229 pytest tests across 21 suites
research/              # Standalone research-track studies (not packaged)
```

Each MPM file is a single concern (≤ 240 lines): `grid.py`, `kernel.py`, `particles.py`, `materials.py`, `transfer.py`, `boundary.py`, `solver.py`. The Scene API lets a user assemble RBD + FEM + MPM bodies without touching solver internals.

---

## Physics

### Rigid-body dynamics

Featherstone spatial-algebra core. ABA gives forward dynamics in O(N), RNEA gives inverse dynamics, CRBA produces the joint-space inertia for control. Joint types: `REVOLUTE`, `PRISMATIC`, `FIXED`, `CONTINUOUS`. Free-floating bodies use a 6-DOF chain (3 prismatic + 3 revolute) so the same algorithms cover both articulated and free objects. Time integration is semi-implicit Euler.

### Finite-element method

| Feature | Detail |
|---|---|
| Elements | Tet4, Tet10, Hex8 |
| Materials | `CorotationalElastic` (polar decomposition `F = RS`), `NeoHookean` |
| Time integration | Implicit Euler with Newton-Raphson (≤ 20 inner iterations, CFL-free) |
| Linear solve | SciPy sparse CSR + direct factorisation |
| Visualization | Per-element von-Mises stress → blue-green-red heat-map |

### Material-point method

RoboSim uses a single-pass explicit MLS-MPM (Hu et al., 2018) with APIC affine velocity transfer (Jiang et al., 2015). One substep:

1. Compute the Kirchhoff stress `τ = P · Fᵀ` from the per-particle deformation gradient `F`.
2. **P2G** in one pass: scatter mass and `m·v + A·(x_node − x_p)` to the 27-cell stencil, with `A = m·C − dt · (4/dx²) · V₀ · τ` folding stress into APIC affine momentum.
3. Normalise grid momentum to velocity, apply gravity, axis-aligned wall BCs, and any kinematic collider.
4. **G2P**: gather grid velocity onto particles, recover the APIC affine matrix `C`.
5. Update the deformation gradient `F ← (I + dt·C) · F` and (for plastic materials) project onto the yield surface.
6. Advect particles `x ← x + dt·v`.

Constitutive models implemented:

| Class | Behaviour |
|---|---|
| `NeoHookean` | Compressible hyperelasticity (jelly, soft tissue) |
| `VonMisesPlastic` | J2 plasticity, return mapping on isochoric trial state (metal-like dough) |
| `DruckerPragerPlastic` | Pressure-sensitive yield (sand, granular media) |
| `DamagedNeoHookean` | Quasi-brittle damage (clay tearing, fracture) |

The 27-stencil scatter / gather is fully vectorised across particles via `np.bincount`; on M1-class hardware this gives ~10× the throughput of the naive 27 × `np.add.at` formulation while staying pure Python.

### Craig-Bampton reduction

`CraigBamptonBody` reduces a full FEM model to `n_b` boundary DOFs plus `n_modes` fixed-interface internal modes. Reduced DOF `n_r = n_b + n_modes` is typically two to three orders of magnitude smaller than the underlying FEM, while preserving the dominant dynamics at coupling interfaces. This is what `--mode cb` selects in the drop and grasp demos.

### Contact

**Broad phase.** Brute force for ≤ 4 bodies; otherwise a BVH with median-split AABBs gives O(N log N) culling. AABBs are computed analytically per shape: box uses the exact OBB projection `extent = |R| · half_extents`; cylinder uses `extent_i = hl·|axis_i| + r·√(1 − axis_i²)`; mesh uses world-transformed vertices.

**Narrow phase.**

| Pair | Algorithm |
|---|---|
| Box–Box | SAT (15 axes) + Sutherland-Hodgman manifold clipping |
| Sphere–Sphere | Centre distance |
| Box–Sphere | Closest point on box + radial test |
| Sphere/Box/Cylinder–Ground | Per-corner / ring SDF |
| Mesh–Ground / Mesh–Box | World-space vertex tests |
| FEM vertex–face | Barycentric projection, threshold `d_hat` |
| FEM edge–edge | Parametric segment closest point (Ericson §5.1.9), per-edge AABB cull |
| MPM particle ↔ rigid box | `KinematicBoxCollider` projects grid velocity each substep |

**Response.**

- *RBD impulse:* `j_n = −(1 + e) · v_n / Σ(1/m + (r×n)·I⁻¹·(r×n))`, position projection without energy injection, Coulomb friction clamped to `μ · |j_n|`.
- *FEM vertex-face / edge-edge:* effective contact mass via barycentric harmonic mean, position correction split by mass ratio.
- *Penalty spring:* `f = k · max(d, 0) · n̂ − c · v_n · n̂` with damping clamped to critical `2√(k·m)` to avoid energy injection on light FEM nodes.
- *MPM kinematic collider:* axis-aligned wall projection or analytic SDF on the grid velocity field.

**Two contact-solver modes (`Scene(contact_solver=...)`).**

- `"penalty"` (default) — explicit penalty + regularised kinetic Coulomb. No static friction. Pair with `Scene.grip(...)` to hold an object via kinematic lock + finger-PD freeze. Cheapest path; used for IL/RL training where wall-time matters.
- `"constraint"` — Coulomb cone with stick/slip transition. When the impulse needed to null tangential velocity within one timestep stays inside the friction disk, the force is set to the exact stick value; otherwise it saturates at `μ · f_n` opposing motion. Provides static friction → grasping is held by finger force alone, no kinematic lock. `Scene.grip(...)` is silently ignored under this mode. Used for sim2real-leaning workloads and ablations that need to remove the kinematic-grip artefact.

Both modes share broadphase / narrow-phase / SDF / pair bookkeeping; the difference is purely in the per-contact friction force computation.

### RBD–FEM coupling

`PenaltyCoupling` ties one or more RBD link frames to a set of FEM boundary nodes through critically-damped springs. The result is bilateral: contact forces in either solver propagate through the spring to the other. This is what powers the `hybrid_test`, `soft_gripper`, and `grasp_scene` examples.

### RBD/CB → MPM coupling

The `mpm_grasp_drop` demo uses a `KinematicBoxCollider` driven by the box's live pose every substep. The MPM pile reacts to the box; the box does not directly feel pile reaction (it stays under the robot's PD control or, in CB mode, its reduced-order dynamics). For grasp-and-drop manipulation this asymmetric one-way coupling is both fast and physically reasonable because the box mass is orders of magnitude above any single particle's contribution.

---

## Reproducing the figures

Each example writes a self-contained GIF / PNG when `--headless` is set and prints diagnostics for verification.

```bash
python examples/pure_simulation/mpm_jello_drop.py --headless        # elastic jelly cube
python examples/pure_simulation/mpm_clay_tear.py --headless         # damage / fracture
python examples/pure_simulation/mpm_plasticity_demo.py --headless   # J2 + Drucker-Prager
python examples/pure_simulation/mpm_grasp_drop.py --mode rigid --headless
python examples/pure_simulation/mpm_grasp_drop.py --mode cb   --material sand --headless
python examples/pure_simulation/grasp_scene.py --mode fem --headless  # FEM grasp via Scene API
python examples/pure_simulation/grasp_scene.py --mode cb  --headless  # CB grasp via Scene API
python examples/pure_simulation/drop_test.py --shape bunny --mode fem --headless
```

Headless mode emits frame-by-frame energy / momentum / max-stress logs that can be diffed against the reference traces stored in the matching test under `tests/`.

---

## Testing

```bash
pytest                         # 229 tests, ~50 s on M1 Pro
pytest tests/test_mpm_solver.py -v
pytest tests/test_fem_solver.py -v
pytest tests/test_contact.py   -v
```

Coverage: SE(3) and spatial algebra, RBD energy and momentum conservation, URDF parsing, FEM mesh / element / material / energy, contact SDF / BVH / SAT, FEM vertex-face & edge-edge, RBD–FEM coupling, MPM transfer / kernel / collider / damage / plasticity.

---

## Citation

If you use RoboSim in academic work, please cite (preprint forthcoming):

```bibtex
@misc{jeong2026robosim,
  author = {Jeong, Jinu},
  title  = {RoboSim: A Hybrid RBD/FEM/MPM Simulator for Contact-Rich Robotics},
  year   = {2026},
  note   = {Software, https://github.com/<your-handle>/Robotics_Simulator},
}
```

The MPM core follows MLS-MPM (Hu et al., SIGGRAPH 2018) with APIC transfers (Jiang et al., SIGGRAPH 2015); please cite the originals when describing the underlying numerics.

---

## Status and roadmap

Stable: RBD, FEM (Tet4/Tet10/Hex8 + Corotational/Neo-Hookean), Craig-Bampton, MPM (Neo-Hookean / J2 / Drucker-Prager / damaged), penalty coupling, BVH+SAT contact, Taichi viewer.

Active: two-way RBD ↔ MPM coupling for fine manipulation, learned reduced-order models for FEM (`research/gnn-boundary-rom/`), GPU port of the MPM transfer kernels.

---

## License

MIT — see [`LICENSE`](LICENSE).
