# RoboSim

A modular, research-grade robotics physics simulator combining **Rigid Body Dynamics (RBD)** and **Finite Element Method (FEM)** deformable bodies in a single framework. Written in pure Python with NumPy/SciPy numerics and Taichi GPU rendering.

---

## Features

| Category | Capabilities |
|---|---|
| **Rigid Body Dynamics** | Featherstone ABA/RNEA, N-DOF articulated chains, URDF loading |
| **Deformable Bodies** | Corotational & Neo-Hookean FEM, Tet4/Tet10/Hex8 elements |
| **Model Reduction** | Craig-Bampton CMS (fixed-interface modes + boundary DOFs) |
| **Contact Detection** | BVH broad phase, SAT box-box, SDF primitives, FEM vertex-face & edge-edge |
| **Contact Response** | Impulse-based (RBD & FEM), penalty spring-damper, Coulomb friction |
| **RBD–FEM Coupling** | Penalty spring connections between rigid links and FEM boundary nodes |
| **Visualization** | Taichi GGUI (Metal/Vulkan), stress colouring, live ImGui parameter tuning |

---

## Installation

**Requirements:** Python 3.11+, macOS (Metal) or Linux (Vulkan)

```bash
# Clone
git clone https://github.com/<your-repo>/Robotics_Simulator.git
cd Robotics_Simulator

# Create conda environment (recommended)
conda create -n robosim python=3.11
conda activate robosim

# Install in editable mode
pip install -e .
```

Core dependencies (`pyproject.toml`):

| Package | Role |
|---|---|
| `numpy ≥ 1.24` | Numerics |
| `scipy ≥ 1.10` | Sparse matrices, Delaunay |
| `trimesh ≥ 3.20` | Mesh loading (STL / OBJ / DAE …) |
| `taichi` | GPU-accelerated GGUI rendering |

---

## Quick Start

```bash
# Drop a single rigid box
python examples/drop_test.py

# Drop 4 FEM deformable boxes and compare FPS
python examples/drop_test.py --mode fem --num-boxes 4

# Drop the Stanford Bunny (rigid)
python examples/drop_test.py --shape bunny

# Drop a deformable FEM Bunny
python examples/drop_test.py --shape bunny --mode fem

# Load and display a URDF robot
python examples/load_urdf_demo.py

# Simple pendulum (energy conservation demo)
python examples/simple_pendulum.py
```

**Viewer controls:** left-drag — orbit, scroll — zoom, `W/S` — zoom, `A/D` — pan, `ESC` — quit.

---

## Project Structure

```
robosim/
├── model/           # Robot definition: links, joints, geometry, URDF parser
├── physics/
│   ├── rbd/         # Rigid Body Dynamics (Featherstone ABA/RNEA)
│   ├── fem/         # FEM solver, mesh types, materials, Craig-Bampton
│   ├── contact/     # Collision detection (BVH, SAT, SDF) + response
│   └── coupling/    # Penalty-based RBD–FEM coupling
├── math/            # SE(3) transforms, spatial algebra, interpolation
├── viz/             # Taichi GGUI viewer, stress colouring, ImGui overlays
└── io/              # Simulation logger, replay, mesh I/O

examples/
├── assets/          # Mesh assets (bunny.obj, …)
├── urdf/            # Robot URDF files
└── *.py             # 11 runnable demos

tests/               # 16 pytest suites (192 tests)
```

---

## Examples

### Drop Test (`examples/drop_test.py`)

Multi-body drop test for comparing solver modes and benchmarking FPS.

```bash
python examples/drop_test.py [OPTIONS]

Options:
  --mode {rbd,fem,cb}   Solver mode (default: rbd)
  --shape {box,bunny}   Object shape (default: box)
  --num-boxes N, -n N   Number of objects (default: 1)
  --tilt DEG            Initial Y-axis tilt for a single object
  --seed INT            Random seed for multi-object tilt angles
```

| Command | Description |
|---|---|
| `--mode rbd` | Rigid box(es), Featherstone ABA, impulse ground/box-box contact |
| `--mode fem` | Deformable box(es), corotational FEM, von Mises stress colouring |
| `--mode cb` | Craig-Bampton reduced model (fast approximation of FEM) |
| `--shape bunny` | Stanford Bunny (2 503 surface vertices) instead of boxes |

### Other Demos

| Script | What it shows |
|---|---|
| `simple_pendulum.py` | RBD energy conservation (KE + PE = const) |
| `load_urdf_demo.py` | Parse URDF, display FK with mesh geometry |
| `fem_fem_drop.py` | Two FEM bodies colliding — vertex-face & edge-edge contacts |
| `hybrid_test.py` | Rigid robot arm + deformable block via penalty coupling |
| `soft_gripper.py` | 9-link rigid gripper grasping a FEM body |
| `grasp_demo.py` | Complex multi-body grasping |
| `interactive_demo.py` | Live ImGui sliders — tune stiffness/damping in real time |
| `fem_composite.py` | Mixed Hex8 + Tet4 composite mesh body |
| `fem_element_compare.py` | Side-by-side Tet4 / Tet10 / Hex8 accuracy comparison |

---

## Physics

### Rigid Body Dynamics

The RBD solver is built on **Featherstone's spatial algebra** and handles arbitrary N-DOF kinematic chains.

| Algorithm | Purpose |
|---|---|
| **ABA** (Articulated Body Algorithm) | Forward dynamics in O(N) — computes joint accelerations from applied forces |
| **RNEA** (Recursive Newton-Euler) | Inverse dynamics — computes required joint torques for a desired motion |
| **CRBA** (Composite Rigid Body) | Joint-space inertia matrix for control applications |
| **Semi-implicit Euler** | Time integration — velocity-implicit, position-explicit |

Joint types: `REVOLUTE`, `PRISMATIC`, `FIXED`, `CONTINUOUS`.
Free-floating bodies use a 6-DOF chain (3 prismatic + 3 revolute) compatible with all single-DOF algorithms.

### Finite Element Method

| Feature | Detail |
|---|---|
| **Elements** | Tet4 (linear), Tet10 (quadratic), Hex8 (trilinear) |
| **Materials** | `CorotationalElastic` (large rotations via polar decomposition), `NeoHookean` (hyperelastic) |
| **Integration** | Implicit Euler + Newton-Raphson (up to 20 iterations, CFL-free) |
| **Assembly** | Sparse CSR stiffness/mass matrices |
| **Visualization** | Per-element von Mises stress → heat-map colour |

**CorotationalElastic** extracts the rotation R from the deformation gradient via polar decomposition (F = RS), then applies linear elasticity in the co-rotated frame. This handles large rigid rotations while keeping the tangent stiffness well-conditioned.

### Craig-Bampton Model Reduction

`CraigBamptonBody` reduces a full FEM model to a small set of degrees of freedom:

- **Boundary DOFs** — physical displacements of user-specified interface nodes (n_b)
- **Interior modes** — fixed-interface normal modes of the interior (n_modes)
- **Reduced DOF** — n_r = n_b + n_modes (typically ≪ full FEM DOF)

This gives a factor of 10–100× speedup vs. full FEM while preserving the dominant dynamic behaviour at coupling interfaces.

### Contact Detection

Two-phase pipeline:

#### Broad Phase
- **O(N²) brute-force** — used when N ≤ 4 bodies (avoids BVH construction overhead)
- **BVH (Bounding Volume Hierarchy)** — binary tree of AABBs; median split along longest axis; recursive cross-pair traversal gives **O(N log N)** average culling for N > 4 bodies

AABB computation:
- **Box**: `extent = |R| @ half_extents` (exact OBB projection)
- **Cylinder**: `extent_i = hl·|axis_i| + r·√(1 − axis_i²)` (tight, not bounding sphere)
- **Sphere**: trivial radius expansion
- **Mesh**: vertex AABB of world-transformed vertices

#### Narrow Phase

| Pair | Algorithm |
|---|---|
| Box–Box | **SAT** — 15 separating axes (6 face normals + 9 edge cross-products); **Sutherland-Hodgman** clipping for contact manifold (up to 4 points) |
| Sphere–Sphere | Center distance |
| Box–Sphere | Closest point on box + radial check |
| Sphere / Box / Cylinder – Ground | Per-corner / ring sampling SDF |
| Mesh – Ground / Box | World-space vertex tests |
| FEM vertex–face | Barycentric projection onto surface triangles; proximity threshold d_hat |
| FEM edge–edge | Parametric segment closest-point (Ericson §5.1.9); per-edge AABB culling |

### Contact Response

#### RBD — Impulse-based
Applied after each sub-step. For each penetrating contact point:

```
j_n = -(1 + e) · v_n / (1/m_a + r_a×n·I_a⁻¹·r_a×n + 1/m_b + …)
```

Position projection removes penetration without injecting energy. Coulomb friction impulse clamped to `μ · |j_n|`.

#### FEM — Vertex-face impulse
- Effective mass of the contact face computed as the barycentric-weighted harmonic mean of nodal masses
- Position correction split by mass ratio; velocity impulse from relative approach velocity

#### FEM — Edge-edge impulse
- Contact point parameterized by (t_a, t_b) on each edge
- Nodal impulse weighted by (1−t, t) interpolation coefficients
- Identical friction formulation to vertex-face

#### Penalty Spring (RBD and RBD–FEM coupling)
```
f = k · max(d, 0) · n̂ − c · v_n · n̂
```
Damping coefficient clamped to critical value `2√(k·m)` to prevent energy injection on light FEM nodes.

---

## Visualization

RoboSim uses **Taichi GGUI** for GPU-accelerated real-time rendering (Metal on macOS, Vulkan on Linux).

```python
from robosim.viz.viewer import SimViewer

viewer = SimViewer(title="My Sim", window_size=(1280, 800))
viewer.initialize()
viewer.add_mesh("robot", vertices, faces, color=np.array([0.8, 0.4, 0.2]))

def step(frame):
    # update physics ...
    viewer.update_mesh_vertices("robot", new_vertices)
    viewer.add_text(f"t = {t:.3f}s")

viewer.add_callback(step)
viewer.show()
```

**Von Mises stress colouring** — computed per element from the Cauchy stress tensor, mapped through a blue-green-red heat-map and applied as per-face vertex colours.

**Interactive parameter tuning** — `interactive_demo.py` shows how to attach ImGui sliders to physics parameters for live tuning without restarting the simulation.

---

## Module API Reference

### `robosim.model`

```python
from robosim.model.robot import Robot
from robosim.model.urdf_parser import URDFParser
from robosim.model.factory import create_free_box, create_free_sphere, create_free_mesh_body
from robosim.model.geometry import Geometry
from robosim.model.scene import Scene
```

| Function | Description |
|---|---|
| `create_free_box(name, size, mass, position)` | 6-DOF free-floating rigid box |
| `create_free_sphere(name, radius, mass, position)` | 6-DOF free-floating rigid sphere |
| `create_free_mesh_body(name, vertices, faces, mass, …)` | 6-DOF rigid body with arbitrary mesh geometry |
| `URDFParser.load(path)` | Parse URDF → `Robot` |

### `robosim.physics.rbd`

```python
from robosim.physics.rbd.solver import RBDSolver

solver = RBDSolver(robot=robot)
solver.initialize(dt=0.001)
solver.step()               # advance by dt
solver.apply_force(link_idx, force_world, torque_world)
```

### `robosim.physics.fem`

```python
from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.materials import CorotationalElastic, NeoHookean
from robosim.physics.fem.solver import FEMSolver, DeformableBody

mesh = TetMesh.create_box(origin, size, divisions=(4, 4, 4))
mat  = CorotationalElastic(young=1e6, poisson=0.3)
body = DeformableBody(name="block", mesh=mesh, material=mat, density=1000.0)

solver = FEMSolver(bodies=[body], gravity=np.array([0, 0, -9.81]), damping=0.02)
solver.initialize(dt=0.001)
solver.step()               # body.x updated in-place
```

### `robosim.physics.contact`

```python
from robosim.physics.contact.solver import ContactSolver
from robosim.physics.contact.detection import GroundPlane

contact = ContactSolver(ground=GroundPlane(height=0.0))

# RBD bodies (registered automatically via ContactDetector)
# FEM bodies
contact.register_fem(fem_solver)

# Per step:
contact.resolve_fem_contact(body, restitution=0.3, friction_mu=0.5)
contact.resolve_fem_fem_all(fem_solver, d_hat=0.005)
```

### `robosim.physics.fem.reduced` — Craig-Bampton

```python
from robosim.physics.fem.reduced import CraigBamptonBody, CraigBamptonSolver

body = CraigBamptonBody(
    mesh=mesh, material=mat, density=1000.0,
    n_modes=6,                              # interior modes to retain
    gravity=np.array([0, 0, -9.81]),
    damping=0.02,
)
solver = CraigBamptonSolver(bodies=[body])
solver.initialize(dt=0.001)
solver.step()
```

---

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific module
pytest tests/test_contact.py -v
pytest tests/test_fem_solver.py -v
```

192 tests across 16 test files covering:

- SE(3) transforms and spatial algebra
- RBD integration and energy conservation
- URDF parsing and kinematic trees
- FEM mesh creation, element shape functions, material models
- Contact SDF primitives, BVH broad phase, SAT box-box
- FEM surface collision (vertex-face, edge-edge)
- RBD–FEM penalty coupling
- Composite and mixed-element meshes

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                     PhysicsWorld                     │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────┐   │
│  │ RBDSolver│  │ FEMSolver│  │ CraigBamptonSol. │   │
│  │  (ABA)   │  │(Impl.Euler│  │  (ROM, n_modes) │   │
│  └────┬─────┘  └────┬─────┘  └────────┬────────┘   │
│       │              │                  │             │
│  ┌────▼──────────────▼──────────────────▼────────┐  │
│  │                ContactSolver                   │  │
│  │  BVH broad phase → SAT / SDF narrow phase     │  │
│  │  RBD impulse   FEM vertex-face  FEM edge-edge  │  │
│  └───────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐   │
│  │            Penalty Coupling                  │   │
│  │     RBD link frames ↔ FEM boundary nodes     │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Taichi GGUI       │
│   SimViewer         │
│   (Metal / Vulkan)  │
└─────────────────────┘
```

---

## License

MIT License — see `LICENSE` for details.
