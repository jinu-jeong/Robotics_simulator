# Literature Survey for RoboSim

This survey positions **RoboSim** — a pure-Python (NumPy/SciPy) simulator that combines rigid-body dynamics (Featherstone), corotational/Neo-Hookean FEM with Craig–Bampton reduction, and explicit MLS-MPM (with Neo-Hookean / J2 / Drucker–Prager / damaged-elastic constitutive models), wired together via penalty (RBD↔FEM) and kinematic-collider (RBD/CB→MPM) couplings — against the existing literature.

Verified against `robosim/physics/{rbd,fem,mpm,coupling}/` and `examples/mpm_grasp_drop.py`. RoboSim ships ~6 kLOC of physics and 229 pytest tests.

---

## 1. General-purpose robotics simulators

| Work | Year | Lang / License | Position vs. RoboSim |
|---|---|---|---|
| **MuJoCo** [Todorov, Erez, Tassa, IROS 2012] | 2012 | C++/Python, Apache 2.0 | The standard for contact-rich rigid-body. No deformable continuum solver beyond soft constraints. RoboSim is far slower for pure RBD but adds FEM and MPM in the same Scene. |
| **MuJoCo MJX / Playground** [Zakka et al., RSS 2025 demo] | 2025 | JAX, Apache 2.0 | Vectorised XLA reimplementation of MuJoCo, GPU/TPU-native. Rigid-only. RoboSim cannot compete on speed but covers regimes MJX does not (plasticity, fracture, granular). |
| **PyBullet / Bullet** [Coumans & Bai, 2016–2021] | 2016+ | C++/Python, ZLib | Mature rigid + soft-body (mass-spring, FEM stub). RoboSim's FEM is a proper implicit-Euler corotational solver with sparse direct factorisation, which Bullet's soft-body lacks. |
| **Drake** [Tedrake & the Drake Team, 2019] | 2019 | C++/Python, BSD-3 | Rigorous multibody with hydroelastic contact and trajectory-opt tooling. No MPM or large-strain plasticity. RoboSim is strictly smaller in scope but covers the deformable side Drake does not. |
| **Gazebo** [Koenig & Howard, IROS 2004] | 2004 | C++, Apache 2.0 | Robot/sensor middleware with pluggable physics. Not a continuum simulator. Orthogonal to RoboSim. |
| **Webots** [Michel, 2004] | 2004 | C++/Python, Apache 2.0 | Education-focused rigid simulator. Same orthogonality as Gazebo. |
| **RaiSim** [Hwangbo, Lee, Hutter, RA-L 2018] | 2018 | C++, free for academic | Per-contact iteration solver; very fast for legged locomotion. Rigid-only. |
| **Isaac Gym** [Makoviychuk et al., NeurIPS Datasets 2021] | 2021 | CUDA/Python, proprietary | GPU-batched rigid sim for RL. Soft-body via PhysX FleX is limited. Closed-source; deprecated for Isaac Lab. |
| **Isaac Lab / Isaac Sim** [Mittal et al., arXiv 2511.04831, 2025] | 2024–25 | Python on PhysX/USD, BSD-3 (Lab) | Photoreal + GPU physics; supports soft-body and particles via PhysX. Heavy GPU/USD stack. RoboSim is the opposite point on the design axis: tiny, no GPU, no USD. |
| **Brax** [Freeman et al., NeurIPS 2021] | 2021 | JAX, Apache 2.0 | Differentiable rigid bodies on TPU/GPU. No deformables. |
| **Project Chrono** [Tasora et al., 2016] | 2016 | C++, BSD-3 | Multi-physics: rigid, FEM, SPH, granular DEM. Closest in feature breadth to Genesis/RoboSim, but C++ and engineering-oriented (vehicle/terramechanics). |
| **Genesis** [Genesis-Embodied-AI consortium, Dec 2024] | 2024 | Python on Taichi, Apache 2.0 | Unifies rigid, MPM, SPH, FEM, PBD, stable-fluid in one engine; differentiable MPM; advertises ≥10⁶ FPS. **The largest direct overlap with RoboSim's "all solvers, one API" framing.** |
| **Dojo** [Howell et al., RSS 2022 / RA-L] | 2022 | Julia, MIT | Variational integrator + interior-point contact, fully differentiable, rigid-only. |

**Where RoboSim sits.** Among general-purpose engines, RoboSim is the smallest hybrid (rigid+FEM+MPM) and the only one whose physics is implemented entirely in NumPy/SciPy (Taichi is used only for the GGUI viewer). Genesis subsumes most of RoboSim's feature list at vastly higher performance but requires Taichi-jitted GPU code; RoboSim is still defensible as a *readable, hackable* alternative with a much shorter learning curve.

---

## 2. Soft / deformable robotics simulators and benchmarks

| Work | Year | What it does | Relation to RoboSim |
|---|---|---|---|
| **SOFA** [Faure et al., 2012; Allard et al., MMVR 2007] | 2007/12 | Modular C++ FEM for medical sim. | RoboSim's FEM is a tiny subset (Tet4/10, Hex8, corotational, NH) but adds CB reduction and MPM. SOFA has no MPM. |
| **PyElastica / Elastica** [Naughton et al., RA-L 2021] | 2021 | Cosserat-rod soft robotics. | Different model class (1D rods). Complementary, no overlap. |
| **ChainQueen** [Hu et al., ICRA 2019] | 2019 | First differentiable MLS-MPM for soft robotics. | RoboSim shares the MLS-MPM kernel but is **not** differentiable; in exchange, RoboSim adds J2, Drucker–Prager and damage models. |
| **DiffTaichi** [Hu et al., ICLR 2020] | 2020 | Differentiable programming for physical sim. | Generic backend used by ChainQueen / PlasticineLab. RoboSim deliberately stays autograd-free. |
| **PlasticineLab** [Huang et al., ICLR 2021] | 2021 | Diff. MPM benchmark for plasticine. | Benchmark, not a simulator framework per se; uses DiffTaichi. RoboSim covers similar plastic-clay regimes (J2 + DP + damage) without diff. |
| **DiSECt** [Heiden et al., RSS 2021] | 2021 | Differentiable FEM cutting via virtual nodes. | RoboSim does not do cutting / topology change in FEM (uses MPM damage instead). Complementary. |
| **DefGraspSim** [Huang et al., RA-L 2022] | 2022 | Isaac Gym FEM grasping benchmark. | Most direct manipulation analog of RoboSim's `mpm_grasp_drop`, but FEM-only and proprietary stack. |
| **SoftGym** [Lin et al., CoRL 2020] | 2020 | NVIDIA FleX cloth/rope/fluid benchmark for RL. | Benchmark, not a framework. RoboSim is closer to PlasticineLab in physics regime. |
| **FluidLab** [Xian et al., ICLR 2023] | 2023 | Differentiable multi-material (MPM + fluid) manipulation. | Similar regime, GPU-based, differentiable. RoboSim is the CPU/Python analog without diff. |
| **DaXBench** [Chen et al., ICLR 2023] | 2023 | Diff. benchmark for cloth/rope/liquid manipulation. | Benchmark layer; complementary. |
| **RoboCraft** [Shi et al., IJRR 2024 / RSS 2022] | 2022/24 | GNN dynamics + MPC for elasto-plastic shaping. | Learning method on top of MPM-style sims; consumer of simulators like RoboSim. |
| **RoboCook** [Shi et al., CoRL 2023] | 2023 | Long-horizon tool use (rolling pin, knife). | Same lineage as RoboCraft; uses real-world data + GNN. |
| **Dojo** [Howell et al., 2022] | 2022 | Differentiable maximal-coordinates rigid. | Rigid only. Listed for diff-sim landscape. |

**Where RoboSim sits.** The diff-sim cluster (ChainQueen / PlasticineLab / DiSECt / FluidLab / DaXBench) dominates this space. RoboSim is conspicuously **not differentiable** and does not target gradient-based control. Its niche is "MPM-as-a-service inside a robotics scene," with the same constitutive coverage as PlasticineLab + Klár 2016 sand + Wolper 2019 damage rolled into one solver, but without the DiffTaichi dependency.

---

## 3. MPM in graphics / physics

| Work | Year | Contribution | Used by RoboSim? |
|---|---|---|---|
| Stomakhin, Schroeder, Chai, Teran, Selle, "A Material Point Method for Snow Simulation" | SIGGRAPH 2013 | First graphics MPM, snow. | Algorithm class only. |
| Jiang, Schroeder, Selle, Teran, Stomakhin, "The Affine Particle-In-Cell Method" | SIGGRAPH 2015 | APIC; momentum-conserving transfers. | **Yes — APIC is the transfer scheme.** |
| Klár, Gast, Pradhana, Fu, Schroeder, Jiang, Teran, "Drucker–Prager elastoplasticity for sand animation" | SIGGRAPH 2016 | Drucker–Prager return mapping. | **Yes — `DruckerPragerPlastic`.** |
| Hu, Fang, Ge et al., "A Moving Least Squares Material Point Method…" | SIGGRAPH 2018 | MLS-MPM; APIC and stress folded into a single fast scatter. | **Yes — solver core.** |
| Wolper, Fang, Li, Lu, Gao, Jiang, "CD-MPM: Continuum Damage MPM for dynamic fracture animation" | SIGGRAPH 2019 | Phase-field / damage in MPM. | RoboSim's `DamagedNeoHookean` is a simpler scalar-damage variant of this idea. |
| Li, Ferguson, Schneider, Langlois, Zorin, Panozzo, Jiang, Kaufman, "Incremental Potential Contact" | SIGGRAPH 2020 | Intersection-/inversion-free FEM contact. | Not used. RoboSim uses simpler vertex-face/edge-edge with `d_hat` threshold (Ericson §5.1.9). |
| Hu, Anderson, Li, Sun, Carr, Ragan-Kelley, Durand, "DiffTaichi" | ICLR 2020 | Differentiable programming for physics. | Not used (RoboSim is not differentiable). |

---

## 4. MPM / continuum methods in robotics

The robotics-side MPM literature is concentrated in **deformable manipulation** (dough, clay, plasticine) and **granular manipulation** (sand, gravel, food).

- **ChainQueen** (Hu et al., ICRA 2019) — first MPM-in-a-robotics-loop, 2D walkers via gradient descent on MLS-MPM.
- **PlasticineLab** (Huang et al., ICLR 2021) — RL/diff-physics benchmark on plasticine; MLS-MPM under DiffTaichi.
- **DiSECt** (Heiden et al., RSS 2021) — virtual-node FEM for cutting; not strictly MPM but in the same "topology-change deformables" basket.
- **FluidLab** (Xian et al., ICLR 2023) — multi-material MPM/fluid manipulation, sim-to-real.
- **DaXBench** (Chen et al., ICLR 2023) — diff-physics for rope/cloth/liquid manipulation.
- **RoboCraft / RoboCook** (Shi et al., RSS 2022 / CoRL 2023 / IJRR 2024) — GNN dynamics learned from real-robot data, evaluated against MPM-class sims.
- **DefGraspSim** (Huang et al., RA-L 2022) — grasping deformables, but FEM-only, on Isaac Gym.

**Gap RoboSim addresses:** none of these ship a *standalone, MIT-licensed, pure-Python* MPM with all four constitutive families (NH, J2, DP, damage) wired into a Featherstone-RBD scene with URDF/PD control. The closest single package is Genesis (Dec 2024); the closest pre-Genesis is PlasticineLab (no rigid robot).

---

## 5. Hybrid / multi-physics couplings

- **Two-way rigid–MPM coupling.** Hu et al. 2018 (CPIC) introduced cut-cell two-way rigid–MPM coupling. RoboSim implements only **one-way** (RBD/CB → MPM via `KinematicBoxCollider`), explicitly because the manipulation tasks have asymmetric mass ratios.
- **FEM–MPM coupling.** Active research direction (e.g., Wang et al., hybrid IPC 2020, Han et al. 2023). RoboSim does **not** couple FEM and MPM; they are independent solvers in the same Scene.
- **IPC family** (Li 2020, Codimensional IPC 2021). State-of-the-art for FEM contact robustness. Not used; RoboSim's FEM contact is the simpler vertex-face/edge-edge with quadratic barrier.
- **RBD–FEM penalty coupling** is standard (cf. SOFA, DefGraspSim). RoboSim's `PenaltyCoupling` is a textbook implementation with critical-damping clamp.

---

## 6. Flexible-link manipulator dynamics (the robotics-side ancestor)

Robotics has used FFR + modal coordinates for *flexible-link / flexible-joint manipulator* dynamics for forty years — but as analytical formulations and bespoke MATLAB/Fortran codes, not as a packaged simulator with URDF, contact, or a learning-friendly Python API. RoboSim's CB body is squarely in this lineage.

| Work | Year | Contribution | Relation to RoboSim |
|---|---|---|---|
| **Book, "Recursive Lagrangian dynamics of flexible manipulator arms," IJRR 1984** | 1984 | Assumed-modes flexible arm — seminal | Same FFR + reduced-modal idea; analytical model, no simulator. |
| **De Luca & Siciliano, IEEE T-SMC 1991** | 1991 | Closed-form planar flexible-link dynamics | Lineage citation. |
| **Theodore & Ghosal, IJRR 1995** | 1995 | Direct comparison: assumed-modes vs FE for flex-link manipulators | Establishes that FE+CMS is more accurate than assumed-modes — RoboSim sits on the FE+CMS side. |
| **Wasfy & Noor, ASME Appl. Mech. Rev. 2003** | 2003 | Review of computational strategies for flexible multibody | Cites FFR+CMS as the dominant industrial method. |
| **Bremer, *Elastic Multibody Dynamics*, Springer 2008** | 2008 | Modern textbook on flexible multibody | Reference text. |

**Where RoboSim sits.** This community produced the equations of motion and bespoke simulations; we package an FFR+CMS body inside an open-source robotics-learning simulator next to URDF rigid bodies, MPM continua, and contact. The contribution is the packaging, not the formulation.

---

## 7. Floating Frame of Reference + Component Mode Synthesis (the CAE ancestor)

The combination FFR + CMS — moving body frame plus reduced internal modes — has been the workhorse of industrial flexible-multibody simulation since the late 1980s. Reviewers from this community will check that we cite it correctly.

| Work | Year | Contribution |
|---|---|---|
| **Likins, AIAA J. 1967** | 1967 | Modal method for spacecraft attitude with flexible appendages — earliest body-frame + modal-coords decomposition. |
| **Craig & Bampton, AIAA J. 1968** | 1968 | Fixed-interface CMS — the reduction we implement. |
| **Yoo & Haug, J. Struct. Mech. 1986** | 1986 | First concrete academic paper combining FFR with linear modal coordinates for articulated multibody. |
| **Wallrapp, Mech. Struct. Mach. 1994** | 1994 | Standard Input Data (SID) format — defines the FE-to-multibody interface that Adams Flex / Simpack still consume. |
| **Shabana, Multibody Syst. Dyn. 1997 / textbook 2005** | 1997/2005 | Canonical academic treatment of FFR + CMS. |
| **Schwertassek & Wallrapp, *Dynamik flexibler Mehrkörpersysteme*, 1999** | 1999 | German-language standard reference. |

**Computer-graphics parallel reinvention** — graphics rediscovered the per-step rigid-rotation extraction without using the multibody vocabulary:

| Work | Year | Note |
|---|---|---|
| **James & Pai, "DyRT," SIGGRAPH 2002** | 2002 | Modal warping with per-frame rotation — closest graphics analogue of FFR+CMS. |
| **Müller et al., "Stable real-time deformations," SCA 2002** | 2002 | "Stiffness warping" — corotational rotation extraction (full FEM, not modal). |
| **Capell et al., SIGGRAPH 2002** | 2002 | Skeleton-driven dynamic deformations — close cousin. |

**Commercial CAE multibody** — Adams Flex (MSC, late 1980s onwards), Simpack, MotionSolve, Recurdyn all consume CB-reduced flex bodies via SID and run them inside FFR. **Project Chrono** (`chrono::modal::ChModalAssembly`, Tasora et al. 2016+) is the open-source CAE counterpart with full CMS support.

**Where RoboSim sits.** Same recipe — fixed-interface CMS inside an FFR via per-step Kabsch rotation, per `robosim/physics/fem/reduced.py`. The implementation is textbook; the contribution is its placement next to a Featherstone articulated robot and an MPM continuum in an open-source robotics-learning simulator, not in the formulation itself.

---

## 8. Reduced-order models in robotics-learning context

- **SOFA + Model Order Reduction plugin** [Goury & Duriez, IEEE T-RO 2018] — POD + ECSW hyper-reduction for soft-robot control. **Not Craig–Bampton**: snapshot-based POD is mathematically distinct from fixed-interface CMS. Closest neighbour in the open-source robotics-simulator space; must be cited and contrasted in the paper.
- **MeshGraphNets** [Pfaff et al., ICLR 2021] — learned mesh-based simulation; not strictly ROM but the headline learned-surrogate baseline.
- **CROM / LiCROM** [Chen et al., 2022; Chang et al., SIGGRAPH Asia 2023] — neural-field-based continuous ROMs. Discretisation-independent.
- **Mathew et al., IJRR 2025** — ROM for hybrid soft-rigid robots. Recent robotics-side ROM application.
- **GNN-based ROMs** — emerging in 2023–2025 for FEM substructures; the user's separate `feat/gnn-boundary-rom` track explores this for boundary DOFs.

**Where RoboSim sits.** RoboSim's ROM offering is *only* classical FFR+CMS Craig–Bampton — verified against full FEM in `tests/test_fem_solver.py` and used in `--mode cb` demos. SOFA-MOR is the closest existing package but uses POD, not CMS. RoboSim is positioned as a substrate against which learned ROMs (the GNN-ROM research track) can be benchmarked head-to-head.

---

## 9. Reference texts

- Featherstone, *Rigid Body Dynamics Algorithms*, Springer 2008 — RoboSim's RBD core (ABA / RNEA / CRBA).
- Bonet & Wood, *Nonlinear Continuum Mechanics for Finite Element Analysis*, CUP 2008 — corotational / Neo-Hookean.
- Jiang, Schroeder, Teran, Stomakhin, Selle, *The Material Point Method for Simulating Continuum Materials*, SIGGRAPH 2016 Course — MPM teaching reference.
- Ericson, *Real-Time Collision Detection*, Morgan Kaufmann 2004 — edge-edge closest point (used in `contact/`).
