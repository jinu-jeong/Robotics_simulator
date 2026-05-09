# Gap Analysis — where RoboSim sits

## Feature matrix

Legend: `Y` = first-class support, `~` = partial / via plugin / limited, `N` = not supported.
"Pure Py" = physics layer in pure Python (no C++/CUDA build step). RoboSim uses Taichi only for the GGUI viewer.

| Simulator | RBD (ABA) | FEM (coro/NH) | MPM elastic | MPM J2 | MPM DP (sand) | MPM damage | Modal/CB ROM | RBD↔FEM | RBD↔MPM | Pure Py | GPU req | License |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **RoboSim** | Y | Y | Y | Y | Y | Y | Y (CB) | Y (penalty) | Y (1-way) | **Y** | N | MIT |
| MuJoCo (Todorov 2012) | Y | N | N | N | N | N | N | N | N | N | N | Apache 2.0 |
| MuJoCo MJX (2025) | Y | N | N | N | N | N | N | N | N | ~ (JAX) | recommended | Apache 2.0 |
| PyBullet (Coumans) | Y | ~ (mass-spring/FEM stub) | N | N | N | N | N | ~ | N | N | N | ZLib |
| Drake (Tedrake 2019) | Y | ~ (hydroelastic) | N | N | N | N | N | ~ | N | N | N | BSD-3 |
| Gazebo (Koenig 2004) | Y (plug-in physics) | N | N | N | N | N | N | N | N | N | N | Apache 2.0 |
| RaiSim (Hwangbo 2018) | Y | N | N | N | N | N | N | N | N | N | N | Free academic |
| Isaac Gym (2021) | Y | ~ (FleX) | ~ (FleX) | N | N | N | N | ~ | ~ | N | **Y (CUDA)** | Proprietary |
| Isaac Lab/Sim (2024) | Y | Y (PhysX FEM) | ~ (PhysX particles) | N | N | N | N | ~ | ~ | N | **Y** | BSD-3 / Prop. |
| Brax (2021) | Y | N | N | N | N | N | N | N | N | ~ (JAX) | recommended | Apache 2.0 |
| Project Chrono (2016) | Y | Y | ~ (CFD/SPH, no full MPM) | N | ~ (DEM) | N | ~ | Y | ~ | N | optional | BSD-3 |
| **Genesis (2024)** | Y | Y | Y | ~ | ~ | ~ | N | Y | Y | ~ (Taichi) | recommended | Apache 2.0 |
| Dojo (2022) | Y | N | N | N | N | N | N | N | N | N (Julia) | N | MIT |
| SOFA (2007/12) | ~ (rigid plug-in) | Y | N | N | N | N | Y | Y | N | N | N | LGPL |
| ChainQueen (2019) | N | N | Y (diff) | N | N | N | N | N | N | ~ (TF/Taichi) | recommended | open |
| PlasticineLab (2021) | N | N | Y (diff) | Y | N | N | N | N | N | ~ (DiffTaichi) | recommended | MIT |
| DiSECt (2021) | N | Y (cutting) | N | N | N | N | N | N | N | N (PyTorch+Warp) | Y | open |
| DefGraspSim (2022) | Y (Isaac) | Y (FEM) | N | N | N | N | N | Y | N | N | Y | open eval data |
| FluidLab (2023) | ~ | N | Y (diff) | ~ | N | ~ | N | N | ~ | ~ (Taichi) | Y | open |

(Cells marked `~` for Genesis on J2/DP/damage reflect that Genesis ships Drucker–Prager-style granular and elasto-plastic materials but the constitutive coverage is still expanding as of 2024–25; treat as a rapidly moving target.)

---

## Narrative analysis

**The honest picture.** Two recent systems already cover most of RoboSim's headline feature list. **Genesis** (Dec 2024) advertises the same "rigid + FEM + MPM + SPH + PBD in one engine" framing and runs millions of FPS on GPU; it is the dominant point of comparison for any "all-physics" pitch. **Isaac Lab** (2024–25) ships GPU-accelerated rigid + PhysX FEM + particles inside a USD/photoreal stack, so any robotics-oriented "we couple rigid and deformable for manipulation" framing must explicitly say what it adds beyond Isaac Lab's particle/FEM tooling. A naive "first hybrid simulator" claim is unsupportable.

**What is genuinely uncrowded.** Three properties of RoboSim are not jointly satisfied by any of the engines above:

1. **Pure-Python physics.** Every other multi-solver engine has a C++/CUDA/Taichi-jitted core. PyBullet is C++; Drake is C++; Genesis is Taichi-jitted; Isaac is PhysX/USD; Chrono is C++. RoboSim's MPM, FEM and RBD are all NumPy/SciPy with `np.bincount`-vectorised P2G/G2P. This is an *educational and research-prototyping* niche — not a performance niche.
2. **Constitutive breadth in MPM with no diff-sim dependency.** PlasticineLab covers NH+J2 but is bound to DiffTaichi; Klár 2016 / Wolper 2019 are graphics references not packaged for robotics. RoboSim is the smallest single package that exposes Neo-Hookean + von-Mises J2 + Drucker–Prager + damaged-elastic behind one robotics Scene API.
3. **Classical CB reduction sitting next to a full FEM and a full MPM.** SOFA has reduction; Isaac/Genesis do not. Combining a robotics-grade CB reduction with MPM coupling for grasp-and-drop is a small but distinctive engineering point, and it is a natural baseline against which GNN-ROM work (the user's separate research track) can be benchmarked.

**Where there is no defensible gap.** RoboSim cannot claim novel physics — every algorithm it uses (ABA, corotational FEM, MLS-MPM, APIC, CB, Drucker–Prager return mapping, damaged Neo-Hookean) is from prior work. It cannot claim performance — Genesis, MJX, Isaac all dominate. It cannot claim differentiability — it is explicitly not differentiable. It cannot claim production-readiness — 6 kLOC, 229 tests, no CI on Linux GPU, no published benchmarks against real hardware.

**Implication for positioning.** A "we built a new simulator" T-RO/IJRR pitch is hard to defend against Genesis. The headline pitch we go with for **ICRA 2027 software-and-tools** (per `positioning.md`) is the narrower, defensible:

> *Among open-source robotics-learning simulators, RoboSim is the first to expose Floating-Frame-of-Reference Craig–Bampton flexible bodies alongside Featherstone articulated rigid-body dynamics and an MLS-MPM continuum solver under one Scene API.*

The qualifier "open-source robotics-learning simulator" is **load-bearing** and must appear verbatim in the abstract. Strip it and three categories of prior art collapse the claim:

1. **CAE multibody simulators** — MSC Adams Flex, Simpack, MotionSolve, Recurdyn, **Project Chrono** (`chrono::modal::ChModalAssembly`) have shipped FFR+CMS flex bodies for ~30 years. These are *not* robotics-learning tools (no URDF-first API, no Python-first scripting, no learning-friendly Scene graph), but a reviewer who works in CAE multibody will reflexively cite them.
2. **Flexible-link manipulator dynamics** — Book 1984, De Luca & Siciliano 1991, Theodore & Ghosal 1995, Wasfy & Noor 2003, Bremer 2008. Robotics has used FFR + assumed-modes / FFR + CMS for forty years for flexible-arm dynamics, but as analytical models and bespoke MATLAB codes — not as a packaged simulator alongside URDF, contact, and a continuum solver.
3. **SOFA + MOR plugin** (Goury & Duriez 2018) — the closest open-source neighbour but uses **POD + ECSW hyper-reduction**, mathematically distinct from fixed-interface CMS. Must be cited and contrasted in one paragraph of related work to pre-empt the obvious reviewer question.

With those three explicitly acknowledged in related work, the narrow claim survives. Without them, a knowledgeable reviewer torpedoes the paper. See `cb_novelty.md` §4 for the stress-test of the abstract sentence and `positioning.md` for the full submission plan.
