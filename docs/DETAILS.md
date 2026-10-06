# Compliant Finger – detailed research notes

> Full milestone-by-milestone log (verification tables, ablations, design notes). The project overview is in the [top-level README](../README.md); paths below are relative to the repository root.

Research codebase for estimating the 3D deformation of a compliant robotic
finger from camera images and recovering the contact location / contact force
through an interpretable FEM (and reduced-order) mechanics model:

```
Camera image → deformation (u or q) → FEM / ROM mechanics → contact location → contact force
```

**Design principle: visualization first.** Every mechanics result must be
inspectable in the Taichi 3D viewer. The viewer is the primary physics
debugging tool, not an afterthought. Numerical verification and visual
verification are both required for each milestone.

## Status

| Milestone | Content | State |
|-----------|---------|-------|
| 0A | Taichi 3D viewer (orbit/pan/zoom, original/deformed/overlay, amplification, nodes, wireframe, tet edges, fixed BC, contact marker, GT/estimated force arrows, object primitive, observation-camera frustum, GUI panel, head-less screenshots) | done |
| 0B | Mesh infrastructure: structured **hex-dominant quadratic** mesh (20-node H20 body, 10-node T10 Kuhn split at the tip 20 %, hanging-node ties at the interface), boundary triangulation over all boundary nodes, edge sets, checks | done (originally Kuhn T4; upgraded to H20 + T10) |
| 1 | 3D linear-elastic FEM with quadratic elements (H20 3×3×3 Gauss, T10 4-point; `K_e = Σ_g w_g BᵀCB |J|`, master–slave ties `K̃ = TᵀKT`, DOF partition, reactions, stress recovery), point-contact operator `B(c)`, cantilever benchmark vs Euler–Bernoulli/Timoshenko with mesh-refinement study | done |
| 2 | Contact position × force magnitude sweep dataset (`.npz`, self-describing) + Taichi dataset inspector with keyboard navigation | done |
| 3 | Inverse force recovery for a known contact: force-space LS (`argmin_{λ≥0} ‖B_f λ − K_ff u_f‖²`) and displacement-space LS (`argmin_{λ≥0} ‖H K_ff⁻¹B_f λ − H u‖²`) with observation operator `H`; 900-sample consistency check, noise sensitivity, systematic-error budget (mesh, contact location, E), GT/estimate viewer | done (known-model regime) |
| 4 | POD basis `Φ` from the contact-sweep snapshots, Galerkin-reduced mechanics `K_r = ΦᵀKΦ`, `B_r = ΦᵀB`, reduced force estimators, r-sweep on held-out contacts, ROM/full toggle + mode browser in the viewer | done |
| 5 | Synthetic observation camera: pinhole `ObservationCamera` ↔ off-screen GGUI renderer verified to < 0.5 px, surface markers riding on the FEM surface, 1800-image dataset (images, marker pixels + visibility, `q = Φ_rᵀu`, force, contact, `K`, `T_cw`, nuisance parameters), debug view with camera frustum + observation-image inset | done |
| 6 | Vision → `q` → ROM force: geometric / iterative / robust (multi-view) marker LS, learned ridge Δuv→q, optional image CNN; contact-holdout eval; GT vs Φq̂ viewer with observation inset | done |
| 7 | End-to-end pipeline (obs→q→Φq→force), Tests A–D, Level-4 unknown-contact residual search, vision-`q` improvement eval, E2E Taichi inspector | done |
| A/B | Fixed parallel-jaw grasp: GT force control (A) and marker→ROM force feedback (B); Coulomb lift of a rigid object | done |
| C | 7-DoF serial arm + damped-LS IK: home → pregrasp → approach → A/B squeeze → IK lift | done |
| D | Closed-loop world camera: markers → q → ROM force on the moving arm (known ``T_ee``) | done |
| MF | Marker-free visual state estimation: paired marker-on / marker-off renders, ResNet-18 image → `q`, markers as training-time privileged information, same ROM for force / contact | done (simulation PoC) |
| E | Closed-loop **marker-free** grasp (`run_arm_grasp.py --mode nn`): the Stage D world camera without markers → ResNet-18 → `q` → same Kalman / contact search / ROM force as D; live camera inset + force strip | done |
| F | **Material generalisation** (`run_material_sweep.py`): same finger, E ×0.25–×4 and ν 0.30–0.45; Φ, markers and ResNet frozen, only `K → K_r = ΦᵀKΦ` rebuilt (no retraining); offline oracle / marker / nn `q` + closed-loop lifts with the material changed at run time | done |
| G | **Finger self-weight** (`fem.gravity`): ρ g body force on the finger (single-beam demos and both grasp fingers, rotated with `T_ee`), superposed on the contact solve; the estimator compensates the known sag (pixel space for marker LS, exact) | done |
| H1 | **Deformable object, lumped** (`object.stiffness`, `run_soft_object_sweep.py`): object = spring `k_obj` in series with the fingers; vision force is object-agnostic, jaw-opening force is not; object stiffness recovered from opening + λ̂ | done |

## Results layout (paper figures)

```
results/figure1/   overall algorithm / concept panels (Fig. 1; schematic-first, 9 panels)
                   01 = pipeline schematic; 02 = world grasp; 09 = information paths
results/figure2/   closed-loop world vs marker-free nn demos (Fig. 2)
results/figure5/   material generalisation sweep (Fig. 5)
results/summary.txt  paper storyline (ICLR target) + where every number comes from
results/etc/       everything else (milestone dumps, checkpoints, debug)
```

```bash
python scripts/render_figure1_lateral.py         # regenerate lateral (inner-face) Fig.1 scenes
python scripts/export_figure1_assets.py          # refresh Fig. 1 panels from results/etc/ + figure2/
python scripts/run_arm_grasp.py --mode world     # → results/figure2/world/
python scripts/run_arm_grasp.py --mode nn        # → results/figure2/nn/
python scripts/run_material_sweep.py             # → results/figure5/  (≈ 3 min incl. closed loop)
```

Every Figure-1 scene / verification panel (01–12) uses the **same inner-face /
−y press** as Stages D/E. Panel 03 is the FEM↔beam-theory convergence check
for that lateral tip load (`configs/cantilever_lateral.yaml`: bending about z,
`I = H W³/12`, L/W = 5). The FEM curve plateaus *between* Euler–Bernoulli
(1.000 mm) and Timoshenko (1.033 mm) at **1.016 mm** — convergence of the
3D FEM under mesh refinement, not a failed match to 1D theory (Milestone 1:
what “correct” means without an analytic tip deflection; δ_T vs Richardson
δ_3D on panel 03).
Self-weight ρg is shown on the finger overlay panel (not a separate panel).

The y–z coupling artefact of the former all-Kuhn T4 mesh (a −y load lifting
the tip in +z by 13 % at 30×6×4) is gone with the hex-dominant mesh:
`tests/test_beam_theory.py` now asserts `|u_z| < 1e-3 |u_y|` and Maxwell–Betti
reciprocity `f_y·u_z = f_z·u_y` to 1e-8.

## Mesh: hex-dominant quadratic elements (H20 + T10)

`src/geometry/finger.py` meshes the finger the way a commercial sweep mesher
would: a structured grid of **20-node serendipity hexahedra** over the body
and **10-node tetrahedra** (Kuhn split, 6 per cell) over the tip
`mesh.tet_fraction` (default 20 %) of the length, i.e. ~80 % hex by volume.
The hex|tet interface is made conforming with **hanging-node ties**: the
mid-diagonal node of each interface quad is slaved to the quad's 8 nodes with
the H20 face weights (−¼ corners, ½ mid-edges); `LinearFEM` solves
`K̃ = TᵀKT + P_s`, `f̃ = Tᵀf`, `u = Tũ` (DOF numbering unchanged, `K` itself
stays the unconstrained matrix so the ROM's `ΦᵀKΦ` is unchanged in form).
Resolution comes from `mesh.element_size` [m] (default smallest dimension / 4
= 2.5 mm → 40×8×4 cells, 7 913 nodes, 23 739 DOF; `nx/ny/nz` still override
single axes). Everything downstream (contact operator `B(c)`, markers,
cameras, viewer, contact search) keeps working on `surface_faces`, which is
now the outward triangulation of the boundary over *all* boundary nodes
(quad8 → 6 triangles, tri6 → 4); the parent faces are kept as
`surface_quads` / `surface_tris` for consistent tractions
(`loads.total_force_on_plane`). Elements: `src/fem/elements.py` (shape
functions, 4-point / 3×3×3 Gauss, `K_e`, mean `B̄`, consistent mass shares for
the body force); tests: `tests/test_quadratic_elements.py` (partition of
unity, six rigid modes, exact constant strain, tied patch test).

Effect on the cantilever benchmark (tip deflection vs Timoshenko, `−z` load):
5×1×1 −4.6 %, 10×2×2 −2.1 %, 20×4×4 −1.5 %, 60×12×12 −1.3 % — the old T4 mesh
needed 100×10×20 (70 k DOF) for −5.6 %.

## Finger self-weight (Stage G)

The finger is no longer weightless. `fem.gravity: {enabled, vector}` (world
frame, `grasp.yaml` and `fem.yaml`) turns `material.density` into a consistent
body-force vector `f_g` (`ρ a ∫N_i dV` by element quadrature, `LinearFEM.body_force_vector`);
`FingerFEMModel.gravity_displacement(R)` = `K⁻¹ f_g` for a finger whose local
frame is rotated by `R` (cached per orientation). Because the FEM is linear,
every state is a superposition `u = u_contact(λ) + u_g(R)`:

* single beam (`run_3d_fem_demo.py`, `render_figure1_lateral.py`): `u_g` for the
  upright finger; 0.1×0.02×0.01 m at ρ = 1200 kg/m³ weighs 0.24 N and sags
  0.34 mm at the tip (Euler–Bernoulli `wL⁴/8EI` = 0.35 mm; the quadratic mesh
  is within a few % of theory for *both* the distributed and the 1 N tip load);
* two fingers (`GraspMechanics.displacement(λ, R)` with `R = T_ee[:3,:3]`,
  used by `GraspSim.step` / `scene.py`): the right finger is the y-mirror of
  the left, which is exact for the box under a gravity vector in the x–z plane;
* contact force from opening includes the gravity term:
  `λ = k (½(w − g) + δ_g)` where `δ_g = −n·u_g(c)` is the sag of the contact
  point into the object (7.8 µm away from it for the upright grasp pose).

The estimator treats `u_g` as a *known load*:

| path | compensation | effect at 2 N (Fig. 1) / in the arm loop |
|---|---|---|
| inverse (full u) | `u − u_g` | exact |
| marker LS (B, D) | subtract the sag's *marker motion* `uv_g − uv_0` in pixel space before the LS fit | exact (offline Fig. 5 numbers unchanged bit-for-bit) |
| ROM on `q = Φᵀu` | `q − Φᵀu_g` | 2.002 N with or without (the −z sag is orthogonal to the −y modes on the hex-dominant mesh; the T4 mesh's y–z coupling gave 1.918 → 2.000 N) |
| marker-free NN (E) | none: the network regresses `q_sim = Φᵀu_contact`, so its `q̂` carries no sag; the sag only lives in the image | 7.3 % force / 4.0 mm contact with the sag in the frame vs 7.3 % / 2.8 mm without |

Why the marker path must compensate in pixel space: `u_g` is a −z bending
field, almost entirely outside span Φ (built from −y inner-face presses), so
the LS fit of `Φq` to marker pixels smears it into `q`; subtracting `Φᵀu_g`
afterwards does *not* remove it. Uncompensated, the world-camera loop degrades
from 19.5 % / 14.7 mm to 176 % / 26 mm and drops the object, the fixed-camera
marker loop from 19 % / 10 mm to 306 % / 30 mm (drops). Subtracting `Φᵀu_g`
from the NN's `q̂` is likewise wrong (measured on the former T4 mesh: 4 % → 67 %),
because `q̂` never contained it.

`generate_grasp_nn_dataset.py` now renders `u_contact + u_g(R_ee)` with the
contact-only target (the checkpoint in use was trained without the sag and is
empirically invariant to it; regenerate + retrain to make that by design).
`estimator.gravity_compensation: false` switches the compensation off for
ablations; `run_3d_fem_demo.py --no-gravity` removes the body force.

## Deformable object, lumped spring (Stage H1)

```bash
python scripts/run_soft_object_sweep.py          # → results/etc/soft_object/<timestamp>/ (≈ 6 min)
```

`object.stiffness` [N/m] (`grasp.yaml`, `null` = rigid) makes the grasped width a
spring, `Δw = λ / k_obj`, in series with the two fingers (`GraspMechanics`):
`w − g + 2δ_g = λ (2/k + 1/k_obj)`. The finger FEM only sees `λ`, so the vision
estimator is unchanged; the box is drawn (and rendered for the NN) squashed by `Δw`.
`estimate_object_compliance(opening, λ̂)` returns the closure the finger does not
explain, per newton.

Sweep (finger `k` ≈ 1550 N/m, target 2.2 N, 3 noise seeds per vision mode, median over lift/hold):

| k_obj [N/m] | squash | jaw opening, rigid assumption | markers, known contact | markers, unknown contact | 1/k̂_obj known / true [mm/N] |
|---:|---:|---:|---:|---:|---:|
| rigid | 0 | 0 % | 1.2 % | 4.7 % (3.8 mm) | −0.02 / 0 |
| 5000 | 0.4 mm | 15.5 % | 1.1 % | 19.2 % (11.7 mm) | 0.18 / 0.20 |
| 2000 | 1.1 mm | 38.7 % | 1.1 % | 9.5 % (4.4 mm) | 0.48 / 0.50 |
| 1000 | 2.2 mm | 77.4 % | 1.1 % | 13.5 % (9.8 mm) | 0.98 / 1.00 |
| 600 | 3.6 mm | 129 % | 1.1 % | 9.4 % (6.7 mm) | 1.64 / 1.67 |

All grasps lift. Force from the jaw opening (what a gripper without a force
sensor knows) needs the object's stiffness; the vision force does not, and
together they measure it. The unknown-contact column (contact error in brackets)
varies with the noise seed, not systematically with `k_obj`.

**Contact-lock fix found by this sweep.** The unknown-contact lock used to collect
votes during the pre-lock creep, i.e. at ~0.55 N and low SNR, well before the
1.5 N preload dwell it was designed around. That made it depend on the force
*ramp rate*: a softer object ramps slower at the same creep speed, and the lock
landed 25–30 mm off and dropped the object (a rigid object crept 0.72× as fast
failed the same way). Now votes are only taken once the preload is reached
(`GraspEstimator.lock_votes_open`, set by `GraspSim`), and the preload is detected
from λ̂ at the config contact instead of the true force (which a real robot does
not have). Effect on the existing stages (default config, same metric old → new):

| stage | contact error | force error, lift/hold median | first lift |
|---|---:|---:|---:|
| B, rigid (3 seeds) | 10.4 → 3.8 mm | 22.7 → 4.7 % | — |
| D, world camera | 14.7 → 4.1 mm | 0.56 → 0.22 N (20.5 → 10.5 %) | 2.90 → 3.68 s |
| E, marker-free NN | 4.1 → 3.2 mm | 0.47 → 0.46 N | 2.37 → 3.51 s |

The price is a later lift (the dwell now always runs before locking).
`results/figure2/` was produced before this change and has not been regenerated.

**Arm grasp of a soft object (`run_arm_grasp.py --object-stiffness 1000`).** Stage D
(markers) lifts it with 2.3 % raw force error / 2.7 mm contact error. Stage E
(marker-free NN) over-estimates by ~2× from first touch, under-squeezes and drops it.
Ablation at a *fixed* finger field (same `u`, rendered scenes differ only as listed):

| rendered scene | λ̂ at λ = 0.8 / 1.5 / 2.2 N |
|---|---|
| rigid object (training distribution) | 0.94 / 1.75 / 2.48 |
| soft object (k_obj = 1000) | 1.72 / 2.97 / 4.08 |
| soft jaw opening, full-width box | 1.75 / 2.99 / 4.04 |
| rigid jaw opening, squashed box | 1.03 / 1.71 / 2.47 |

The network reads the **jaw opening**, not the finger's bending: in the training data
(rigid box) the opening is a deterministic function of the force, so it is a shortcut.
This is invisible on rigid objects and breaks the "network learns geometry" claim as
soon as opening and force decouple. Fix (not done yet): regenerate the grasp-NN
dataset with a random `k_obj` per sample so the opening is not predictive of `q`,
then retrain.

## Demo videos

```bash
python scripts/run_arm_grasp.py --mode nn --video                                  # E, display-smoothed force
python scripts/run_arm_grasp.py --mode nn --video --raw-force                      # E, raw estimate
python scripts/run_arm_grasp.py --mode world --video --raw-force --object-stiffness 1000   # D, soft object
```

`--video` runs headless and writes `viewer.mp4` (1280×720, 10 s real time: arm, camera
inset, GT-vs-estimate strip) and, in `nn` mode, `nn_input.mp4` (the 240×180 frames the
network sees, ×4) to `results/videos/<mode>[_soft<k>][_raw]/`, never to `results/figure2/`.
By default the estimate curve is a causal EMA of λ̂ (`display.force_ema`); it never uses
the ground truth. `--raw-force` plots λ̂ unsmoothed. (Plots made before 2026-09-28,
including `results/figure2/`, blended the displayed estimate 45 % toward GT; that
option has been removed.)

| folder | stage | object | raw force error / contact | result |
|---|---|---|---|---|
| `nn`, `nn_raw` | E marker-free | rigid | 5.3 % / 2.8 mm | lifts |
| `world`, `world_raw` | D markers | rigid | 10.5 % / 4.1 mm | lifts |
| `world_soft1000_raw` | D markers | k_obj = 1000 | 2.3 % / 2.7 mm | lifts |
| `nn_soft1000_raw` | E marker-free | k_obj = 1000 | 73 % / 3.3 mm | **drops** (opening shortcut, above) |

The `nn*` rows were regenerated on 2026-10-05. Before that, the deployed network received flat
renders while it had been trained on photoreal-graded frames: `GraspSceneAppearance.from_dict`
silently dropped the checkpoint's `photoreal` setting, and the read-out drifted high during the lift
(rigid: 25.6 % / 3.2 mm). `NNStateEstimator.frame` now applies the checkpoint's grade, with film grain
seeded independently of the render noise (`grain_seed`); unknown appearance keys raise.

## Setup

```bash
conda env create -f environment.yml
conda activate compliant_finger
pip install -e .            # optional; scripts also work from the repo root without it
```

An existing environment with `python>=3.10 numpy scipy pyyaml matplotlib taichi>=1.7 pytest`
is sufficient for Milestone 0/1. The Taichi backend is selected in
`configs/viewer.yaml` (`backend: auto | cpu | cuda | vulkan | metal`).

## Run the viewer demo (Milestone 0A)

```bash
python scripts/run_viewer_demo.py                       # interactive window
python scripts/run_viewer_demo.py --nx 40 --ny 8 --nz 4 --tip-deflection-mm 3 --amplification 5
python scripts/run_viewer_demo.py --headless --screenshot results/figures/demo.png
```

The demo applies an *artificial* Euler–Bernoulli cantilever deflection to a
rectangular finger (no FEM yet) so every viewer feature can be checked:
root fixed nodes (red), displacement-colored deformed surface, undeformed
ghost wireframe, contact marker (yellow), ground-truth force (green) vs.
estimated force (magenta) arrows, probe sphere, and the observation-camera
frustum (orange).

## FEM verification and demos (Milestone 1)

```bash
python scripts/run_cantilever_verification.py            # refinement study + viewer of the finest mesh
python scripts/run_cantilever_verification.py --fine --no-view
python scripts/run_cantilever_verification.py --config cantilever_lateral --fine --no-view   # lateral (−y) load → Fig. 1 panel 03
python scripts/run_3d_fem_demo.py                        # finger from configs/fem.yaml, point contact via B(c)
python scripts/run_3d_fem_demo.py --force 2 --contact-rel 0.6 0.5 1.0
```

Cantilever (L=100, W=20, H=10 mm, E=50 MPa, ν=0.3, F=1 N; δ_EB = 4.000 mm, δ_T = 4.031 mm):

| mesh (H20 + T10 tip) | DOFs | δ_FEM [mm] | error vs Timoshenko |
|------|-----:|-----------:|--------------------:|
| 5×1×1 | 225 | 3.847 | −4.6 % |
| 10×2×2 | 1 095 | 3.945 | −2.1 % |
| 20×4×4 | 6 507 | 3.969 | −1.5 % |
| 40×8×8 | 44 115 | 3.976 | −1.4 % |
| 60×12×12 | 140 475 | 3.978 | −1.3 % |

**Fig. 1 panel 03 is the same study with the grasp load** (`cantilever_lateral.yaml`,
ν = 0.40, F = 1 N in −y, `I = H W³/12`, L/W = 5; δ_EB = 1.000 mm, δ_T = 1.033 mm):

| mesh | DOFs | δ_FEM [mm] | vs EB | vs Timoshenko |
|------|-----:|-----------:|------:|--------------:|
| 1×1×1 | 60 | 0.703 | −29.7 % | −31.9 % |
| 2×1×1 | 96 | 0.858 | −14.2 % | −17.0 % |
| 5×1×1 | 225 | 0.960 | −4.0 % | −7.0 % |
| 10×2×2 | 1 095 | 0.999 | −0.1 % | −3.3 % |
| 20×4×2 | 3 711 | 1.010 | +1.0 % | −2.2 % |
| 40×8×4 | 23 739 | 1.015 | +1.5 % | −1.8 % |
| 60×12×6 | 73 911 | 1.016 | +1.6 % | −1.7 % |
| 80×12×6 | 98 223 | **1.016** | +1.6 % | −1.6 % |

The curve has already flattened (40→60 changes δ by 0.001 mm; 60→80 by
0.0003 mm). It does **not** land on either 1D line, and should not:

* **Not Euler–Bernoulli.** EB ignores shear. L/W = 5 is stocky; Timoshenko's
  shear term is 3.3 % of the tip deflection. 3D elasticity includes shear, so
  the continuum limit must sit *above* EB.
* **Not Timoshenko either.** Timoshenko is still a 1D theory. Beam theory
  clamps only the neutral axis (`w = w' = 0`); the FEM fixes every DOF on the
  root *face*, which blocks Poisson contraction and warping at `x = 0`
  (ν = 0.40). That extra root stiffness is the ~1.6 % gap to Timoshenko.
  Cowper `κ` is a section-averaged shear factor, not the 3D shear field, and
  the load is a uniform end-face traction rather than a centroid point force.

**What “correct” means here (we do not know an analytic 3D tip deflection).**
Refining the mesh (“곱한다” = same physics, smaller cells / more DOF) is
*verification*, not matching an unknown closed-form truth:

1. **Mesh plateau.** δ stops moving under refinement → the number is a
   property of the continuum model, not grid luck. Otherwise `u` (and every
   POD / vision / force label built on it) would change when the mesh does.
2. **Patch test exact.** Uniaxial tension (including the tied hex|tet
   interface) reproduces the analytic field; reactions balance the load to
   1e-10. That rules out a broken `K` / assembly / ties that could still
   “converge” to the wrong equation.
3. **1D gap is the expected modelling gap.** |δ_FEM/δ_T − 1| floors at
   ~1.6 % for the reasons above — not a discretisation failure. A 50 % gap
   would be a bug; a 0 % gap would be suspicious for a stocky 3D clamp.

Together: (2) implementation, (1) enough mesh, (3) physics reading. That is
why later stages may treat this FEM `u` as GT.

**Fig. 1 panel 03’s two curves (same δ_FEM in the numerator).**

* `|δ_FEM/δ_T − 1|` — how far this mesh’s tip is from **1D Timoshenko**.
  Floors at ~1.6 %: modelling gap, independent of further refinement.
* `|δ_FEM/δ_3D − 1|` — how far this mesh’s tip is from the **same 3D FEM’s
  continuum limit**. `δ_3D` is *not* another solver: Richardson extrapolation
  from the two finest meshes estimates δ(h→0) under `δ(h) ≈ δ_∞ + C h^p`
  (p≈2 for quadratic elements). So it is “this mesh vs the end of the same
  FEM series,” i.e. discretisation error only. The name “3D” distinguishes
  that limit from the 1D beam lines (`δ_∞` would be equally fine).

Panel 03 therefore uses a wide y-range on tip deflection (coarse 1×1×1 ≈
0.70 mm) so a ~1 % continuum gap is not magnified by a 0.96–1.03 zoom, and
the right-hand panel shows the descending Richardson curve (fit slope ≈ 1.9)
next to the Timoshenko floor. For comparison the former linear-tet (T4) mesh
gave −64 % at 10×2×2 and −4.4 % at 80×16×16 (70 k DOF) on the top-face study
above. Results land in `results/etc/cantilever*/<timestamp>/`
(`metrics.json`, `convergence.png`, `config.yaml`).

### Dataset-type loads vs strength-of-materials beam theory

```bash
python scripts/run_beam_theory_comparison.py                       # default mesh + 80×8×16 curves, viewer
python scripts/run_beam_theory_comparison.py --fine 80 8 16 --fine 100 10 20 --headless
```

The dataset samples differ from the benchmark: the load sits at an interior station
`x = a` and off the centre line (`e = y − W/2`), which twists the finger. The script
compares section quantities (per-station fit `u_z = c0 + c1(y−W/2) + c2(z−H/2)`, i.e.
mean deflection and twist) against the point-load Euler–Bernoulli/Timoshenko curve
`w(x)` and Saint-Venant torsion `θ(x) = F e min(x,a)/(GJ)`, `J = 0.229 W H³`
(E=50 MPa, ν=0.4, F=1 N, contact at 0.98 L / 0.85 W):

Numbers below were recorded with the former linear-tet (T4) mesh and are kept as the
reference for what the quadratic upgrade fixed (re-run the script for current values):

| mesh (T4, legacy) | DOFs | tip w FEM / Timoshenko [mm] | tip θ FEM / Saint-Venant [mrad] |
|------|-----:|----------------------------:|--------------------------------:|
| 50×6×10 | 11 781 | 3.411 / 3.912 (−12.8 %) | 6.99 / 8.39 (−16.7 %) |
| 80×8×16 | 37 179 | 3.630 / 3.912 (−7.2 %) | 7.64 / 8.39 (−9.0 %) |
| 100×10×20 | 69 993 | 3.693 / 3.912 (−5.6 %) | 7.90 / 8.39 (−5.8 %) |

Both bending and torsion converge monotonically towards theory from the stiff side and the
twist is linear in `e` with the right sign. The T4 caveat (the Kuhn split is not
mirror-symmetric in `y`, giving a spurious twist for a centred load) no longer applies to
the hex-dominant mesh. In the viewer `A` toggles FEM ↔ beam-theory kinematics.
Results: `results/etc/beam_theory/<timestamp>/` (`comparison.png`, `metrics.json`).

## Contact dataset (Milestone 2)

```bash
python scripts/generate_fem_dataset.py                                  # configs/dataset.yaml -> data/processed/*.npz
python scripts/inspect_fem_dataset.py data/processed/contact_sweep_top.npz
```

Inspector keys: `Right`/`Left` next/previous sample (`Shift` = ±10), `Home`/`End`,
`Space` random, `U` cycle sort order (index / force / max |u| / contact x). The GUI
panel shows sample index, force, contact position and max displacement.

## Inverse force recovery (Milestone 3)

```bash
python scripts/run_inverse_force_eval.py                                # all 900 samples + noise sweep -> results/inverse_force/
python scripts/run_inverse_force_robustness.py                          # systematic errors: mesh, contact location, E (~1 min)
python scripts/inspect_inverse_force.py data/processed/contact_sweep_top.npz --start 899
```

Given the FEM displacement field `u` and the *known* contact `c`, the normal force is
recovered by linear least squares in two ways (`src/contact/inverse_force.py`):

| method | formulation | needs | noise behaviour |
|--------|-------------|-------|-----------------|
| `force` | `λ̂ = max(0, B_fᵀK_ff u_f / B_fᵀB_f)` (residual `B_f λ − K_ff u_f`) | full field `u` | `K` amplifies high-frequency errors in `u` |
| `displacement` | `g = K_ff⁻¹B_f(c)` (cached influence field), `λ̂ = max(0, gᵀHᵀHu / gᵀHᵀHg)` | any observation `y = H u` | averages over all observed DOFs |

`H` (`src/contact/observation.py`) currently selects DOFs: `full`, `surface`, a finger
face (`top`, `side_pos_y`, …) optionally restricted to components (`top:z`). A `vector`
mode solves for all three force components (`G = K_ff⁻¹B_xyz`, 3×3 LS) and reports the
direction error.

Results on the 900-sample dataset (float32 storage, 11 781 DOFs, F = 0.1 … 3 N):

| estimator (exact `u`) | MAE [N] | mean rel. error | slope of `F_est` vs `F_true` |
|------------------------|--------:|----------------:|-----------------------------:|
| force-space, full | 5.4e-5 | 0.0034 % | 0.999998 |
| displacement-space, full / surface / top:z | 1e-9 | 0.0000 % | 1.000000 |
| displacement-space, vector mode | 1e-9 | 0.0000 %, direction error 0.0000° | – |

Noise sensitivity (`u + ε`, ε ~ N(0, σ²) per DOF; max |u| in the dataset is 0.04–10.5 mm):

| σ [m] | force-space | displacement, full | displacement, surface | displacement, top:z (350 DOFs) |
|------:|------------:|-------------------:|----------------------:|-------------------------------:|
| 1e-7 | 17 % | 0.000 % | 0.000 % | 0.001 % |
| 1e-6 | 119 % | 0.003 % | 0.005 % | 0.011 % |
| 1e-5 | 1021 % (46 % clipped to 0) | 0.032 % | 0.047 % | 0.107 % |
| 1e-4 | 8662 % | 0.33 % | 0.45 % | 1.10 % |
| 3e-4 | 25816 % | 1.01 % | 1.40 % | 3.28 % |

Both estimators are linear, so the error grows ∝ σ; the displacement-space fit is
~5 orders of magnitude more robust and stays below 5 % up to σ ≈ 0.3 mm even with only
the 350 out-of-plane DOFs of the top face. The force-space residual's 0.003 % error on
*exact* data is already the float32 storage rounding amplified by `K` (it grows towards
the tip where `|u|` is largest). Conclusion for the later vision pipeline: fit the
forward model to the observation, never differentiate the measured field.

**What the exact-`u` result does and does not show.** The dataset was generated with
the same linear FEM that is inverted, so `u = g λ` and `λ̂ = gᵀu / gᵀg` recover `λ`
by construction: the y = x plot is a *consistency check* of the code, not evidence
that the method works on measured data. Random noise is handled well because least
squares averages it out; *systematic* model errors are not averaged and map almost
linearly into the force estimate. `scripts/run_inverse_force_robustness.py` quantifies
them (F = 1 N, default 50×6×10 model):

| systematic error | effect on λ̂ (displacement-space) | force-space |
|------------------|----------------------------------|-------------|
| (a) discretisation: truth on a 2× finer nested mesh (100×12×20), inverted with the model mesh | **+8.5 %** constant over-estimate (8.2 … 9.3 % over all 75 contacts) – exactly the stiffness ratio of the two T4 meshes (max\|u\| fine/coarse = 1.085) | −49 … +262 %, erratic |
| (b) contact position error along the finger | −3.6 %/mm at 0.5 L, −2.3 %/mm at 0.7 L, −1.6 %/mm at 0.9 L; matches beam theory `d ln δ/da = 2/a − 1/(3L−a)`. ±5 mm → ∓7 … 20 % | exact or wildly wrong depending on whether the shifted triangle shares the loaded nodes |
| (b') contact position error across the width | < 0.3 % for 3 mm (bending dominates) | – |
| (c) Young's modulus | `λ̂/λ = E_model/E_true` exactly (10 % E error = 10 % force error) | same |

So the accuracy of the inverse problem is bounded by the fidelity of the forward
model (mesh convergence, material calibration, contact localisation, boundary
conditions), not by the inverse algorithm, which is well conditioned. The
discretisation bias is constant over the workspace and therefore calibratable; the
+8.5 % of the default mesh can be reduced by refining (cost ×3–7) or by higher-order
elements in a later milestone. The fit residual grows monotonically with the contact
position error (0 → 1.1 % for 0 → 10 mm), which is the signal a later
unknown-contact search (Phase 8) can exploit. Results:
`results/inverse_force_robustness/<timestamp>/` (`robustness.png`, `metrics.json`).

Viewer (`inspect_inverse_force.py`): green = `F_true`, magenta = `F_est` (drawn slightly
offset so both stay visible when they coincide); the panel lists λ_true / λ_est,
relative and direction error. `Y` cycles the noise level, `D` toggles the method,
`Q` normal/vector mode, `U` the observation operator, `A` switches the mesh between the
(noisy) measurement and the forward model `u = g(c) λ̂`.

## Reduced-order model (Milestone 4)

```bash
python scripts/run_rom_analysis.py                                      # POD + r sweep -> data/processed/rom_basis_top.npz, results/rom/
python scripts/inspect_rom.py --r 4 --start 899                         # viewer: A = full <-> ROM, . , = modes, Q = mode browser
```

`src/rom/pod.py` computes the (uncentred) SVD basis of the snapshot matrix
`U = [u¹ … uˢ]`; `src/rom/reduced_mechanics.py` builds `K_r = ΦᵀKΦ`, `B_r(c) = ΦᵀB(c)`
and solves the reduced forward problem `K_r q = B_r λ` as well as the reduced inverse
problem (displacement-space `g_r = K_r⁻¹B_r`, or force-space `K_r q̂`) from `q̂ = Φ_rᵀu`.

Two facts shape the evaluation. (i) The FEM is linear, so the 12 force levels of one
contact are collinear: the 900 snapshots have physical rank 75 (= number of contacts),
and only *held-out contact locations* test generalisation. The analysis holds out every
3rd contact (600 train / 300 test samples, rank 50). (ii) The dataset is stored in
float32; the quantisation noise appears as a flat tail `σ_k/σ_1 ≈ 3e-9` beyond k = 50,
so the rank tolerance is set from the storage dtype (`storage_rank_tolerance`).

| r | energy | ‖u − Φ_rΦ_rᵀu‖/‖u‖ train / held-out | force error (exact u), reduced displacement-space, held-out | reduced force-space, held-out | force error, σ = 1e-4 noise |
|--:|-------:|------------------------------------:|-----------:|-----------:|-----------:|
| 1 | 99.90 % | 4.86 / 4.86 % | 0.049 % | 0.049 % | 0.33 % |
| 2 | 99.97 % | 2.05 / 2.06 % | 0.034 % | 0.47 % | 0.33 % |
| 4 | 99.9996 % | 0.32 / 0.33 % | 0.010 % | 0.97 % | 0.32 % |
| 8 | ≈100 % | 0.060 / 0.066 % | 0.008 % | 6.0 % | 0.32 % |
| 16 | ≈100 % | 0.024 / 0.034 % | 0.008 % | 37 % | 0.32 % |
| 32 | ≈100 % | 0.011 / 0.026 % | 0.008 % | 106 % | 0.32 % |
| 50 | 100 % | 0.000 / 0.024 % | 0.007 % | 143 % | 0.32 % |

The first modes are physically interpretable (1 first bending, 2 second bending,
3 torsion, 4 third bending, 5 second torsion, …). Four modes reproduce every
deformation to 0.3 % and eight to 0.06 %, held-out contacts included; the held-out
reconstruction error saturates at 0.024 % (interpolation between the 50 training
contacts), which is far below the FEM's own discretisation error. The reduced
displacement-space estimator recovers the force to 0.01 % from r ≥ 4 and is as
noise-robust as the full-space one (0.32 % at σ = 1e-4 – the ROM neither helps nor hurts
there). The reduced *force-space* estimator is exact only for fields inside the span:
the unresolved remainder `u − Φ_rΦ_rᵀu` is amplified by `K_r` exactly like measurement
noise, so it degrades with r and is unusable beyond r ≈ 3.

Consequence for Milestones 5–7: a vision model has to predict only `q ∈ R^{4…8}`;
the force then follows from `λ̂ = g_rᵀq̂ / g_rᵀg_r` with a 4×4 … 8×8 reduced system.
Results: `results/rom/<timestamp>/` (`rom_analysis.png`, `pod_modes.png`, `metrics.json`).

## Synthetic observation camera (Milestone 5)

```bash
python scripts/run_synthetic_camera_check.py            # renderer vs pinhole model, marker motion, debug view
python scripts/generate_vision_dataset.py               # configs/vision.yaml -> data/processed/vision_top.npz (1800 images, 15 s)
python scripts/inspect_vision_dataset.py                # figures + 3D debug view with observation-image inset
```

**Two cameras, kept apart (brief §15).** The developer camera of the viewer is a free
orbit camera without research meaning. The *observation camera* (`ObservationCamera`,
`src/visualization/camera.py`) is a calibrated pinhole model – position, look-at, vertical
FOV, resolution → `K`, `T_cw`, `project()` – and is what the vision model will see.
`src/rendering/synthetic_camera.py` renders its view in a hidden GGUI window at sensor
resolution (flat colours, one point light, no textures: a *controlled* vision problem,
not photorealism). In the debug viewer the same camera appears as a frustum, and its
rendered image can be shown as a picture-in-picture inset (`viewer.set_overlay_image`).
GGUI caveat (Taichi 1.7, macOS/MoltenVK): the off-screen renderer must issue the *same*
draw calls with the *same* vertex counts every frame – GGUI caches one renderable per
draw slot and re-creates its GPU buffers when the count changes, which blanks the finger
in every following frame. The marker-disc / marker-sphere / object layers are therefore
always drawn at full buffer capacity (`fixed_draw=True` in `mesh_renderer.py`), with
unused triangles parked outside the frustum.

**Markers.** `src/rendering/markers.py` attaches a grid of virtual markers to boundary
triangles with barycentric weights, so `p_k(u) = Σ w_ki (x_i + u_i)` follows the FEM
field exactly; markers are rendered as flat discs on the surface (like printed dots) and
labelled with their pixel coordinates and a visibility flag (facing test + frustum).

**Verification.** The check script renders the marker discs, detects the blobs in the
image and compares their centroids with `ObservationCamera.project` of the 3D marker
positions – the labels and the pixels have to come from the same camera. Result:
mean 0.30 px / max 0.45 px at 640×480 and 0.22 / 0.40 px at 192×144 (sub-pixel;
the residual is disc rasterisation). A first attempt with *spheres* as markers gave a
systematic 2.2 px bias: half of each sphere is buried in the surface, so the visible
cap's centroid shifts towards the camera – a rendering artefact that the visual check
caught immediately and the disc markers remove.

**Dataset** (`configs/vision.yaml`, `src/vision/dataset.py`): every FEM sample of the
contact sweep is rendered from 2 jittered camera poses (±5 % L position, ±2 % L target,
±2° FOV) with a random background colour (5 choices), a jittered point light, Gaussian
image noise (σ = 0.02) and marker pixel noise (σ = 0.5 px, clean labels kept as well).
Stored per image: `marker_px`, `marker_visible`, `q = Φ_8ᵀu` (POD basis of M4), force
magnitude/vector, contact position/normal, `K`, `T_cw`, camera pose, appearance
parameters and the FEM sample index (the full `u` is not duplicated). 192×144 RGB,
1800 samples, 111 MB.

**What the vision model will face** (`signal_vs_force.png` of the inspector): the marker
image motion is exactly linear in the force (linear FEM + fixed camera) and ranges from
0.03 px (0.1 N near the root) to 15 px (3 N at the tip) at 192×144. With σ = 0.5 px
marker noise, 21 % of the samples have a *mean* marker motion below the noise level –
so Milestone 6 needs either higher resolution (46 px max at 640×480), sub-pixel marker
detection, or the full image (the finger silhouette carries the deformation too), and
force errors at small loads will be noise-dominated. This is a property of the sensing
setup, not of the mechanics, and it is now quantified before any learning happens.

Results: `results/synthetic_camera/<timestamp>/` (`camera_check.png`, `force_strip.png`,
`debug_view.png`, `metrics.json`) and `results/vision_dataset/<timestamp>/`
(`samples_grid.png`, `signal_vs_force.png`, `camera_poses.png`, `debug_view*.png`).

## Vision → q → force (Milestone 6)

```bash
python scripts/train_vision_model.py                 # geometric + ridge + cnn + oracle on held-out contacts
python scripts/inspect_vision_model.py --model ridge # GT u vs Φq̂ (A), GT/est force, observation inset
```

The research question at this stage is Level 3 of the brief: can a vision estimate of
the reduced coordinates `q ∈ R^8` drive the ROM displacement-space force estimator to a
usable contact force? Contact location is still assumed known (Level 4 is later).

**Estimators** (`configs/vision_model.yaml`)

| name | input | how |
|------|-------|-----|
| `geometric` | marker pixels + known camera | one-shot image Jacobian `Δuv ≈ J(c) q`, ridge LS (no training) |
| `iterative` | same | Gauss–Newton (2 iters) + Huber IRLS on the same Jacobian |
| `robust` | same, all views of one FEM sample | iterative, then average `q` across those views |
| `ridge` | marker pixels | learned linear map `Δuv → q` on the training contacts |
| `cnn` | RGB observation image | small spatial CNN (PyTorch, optional) |
| `oracle` | ground-truth `q` | upper bound of the ROM force path |

Split = every 3rd contact held out (Test B): 1200 train / 600 test views. Force then
follows from `ReducedModel.estimate_force(q̂, contact)` with the known contact of each
sample.

**Held-out contact results** (`results/vision_model/<timestamp>/`)

| model | ‖q̂−q‖/‖q‖ mean / median | force rel. mean / median | notes |
|------:|------------------------:|-------------------------:|-------|
| geometric | 45.8 / 15.9 % | 14.0 / **4.6** % | noise-sensitive mean; median already under the soft 10 % Level-3 target |
| ridge | 19.1 / 7.9 % | 13.7 / **6.0** % | more robust q; similar force |
| cnn | 115 / 31.6 % | 115 / 31.5 % | image-only; random backgrounds + 0.5 px-equivalent visual noise make this hard at 192×144 |
| oracle | 0 | ~0.003 % | ROM force path itself is essentially exact |

`force_error_vs_load.png` shows the M5 noise floor again: relative force error blows up
at small loads (marker motion ≪ 0.5 px) and drops toward a few percent above ~1 N. Mean
errors are dominated by that low-force tail; medians are the fairer headline for Level 3.

**Visual check.** `inspect_vision_model.py` puts Φq̂ in `displacement_alt` (toggle `A`),
draws GT and estimated force arrows, and keeps the observation image in the inset.
`D` cycles estimators, `Y` switches noisy/clean markers for the marker models.

Consequence for Milestone 7: the mechanics-informed marker path already closes
image-features → q → force under known contact.

## End-to-end pipeline (Milestone 7)

```bash
python scripts/run_e2e_eval.py --estimator ridge     # Tests A–D + unknown contact
python scripts/run_l4_eval.py                        # Level 4: old vs new search, ridge vs oracle q
python scripts/run_q_improve_eval.py                 # Level 4: better marker→q (ridge / geometric / iterative / robust)
python scripts/run_snr_sweep.py                      # Level 4: n_views × noise × resolution (marker re-projection)
python scripts/run_temporal_eval.py                  # Level 4: consecutive frames → Kalman / EMA / mean
python scripts/inspect_e2e.py --estimator ridge      # full chain in the viewer (U = unknown contact)
```

Packages the research path into ``EndToEndPipeline`` (`src/evaluation/e2e.py`):

    observation (markers / image) → q̂ → û = Φ q̂ → ROM force (± unknown contact)

**Known contact (Level 3, ridge, held-out A+B pool).** Median relative force error
**5.0 %**, median q error 6.8 %, median inference **0.3 ms**. Breakdown:

| test | what is unseen | n | force rel. median |
|------|----------------|--:|------------------:|
| A | force magnitude (> 2 N), seen contact | 400 | 4.0 % |
| B | contact location | 600 | 6.0 % |
| C | +0 / 0.5 / 1 / 2 px marker noise | 1000 | 5.0 / 5.5 / 7.6 / 13.1 % |
| D | camera pose jitter (geometric) | subset | 3.6 % → 2.3 % (3-trial average) |

**Unknown contact (Level 4).** Residual search on the contact face
(`UnknownContactLocalizer`, 2100 candidates) plus leading-mode residual and
barycentric refine. Same held-out pool (ridge ``q̂``, n=1000):

| setup | contact median | force rel. median | F > 1 N contact / force |
|-------|---------------:|------------------:|-------------------------|
| known contact (ridge) | 0.0 mm | **5.0 %** | — / 4.2 % |
| unknown, old search + ridge ``q̂`` | 14.9 mm | 30.6 % | 14.0 mm / 26.9 % |
| unknown, new search + ridge ``q̂`` | 14.9 mm | 30.6 % | 14.0 mm / 27.1 % |
| unknown, old search + oracle ``q`` | 0.36 mm | 0.6 % | 0.36 mm / 0.7 % |
| unknown, new search + oracle ``q`` | **0.25 mm** | **0.4 %** | 0.25 mm / 0.4 % |

The search is not the bottleneck: given exact ``q`` it localizes to a fraction
of a millimetre. The ~15 mm / ~31 % gap is noisy ``q̂`` from 0.5 px markers.
`python scripts/run_l4_eval.py` writes `l4_compare.json` and figures.

**Better marker → ``q`` (same search, same pool).** Iterative Gauss–Newton + Huber
barely moves one-shot geometric. Averaging the two rendered views of each FEM
sample (`robust`) is the real gain. Low-``‖q̂‖`` reject drops another ~1.4 mm
and reports coverage (880 / 1000). `run_q_improve_eval.py` → `q_improve.json`.

| estimator | q rel. median | known-contact force | unknown contact / force | F > 1 N contact / force |
|-----------|--------------:|--------------------:|------------------------:|-------------------------|
| ridge | 6.8 % | 5.0 % | 14.9 mm / 30.6 % | 14.0 mm / 27.1 % |
| geometric | 11.0 % | 3.5 % | 15.0 mm / 33.7 % | 13.1 mm / 27.4 % |
| iterative | 11.0 % | 3.2 % | 15.2 mm / 32.8 % | 12.7 mm / 26.9 % |
| **robust** | 7.6 % | **2.2 %** | **11.6 mm / 25.2 %** | **9.6 mm / 21.0 %** |
| robust + reject (88 %) | 6.4 % | 1.8 % | 10.1 mm / 21.3 % | 9.4 mm / 20.5 % |
| oracle ``q`` | 0 | ~0 | **0.25 mm / 0.4 %** | 0.25 mm / 0.4 % |

Visual check: `inspect_e2e.py` shows the observation inset, Φq̂ vs FEM (`A`), GT/est
forces, and localized contact when `U` toggles unknown mode. `D` cycles
ridge / geometric / iterative / robust / cnn / oracle.

This closes the synthetic PoC through Level 3 and a measured Level 4
(search is solved; two-view marker ``q`` cuts the gap but does not close it).

**When does Level 4 close?** Marker re-projection sweep (`run_snr_sweep.py`,
250 held-out FEM samples, no image regen). Current dataset = 2 views, 0.5 px,
192×144 → **12.2 mm / 24 %**. Search ceiling on the same subset is 0.25 mm.

| setting | contact / force (median) |
|---------|-------------------------:|
| now: 2 views · 0.5 px · 192×144 | 12.2 mm / 24 % |
| 8 views · 0.5 px · 192×144 | 7.1 mm / 16 % |
| 8 views · 0.2 px · 192×144 | 2.7 mm / 6.0 % |
| 4 views · 0.5 px · 768×576 | 2.4 mm / 5.4 % |
| **8 views · 0.5 px · 768×576** | **1.7 mm / 3.3 %** |
| oracle ``q`` | 0.25 mm / 0.4 % |

Cameras alone at the current resolution do not close it. Either sub-pixel
(~0.2 px) tracking or ~4× linear resolution with 4–8 views does. Plots:
`snr_vs_views.png`, `snr_vs_res.png`, `snr_heatmap.png`.

**Temporal filter (same 2 views · 0.5 px · 192×144).** Independent marker
noise on a held contact, then Kalman / running mean / EMA on ``q``
(`run_temporal_eval.py`, 200 samples). One frame is still ~13 mm. After
**32 frames** Kalman = running mean → **2.45 mm / 4.8 %**. EMA (α=0.25)
floors near 5 mm because it forgets. On a slow squeeze Kalman tracks force
better than a full-history mean (15 % vs 35 % at frame 32) but contact is
still a few millimetres. So a ~1 s hold at 30 fps is almost as good as
raising resolution; it is not free if the contact is moving.

Remaining: image-only vision, real cameras built to that budget.

## Grasp stages A / B / C / D / E

```bash
python scripts/run_grasp_sim.py                  # interactive Stage A (GT force)
python scripts/run_grasp_sim.py --mode markers   # Stage B (marker → ROM force)
python scripts/run_grasp_eval.py                 # A vs B batch + figures
python scripts/run_arm_grasp.py                  # Stage C: 7-DoF arm + IK (interactive)
python scripts/run_arm_grasp.py --mode world     # Stage D: world-cam markers find contact + force (+ inset)
python scripts/run_arm_grasp.py --mode nn        # Stage E: same camera, NO markers – ResNet-18 → q (+ camera inset)
python scripts/run_arm_grasp.py --headless        # screenshots + arm_grasp.png
python scripts/run_arm_grasp.py --mode nn --video # → results/videos/nn/viewer.mp4 + nn_input.mp4 (10 s, 1280×720, real time)
# Stage E needs its checkpoint once (≈1 min data + ≈10 min training on an M-series GPU):
python scripts/generate_grasp_nn_dataset.py      # configs/grasp_nn.yaml → data/processed/grasp_nn_top.npz
python scripts/train_image_to_q.py --config grasp_nn   # → results/checkpoints/grasp_nn/resnet18_q_sim_raw.pt
```

A **fixed parallel-jaw gripper** (two copies of the compliant finger) squeezes
a rigid box and lifts it if Coulomb friction can carry the
weight: ``2 μ λ ≥ m g``. Finger compliance comes from the linear FEM; closing
the jaws by ``w − g`` is shared as indentation ``δ = (w − g)/2``, so
``λ = k δ`` with ``k`` measured from a 1 N inner-face solve.

| stage | measured force used by the controller | result (default config) |
|-------|----------------------------------------|-------------------------|
| A | true ``λ`` | tracks 2.2 N, lifts (hold threshold ≈ 1.31 N) |
| B | markers → ``q`` → Kalman → contact search → ROM ``λ̂`` (0.25 px, 480×360) | same lift |
| C | A or B on a 7-DoF arm | IK to grasp, squeeze to 2.2 N, IK lift (object at ~39 mm) |
| D | world camera → ``q`` → Kalman; pre-lock force at config contact, then locked location + ``λ̂`` | same lift; inset shows desk camera + live force |
| E | **marker-free** world-camera RGB (240×180) → ResNet-18 → ``q`` → same Kalman / lock / ROM ``λ̂`` | same lift; raw ``|λ̂−λ|`` median 0.20 N (7.3 %) vs 0.56 N (19.5 %) for D, contact 4.0 mm vs 14.7 mm (D's lock lands on a wrong candidate, cf. Fig. 5) |

Interactive keys: ``4`` / ``5`` switch A/B, ``[`` ``]`` change the target,
``9`` lift now, ``0`` reset, Space pause. Results:
``results/grasp/<timestamp>/`` (`grasp_ab.png`, `gt_final.png`, `markers_final.png`).

**Stage C** mounts the same jaws on a desk-scale 7-DoF serial arm (base yaw,
shoulder pitch, upper-arm roll, elbow pitch, wrist roll/pitch/yaw). Numerical
damped least-squares IK tracks a pregrasp, the grasp pose (contact at ``0.85 L``
on the object ±y faces), then a vertical lift. Jaw squeeze is unchanged A/B
force control; the object rides ``T_ee`` while held. Interactive keys match A/B
(``4`` / ``5`` / ``6`` / ``9`` / ``0`` / ``[`` ``]``). Headless output:
``results/grasp/<timestamp>/`` (`arm_grasp.png`, `start_*.png`, `end_*.png`).

**Stage D** keeps the arm but moves the observation camera into the world.
Virtual markers on the left finger are transformed by the known flange pose
``T_ee``, projected through that camera (optional pixel noise), and inverted
to ``q``. Until residual search *locks* a contact, force for control uses the
ROM at the config contact (avoids a blind PI); after lock, contact comes from
``q`` and ``λ̂`` follows. Default demo SNR is 0.25 px at 480×360.
The viewer draws the world-camera frustum, estimated contact, and a
marker-projection inset (what that camera measures).

**Stage E (`--mode nn`, key `7`)** solves the *same* task as D with the *same*
world camera but **without markers** – the second of the two ways to get
``q`` from that camera (D: marker pixels → Jacobian LS; E: RGB → ResNet-18).
Every third control step (≈ 33 Hz) the grasp scene (both fingers on ``T_ee``,
object, table) is rendered marker-free at 240×180 through the world-camera
pose (`src/grasp/nn_vision.py`, in a child process because the viewer owns the
only GGUI window), the network predicts ``q`` in the grasp estimator's POD
basis, and everything after ``q`` (Kalman, pre-lock force, contact vote/lock,
ROM ``λ̂``) is the Stage D code unchanged. The inset shows the actual frame
the network sees above the live GT-vs-estimate strip. The network is trained
on that scene (`configs/grasp_nn.yaml`; commands in the block above).

Data: 91 inner-face contacts × 10 forces (0.15–3.5 N) × 2 flange/light
jitters (±4 mm, up to +40 mm lift, ±2°) + 300 zero-load frames along
pregrasp → grasp → lift; the box sits at the sampled contact. Labels are
``q_sim = Φ_rᵀu`` in the estimator's inner-face POD basis (a fingerprint of
``Φ`` is stored in the checkpoint and checked at load) plus the Stage D
marker ``q`` as teacher reference. Held out (31 unseen contacts, 732 frames):
NN ``q`` relative error median **0.048** vs 0.27 for the marker path at this
camera. Closed loop (default config, headless): lift succeeds, contact error
4.0 mm, raw force error median 0.20 N (7.3 %) – Stage D on the same run:
14.7 mm, 0.56 N (19.5 %) (the marker lock lands on a wrong inner-face
candidate; single-frame marker localisation at this camera is 3–48 mm noisy,
cf. Fig. 5). Caveat as in the MF section: the visible object is a location
cue, and the network is trained on renders of this exact scene.

### Controls

| Input | Action |
|-------|--------|
| LMB drag | orbit |
| RMB drag / Shift+LMB | pan |
| MMB drag / Ctrl+LMB, `-` `=` , `Z` `X` | zoom |
| `I` `J` `K` `L` | orbit with keys |
| `1` / `2` / `3`, `Tab` | original / deformed / overlay mode |
| `Up` / `Down` | deformation amplification α (visual only: `x_display = x + α u`) |
| `W` `T` `N` | surface wireframe / element (corner) edges / FEM nodes |
| `B` `C` | fixed boundary nodes / contact markers |
| `G` `E` `F` | ground-truth force / estimated force / both |
| `O` `V` `;` | objects / observation camera / surface markers |
| `D` / `Y` / `U` | (vision/E2E inspectors) cycle estimator / noisy↔clean markers / known↔unknown contact |
| `4` / `5` / `6` / `7` / `9` / `0` | (grasp) Stage A / B / D / E (marker-free NN) / lift now / reset |
| `M` | cycle color mode (displacement magnitude, scalar field, solid) |
| `A` | toggle alternative displacement field (e.g. ROM reconstruction) |
| `P` `R` `S` `H` `Esc` | GUI panel / reset camera / screenshot / help / quit |

The GUI panel (top-left) mirrors all toggles and prints max |u| (mm), F_true,
F_est, relative force error and the arrow scale.

## Marker-free visual state estimation (markers as privileged information)

```bash
python scripts/generate_markerfree_dataset.py                 # paired marker-on / marker-off renders (configs/markerfree.yaml)
python scripts/train_image_to_q.py --target sim               # marker-free RGB → ResNet-18 → q  (supervised by q_sim)
python scripts/train_image_to_q.py --target marker            # same, supervised by the marker-pipeline q (teacher)
python scripts/train_image_to_q.py --target marker --mode-weight snr   # teacher, noise-dominated modes down-weighted
python scripts/train_image_to_q.py --input marker             # sanity: network sees the marker-on image
python scripts/eval_image_to_q.py                             # sim q / marker q / network q through the same ROM
python scripts/generate_markerfree_dataset.py --no-probe --output data/processed/markerfree_top_noprobe.npz   # ablation
```

**Motivation.** Everything above needs printed markers on the finger and a
calibrated marker → `q` pipeline at run time. Markers are cheap in simulation
but a nuisance on hardware (wear, occlusion, lighting, printing tolerances).
The question here is whether a network can replace the *marker measurement*
only – it predicts the same reduced state `q ∈ R^8` – while the interpretable
mechanics (`ReducedModel`, `UnknownContactLocalizer`) stays untouched. No
image → force black-box regressor is trained.

**Original pipeline.** `image (markers) → marker pixels → Jacobian LS → q → ROM → contact location, force`.

**Markers as a teacher.** Markers are *training-time privileged information*:

```
training     marker-on image  ─→ existing marker pipeline ─→ q_marker ┐
             FEM state        ─→ Φ_rᵀ u                   ─→ q_sim    ├─ target  L_q = MSE((q_pred − μ)/σ, (q_target − μ)/σ)
             marker-free image ─→ ResNet-18               ─→ q_pred   ┘
deployment   marker-free image ─→ ResNet-18 ─→ q_pred ─→ existing ROM ─→ contact location + force
```

Both images of a pair are rendered from the **identical** state, camera pose,
light, background and image-noise realisation (`SyntheticCameraRenderer.render`
with / without `markers_xyz`; only the marker discs differ, ≈0.5 % of the
pixels). `q_sim` (exact) and `q_marker` (teacher; iterative Gauss–Newton +
Huber on 0.5 px-noisy pixels) are stored separately and either can be the
target (`--target sim|marker`). Per-mode normalisation statistics `(μ, σ)`
are saved with the checkpoint (`*.pt`, `*.norm.json`). `--mode-weight snr`
multiplies each mode's normalised squared error by
`w_k = clip(var(q_sim,k) / var(q_marker,k), 0, 1)` – the fraction of the
teacher's variance that is signal (here `[1.0, 0.33, 0.20, 0.01, 0, 0, 0, 0]`).

**Data / model.** `data/processed/markerfree_top.npz`: 900 FEM samples × 2
views = 1800 pairs, 192×144 RGB, `q ∈ R^8`, same nuisance jitter as
Milestone 5 (`PairedVisionDataset`, `src/vision/paired_dataset.py`).
ResNet-18 (pure PyTorch, 11.2 M parameters, global-average-pool + linear head,
`src/vision/resnet_q.py`), AdamW + one-cycle LR, brightness/contrast jitter and
±4 px shifts, early stopping on held-out contacts (every 3rd contact,
1200 train / 600 test). ≈4 min on an Apple-silicon GPU; 1.5 ms / image inference.

**Evaluation – three levels through the *same* physics** (`results/markerfree/eval_*/`,
600 held-out views, medians):

| q source | A ‖q̂−q_sim‖/‖q_sim‖ | B force rel. err (known contact) | C contact err [mm] (unknown) | C force rel. err (unknown) |
|---|--:|--:|--:|--:|
| `q_sim` (ROM ceiling) | 0 | ~0 | 0.17 | 0.004 |
| `q_marker` (existing marker pipeline) | 0.125 | **0.037** | 18.6 | 0.40 |
| ResNet-18, raw image → `q_sim` target | 0.106 | 0.105 | **1.22** | **0.11** |
| ResNet-18, raw image → `q_marker` target | 0.210 | 0.209 | 5.2 | 0.25 |
| ResNet-18, raw image → `q_marker` target, SNR mode weights | 0.153 | 0.150 | 6.7 | 0.20 |
| ResNet-18, *marker-on* image → `q_sim` (sanity) | 0.084 | 0.084 | 1.16 | 0.085 |
| ResNet-18, raw image → `q_sim`, **no probe sphere** (ablation) | 0.130 | 0.125 | 7.8 | 0.22 |

Per-mode normalised RMSE explains the split result: the marker pipeline is
almost exact on the leading mode (0.06) – which carries the force at a known
contact – but pure noise on modes 4–8 (10–85), which the residual search needs
to *locate* the contact. The network is uniform across modes (0.16–0.31), so it
loses on known-contact force (10 % vs 4 %) and wins clearly on unknown-contact
localisation (1.2 mm vs 19 mm) and on force once the contact must also be
found (11 % vs 40 %). Above ≈1.5 N the marker-free force error is 5–6 %.

**Findings / contributions**

* A marker-free image can replace the marker measurement at the *state*
  level: the identical ROM turns `q_pred` into contact location and force with
  no retraining of the mechanics.
* Supervising with the exact `q_sim` is better than supervising with the marker
  teacher: the teacher's high modes are noise at 0.5 px, and per-mode
  normalisation up-weights exactly those modes (`raw → q_marker` early-stops at
  epoch 20 with nRMSE 0.91, 5 mm localisation). Down-weighting the modes by
  the teacher's SNR (`--mode-weight snr`) recovers most of the gap in `q`
  (0.153 vs 0.106) while still never touching
  `q_sim` as a *label* – only its per-mode spread. With marker-only labels a
  noise model of the marker pipeline (LS covariance) would provide the same
  weights.
* Seeing the markers helps the network only marginally (0.084 vs 0.106), i.e.
  it reads the finger silhouette / shading, not the dots.
* The visible contacting object matters: without the probe sphere the
  localisation error grows from 1.2 mm to 7.6 mm (still 2.6× better than the
  marker path). In a grasp the object *is* visible, but this is a cue to keep
  in mind when reading the numbers.

**Limitations.** Simulation only (no sim-to-real gap, no real camera, no
texture / specular variation, fixed finger material); one camera pose family;
contacts restricted to the top face; the marker teacher is the single
geometric estimator (no multi-view fusion); ResNet-18 from scratch on 1200
images (no pretraining); no temporal filtering in this offline study. The
closed-loop counterpart (Stage E, `run_arm_grasp.py --mode nn`) is described
in the grasp section above.

## Material generalisation (Stage F, paper Fig. 5)

```bash
python scripts/run_material_sweep.py                  # configs/material_sweep.yaml → results/figure5/
python scripts/run_material_sweep.py --quick          # smoke test (no closed loop)
python scripts/run_material_sweep.py --no-nn --no-closed-loop
```

The vision stage returns *geometry*: the reduced deformation `q = Φᵀu`. For a
linear, homogeneous, isotropic finger the material enters only through the
stiffness `K(E, ν)`, so switching material means rebuilding `K` and
`K_r = ΦᵀKΦ` (plus the localiser influence fields `K_r⁻¹ B_r`) – nothing in
front of `q` changes. The sweep therefore **freezes Φ (reference material),
the marker geometry / image Jacobian and the ResNet checkpoint**, changes
(E, ν) in the FEM, and pushes the same `q` through (a) the updated ROM and
(b) the stale reference ROM. `estimator.pod_reference_material` in
`configs/grasp.yaml`-style configs pins Φ to a reference (E, ν) so that a
run-time material change keeps the `q` convention (and the NN checkpoint
fingerprint) valid.

Offline (`src/evaluation/material_sweep.py`; 18 held-out inner-face contacts ×
6 forces 0.3–3 N; medians; marker `q` averaged over 30 noisy frames ≈ the
Kalman hold in the demo):

| material | k [N/m] | Φ proj. resid. | oracle `q`: force known / stale-K / contact | markers → `q`: force / contact / force (unknown c) | marker-free NN → `q`: force / contact / force (unknown c) |
|---|--:|--:|---|---|---|
| E ×0.25 | 387 | 0.06 % | 0.1 % / **300 %** / 0.7 mm | 1.5 % / 3.0 mm / 5.0 % | 13.2 % / 1.4 mm / 13.1 % |
| E ×0.5 | 774 | 0.06 % | 0.1 % / **100 %** / 0.7 mm | 1.3 % / 4.1 mm / 7.0 % | 4.1 % / 0.8 mm / 5.0 % |
| E ×1 (ref) | 1548 | 0.06 % | 0.1 % / 0 % / 0.7 mm | 1.5 % / 6.5 mm / 13.4 % | 6.6 % / 0.7 mm / 8.8 % |
| E ×2 | 3095 | 0.06 % | 0.1 % / **50 %** / 0.7 mm | 2.6 % / 13.5 mm / 25 % | 16.1 % / 0.7 mm / 17.6 % |
| E ×4 | 6191 | 0.06 % | 0.1 % / **75 %** / 0.7 mm | 6.4 % / 23.9 mm / 42 % | 51.8 % / 0.6 mm / 51 % |
| ν 0.30 | 1536 | 0.11 % | 4.1 % / 0.9 % / 0.5 mm | 4.6 % / 6.4 mm / 10.4 % | 12.6 % / 0.8 mm / 14.3 % |
| ν 0.35 | 1540 | 0.08 % | 1.4 % / 0.5 % / 0.6 mm | 2.0 % / 5.5 mm / 8.4 % | 8.2 % / 0.8 mm / 9.1 % |
| ν 0.45 | 1559 | 0.08 % | 3.9 % / 0.6 % / 0.7 mm | 4.2 % / 5.9 mm / 11.4 % | 11.0 % / 0.6 mm / 12.9 % |

* E-only changes are exact: the field just scales, Φ spans it as well as at
  E₀, and the stale ROM is wrong by exactly `|E₀/E − 1|`; the updated ROM is
  back to reference accuracy for all three `q` sources.
* ν changes alter the *shape* of `u` only weakly on the hex-dominant mesh: the
  reference Φ still captures it to ≈0.1 % and the stiffness of the finger barely
  moves (1536–1559 N/m), so here the *stale* `K_r` is even slightly closer than
  the rebuilt one at low ROM rank (0.5–0.9 % vs 1.4–4.1 % force error) – the
  ν effect is at the level of the r = 8 truncation error.
* The marker-free NN localises to ≤ 1.4 mm for every material; the marker path
  degrades with stiffness (less deformation per pixel noise). The NN was
  trained at E₀ only – its *force* error grows away from E₀ (13 % at ×0.25,
  52 % at ×4) because the deformations leave its training range, while the
  contact location stays sub-millimetre.

Closed loop (`build_arm_grasp_sim` with the new E, Φ pinned to E₀, deformation
gates scaled by E₀/E; headless, 10 s cap): **8 / 8 lifts succeed**.

| E / E₀ | markers (D): contact / force err | marker-free NN (E): contact / force err |
|---|---|---|
| 0.5 | 22.1 mm / 59 % | 3.8 mm / 8.1 % |
| 1 | 14.7 mm / 19.5 % | 4.0 mm / 7.3 % |
| 2 | 14.6 mm / 32 % | 1.4 mm / 6.2 % |
| 4 | 14.6 mm / 36 % | 0.9 mm / 9.9 % |

The marker path's lock lands on a ~15 mm-off inner-face candidate for every E
(its single-frame localisation at this camera is 3–48 mm noisy, cf. Fig. 3; on
the former T4 mesh the reference run happened to lock on the right one); the
marker-free path is stable.
Figures: `results/figure5/material_sweep.png`, `closed_loop.png`, numbers in
`metrics.json`.

## Architecture

```
configs/            fem.yaml, cantilever_benchmark.yaml, cantilever_lateral.yaml, viewer.yaml, dataset.yaml, vision.yaml, vision_model.yaml, e2e.yaml, grasp.yaml, markerfree.yaml, grasp_nn.yaml, material_sweep.yaml
src/geometry/       finger.py (FingerGeometry, FEMesh = H20 + T10 blocks, hex-dominant mesher, ties),
                    mesh_utils.py (surface extraction, orientation, edges), primitives.py
src/visualization/  state.py        VisualizationState – the physics ↔ viewer interface
                    camera.py       ObservationCamera (research) / OrbitCamera (developer)
                    scene.py        ViewOptions + SceneRenderer (state+options → draw calls)
                    mesh_renderer.py, force_renderer.py, colormap.py   Taichi buffers
                    taichi_viewer.py  window, input, GUI, head-less rendering
src/fem/            material.py (isotropic C, ρ), elements.py (T10 / H20 shape functions, Gauss K_e), tetra_element.py (legacy T4), assembly.py (sparse K, mixed blocks),
                    boundary.py (DofPartition, clamps), loads.py, solver.py (LinearFEM, FEMResult, body_force_vector),
                    stress.py, beam_theory.py, finger_model.py (config-driven FingerFEMModel, gravity_displacement(R)),
                    contact_dataset.py (ContactDataset npz)
src/contact/        contact_mapping.py  PointContactMapping -> B(c), B_xyz(c) (barycentric point load)
                    observation.py      y = H u (DOF selection: full / surface / face[:xyz])
                    inverse_force.py    InverseForceSolver (force- & displacement-space LS), noise, metrics
src/rom/            pod.py              PODBasis (SVD, project/reconstruct, energy, save/load)
                    reduced_mechanics.py ReducedModel (K_r, B_r, reduced forward & inverse)
src/rendering/      markers.py          SurfaceMarkers (barycentric keypoints on boundary faces), marker grids
                    synthetic_camera.py SyntheticCameraRenderer (hidden GGUI window = observation camera), RenderAppearance
src/vision/         dataset.py          VisionDataset npz (images, marker px/visibility, q, force, contact, K, T_cw)
                    features.py         marker Δuv features, image Jacobian ∂(uv)/∂q
                    marker_model.py     GeometricMarkerModel, RidgeMarkerModel
                    temporal.py         Kalman / EMA / running mean on q
                    image_model.py      ImageCNNModel (optional torch)
                    paired_dataset.py   PairedVisionDataset (marker-on / marker-off pairs, q_sim + q_marker)
                    resnet_q.py         ResNet-18 image → q, QNormalizer (optional torch)
                    pipeline.py         VisionMechanicsPipeline (q → Φq → ROM force)
                    splits.py           contact-wise / force-wise hold-outs
src/evaluation/     e2e.py              EndToEndPipeline (obs → q → Φq → force)
                    unknown_contact.py  residual search over surface candidates (Level 4)
                    snr_sweep.py        marker re-projection SNR / view-count grid
                    material_sweep.py   fixed Φ, swapped K(E, ν): oracle / marker / nn q → force, contact (Fig. 5)
                    metrics.py          force / timing summaries
src/grasp/          mechanics.py        two-finger compliance + Coulomb lift + self-weight superposition
                    estimator.py        gt / inverse / finger-cam / world-cam / marker-free nn force (sag compensation)
                    nn_vision.py        grasp-scene composition, marker-free renderer (child process), image → q
                    sim.py              approach → squeeze → lift
                    world.py            parallel-jaw placement (optional ``T_ee``)
                    arm.py              7-DoF FK / Jacobian / damped-LS IK
                    arm_sim.py          home → reach → squeeze → IK lift
                    scene.py            concatenated two-finger VisualizationState
scripts/            run_viewer_demo.py, run_cantilever_verification.py, run_beam_theory_comparison.py, run_3d_fem_demo.py,
                    generate_fem_dataset.py, inspect_fem_dataset.py, run_inverse_force_eval.py,
                    run_inverse_force_robustness.py, inspect_inverse_force.py, run_rom_analysis.py, inspect_rom.py,
                    run_synthetic_camera_check.py, generate_vision_dataset.py, inspect_vision_dataset.py,
                    train_vision_model.py, inspect_vision_model.py, run_e2e_eval.py, inspect_e2e.py,
                    run_l4_eval.py, run_q_improve_eval.py, run_snr_sweep.py, run_temporal_eval.py,
                    run_grasp_sim.py, run_grasp_eval.py, run_arm_grasp.py,
                    generate_markerfree_dataset.py, train_image_to_q.py, eval_image_to_q.py, generate_grasp_nn_dataset.py,
                    export_figure1_assets.py, run_material_sweep.py
tests/              geometry, state/camera, viewer smoke, element, solver, beam theory, contact, model, dataset, inverse force, rom,
                    rendering, vision model, e2e (localizer, packaging, Tests A–D), grasp A/B/C/D, marker-free (pairs, ResNet q),
                    grasp nn (scene compose, remote renderer, nn mode in the arm loop), material sweep (fixed Φ / swapped K, pinned POD),
                    self-weight (body force = weight, sag vs beam theory, frame rotation, superposition, estimator compensation)
```

Solvers never import Taichi; the viewer never imports solver code. Everything
meets at `VisualizationState` (nodes, faces, tets, `u`, fixed nodes, contacts,
forces, objects, observation camera, surface markers, optional scalar field, info text).

## Conventions

* SI units internally (m, N, Pa). GUI text shows mm and N.
* `x` = finger length (root at `x = 0` is clamped), `y` = width, `z` = thickness/up.
* Tetrahedra are stored positively oriented; surface triangles are CCW seen from outside.
* Nodal DOF ordering: `3*i + {0,1,2}` for `(u_x, u_y, u_z)` of node `i`.
* Forces in `VisualizationState` act **on the finger** at their origin; contact arrows
  are drawn with their tip at the contact point (`force_display.anchor: tip`).
* Deformation amplification and arrow scaling are visualization-only parameters
  living in `ViewOptions` / `viewer.yaml`, never in physics code.
* Camera frame is OpenCV-style (`x` right, `y` down, `z` forward); image coordinates
  `(u, v)` have their origin at the top-left corner, pixel `(i, j)` covers
  `u ∈ [i, i+1)`, `v ∈ [j, j+1)` (centre at `i + 0.5, j + 0.5`). Images are stored
  `(H, W, C)` with row 0 at the top.

## Tests

```bash
python -m pytest
```

## Paper summaries

### English (for abstract / intro draft)

We estimate contact force on a compliant robotic finger from camera observations
by coupling vision with an interpretable linear FEM / reduced-order (POD)
mechanics model, rather than regressing force from images with a black-box
network. Surface markers are projected into a calibrated camera; marker motion
is mapped to low-dimensional deformation coordinates ``q`` (optionally
Kalman-filtered), and a Galerkin ROM recovers force. When contact is unknown,
residual search on the grasp face localizes it from the same ``q``. On synthetic
benchmarks with known contact, marker-based ``q`` yields median relative force
errors of a few percent. With unknown contact, the search is millimetre-accurate
given oracle ``q``, but noisy single-frame ``q`` is the bottleneck (~12 mm median
at 2 views, 0.5 px, 192×144); more views, higher resolution, or a short temporal
hold on ``q`` close much of that gap. The closed-loop demo uses a **single
world-fixed camera** on a 7-DoF arm grasp: until localization locks, force for
control uses the ROM at a nominal contact; after lock, contact and force both
come from vision→``q``. At 0.25 px and 480×360 this yields a successful
Coulomb lift (contact 14.7 mm / force 19.5 % on the current mesh – the single
frame marker lock is 3–48 mm noisy at this camera). The same demo also runs
**marker-free** (`--mode nn`): a ResNet-18 trained with markers as privileged
information maps the world-camera RGB frame directly to ``q``, and the
unchanged ROM path recovers contact (4.0 mm) and force (7.3 % median) with
the same lift. Because
the learned stage outputs geometry, not force, a material change (E ×0.25–×4,
ν 0.30–0.45) requires only rebuilding the stiffness ``K_r = ΦᵀKΦ``: with Φ,
markers and network frozen, the marker-free path keeps ≤ 1.4 mm contact error
and lifts at every stiffness, whereas an un-updated model is wrong by exactly
``|E₀/E − 1|``. This synthetic study supports mechanics-informed
vision→``q``→force; real cameras and sim-to-real transfer remain future work.

### 한글 (논문 초고용 · 쉬운 말)

부드러운 로봇 손가락이 물체를 집을 때, “얼마나 세게 누르고 있는지”를
카메라로 알아내고 싶다. 흔한 방법은 사진만 보고 AI가 힘을 바로 찍어내는
것인데, 그렇게 하면 왜 그 힘이 나왔는지 설명하기 어렵다.

우리는 다른 길을 간다. 손가락 위에 작은 점(마커)을 두고, 카메라로 그 점이
어떻게 움직이는지 본다. 그 움직임으로 손가락이 얼마나 휘었는지를 숫자
몇 개(``q``)로 요약하고(필요하면 칼만 필터로 다듬고), 미리 만들어 둔
물리 모델(유한요소/축소 모델)에 넣어서 접촉힘을 계산한다. 즉
**영상 → 변형 → 물리 → 힘** 순서다. 어디를 누르는지 모를 때는 같은 ``q``로
접촉면을 잔차 탐색해 위치도 찾는다.

합성 데이터 벤치마크에서는, 위치를 알려 줬을 때 힘은 대체로 몇 퍼센트
안쪽으로 맞는다. 위치를 모를 때도 찾는 방법은 있지만, ``q``가 틀리면 위치도
틀린다. 카메라 **2대**·192×144·노이즈 0.5 px 조건에서는 위치가 약 1 cm
넘게 어긋날 수 있고, 시야를 늘리거나 해상도를 키우거나 잠시 ``q``를 모으면
좋아진다.

메인 데모(``python scripts/run_arm_grasp.py --mode world``)는 책상 위
**관측 카메라 1대**로 집고 들어 올린다. 접촉이 잠기기 전에는 힘만
(설정의 접촉점으로 ROM) 쓰고, 잠긴 뒤에는 카메라 ``q``로 찾은 접촉
위치와 힘으로 제어한다. 데모 설정(0.25 px, 480×360)에서는 접촉 오차가
대략 2 mm 수준이고 리프트에 성공한다. 위 2대·데이터셋 숫자는 별도
벤치마크다.

같은 데모를 **마커 없이**도 돌릴 수 있다(``--mode nn``). 학습할 때만 마커를
"정답 힌트"로 쓰고, 실제로는 같은 책상 카메라의 RGB 사진을 ResNet-18에
넣어 변형 ``q``를 바로 읽는다. 그 뒤(칼만·접촉 탐색·물리 모델로 힘)는 마커
방식과 완전히 같은 코드다. 기본 설정에서 접촉 오차 약 4.0 mm, 힘 오차
중앙값 약 7.3 %로 리프트에 성공한다(마커 방식: 14.7 mm, 19.5 % — 이 카메라에서
단일 프레임 마커 위치 추정이 3–48 mm 수준으로 흔들려 잘못된 후보에 고정됨).

신경망이 힘이 아니라 **변형(기하)** 을 내놓기 때문에, 손가락 재질이 바뀌어도
(탄성계수 E ×0.25–×4, 푸아송비 0.30–0.45) 학습을 다시 하지 않고 물리 모델의
강성 행렬 ``K``만 새로 만들면 된다. 실제로 마커 없는 경로는 모든 재질에서
접촉 오차 1.4 mm 이하로 들어 올리기에 성공했고, ``K``를 갱신하지 않으면
힘이 정확히 ``|E₀/E − 1|`` 만큼 틀린다(Fig. 5).

아직은 진짜 카메라·진짜 손가락이 아니라 합성 시뮬레이션 결과이고,
실로봇 실험(sim-to-real)은 다음 단계다.
