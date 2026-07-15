# RoboSim — Visual Force Sensing

Estimate gripper contact force from **finger bending observed in a palm-mounted camera**, and validate it against simulation ground truth.

Deformable Craig-Bampton fingers bend under grasp load. The sim logs tip deflection and elastic normal force directly from the mesh. The same run can render solid-red fingers from a fixed palm “antenna” camera; a CV estimator recovers the deflection profile \(u(s)\) and maps it back to force for comparison.

---

## Pipeline

```
grasp → deformable CB fingers bend
      → sim probe: δ, F_n, contact station s_c   (ground truth)
      → palm-cam frames (solid red)
      → CV: silhouette ridge → u(s) → F_n estimate
      → CSV + plots under runs/grasp_d<depth>/
```

| Stage | Module | Role |
|---|---|---|
| Sim probes | `robosim/scene/finger_probe.py` | Tip / station deflection, contact \(s_c\), elastic \(F_n\) from CB centerline |
| Capture | `robosim/viz/finger_capture.py` | Lock palm camera, save PNG frames, call CV |
| Vision | `robosim/viz/finger_cv.py` | Outer-silhouette ridge → clamped \(u(s)\) → tip / station metrics |
| Plotting | `robosim/viz/finger_force_plot.py` | Sim vs CV time series, deflection profiles, depth sweep |

---

## Quick start

```bash
conda create -n robosim python=3.11
conda activate robosim
pip install -e .
pip install taichi   # viewer + frame capture
```

**Deformable fingers + force log + palm-cam CV** (default capture when `--finger cb`):

```bash
python examples/pure_simulation/grasp_scene.py --object rigid --finger cb
```

**Grasp depth** places the box along the finger span (`0` = tip-only / near face, `1` = full-span / far face):

```bash
python examples/pure_simulation/grasp_scene.py --object rigid --finger cb --grasp-depth 0.10
python examples/pure_simulation/grasp_scene.py --object rigid --finger cb --grasp-depth 1.00
```

**Headless** still writes the sim force CSV/plot; CV frames need a GPU window (`--capture`):

```bash
python examples/pure_simulation/grasp_scene.py --object rigid --finger cb --headless --no-capture
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--finger cb` | Deformable CB fingers (enables visual sensing path) |
| `--grasp-depth` | Box placement along finger X ∈ `[0, 1]` |
| `--log PATH` | Override force CSV path |
| `--capture` / `--no-capture` | Force on/off palm-cam PNG + CV |
| `--capture-dir` / `--capture-every` | Frame directory and stride |

---

## Outputs

Each run writes under `runs/grasp_d<depth>/`:

```
runs/grasp_d0.55/
├── finger_force_log.csv      # sim δ / F_n (+ CV columns when capturing)
├── finger_force_log.png      # time-series plot
└── frames/                   # palm-cam PNGs + CV profile side files (if capture on)
```

CSV columns include sim and (when available) CV tip deflection [mm], normal force [N], and contact station \(s_c\).

---

## What the CV does

Palm frames are oriented so finger length runs along image **X** and left/right fingers stack in **Y**. For each finger ridge:

1. Track the **outer** silhouette along \(s\).
2. Remove rigid root motion: \(u(s) = y(s) - \mathrm{mean}(y_{\mathrm{root}})\).
3. Fit a cantilever point-load shape for contact station \(s_c\).
4. Map the centerline \(u(s)\) through the CB reduced stiffness to an elastic normal force.

Fingers are rendered with a palm→tip white–red gradient / solid-red capture mode so the silhouette is easy to segment.

---

## Layout (sensing-related)

```
robosim/scene/finger_probe.py      # sim deflection / force probes
robosim/viz/finger_capture.py      # palm cam + frame session
robosim/viz/finger_cv.py           # gradient / silhouette estimator
robosim/viz/finger_force_plot.py   # validation plots
examples/pure_simulation/grasp_scene.py
```

Underlying dynamics (RBD + CB soft fingers, contact, grasp trajectory) live in `robosim/physics/` and `robosim/scene/` — this branch’s docs focus on the sensing/validation loop above.

---

## License

MIT — see [`LICENSE`](LICENSE).
