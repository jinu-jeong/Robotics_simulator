Paper Figure 1 – 9 panels (schematic first)

Shared resolution: src/utils/figure_res.py
  viewer 12800×7200 · matplotlib dpi 1120
  regenerate all: python scripts/regenerate_paper_figures.py

01  pipeline schematic (method overview)           ← FIRST
02  world grasp scene (arm + gripper + object)
    early approach_ee, photo only
    regenerate: python scripts/render_figure1_grasp_scene.py
03  paired marker-on vs marker-free (|u| colormap)
04  finger close-up + F_true / F_est / self-weight
05  FEM mesh convergence vs beam theory
06  POD modes Φ
07  ROM vs full (tip |δ| vs rank; λ(x) at fixed tip δ*)
08  held-out: sim / marker / NN → same ROM
09  information paths — same state, 3 routes to q/λ
    A privileged uv | B marker RGB+LS | C marker-free ResNet

Dropped as redundant:
  marker_projection, force_recovery, self_weight alone,
  old pipeline_gt / pipeline_vision_q

Regenerate: python scripts/render_figure1_lateral.py
Export:     python scripts/export_figure1_assets.py
