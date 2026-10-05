Paper Figure 5 – material generalisation (same finger, different E / ν)

Frozen: POD basis Φ (reference material), marker geometry / image Jacobian, marker-free ResNet.
Rebuilt: FEM K → ROM K_r = ΦᵀKΦ and contact-localiser influence fields. No retraining.

material_sweep.png  offline: oracle / marker / nn q → force & contact, updated vs stale K
closed_loop.png     arm grasp with the material changed at run time (world vs nn)
metrics.json        all numbers (per-material medians, closed-loop rows)

Regenerate: python scripts/run_material_sweep.py
