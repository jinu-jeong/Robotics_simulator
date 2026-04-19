"""Material Point Method (MPM) solver.

Hybrid Eulerian–Lagrangian solver for large-deformation, inelastic, and
topology-changing continua (plasticity, fracture, granular).

Complements the FEM solver (small/medium elastic and plastic deformation with
fixed topology) — MPM owns anything with large plastic flow, tearing, or
granular behaviour.
"""
