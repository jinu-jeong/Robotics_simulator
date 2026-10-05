results/
  summary.txt   paper storyline (ICLR target; ICML / ICRA variants) + figure → file map
  figure1/      paper Fig.1 concept panels (9 PNGs + README; schematic-first)
                01 = pipeline schematic; 02 = world grasp; 09 = information paths
  figure2/      paper Fig.2 closed-loop demos (world vs nn): stage screenshots + force plot
  figure5/      paper Fig.5 material generalisation: material_sweep.png, closed_loop.png, metrics.json
  etc/          all other milestone dumps, checkpoints, debug runs (Fig.3 / Fig.4 / appendix sources)
  checkpoints -> etc/checkpoints   (compat symlink)

Export Fig.1:  python scripts/export_figure1_assets.py
Fig.2 demos:   python scripts/run_arm_grasp.py --mode world|nn
Fig.5 sweep:   python scripts/run_material_sweep.py
