# Changelog

## 1.0.0 — first public release
- `src/planner.py`: DirectCrystalFlow (model, data loader, three-stage training). Relative to the
  research version, only the command line changed: `--stored-basis-symmetry`, `--stage3-metrics`,
  `--save-config` (writes the resolved config to `run_config.json`). Defaults reproduce the original run.
- `src/symfix.py`: stored-basis symmetry operations (was `symfix__1_.py`).
- `scripts/evaluate.py` (evaluator v6): aligned with the composition planner of this model; CDVAE gate
  on the smact14 list; MLFF-energy hull; S.U.N. over the generated pool; unconditional baseline;
  planner ablation; confidence intervals. Environment overrides: `EMF_CHECKPOINT`, `MP20_ROOT`,
  `EMF_OUTPUT_PREFIX`, `EMF_NUM_GEN`, `EMF_SEED`, `EMF_SYMFIX`, `EMF_RUN_STABILITY`.
- `scripts/make_figures.py` (was `UMAP.py`), `scripts/sample.py`, `scripts/results_table.py`.
- `figures/architecture/`: source of the architecture figure.
