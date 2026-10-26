# Before making the repository public

## Fill in (search for `<` placeholders)
- [ ] `LICENSE`: author names (or choose another license)
- [ ] `CITATION.cff`: authors, affiliation, ORCID, repository URL, release date, paper reference
- [ ] `README.md`: GitHub user in the clone URL; BibTeX entry

## Required for reproducibility
- [ ] Add the preprocessing script (`dataupgrade.py`) as `data/dataupgrade.py`, with the exact command
      that produced the training data (see `data/README.md`)
- [ ] Commit `run_config.json` of the reported run (written by `src/planner.py --save-config`)
- [ ] Confirm whether the reported checkpoint was trained with `--stored-basis-symmetry`; if not,
      report both `EMF_SYMFIX=1` and `EMF_SYMFIX=0` evaluation rows
- [ ] Make sure the architecture figure's training numbers (panel f) match `run_config.json`
      (the command line defaults are 30 / 10 / 8, `default_config()` is 80 / 50 / 16)
- [ ] Run `scripts/evaluate.py` at `EMF_NUM_GEN=10000`, then fill the README table with
      `scripts/results_table.py` (never by hand)
- [ ] Upload the checkpoints (`encoder_pretrained.pt`, `direct_flow.pt`, `stage3_finetune.pt`),
      `*_generated.npz` and `*_eval_results.json` to Zenodo; put the DOI in the README

## Check
- [ ] CI is green (`.github/workflows/ci.yml` runs `pytest` incl. `planner.py --planner-selftest`)
- [ ] `python scripts/sample.py --checkpoint ... --n 8 --no-symfix` writes CIFs on a clean machine
- [ ] Tag the release (`git tag v1.0.0`) and enable Zenodo's GitHub integration for a software DOI
