# DirectCrystalFlow

**Symmetry-aware crystal generation by flow matching over a latent set of asymmetric-unit sites, with an exact composition planner.**

DirectCrystalFlow generates periodic crystals conditioned on a target property (formation energy) and, optionally, a space group. A flow model moves 20 latent site tokens from noise to a crystal latent. A decoder reads out one asymmetric unit with Wyckoff positions, an autoregressive planner chooses how many sites each element occupies, and whole-orbit expansion builds the full cell. The requested space group is built in by construction rather than repaired afterwards.

<p align="center">
  <img src="docs/figures/directcrystalflow_architecture.png" width="760" alt="DirectCrystalFlow architecture">
</p>

## What is different

- **Latent site-set flow.** A frozen equivariant GNN encoder maps each real crystal to 20 tokens × 64 dimensions, one token per asymmetric-unit site, in a canonical order, with learned "empty" tokens for unused slots. The flow matches noise to these encoder latents, so it learns over sites, not atoms.
- **Wyckoff-projected decoding.** Each site gets a Wyckoff class (masked to the classes of the requested group), and its position is projected onto that class's subspace: *f* = *P*(*a* − *o*) + *o*. Multiplicities come from an exact table. Fixed-point classes are used at most once.
- **Composition planner over sites, drawn after conflict resolution.** A GRU emits (element, number of sites) pairs in ascending electronegativity, under masks that make the plan fill the site count exactly. The plan is drawn *after* the conflict resolver has relocated or dropped clashing sites, so it is written for the sites that actually exist. It is then realised exactly, with high-multiplicity sites choosing first.
- **Guidance on property and symmetry.** Classifier-free guidance on *y* and on the space group in the flow; guidance on *y* on the planner's logits before masking.

See [`docs/architecture.md`](docs/architecture.md) for the full description and every default hyperparameter.

## Repository layout

```
src/planner.py            model, data loader, three-stage training (python src/planner.py --help)
src/symfix.py             stored-basis symmetry operations (mined from the training data)
scripts/sample.py         generate crystals -> CIF files
scripts/evaluate.py       full MP-20 evaluation (CDVAE protocol + MLFF stability + S.U.N.)
scripts/make_figures.py   post-training figures (training curves, conditioning, latent maps, ...)
scripts/results_table.py  results JSON -> markdown table for the paper / this README
figures/architecture/     source of the architecture figure (python make_fig.py)
docs/                     architecture, evaluation protocol, figures
data/README.md            dataset format the loader expects
tests/                    CPU tests (run in CI)
```

## Installation

```bash
git clone https://github.com/<USER>/DirectCrystalFlow.git
cd DirectCrystalFlow
conda env create -f environment.yml && conda activate dcflow
# or, with an existing PyTorch (CUDA) install:
pip install -r requirements.txt -r requirements-eval.txt
```

Tested with Python 3.10–3.11. Training needs one GPU; sampling and the tests run on CPU. The evaluation's stability section additionally needs one machine-learned force field (MLFF): `pip install chgnet` (or `mattersim`, `mace-torch`, `orb-models`).

## Data

The model trains on **MP-20** (the CDVAE train/val/test split of 45,231 Materials Project structures with at most 20 atoms per cell). The loader reads a *pre-processed* version of it, in which every structure carries its mined symmetry: space group, operations, Wyckoff letters, projectors and offsets, multiplicities, the asymmetric-unit mask, and a periodic graph. The expected on-disk format is documented field by field in [`data/README.md`](data/README.md).

```
data/mp_20/{train,val,test}/{split}.npz           graph_dict: material_id -> record
data/mp_20/{train,val,test}/{split}.csv           material_id, formation_energy_per_atom, space_group
data/mp_20/{train,val,test}/{split}_config.json   atomic_numbers, node_vectors, element_properties
```

## Training

Training runs in three stages from one command. Checkpoints and logs are written to the working directory.

| stage | what trains | output |
|---|---|---|
| 1 autoencoder pre-training | encoder, decoder, all heads, planner | `encoder_pretrained.pt`, `encoder_pretrain_metrics.csv` |
| 2 flow matching | flow, conditioner, coarse space-group head (encoder frozen) | `direct_flow.pt` |
| 3 generative fine-tuning | everything except encoder and property head | `stage3_finetune.pt` |

```bash
mkdir -p runs/main && cd runs/main
python ../../src/planner.py --data-root ../../data/mp_20 \
    --flow-epochs 80 --s3-epochs 50 --s3-steps 16 \
    --stage3-metrics stage3_metrics.csv \
    2>&1 | tee train.log
```

**Pass `--flow-epochs`, `--s3-epochs` and `--s3-steps` explicitly.** Without them the command line uses its own defaults (30 / 10 / 8), which differ from `default_config()` (80 / 50 / 16) because the command line overwrites those keys. The values above match `default_config()` and the architecture figure; whatever you use, report the values from the `run_config.json` written next to the checkpoints.

Other options:
- `--encoder-path encoder_pretrained.pt` resumes from a stage-1 checkpoint and skips stage 1.
- `--stored-basis-symmetry` trains with `src/symfix.py`. See the note under *Evaluation*.
- `python src/planner.py --planner-selftest` runs the planner's CPU self-checks.

## Sampling

```bash
python scripts/sample.py --checkpoint runs/main/stage3_finetune.pt --data-root data/mp_20 \
    --n 64 --target-y -1.5 --out samples/            # add --target-sg 225 to fix the group
```

This writes one CIF per crystal plus `samples.csv`. The CIFs are raw model output; nothing is relaxed. From Python:

```python
import sys, torch; sys.path.insert(0, "src")
import planner
ck = torch.load("stage3_finetune.pt", map_location="cpu", weights_only=False)
model = planner.DirectCrystalFlow(planner.config_from_dict(ck["config"]))
planner.load_checkpoint("stage3_finetune.pt", model, map_location="cpu", strict=False)
out = model.eval().sample(16, target_y=-1.5, steps=100)   # frac, lattice, sampled_types, mask, ...
```

## Evaluation

```bash
EMF_CHECKPOINT=runs/main/stage3_finetune.pt MP20_ROOT=data/mp_20 \
EMF_OUTPUT_PREFIX=results/main python scripts/evaluate.py
python scripts/results_table.py results/main_eval_results.json
```

`scripts/evaluate.py` reports, with 95% confidence intervals (details in [`docs/evaluation.md`](docs/evaluation.md)):

- **Validity.** CDVAE structural gate (0.5 Å) and the CDVAE composition gate on the `smact14` oxidation-state list.
- **Coverage.** COV-R / COV-P with CrystalNN + Magpie fingerprints, at 10,000 samples.
- **Distribution distances.** Density, number of elements, atoms per cell, volume per atom.
- **Symmetry.** Space-group statistics and space-group controllability against a shuffled-condition control.
- **Uniqueness and novelty.** CDVAE StructureMatcher tolerances.
- **Stability and S.U.N.** MLFF relaxation, a hull built from MLFF energies of the reference phases, and S.U.N. over the generated pool.
- **Conditioning.** MLFF formation energy versus target, compared with the unconditional chance level.
- **Planner ablation.** `use_planner=False`, both unconditional and conditional.

Useful switches: `EMF_NUM_GEN` (default 10000), `EMF_SEED`, `EMF_RUN_STABILITY=0`, `EMF_SYMFIX=0`.

> **Symmetry basis.** `src/planner.py` trains with `stored_basis_symmetry=False` by default, and the checkpoint does not record this flag. The evaluator installs stored-basis symmetry by default. If you trained without `--stored-basis-symmetry`, evaluate both ways (`EMF_SYMFIX=1` and `EMF_SYMFIX=0`) and report both rows.

A full run takes many GPU hours. Most of the time goes to CrystalNN fingerprints and MLFF relaxations, both cached on disk.

## Figures

```bash
python scripts/make_figures.py --checkpoint runs/main/stage3_finetune.pt --data-root data/mp_20 \
    --log runs/main/train.log --stage1_csv runs/main/encoder_pretrain_metrics.csv \
    --stage3_csv runs/main/stage3_metrics.csv --guidance_sweep 1 1.5 2 3 --out_dir results/figures
python scripts/make_figures.py --from_results results/figures/results.pkl --out_dir results/figures_v2  # no GPU
python figures/architecture/make_fig.py      # regenerates docs/figures/directcrystalflow_architecture.*
```

Property values in `make_figures.py` are read by the model's own frozen encoder and serve as internal consistency checks. The independent grade is the MLFF oracle in `evaluate.py`.

## Results

MP-20 test split, 10,000 unconditional samples, seed 0. Fill this table with `scripts/results_table.py`; do not copy numbers by hand.

| | Validity (struct / comp) | COV-R / COV-P | d_ρ / d_Nel | Unique / novel | Stable / S.U.N. (MLFF) | Cond. Spearman (MLFF) |
|---|---|---|---|---|---|---|
| DirectCrystalFlow | — | — | — | — | — | — |
| DirectCrystalFlow, no planner | — | — | — | — | — | — |

## Limitations

- **No crystal structure prediction.** The flow is not conditioned on composition, and the planner counts sites rather than atoms, so the DiffCSP / FlowMM CSP protocol does not apply directly.
- **Asymmetric-unit size.** It is capped at `n_sites = 20` tokens; training structures with more sites cannot be represented.
- **Stability is estimated, not confirmed.** It comes from an MLFF and a hull of MP-20 phases (plus Materials Project phases when `MP_API_KEY` is set). Confirm candidate structures with DFT.
- **Single dataset and property.** Results cover MP-20 and formation energy only.

## Citation

```bibtex
@article{<KEY>,
  title   = {<TITLE>},
  author  = {<AUTHORS>},
  journal = {<JOURNAL or arXiv>},
  year    = {<YEAR>}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License and acknowledgements

Code: MIT ([`LICENSE`](LICENSE)). MP-20 is derived from the Materials Project (CC BY 4.0) via the CDVAE split.

The evaluation follows the protocols of CDVAE (Xie et al., 2022), DiffCSP (Jiao et al., 2023) and MatterGen (Zeni et al., 2025). It builds on pymatgen, spglib, matminer, SMACT, ASE, PyTorch and PyTorch Geometric.
