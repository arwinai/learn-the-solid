# Learn the Solid, Not the File

Reference implementation of the **canonical region graph** and the invariance
measurement suite from

> *Learn the Solid, Not the File: Canonical Inputs for Neural Networks on CAD
> Boundary Representations.*

A B-rep does not determine the solid it bounds. The same part can be written
with a different face partition, in a different coordinate frame, or with its
analytic surfaces re-expressed as splines — and published B-rep encoders change
their predictions when it is. This repository contains

1. the **canonical region graph**: an input representation whose nodes,
   features and coordinate frame are functions of the solid rather than of the
   file, with a graph-transformer segmentation model trained on it;
2. the **perturbation suite** that produces geometrically identical variants of
   standard benchmark solids (re-partitioning, rigid motion, spline
   re-expression, and their composition);
3. the **evaluation protocol**: accuracy on perturbed cells, per-solid
   prediction stability, label-free retrieval, and representation-level
   invariance checks.

Datasets, checkpoints and the human-modeller study data will be released
separately.

## Layout

```
cinv/
  # the representation
  region_graph.py          faces -> maximal analytic regions -> graph (the construction)
  surface_graph.py         canonical surface identity; analytic recognition of spline-stored quadrics
  region_feats.py          per-region integral features
  region_canon_grid.py     canonical inertia-frame spatial grid
  shape_io.py              one solid out of a STEP file
  paths.py                 filesystem locations, resolved from the environment
  fixtures.py              construction-variant pairs (same solid, two constructions), no data needed

  # extraction and training
  region_extract.py        region graphs for a dataset split (clean)
  region_train.py          graph transformer + training loop
  eval_region_ckpt.py      clean test accuracy from a checkpoint
  eval_perclass.py         per-class IoU with support counts
  train_baseline.py        AAGNet / UV-Net baseline training on the same protocol
  uvnet_families.py        UV-Net segmentation head used by the baseline scripts
  pc_extract.py            dataset split enumeration (shared by every extractor); point-cloud baseline inputs

  # perturbations (same solid, different file)
  repartition_gen.py       re-partitioning families A (axis), B (diagonal), C (merge)
  repartition_testset.py   the perturbed test set as STEP files
  repart_extract.py        region graphs + baseline graphs for a perturbed directory, labels transferred geometrically
  rot_extract.py           random SO(3) cell
  nc_extract.py            spline re-expression (NurbsConvert) cell
  combo_extract.py         all nuisances composed
  nuisance_seed.py         fixed per-part draws so every model sees the same perturbation
  emit_nuisance_steps.py   rotated / converted STEP directories (+ labels) for BRepNet
  emit_combo_bn.py         the composed cell as a STEP directory

  # measurement
  graph_agreement.py       representation-level invariance: matched-feature residuals, clean vs perturbed
  fixture_agreement.py     the same protocol on the in-process fixtures
  curved_split_stress.py   feature residuals when curved faces are split (per feature group)
  verification_musd.npz    per-channel training statistics used to scale residuals
  clean_bitident_check.py  does a code change move any clean graph?
  verify_exactfit.py       surface keys agree clean vs spline-converted
  region_check.py          invariance and label-purity go/no-go checks
  eval_region_pkldir.py    accuracy on a perturbation directory
  eval_baselines_pkldir.py AAGNet / UV-Net accuracy on a perturbation directory
  eval_cells.py            every model on a list of cells
  eval_matrix.py           dataset x model x {clean, famA, famB} table from checkpoints
  paired_consistency.py    per-solid prediction stability, clean vs perturbed
  f360_paired_consistency.py   the same against a clean reference directory
  paired_churn_all.py      prediction churn on every cell
  report_rejections.py     rejected label transfers per cell
  retrieval_invariance.py  label-free retrieval under re-partitioning
  aagnet_aug_train.py      baseline training on clean + augmented mixes
  uvnet_aug_train.py
  famC_aug_eval.py         augmented checkpoints on any perturbation cell

  # BRepNet path
  brepnet_extract_robust.py  fault-isolated driver for BRepNet's STEP extractor
  brepnet_prep.py            BRepNet inputs for a cell
  brepnet_rp_dataset.py      BRepNet dataset over a perturbed directory
  bn_parse_metrics.py        BRepNet test metrics as one line
```

## Install

Two environments, because the geometry kernel and the deep-learning stack do
not coexist comfortably:

```bash
# geometry: building region graphs, generating perturbations
conda create -n cinv-geom python=3.11 pythonocc-core=7.8 numpy scipy
# learning: training and evaluating (torch 2.4 / dgl 2.4, CUDA 12.1 in the paper)
conda create -n cinv-dl python=3.11 numpy scipy
conda activate cinv-dl && pip install torch==2.4.0 dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

The baselines are used as published; clone them into `third_party/`
(or point the variables below at existing checkouts):

- [AAGNet](https://github.com/whjdark/AAGNet) — required. Besides being a
  baseline, its STEP→graph extractor and unit-box normalisation are reused so
  that every model sees identically normalised solids.
- [UV-Net](https://github.com/AutodeskAILab/UV-Net), [BRepNet](https://github.com/AutodeskAILab/BRepNet) — optional, only for those baselines.

## Paths

Nothing is hard-coded. `cinv/paths.py` resolves every location from the
environment:

| variable | default | purpose |
|---|---|---|
| `CINV_DATA` | `./data` | datasets root, one subdirectory per dataset |
| `CINV_CKPT` | `./ckpt` | checkpoints |
| `CINV_AAGNET` | `./third_party/AAGNet` | AAGNet checkout |
| `CINV_UVNET` | `./third_party/UV-Net` | UV-Net checkout (optional) |
| `CINV_BREPNET` | `./third_party/BRepNet` | BRepNet checkout (optional) |

A dataset directory holds `steps/`, `labels/` and `train/val/test.txt` in the
layout of the original benchmark; the extraction scripts create
`region_graphs*/` and `eval_*/` beside them. Several scripts also accept
`DS_ROOT=<dataset dir>` to select the dataset.

## Feature flags

The representation is assembled from independently gated pieces, so an ablation
is an environment variable rather than a code branch. The configuration used
throughout the paper is

```bash
export RECOG=1 PART_FRAME=1 BHIST=1 EHIST=1 CANON_GRID=1 \
       ISO_SCALE=1 CONE_INTRINSIC=1 SA_ABS=1 SWEEP_RECOG=1
```

`PART_FRAME=0` gives the file-frame ablation, `CANON_GRID=0` drops the
spatial grid, `RECOG=0` disables analytic recognition of spline-stored
quadrics. Flags are read at extraction time and baked into the `.npz` graphs.

## Quick check (no data needed)

Two constructions of the same ring — one revolved profile vs. a cylinder minus
a cylinder — produce the same region graph:

```python
# geometry environment, from cinv/
import sys, os; sys.path.insert(0, os.environ["CINV_AAGNET"])
from fixtures import ring_pair
from shape_io import as_solid
from dataset.AAGExtractor import scale_solid_to_unit_box
from region_graph import build

A, B = ring_pair(1.0, 0.4, 0.5)
f2r_a, feats_a, edges_a, ef_a = build(scale_solid_to_unit_box(as_solid(A)))
f2r_b, feats_b, edges_b, ef_b = build(scale_solid_to_unit_box(as_solid(B)))
# same region count; matched node features agree to ~1e-17
```

`python fixture_agreement.py` runs the full representation-level protocol on
24 such pairs; `python surface_graph.py` and `python region_check.py` run the
surface-identity and label-purity self-tests.

## Reproducing the paper

All commands are run from `cinv/` with the feature flags above exported.
Geometry-environment steps are marked *(geom)*, learning-environment steps
*(dl)*.

**1. Clean region graphs** *(geom)*

```bash
for s in train val test; do
  DS_ROOT=$CINV_DATA/mfinstseg RG_DIR=region_graphs python region_extract.py $s
done
```

**2. Train** *(dl)* — the paper's configuration for the synthetic benchmarks:

```bash
python region_train.py --root $CINV_DATA/mfinstseg --tag mfinstseg \
    --d 512 --layers 8 --heads 8 --epochs 80 --cb 0 --lr 1e-3 --seed 0
python eval_region_ckpt.py $CINV_CKPT/region_mfinstseg.pt $CINV_DATA/mfinstseg
```

Fusion 360 segmentation uses `--epochs 200 --cb 0.3`. Model selection is on the
validation split; the test split is scored once.

**3. Perturbation cells** *(geom)* — same solids, different files:

```bash
RP_SPLIT=test RP_FAM=A python repartition_gen.py      # family A (axis pattern)
RP_SPLIT=test RP_FAM=B python repartition_gen.py      # family B (diagonal pattern)
python repart_extract.py                              # graphs + geometric label transfer
python rot_extract.py test                            # random SO(3)
python nc_extract.py test                             # spline re-expression
python combo_extract.py test                          # everything composed
```

Each cell records, per solid, the child→parent face map and the faces whose
label transfer was rejected (reported, never silently dropped); volume and
area of every perturbed solid are checked against the original to `1e-9`.

**4. Measure** *(dl)*

```bash
python graph_agreement.py                 # representation-level residual table
python eval_region_pkldir.py ...          # accuracy on a cell
python eval_baselines_pkldir.py ...       # the same for AAGNet / UV-Net
python paired_consistency.py --region $CINV_CKPT/region_mfinstseg.pt
python retrieval_invariance.py --region $CINV_CKPT/region_mfinstseg.pt
python paired_churn_all.py                # churn on every cell
```

Each script prints its `Usage:` line when run without arguments or in its
module docstring.

## Citation

```bibtex
@article{learnthesolid2026,
  title  = {Learn the Solid, Not the File: Canonical Inputs for Neural Networks on CAD Boundary Representations},
  year   = {2026},
}
```

## License

MIT (see `LICENSE`). The baselines this code compares against carry their own
licenses, which govern their weights and extractors.
