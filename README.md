# WildfireSpreadBench: The Metric Decides the Model in Wildfire Spread Prediction

Code and configurations for a unified benchmark of six architectures on
next-day wildfire spread prediction, all scored through a single shared
evaluation pipeline.

Most work in this area is ranked by Average Precision, which summarizes
performance across every decision threshold. But a fire manager drawing an
evacuation boundary has to pick one threshold. This benchmark measures what
happens when you do: **AP and threshold-dependent metrics disagree about which
model is best.** The highest-AP model predicts 4–5× more area than actually
burned and places fifth of six on F1 and IoU.

## Results

Test set is the held-out 2021 season; threshold metrics computed at 0.5.
Rows are grouped by operating-point profile.

| Model | AP (Veg) | AP (All) | F1 (Veg) | F1 (All) | IoU (Veg) | IoU (All) | P (Veg) | P (All) | R (Veg) | R (All) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BCE U-Net | **0.562** | **0.529** | 0.308 | 0.329 | 0.182 | 0.197 | 0.186 | 0.203 | 0.896 | 0.861 |
| UTAE | 0.383 | 0.322 | 0.115 | 0.080 | 0.061 | 0.042 | 0.061 | 0.042 | **0.969** | **0.964** |
| ResNet18 U-Net | 0.375 | 0.401 | 0.568 | **0.585** | 0.396 | **0.413** | 0.558 | **0.599** | 0.578 | 0.572 |
| ConvLSTM | 0.403 | 0.390 | **0.583** | 0.570 | **0.411** | 0.398 | **0.565** | 0.543 | 0.602 | 0.599 |
| Logistic Regression | 0.352 | 0.371 | 0.548 | 0.566 | 0.378 | 0.394 | 0.501 | 0.551 | 0.606 | 0.581 |
| Flow Matching | 0.322 | 0.350 | 0.402 | 0.425 | 0.252 | 0.270 | 0.448 | 0.465 | 0.365 | 0.392 |

Three findings shape how this repository is organized:

**The metric outranks the architecture.** BCE U-Net wins on AP in both feature
configurations and loses badly on F1 and IoU. ConvLSTM and ResNet18 U-Net take
F1, IoU, and precision. Reporting AP alone would pick a different winner than
reporting F1 alone.

**Architectures fall into three operating-point profiles** that AP cannot
distinguish: recall-maximizing (BCE U-Net, UTAE), balanced (ConvLSTM, ResNet18
U-Net, Logistic Regression), and precision-leaning / under-predicting (Flow
Matching). At threshold 0.5, ConvLSTM's precision is roughly nine times UTAE's.

**More input channels barely matter.** Going from 7 channels to 23 moves AP by
0.03 on average, against a 0.21–0.24 spread across architectures. The
seven-channel Vegetation configuration is far cheaper and performs comparably.

## Models

Five discriminative and one generative, all trained on the same data splits.

| Model | Kind | Implementation | How to run |
| --- | --- | --- | --- |
| Logistic Regression | discriminative | `models/LogisticRegression.py` | `cfgs/LogisticRegression/full_run.yaml` |
| ResNet18 U-Net | discriminative | `models/SMPModel.py` | `cfgs/unet/res18_monotemporal.yaml` |
| ConvLSTM | discriminative | `models/ConvLSTMLightning.py` | `cfgs/convlstm/full_run.yaml` |
| UTAE | discriminative | `models/UTAELightning.py` | `cfgs/UTAE/all_features.yaml` |
| BCE U-Net | discriminative | `generative/unet_bce.py` | `--model unet_bce` |
| Flow Matching | generative | `generative/flow_matching.py` | `--model flow` |

The first four are PyTorch Lightning modules driven by YAML configs through
`src/train.py`. BCE U-Net and Flow Matching are standalone and run through
`src/train_generative.py`. BCE U-Net lives under `src/generative/` for
historical reasons — it is a discriminative segmentation model trained with a
positive-weighted binary cross-entropy objective, not a generative one.

## Data

[WildfireSpreadTS](https://doi.org/10.5281/zenodo.8006177): 13,607 daily images
across 607 U.S. fire events, January 2018 to October 2021, at 375 m resolution
over 23 channels. Train and validate on 2018–2020, test on the held-out 2021
season, using 128×128 crops.

Two input configurations isolate architecture from input data:

| Configuration | Channels | Feature indices |
| --- | --- | --- |
| Vegetation | 7 | `[0, 1, 2, 3, 4, 38, 39]` (vegetation indices + active fire) |
| All | 23 | all channels (`features_to_keep: null`) |

### Preparing the dataset

Download from [Zenodo](https://doi.org/10.5281/zenodo.8006177) (CC-BY-4.0),
then convert the GeoTIFFs to HDF5. HDF5 takes about twice the disk space but is
far faster to train from:

```bash
python src/preprocess/CreateHDF5Dataset.py --data_dir YOUR_DATA_DIR --target_dir YOUR_TARGET_DIR
```

You can skip conversion with `--data.load_from_hdf5=False`, but training will
be impractically slow.

If your HDF5 files sit on a network or cloud-synced mount, build an index once
so dataset setup does not have to stat every file:

```bash
python src/preprocess/BuildHDF5Index.py --data_dir YOUR_HDF5_DIR
```

Files on Google Drive or OneDrive mounts are copied to local disk on first read
(`src/dataloader/hdf5_cache.py`), because those filesystems cannot serve HDF5
random reads reliably. Override the cache location with `HDF5_CACHE_DIR` or
disable it with `HDF5_CACHE_DISABLE=1`.

## Setup

```bash
pip install -r requirements.txt
```

On Google Colab use `requirements-colab.txt`, which installs only what the
Colab runtime is missing, and see `colab_train.ipynb`.

## Running the experiments

Run everything from the repository root; both entry points import relative to
`src/`.

### Lightning models

Runs are assembled from three configs — model, trainer, and data. Command-line
arguments override config files, and later arguments override earlier ones.

```bash
python src/train.py \
  --config=cfgs/convlstm/full_run.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_multitemporal_full_features.yaml \
  --data.data_dir=YOUR_HDF5_DIR \
  --seed_everything=0 \
  --do_test=True
```

Select the feature configuration with `--data.features_to_keep`:

```bash
# Vegetation (7 channels)
--data.features_to_keep="[0, 1, 2, 3, 4, 38, 39]"

# All (23 channels)
--data.features_to_keep=null
```

Logistic Regression and ResNet18 U-Net use
`cfgs/data_monotemporal_full_features.yaml`; ConvLSTM and UTAE use
`cfgs/data_multitemporal_full_features.yaml`.

With `--do_test=True`, testing is followed automatically by a unified-eval pass
and a prediction visualization, so these results land in the same metric tables
and W&B panels as BCE U-Net and Flow Matching.

### BCE U-Net and Flow Matching

```bash
python src/train_generative.py \
  --model [unet_bce|flow] \
  --data_dir YOUR_HDF5_DIR \
  --ckpt_dir YOUR_CKPT_DIR \
  --feature_set [vegetation|all] \
  --epochs 200 \
  --eval_every 10
```

## Evaluation

`src/evaluation/unified_eval.py` is the single source of truth for every number
in the results table. All six models pass through the same metric code, the
same thresholds, and the same test data, which is what makes the cross-model
comparison meaningful.

It computes AP, F1, IoU, precision, and recall, with the threshold metrics at
0.5. Reporting all five together is the point: AP alone hides the
over-prediction that makes a high-AP model unusable for drawing an operational
boundary.

Metrics accumulate incrementally through `torchmetrics` rather than by
concatenating every pixel score across the test set, so memory stays bounded on
the full 2021 season.

Two entry points:

- `evaluate_lightning_module()` for Lightning models. It calls each model's own
  `get_pred_and_gt`, so temporal flattening, day-of-year features, and tiled
  inference match the training-time path exactly.
- `evaluate_model()` for anything else. Pass a `predict_fn(x0) -> probability
  map` callable and it computes the identical metrics.
  `evaluate_bce_unet()` and `evaluate_flow()` are thin wrappers around it.

## Repository layout

```
cfgs/                       Model, trainer, data, and sweep configs
src/train.py                LightningCLI entry point
src/train_generative.py     Entry point for BCE U-Net and Flow Matching
src/dataloader/             FireSpreadDataset, datamodule, HDF5 caching/indexing
src/models/                 Lightning models + prediction visualization
src/generative/             BCE U-Net and Flow Matching
src/evaluation/             Unified metric harness and in-training eval callback
src/preprocess/             GeoTIFF -> HDF5 conversion and indexing
scripts/                    Checkpoint inspection, standalone visualization
colab_train.ipynb           End-to-end Colab workflow
```

## Logging

Results are logged to Weights & Biases, defaulting to project
`WildfireSpreadBench` and entity `ram-algoverse`. Both are set with
`os.environ.setdefault`, so exporting `WANDB_PROJECT` or `WANDB_ENTITY`
overrides them without editing code. Set `WANDB_MODE=disabled` to turn logging
off entirely and write results locally.

For a machine-local setup, copy `cfgs/trainer_local.example.yaml` to
`cfgs/trainer_local.yaml` (gitignored) and point the checkpoint `dirpath` at
fast local disk. Do not write checkpoints into a cloud-synced folder; partial
syncs corrupt them.

The `cfgs/**/wandb_*.yaml` files are W&B sweep configurations covering the
cross-validation folds and feature-set ablations inherited from
WildfireSpreadTS. Run one with `wandb sweep cfgs/unet/wandb_table3.yaml`.

## Relationship to WildfireSpreadTS

This repository is a derivative of
[SebastianGer/WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS)
(MIT), the code released with the WildfireSpreadTS dataset. The dataset classes,
the datamodule, and the Logistic Regression, ResNet18 U-Net, ConvLSTM, and UTAE
baselines originate there. `src/models/utae_paps_models/` is in turn an
attributed copy from [utae-paps](https://github.com/VSainteuf/utae-paps) (MIT).

Added for this work:

- `src/generative/` and `src/train_generative.py` — BCE U-Net and Flow Matching.
- `src/evaluation/` — the unified metric harness that makes all six models
  comparable, plus a callback that runs it periodically during training.
- `src/dataloader/hdf5_cache.py` and `src/preprocess/BuildHDF5Index.py` — local
  caching and indexing so training works from cloud-mounted datasets.
- Checkpoint-resume fixes in `ConvLSTMLightning` and `UTAELightning`, and
  hyperparameter capture fixes in `BaseModel`, for current PyTorch Lightning.

Two upstream caveats carry over and matter when comparing against published
WildfireSpreadTS numbers:

- A dataset-class bug was fixed upstream in commit `ab3c8f35`. Corrected numbers
  run slightly higher than the 2023 paper's; the trends are unchanged.
- Angle features are transformed with `sin` only, losing information a
  `sin`/`cos` pair would preserve
  ([upstream note](https://github.com/SebastianGer/WildfireSpreadTS/blob/main/src/dataloader/FireSpreadDataset.py#L339)).

## Citation

Please cite the WildfireSpreadTS dataset this benchmark is built on:

```bibtex
@inproceedings{
    gerard2023wildfirespreadts,
    title={WildfireSpread{TS}: A dataset of multi-modal time series for wildfire spread prediction},
    author={Sebastian Gerard and Yu Zhao and Josephine Sullivan},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://openreview.net/forum?id=RgdGkPRQ03}
}
```

## License

MIT. See `LICENSE` for the full notice, including upstream copyright.
