# WildfireSpreadBench

A benchmark comparing **generative** and **discriminative** models for next-day
wildfire spread prediction, built on the
[WildfireSpreadTS](https://doi.org/10.5281/zenodo.8006177) dataset.

Every model — discriminative or generative — is scored through the same
evaluation harness (`src/evaluation/unified_eval.py`), so the reported AP, F1,
precision, recall, and IoU are directly comparable across the board.

> **Paper:** _TODO: title_ — _TODO: authors_, _TODO: venue / year_.
> See [Citation](#citation).

## Models

**Discriminative baselines** (PyTorch Lightning, YAML-configured):

| Model | Class | Config | Temporal |
| --- | --- | --- | --- |
| Persistence | `PersistenceModel` | `cfgs/persistence/full_run.yaml` | multi |
| Logistic regression | `LogisticRegression` | `cfgs/LogisticRegression/full_run.yaml` | mono |
| ResNet18 U-Net | `SMPModel` | `cfgs/unet/res18_monotemporal.yaml` | mono |
| ConvLSTM | `ConvLSTMLightning` | `cfgs/convlstm/full_run.yaml` | multi |
| U-TAE | `UTAELightning` | `cfgs/UTAE/all_features.yaml` | multi |

**Generative / segmentation models** (argparse-configured, `src/generative/`):

| Model | Module | `--model` |
| --- | --- | --- |
| BCE segmentation U-Net | `generative/unet_bce.py` | `unet_bce` |
| Conditional flow matching | `generative/flow_matching.py` | `flow` |
| Classifier-free DDPM | `generative/ddpm/` | `ddpm` |

Feature sets: `vegetation`, `multi`, `all` (matching WildfireSpreadTS Table 5).

## Setup

```bash
pip install -r requirements.txt
```

On Google Colab, use `requirements-colab.txt` — it installs only what the Colab
runtime is missing — and see `colab_train.ipynb`.

## Preparing the dataset

Download the dataset from
[Zenodo](https://doi.org/10.5281/zenodo.8006177) (CC-BY-4.0), then convert the
GeoTIFFs to HDF5. HDF5 takes roughly twice the disk space but is far faster to
train from:

```bash
python src/preprocess/CreateHDF5Dataset.py --data_dir YOUR_DATA_DIR --target_dir YOUR_TARGET_DIR
```

You can skip conversion with `--data.load_from_hdf5=False`, but training will be
impractically slow.

If your HDF5 files live on a network or cloud-synced mount, build an index once
so dataset setup does not have to stat every file:

```bash
python src/preprocess/BuildHDF5Index.py --data_dir YOUR_HDF5_DIR
```

Files on Google Drive or OneDrive mounts are copied to local disk on first read
(`src/dataloader/hdf5_cache.py`), because those filesystems cannot serve HDF5
random reads reliably. Override the cache location with `HDF5_CACHE_DIR`, or
disable it with `HDF5_CACHE_DISABLE=1`.

## Running the discriminative baselines

These are parameterized by YAML in `cfgs/` and parsed with the
[LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).
A run is assembled from three configs — model, trainer, and data. Later
arguments override earlier ones, and command-line arguments override configs.

Run from the repository root:

```bash
python src/train.py \
  --config=cfgs/unet/res18_monotemporal.yaml \
  --trainer=cfgs/trainer_single_gpu.yaml \
  --data=cfgs/data_monotemporal_full_features.yaml \
  --data.data_dir=YOUR_HDF5_DIR \
  --seed_everything=0 \
  --do_test=True
```

Multi-temporal models (ConvLSTM, U-TAE, Persistence) use
`cfgs/data_multitemporal_full_features.yaml`. You can also set `data_dir`
permanently in the data config instead of passing it each time.

When `--do_test=True`, training is followed automatically by a unified-eval pass
and a prediction visualization, so discriminative results land in the same
metric tables and W&B panels as the generative ones.

For a machine-local setup, copy `cfgs/trainer_local.example.yaml` to
`cfgs/trainer_local.yaml` (gitignored) and point the checkpoint `dirpath` at
fast local disk. Do not write checkpoints into a cloud-synced folder; partial
syncs corrupt them.

## Running the generative models

```bash
python src/train_generative.py \
  --model [unet_bce|flow|ddpm] \
  --data_dir YOUR_HDF5_DIR \
  --ckpt_dir YOUR_CKPT_DIR \
  --feature_set [vegetation|multi|all] \
  --epochs 200 \
  --eval_every 10
```

## Logging

Results are logged to Weights & Biases, defaulting to project
`WildfireSpreadBench` and entity `ram-algoverse`. Both are set with
`os.environ.setdefault`, so exporting `WANDB_PROJECT` / `WANDB_ENTITY`
overrides them without editing code. Disable W&B entirely with
`WANDB_MODE=disabled`, in which case results are written locally.

## Sweeps

The `cfgs/**/wandb_*.yaml` files are W&B sweep configurations covering the
cross-validation folds and feature-set ablations. Run them with
`wandb sweep cfgs/unet/wandb_table3.yaml`; see the
[W&B sweep docs](https://docs.wandb.ai/guides/sweeps). To run the same
experiments without W&B, pass the sweep's parameters directly on the command
line.

## Repository layout

```
cfgs/                       Model, trainer, data, and sweep configs
src/train.py                LightningCLI entry point (discriminative)
src/train_generative.py     Dispatcher for the generative models
src/dataloader/             FireSpreadDataset, datamodule, HDF5 caching/indexing
src/models/                 Discriminative baselines + prediction visualization
src/generative/             BCE U-Net, flow matching, DDPM
src/evaluation/             Unified metric harness and in-training eval callback
src/preprocess/             GeoTIFF -> HDF5 conversion and indexing
scripts/                    Checkpoint inspection, standalone visualization
colab_train.ipynb           End-to-end Colab workflow
```

Both entry points import relative to `src/`, so run them from the repository
root as shown above.

## Relationship to WildfireSpreadTS

This repository is a derivative of
[SebastianGer/WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS)
(MIT), the code accompanying the WildfireSpreadTS dataset paper. The dataset
classes, datamodule, and the discriminative baselines originate there.
`src/models/utae_paps_models/` is in turn an attributed copy from
[utae-paps](https://github.com/VSainteuf/utae-paps) (MIT).

Added here for WildfireSpreadBench:

- `src/generative/` and `src/train_generative.py` — the DDPM, flow matching, and
  BCE U-Net models, which are the novel contribution of this work.
- `src/evaluation/` — a unified metric harness shared by discriminative and
  generative models, plus a callback that runs it periodically during training.
- `src/dataloader/hdf5_cache.py` and `src/preprocess/BuildHDF5Index.py` — local
  caching and indexing so training works from cloud-mounted datasets.
- Checkpoint-resume fixes in `ConvLSTMLightning` and `UTAELightning`, and
  hyperparameter capture fixes in `BaseModel`, for current PyTorch Lightning.
- `colab_train.ipynb` and Colab-specific configs.

Two upstream caveats carry over and are worth knowing when comparing numbers:

- A dataset-class bug was fixed upstream in commit `ab3c8f35`. Corrected numbers
  run slightly higher than those in the 2023 paper; the trends are unchanged.
- Angle features are transformed with `sin` only, losing information that a
  `sin`/`cos` pair would preserve
  ([upstream note](https://github.com/SebastianGer/WildfireSpreadTS/blob/main/src/dataloader/FireSpreadDataset.py#L339)).

## Citation

If you use this code, please cite our paper (see `CITATION.cff`) and the
original dataset:

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
