# WildfireBenchBaselines

Discriminative baselines for **WildfireSpreadBench**, a benchmark for next-day
wildfire spread prediction. This repository provides the reference
implementations, configs, and a unified evaluation harness used to produce the
baseline numbers reported in the accompanying paper.

> **Paper:** _TODO: title_ — _TODO: authors_, _TODO: venue / year_.
> See [Citation](#citation).

All baselines are trained and evaluated on the
[WildfireSpreadTS](https://doi.org/10.5281/zenodo.8006177) dataset (Gerard et
al., NeurIPS 2023 Datasets & Benchmarks), from which this codebase is derived —
see [Relationship to WildfireSpreadTS](#relationship-to-wildfirespreadts).

## Baselines

| Model | Class | Config | Temporal |
| --- | --- | --- | --- |
| Persistence | `PersistenceModel` | `cfgs/persistence/full_run.yaml` | multi |
| Logistic regression | `LogisticRegression` | `cfgs/LogisticRegression/full_run.yaml` | mono |
| ResNet18 U-Net | `SMPModel` | `cfgs/unet/res18_monotemporal.yaml` | mono |
| ConvLSTM | `ConvLSTMLightning` | `cfgs/convlstm/full_run.yaml` | multi |
| U-TAE | `UTAELightning` | `cfgs/UTAE/all_features.yaml` | multi |

Every model is scored through the same code path
(`src/evaluation/unified_eval.py`), so reported AP, F1, precision, recall, and
IoU are directly comparable across baselines. The harness also accepts a plain
`predict_fn(x) -> probability map` callable, which lets non-Lightning models
(e.g. diffusion or flow-matching baselines) be evaluated with identical metrics.

## Setup

```bash
pip install -r requirements.txt
```

On Google Colab, use `requirements-colab.txt` instead — it installs only the
packages missing from the Colab runtime — and see `colab_train.ipynb`.

## Preparing the dataset

Download the dataset from
[Zenodo](https://doi.org/10.5281/zenodo.8006177) (CC-BY-4.0), then convert the
GeoTIFFs to HDF5. HDF5 takes roughly twice the disk space but is far faster to
train from:

```bash
python src/preprocess/CreateHDF5Dataset.py --data_dir YOUR_DATA_DIR --target_dir YOUR_TARGET_DIR
```

You can skip conversion with `--data.load_from_hdf5=False`, but training will
be impractically slow.

If your HDF5 files live on a network or cloud-synced mount, build an index once
so dataset setup does not have to stat every file:

```bash
python src/preprocess/BuildHDF5Index.py --data_dir YOUR_HDF5_DIR
```

Files on Google Drive or OneDrive mounts are copied to local disk on first read
(`src/dataloader/hdf5_cache.py`), because those filesystems cannot serve HDF5
random reads reliably. Override the cache location with `HDF5_CACHE_DIR`, or
disable it with `HDF5_CACHE_DISABLE=1`.

## Training and evaluation

Experiments are parameterized by YAML in `cfgs/` and parsed with the
[LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).
A run is assembled from three configs: a model config, a trainer config, and a
data config. Later arguments override earlier ones, and command-line arguments
override config files.

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

For a machine-local setup, copy `cfgs/trainer_local.example.yaml` to
`cfgs/trainer_local.yaml` (gitignored) and point the checkpoint `dirpath` at
fast local disk. Do not check checkpoints into a cloud-synced folder; partial
syncs corrupt them.

### Logging

Results are logged to Weights & Biases under the project
`wildfire-spread-bench`. Override it with
`--trainer.logger.init_args.project=YOUR_PROJECT`, or disable W&B entirely with
`WANDB_MODE=disabled`, in which case results are written locally.

### Sweeps

The `cfgs/**/wandb_*.yaml` files are W&B sweep configurations covering the
cross-validation folds and feature-set ablations. Run them with
`wandb sweep cfgs/unet/wandb_table3.yaml`; see the
[W&B sweep docs](https://docs.wandb.ai/guides/sweeps). To run the same
experiments without W&B, pass the sweep's parameters directly on the command
line.

## Repository layout

```
cfgs/                  Model, trainer, data, and sweep configs
scripts/               Checkpoint inspection and prediction visualization
src/train.py           LightningCLI entry point
src/dataloader/        FireSpreadDataset, datamodule, HDF5 caching/indexing
src/models/            Baseline implementations
src/preprocess/        GeoTIFF -> HDF5 conversion and indexing
src/evaluation/        Unified metric harness and in-training eval callback
colab_train.ipynb      End-to-end Colab workflow
```

## Relationship to WildfireSpreadTS

This repository is a derivative of
[SebastianGer/WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS)
(MIT), the code accompanying the WildfireSpreadTS dataset paper. The dataset
classes, datamodule, and the U-Net / ConvLSTM / U-TAE / logistic regression /
persistence baselines originate there. `src/models/utae_paps_models/` is in turn
an attributed copy from
[utae-paps](https://github.com/VSainteuf/utae-paps) (MIT).

Changes made here for WildfireSpreadBench:

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
