"""
src/models/visualize_predictions.py

Post-training visualization for discriminative models
(SMPModel, ConvLSTMLightning, LogisticRegression, UTAELightning, etc.).

Produces a multi-panel figure per sample showing:
  - Active-fire input channel (last timestep, channel index -1 in flattened tensor)
  - Ground truth fire mask (binary)
  - Model probability map (sigmoid output)
  - Binary prediction at threshold=0.5

Also logs a Precision-Recall curve and per-sample W&B images when W&B is active.

Designed to be called at the end of train.py, after evaluate_lightning_model().

Usage (standalone):
-------------------
python src/models/visualize_predictions.py \\
    --config cfgs/unet/res18_monotemporal.yaml \\
    --data_config cfgs/data_monotemporal_full_features.yaml \\
    --ckpt_path /path/to/best.ckpt \\
    --output_dir /path/to/output \\
    --n_samples 8

Usage (from train.py after do_test):
--------------------------------------
    from models.visualize_predictions import run_visualization
    run_visualization(
        model=cli.model,
        datamodule=cli.datamodule,
        output_dir=cli.config.trainer.default_root_dir,
        model_name=model_name,
        n_samples=8,
        wandb_log=True,
    )
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for Colab / headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

# ---------------------------------------------------------------------------
# Feature channel name lookup (indices into the 23-channel per-timestep HDF5
# tensor). Index -1 in the flattened input = active fire of the last timestep.
# ---------------------------------------------------------------------------
CHANNEL_NAMES = {
    0: "I1 (0.64µm)", 1: "I2 (0.86µm)", 2: "M11 (2.25µm)",
    3: "NDVI", 4: "EVI2", 5: "LAI", 6: "FPAR",
    7: "LST_Day", 8: "LST_Night", 9: "Emis31", 10: "Emis32",
    11: "SWIR_1", 12: "SWIR_2", 13: "SWIR_3", 14: "VH", 15: "VV",
    16: "Elevation", 17: "Slope", 18: "Aspect", 19: "ERC",
    20: "PDSI", 21: "FFMC", 22: "ActiveFire",
}

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def _get_active_fire_channel(x_batch: torch.Tensor) -> np.ndarray:
    """
    Extract the active-fire channel from a [B, T*C, H, W] or [B, T, C, H, W] tensor.
    Returns shape [B, H, W].
    """
    if x_batch.dim() == 5:
        # [B, T, C, H, W] → use last timestep, fire channel (index 22)
        fire = x_batch[:, -1, 22, :, :]
    else:
        # [B, T*C, H, W] flattened → fire channel is at position -1 (last channel)
        fire = x_batch[:, -1, :, :]
    return _to_numpy(fire)  # [B, H, W]


@torch.no_grad()
def _collect_samples(model, datamodule, device, n_samples: int, threshold: float = 0.5):
    """
    Run inference over the test set and collect up to n_samples examples.
    Prefers samples that contain at least one fire pixel in the GT.

    Returns lists of length n_samples:
        fire_inputs  – [H, W] active-fire input arrays
        prob_maps    – [H, W] sigmoid probability maps
        pred_masks   – [H, W] binary predictions at `threshold`
        gt_masks     – [H, W] ground-truth binary masks
        all_targets  – flat array of all GT labels  (for PR curve)
        all_scores   – flat array of all probabilities (for PR curve)
    """
    model.eval()
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    fire_inputs, prob_maps, pred_masks, gt_masks = [], [], [], []
    all_targets, all_scores = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Handle temporal 5-D input identical to BaseModel.forward()
        if x.dim() == 5:
            x_flat = x[:, 0, :, :, :]  # monotemporal: use first (only) time step
        else:
            x_flat = x

        logits = model(x_flat)          # [B, 1, H, W]  or  [B, H, W]
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        prob = torch.sigmoid(logits)    # [B, 1, H, W]

        # Accumulate for PR curve
        all_targets.append(_to_numpy(y).flatten())
        all_scores.append(_to_numpy(prob).flatten())

        # Harvest visualizable samples — prefer fire-containing GT patches
        for b in range(x.size(0)):
            gt_b = _to_numpy(y[b])              # [H, W]
            prob_b = _to_numpy(prob[b, 0])      # [H, W]
            fire_b = _get_active_fire_channel(x)[b]  # [H, W]

            if len(fire_inputs) < n_samples:
                if gt_b.sum() > 0 or len(fire_inputs) < n_samples // 2:
                    fire_inputs.append(fire_b)
                    prob_maps.append(prob_b)
                    pred_masks.append((prob_b >= threshold).astype(np.float32))
                    gt_masks.append(gt_b)

        if len(fire_inputs) >= n_samples:
            break

    all_targets = np.concatenate(all_targets)
    all_scores = np.concatenate(all_scores)

    return fire_inputs, prob_maps, pred_masks, gt_masks, all_targets, all_scores


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _plot_sample_grid(
    fire_inputs, prob_maps, pred_masks, gt_masks,
    model_name: str, output_dir: Path
) -> Path:
    """
    4-column grid: Input Fire | GT | Prob Map | Prediction
    One row per sample.
    """
    n = len(fire_inputs)
    fig = plt.figure(figsize=(16, 4 * n))
    gs = gridspec.GridSpec(n, 4, figure=fig, hspace=0.35, wspace=0.08)

    col_titles = ["Input: ActiveFire (t)", "Ground Truth (t+1)",
                  "Probability Map", f"Prediction (thr=0.5)"]
    cmaps = ["magma", "Reds", "YlOrRd", "Reds"]

    for row in range(n):
        data_cols = [fire_inputs[row], gt_masks[row], prob_maps[row], pred_masks[row]]
        for col, (dat, cmap, ctitle) in enumerate(zip(data_cols, cmaps, col_titles)):
            ax = fig.add_subplot(gs[row, col])
            vmax = 1.0 if col != 2 else None   # prob map: let matplotlib auto-scale
            im = ax.imshow(dat, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(ctitle, fontsize=11, fontweight="bold", pad=6)
            if col == 2:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # Annotate fire pixel count
            if col in (1, 3):
                n_fire = int(dat.sum())
                ax.text(0.02, 0.96, f"fire px: {n_fire}",
                        transform=ax.transAxes, fontsize=7,
                        va="top", color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45))

    fig.suptitle(f"{model_name} — Test-set Predictions", fontsize=14, fontweight="bold", y=1.01)
    out_path = output_dir / f"{model_name}_sample_predictions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_pr_curve(all_targets, all_scores, model_name: str, output_dir: Path) -> Path:
    """
    Precision-Recall curve with AP annotation and persistence baseline.
    """
    precision, recall, _ = precision_recall_curve(all_targets, all_scores)
    ap = average_precision_score(all_targets, all_scores)
    fire_rate = float(all_targets.mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, color="#E07B39", label=f"{model_name} (AP={ap:.4f})")
    ax.axhline(fire_rate, ls="--", lw=1.2, color="#888", label=f"No-skill baseline ({fire_rate:.4f})")
    ax.axvline(0.0, lw=0, color="none")  # padding

    # Shade area under curve
    ax.fill_between(recall, precision, alpha=0.12, color="#E07B39")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    out_path = output_dir / f"{model_name}_pr_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_score_histogram(all_scores, all_targets, model_name: str, output_dir: Path) -> Path:
    """
    Histogram of predicted probabilities, split by GT class.
    Useful for diagnosing calibration and class collapse.
    """
    pos_scores = all_scores[all_targets == 1]
    neg_scores = all_scores[all_targets == 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 51)
    ax.hist(neg_scores, bins=bins, alpha=0.55, color="#4472C4", label="GT=0 (background)", density=True)
    ax.hist(pos_scores, bins=bins, alpha=0.75, color="#FF4B4B", label="GT=1 (fire)", density=True)
    ax.axvline(0.5, ls="--", lw=1.5, color="#333", label="threshold=0.5")
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Score Distribution by Class — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    out_path = output_dir / f"{model_name}_score_histogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# W&B logger
# ---------------------------------------------------------------------------

def _log_to_wandb(sample_grid_path, pr_curve_path, histogram_path, model_name: str):
    try:
        import wandb
        if wandb.run is None:
            print("[visualize] W&B run not active — skipping W&B logging.")
            return
        wandb.log({
            f"viz/{model_name}/sample_predictions": wandb.Image(str(sample_grid_path)),
            f"viz/{model_name}/pr_curve":           wandb.Image(str(pr_curve_path)),
            f"viz/{model_name}/score_histogram":    wandb.Image(str(histogram_path)),
        })
        print(f"[visualize] Logged figures to W&B under viz/{model_name}/")
    except ImportError:
        print("[visualize] wandb not installed — skipping W&B logging.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_visualization(
    model,
    datamodule,
    output_dir: str,
    model_name: str = "model",
    n_samples: int = 8,
    threshold: float = 0.5,
    wandb_log: bool = True,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Generate and save prediction visualizations for a trained discriminative model.

    Parameters
    ----------
    model : BaseModel subclass with weights already loaded.
    datamodule : FireSpreadDataModule, configured with the correct data splits.
    output_dir : Directory where PNG files are written.
    model_name : Used in file names and W&B keys (e.g. "SMPModel", "ConvLSTM").
    n_samples : How many test samples to include in the sample grid.
    threshold : Binary decision threshold (default 0.5).
    wandb_log : Whether to push figures to the active W&B run.
    device : Inference device. Defaults to cuda if available.

    Returns
    -------
    dict with keys: sample_grid, pr_curve, score_histogram (paths to saved PNGs)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    print(f"\n[visualize] Running visualization for {model_name} on {device} ...")

    # ── 1. Collect samples & scores ──────────────────────────────────────────
    fire_inputs, prob_maps, pred_masks, gt_masks, all_targets, all_scores = \
        _collect_samples(model, datamodule, device, n_samples, threshold)

    n_collected = len(fire_inputs)
    if n_collected == 0:
        print("[visualize] No samples collected — check that datamodule.setup('test') works.")
        return {}

    print(f"[visualize] Collected {n_collected} samples for visualization.")
    ap = average_precision_score(all_targets, all_scores)
    print(f"[visualize] AP over full test set: {ap:.4f}  (persistence baseline ≈ 0.193)")

    # ── 2. Build figures ──────────────────────────────────────────────────────
    grid_path  = _plot_sample_grid(fire_inputs, prob_maps, pred_masks, gt_masks,
                                   model_name, output_dir)
    pr_path    = _plot_pr_curve(all_targets, all_scores, model_name, output_dir)
    hist_path  = _plot_score_histogram(all_scores, all_targets, model_name, output_dir)

    print(f"[visualize] Saved: {grid_path.name}, {pr_path.name}, {hist_path.name}")

    # ── 3. W&B ───────────────────────────────────────────────────────────────
    if wandb_log:
        _log_to_wandb(grid_path, pr_path, hist_path, model_name)

    return {
        "sample_grid":     str(grid_path),
        "pr_curve":        str(pr_path),
        "score_histogram": str(hist_path),
    }


# ---------------------------------------------------------------------------
# Standalone CLI (for calling after training, outside of train.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Add src/ to path so imports resolve the same way as train.py
    src_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(src_dir))

    parser = argparse.ArgumentParser(
        description="Post-training visualization for WildfireSpreadBench discriminative models."
    )
    parser.add_argument("--config",      required=True,
                        help="Model YAML config (e.g. cfgs/unet/res18_monotemporal.yaml)")
    parser.add_argument("--data_config", required=True,
                        help="Data YAML config (e.g. cfgs/data_monotemporal_full_features.yaml)")
    parser.add_argument("--ckpt_path",   required=True,
                        help="Path to .ckpt file to load weights from")
    parser.add_argument("--output_dir",  default="./viz_output",
                        help="Directory for output PNGs (default: ./viz_output)")
    parser.add_argument("--n_samples",   type=int, default=8,
                        help="Number of samples to include in the prediction grid (default: 8)")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Binary decision threshold (default: 0.5)")
    parser.add_argument("--wandb_log",   action="store_true",
                        help="Push figures to the active W&B run")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="W&B run ID to resume (optional)")
    args = parser.parse_args()

    # ── Lazy imports after path setup ────────────────────────────────────────
    import yaml
    from pytorch_lightning.cli import LightningArgumentParser
    from dataloader.FireSpreadDataModule import FireSpreadDataModule
    from dataloader.FireSpreadDataset import FireSpreadDataset
    from dataloader.utils import get_means_stds_missing_values
    from models import SMPModel, BaseModel, ConvLSTMLightning, LogisticRegression  # noqa

    # ── Parse configs ─────────────────────────────────────────────────────────
    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    # ── Resolve n_channels (same logic as train.py before_instantiate_classes) ──
    n_leading = data_cfg.get("n_leading_observations", 1)
    features  = data_cfg.get("features_to_keep", None)
    rm_dup    = data_cfg.get("remove_duplicate_features", False)
    n_channels = FireSpreadDataset.get_n_features(n_leading, features, rm_dup)
    model_cfg["model"]["init_args"]["n_channels"] = n_channels

    # Compute pos_class_weight
    train_years, _, _ = FireSpreadDataModule.split_fires(data_cfg.get("data_fold_id", 0))
    _, _, missing_values_rates = get_means_stds_missing_values(train_years)
    fire_rate = 1 - missing_values_rates[-1]
    model_cfg["model"]["init_args"]["pos_class_weight"] = float(1 / fire_rate)

    # ── Instantiate model from config ─────────────────────────────────────────
    import importlib
    class_path = model_cfg["model"]["class_path"]           # e.g. "models.SMPModel"
    module_name, class_name = class_path.rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_name), class_name)
    model = ModelClass.load_from_checkpoint(
        args.ckpt_path,
        **{k: v for k, v in model_cfg["model"]["init_args"].items()
           if k not in ("n_channels", "pos_class_weight")},  # already set above
        n_channels=n_channels,
        pos_class_weight=float(1 / fire_rate),
    )

    # ── Instantiate datamodule ────────────────────────────────────────────────
    datamodule = FireSpreadDataModule(**data_cfg)

    # ── Optional W&B init ─────────────────────────────────────────────────────
    if args.wandb_log:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "WildfireSpreadBench"),
            entity=os.environ.get("WANDB_ENTITY", "ram-algoverse"),
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    model_name = class_name
    run_visualization(
        model=model,
        datamodule=datamodule,
        output_dir=args.output_dir,
        model_name=model_name,
        n_samples=args.n_samples,
        threshold=args.threshold,
        wandb_log=args.wandb_log,
    )

    if args.wandb_log:
        import wandb
        wandb.finish()
