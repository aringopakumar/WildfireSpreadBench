"""
Save side-by-side prediction maps for a trained WildfireSpreadTS model.

Supports UTAE and ConvLSTM checkpoints via --model.

Example (UTAE vegetation-only checkpoint):

  python src/evaluation/visualize_predictions.py ^
    --model utae ^
    --ckpt C:/data/wildfire_runs/checkpoints_utae_vegetation/epoch=46-val_loss=1.7412.ckpt ^
    --data_dir C:/data/newHDF5Data ^
    --out_dir C:/data/wildfire_runs/viz_utae_vegetation ^
    --features_to_keep 0,1,2,3,4,38,39 ^
    --n_samples 24

Example (ConvLSTM full-features checkpoint):

  python src/evaluation/visualize_predictions.py ^
    --model convlstm ^
    --ckpt C:/data/wildfire_runs/checkpoints_convlstm/last.ckpt ^
    --data_dir C:/data/newHDF5Data ^
    --out_dir C:/data/wildfire_runs/viz_convlstm ^
    --n_samples 24
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

# Allow running as `python src/evaluation/visualize_predictions.py` from repo root.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataloader.FireSpreadDataModule import FireSpreadDataModule  # noqa: E402
from baselines.UTAELightning import UTAELightning  # noqa: E402
from baselines.ConvLSTMLightning import ConvLSTMLightning  # noqa: E402

# Registry of supported model architectures. Maps the --model choice to its
# Lightning class. get_pred_and_gt handles doy/cropping internally based on the
# checkpoint's saved hparams, so the rest of the script stays model-agnostic.
MODEL_REGISTRY = {
    "utae": UTAELightning,
    "convlstm": ConvLSTMLightning,
}


def parse_features(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "" or raw.lower() == "none":
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize wildfire model predictions as PNG maps.")
    p.add_argument(
        "--model",
        type=str,
        default="utae",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model architecture of the checkpoint.",
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to Lightning .ckpt")
    p.add_argument("--data_dir", type=str, required=True, help="HDF5 dataset directory")
    p.add_argument("--out_dir", type=str, required=True, help="Directory for PNG outputs")
    p.add_argument(
        "--features_to_keep",
        type=str,
        default=None,
        help="Comma-separated channel indices, or 'none' for all features. "
        "Default (unset): 'none' (all 40 features) for convlstm, "
        "'0,1,2,3,4,38,39' (vegetation subset) for utae. Must match training.",
    )
    p.add_argument("--n_leading_observations", type=int, default=5)
    p.add_argument("--crop_side_length", type=int, default=128)
    p.add_argument("--data_fold_id", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=24, help="Number of test samples to save")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binary decision threshold on sigmoid probs (strict >). "
        "Default (unset): 0.99 for utae, 0.5 for convlstm. "
        "UTAE's BCE pos_weight training leaves background near 0.5, so 0.5-0.9 "
        "still over-predicts heavily; ConvLSTM (Jaccard loss) separates cleanly "
        "around 0.5.",
    )
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    p.add_argument(
        "--skip_empty",
        action="store_true",
        help="Skip samples with no fire pixels in target (often more informative gallery)",
    )
    p.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated test-set indices to render directly (e.g. the "
        "'large fires' set 827,883,1019,1020,1031,1440). When set, only these "
        "samples are produced, named large_tgt_idxNNNN.png, and --n_samples / "
        "--skip_empty are ignored.",
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Comma-separated thresholds for a sweep, e.g. "
        "'0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9'. When set, each figure shows one "
        "binary panel per threshold (instead of a single-threshold binary panel).",
    )
    return p.parse_args()


def infer_convlstm_kwargs(ckpt_path: str) -> dict:
    """ConvLSTM's constructor args (img_height_width, kernel_size, num_layers)
    are not saved in the checkpoint hparams (BaseModel only persists a fixed
    subset), so load_from_checkpoint can't rebuild the model. Recover them from
    the state_dict shapes and required_img_size instead.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", {})
    hp = ck.get("hyper_parameters", {}) or {}

    layer_idxs: set[int] = set()
    kernel_hw = None
    for k, v in sd.items():
        m = re.search(r"cell_list\.(\d+)\.conv\.weight", k)
        if m:
            layer_idxs.add(int(m.group(1)))
            kernel_hw = tuple(int(s) for s in v.shape[-2:])
    num_layers = (max(layer_idxs) + 1) if layer_idxs else 1
    kernel_size = list(kernel_hw) if kernel_hw else [3, 3]

    req = hp.get("required_img_size")
    img_hw = [int(req[0]), int(req[1])] if req is not None else [128, 128]

    return {
        "img_height_width": img_hw,
        "kernel_size": kernel_size,
        "num_layers": num_layers,
    }


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_figure(
    fire_in: np.ndarray,
    prob: np.ndarray,
    binary: np.ndarray,
    target: np.ndarray,
    title: str,
) -> plt.Figure:
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2))
    # Binary / fire masks: dark background so all-white predictions stay visible.
    mask_cmap = ListedColormap(["#111111", "#ffcc33"])
    panels = [
        (fire_in, "Input fire (t)", mask_cmap, 0, 1),
        (prob, "Prediction (prob)", "inferno", 0, 1),
        (binary, "Prediction (binary)", mask_cmap, 0, 1),
        (target, "Target (t+1)", mask_cmap, 0, 1),
    ]
    for ax, (img, panel_title, cmap, vmin, vmax) in zip(axs, panels):
        ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#444444")
            spine.set_linewidth(0.8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def make_threshold_figure(
    fire_in: np.ndarray,
    prob: np.ndarray,
    target: np.ndarray,
    thresholds: list[float],
    title: str,
) -> plt.Figure:
    """Large-fire figure: fixed reference panels (input / prob / target) plus one
    binary panel per threshold. Uses the same colormaps as make_figure so the
    output matches the other models' galleries.
    """
    # Same palette as the standard gallery: dark background, yellow fire mask,
    # inferno for the probability heatmap.
    mask_cmap = ListedColormap(["#111111", "#ffcc33"])

    panels = [
        (fire_in, "Input fire (t)", mask_cmap, 0, 1),
        (prob, "Prediction (prob)", "inferno", 0, 1),
        (target, "Target (t+1)", mask_cmap, 0, 1),
    ]
    for thr in thresholds:
        binary = (prob > thr).astype(np.float32)
        panels.append(
            (binary, f"binary @ {thr:.2f}\npred={binary.sum():.0f} px", mask_cmap, 0, 1)
        )

    cols = 5
    rows = int(np.ceil(len(panels) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.7 * cols, 3.4 * rows))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.axis("off")
    for ax, (img, panel_title, cmap, vmin, vmax) in zip(axs, panels):
        ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
        ax.set_title(panel_title, fontsize=9)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#444444")
            spine.set_linewidth(0.8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def predict_sample(model, dataset, idx: int, device: torch.device):
    """Run the model on a single test-set sample (by index) and return the
    numpy (fire_in, prob, target) triplet. Mirrors the collate the dataloader
    would do (add batch dim), so get_pred_and_gt's doy/cropping logic applies.
    """
    sample = dataset[idx]
    tensors = []
    for j, s in enumerate(sample):
        t = torch.as_tensor(np.asarray(s)).unsqueeze(0).to(device)
        if j < 2:  # x and y to float; doys (if present) left as-is
            t = t.float()
        tensors.append(t)
    batch = tuple(tensors)

    with torch.no_grad():
        y_hat, y = model.get_pred_and_gt(batch)

    fire_in = batch[0][0, -1, -1].detach().float().cpu().numpy()
    prob = torch.sigmoid(y_hat[0].detach().float().cpu()).numpy()
    target = y[0].detach().float().cpu().numpy()
    return fire_in, prob, target


def build_contact_sheet(pngs: list[Path], out_path: Path, cols: int = 2) -> None:
    if not pngs:
        return
    n = len(pngs)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(7.5 * cols, 2.6 * rows))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.axis("off")
    for ax, path in zip(axs, pngs):
        ax.imshow(plt.imread(path))
        ax.set_title(path.name, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_large_fires(model, dataset, indices, thresholds, out_dir, device) -> None:
    """Render specific test indices (the 'large fires' set) with a threshold sweep."""
    for idx in indices:
        fire_in, prob, target = predict_sample(model, dataset, idx, device)
        title = (
            f"test_idx={idx} | fire_in={fire_in.sum():.0f} px | "
            f"tgt={target.sum():.0f} px"
        )
        fig = make_threshold_figure(fire_in, prob, target, thresholds, title)
        out_path = out_dir / f"large_tgt_idx{idx:04d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")

    pngs = sorted(out_dir.glob("large_tgt_idx*.png"))
    build_contact_sheet(pngs, out_dir / "large_fires_contact_sheet.png", cols=2)


def main() -> None:
    args = parse_args()

    # Model-aware defaults for features and threshold when the user leaves them unset.
    if args.features_to_keep is None:
        args.features_to_keep = "none" if args.model == "convlstm" else "0,1,2,3,4,38,39"
    if args.threshold is None:
        args.threshold = 0.5 if args.model == "convlstm" else 0.99

    features = parse_features(args.features_to_keep)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {args.model} | features: {args.features_to_keep} | threshold: {args.threshold}")
    print(f"Loading checkpoint: {args.ckpt}")
    model_cls = MODEL_REGISTRY[args.model]
    # ConvLSTM needs constructor args rebuilt from the checkpoint (not in hparams).
    extra_kwargs = infer_convlstm_kwargs(args.ckpt) if args.model == "convlstm" else {}
    if extra_kwargs:
        print(f"Inferred ConvLSTM args: {extra_kwargs}")
    model = model_cls.load_from_checkpoint(args.ckpt, map_location=device, **extra_kwargs)
    model.eval()
    model.to(device)

    # The datamodule must return day-of-year iff the model consumes it (UTAE does,
    # ConvLSTM does not). Deriving this from the checkpoint's saved hparams keeps
    # get_pred_and_gt's (x, y[, doys]) unpacking in sync with the batch shape.
    return_doy = bool(model.hparams.get("use_doy", False))
    print(f"return_doy (from checkpoint hparams.use_doy): {return_doy}")

    dm = FireSpreadDataModule(
        data_dir=args.data_dir,
        batch_size=1,
        n_leading_observations=args.n_leading_observations,
        n_leading_observations_test_adjustment=args.n_leading_observations,
        crop_side_length=args.crop_side_length,
        load_from_hdf5=True,
        num_workers=0,
        remove_duplicate_features=False,
        features_to_keep=features,
        return_doy=return_doy,
        data_fold_id=args.data_fold_id,
    )
    dm.setup("test")
    print(f"Test set size: {len(dm.test_dataset)}")

    # Large-fire + threshold-sweep mode: render specific indices directly.
    if args.indices:
        indices = [int(i) for i in args.indices.split(",") if i.strip() != ""]
        thr_raw = args.thresholds or "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
        thresholds = [float(t) for t in thr_raw.split(",") if t.strip() != ""]
        print(f"Large-fire indices: {indices}")
        print(f"Thresholds: {thresholds}")
        run_large_fires(model, dm.test_dataset, indices, thresholds, out_dir, device)
        print(f"Done. -> {out_dir}")
        return

    loader = dm.test_dataloader()

    saved = 0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if saved >= args.n_samples:
                break
            seen += 1
            batch = tuple(t.to(device) for t in batch)
            x = batch[0]
            y_hat, y = model.get_pred_and_gt(batch)

            # Last day, last channel = binary active fire after vegetation subset.
            fire_in = x[0, -1, -1].detach().float().cpu().numpy()
            logits = y_hat[0].detach().float().cpu()
            prob = torch.sigmoid(logits).numpy()
            target = y[0].detach().float().cpu().numpy()
            # Strict > so background logits==0 (sigmoid==0.5) are not counted as fire.
            # UTAE's final head often floors negatives at 0, so >=0.5 lights up the whole map.
            binary = (prob > args.threshold).astype(np.float32)

            if args.skip_empty and target.sum() < 1:
                continue

            title = (
                f"sample {saved:04d} | fire_in={fire_in.sum():.0f} px | "
                f"pred={binary.sum():.0f} px | tgt={target.sum():.0f} px"
            )
            fig = make_figure(fire_in, prob, binary, target, title)
            out_path = out_dir / f"sample_{saved:04d}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out_path}")
            saved += 1

    # Also write a simple contact sheet of the first few saved images for quick review.
    pngs = sorted(out_dir.glob("sample_*.png"))[:12]
    if pngs:
        n = len(pngs)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.4 * rows))
        axs = np.atleast_1d(axs).ravel()
        for ax in axs:
            ax.axis("off")
        for ax, path in zip(axs, pngs):
            ax.imshow(plt.imread(path))
            ax.set_title(path.name, fontsize=8)
            ax.axis("off")
        sheet = out_dir / "contact_sheet.png"
        fig.tight_layout()
        fig.savefig(sheet, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {sheet}")

    print(f"Done. Saved {saved} maps (scanned {seen} test batches) -> {out_dir}")


if __name__ == "__main__":
    # Avoid OpenMP / HDF5 worker contention noise on Windows.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    main()
