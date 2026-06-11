"""
src/generative/ddpm/visualize.py

Visualize SDF-DDPM next-day fire predictions vs. ground truth.

Rewritten for the actual dataset format and the SDF-DDPM pipeline:

- DATA: loads through FireSpreadDataset, i.e. the SAME .hdf5 files
  (one file per fire, "data" array of shape [T, 23, H, W]) and the SAME
  preprocessing (fixed 2018+2019 standardization, vegetation channel subset
  [0,1,2,3,4,38,39], binary active-fire channel) that the model trains and
  evaluates on. The previous version expected per-day .h5 files with an
  "imagery" key and skipped all preprocessing, so it found zero files on
  this dataset and did not represent what the model actually sees.
- MODEL/SAMPLER: loads the training checkpoint (dict with model weights AND
  the SDF normalization stats), rebuilds Diffusion with the stored
  sdf_mean/sdf_std/sdf_trunc, and recovers predictions via
  diffusion.sample_to_prob. The raw sampler output is a normalized SDF in
  roughly [-9, +0.1], NOT a [0,1] image, so it must go through this recovery.
- SIZE: scenes are center-cropped to 128x128 (the training crop size).
  Full-scene sampling is intractable: 1000 reverse steps x 2 CFG forwards
  per image, with cross-attention compute quadratic in pixel count.

Output: a grid (one row per example) with columns
  [today's fire (input) | predicted probability | predicted mask | ground truth]
saved to <ckpt_dir>/ddpm_predictions.png.

Run from the repo root:
  PYTHONPATH=. python src/generative/ddpm/visualize.py \
      --data_dir ~/RAM/data/hdf5_new --ckpt_dir ckpts/ddpm_vegetation
"""

import os, sys, argparse, contextlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

sys.path.insert(0, 'src')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddpm.unet import Unet
from ddpm.diffusion_proc import Diffusion, SDF_TRUNC
from dataloader.FireSpreadDataset import FireSpreadDataset

FEATURE_SETS = {
    "vegetation": [0, 1, 2, 3, 4, 38, 39],
    "multi": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
              27, 28, 29, 30, 31, 32, 38, 39],
    "all": None,
}
CHANNELS = {"vegetation": 7, "multi": 34, "all": 40}

TEST_YEARS  = [2021]
STATS_YEARS = [2018, 2019]


def load_checkpoint(ckpt_dir, ckpt_name, device):
    """Load model weights + SDF stats.

    fire_ddpm_last.pt is a dict holding both the weights and the SDF stats.
    fire_ddpm_best.pt is weights-only; in that case the stats are pulled from
    the companion fire_ddpm_last.pt (training always stores them there).
    Returns (state_dict, sdf_mean, sdf_std, sdf_trunc).
    """
    path = os.path.join(ckpt_dir, ckpt_name)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return (ckpt["model"], float(ckpt["sdf_mean"]), float(ckpt["sdf_std"]),
                float(ckpt.get("sdf_trunc", SDF_TRUNC)))

    last = os.path.join(ckpt_dir, "fire_ddpm_last.pt")
    if not os.path.exists(last):
        raise FileNotFoundError(
            f"{ckpt_name} contains no SDF stats and {last} was not found. "
            "Point --ckpt_name at fire_ddpm_last.pt instead.")
    meta = torch.load(last, map_location="cpu")
    return (ckpt, float(meta["sdf_mean"]), float(meta["sdf_std"]),
            float(meta.get("sdf_trunc", SDF_TRUNC)))


def find_examples(dataset, crop, num_examples, min_fire_pixels):
    """Center-crop test scenes and keep crops that actually contain fire
    (in the target, or failing that the input), so the panels are meaningful.
    Center crops of fire-centered scenes usually qualify within a few items.
    """
    found = []
    for i in range(len(dataset)):
        x, y = dataset[i]                                  # x: [T,C,H,W], y: [H,W]
        x = TF.center_crop(x, [crop, crop])
        y = TF.center_crop(y.unsqueeze(0), [crop, crop]).squeeze(0)
        target_fire = int(y.sum().item())
        input_fire  = int(x[0, -1].sum().item())           # last channel = binary AF
        if target_fire >= min_fire_pixels or input_fire >= min_fire_pixels:
            found.append((x, y))
            print(f"  example {len(found)}: dataset idx {i} "
                  f"(input fire px={input_fire}, target fire px={target_fire})")
            if len(found) >= num_examples:
                break
    return found


@torch.no_grad()
def visualize(model, diffusion, examples, device, guidance_w, save_path):
    xs = torch.stack([x[0] for x, _ in examples]).to(device)    # [N, C, H, W]
    ys = torch.stack([y for _, y in examples]).float()          # [N, H, W]

    print(f"Sampling {len(examples)} predictions "
          f"({diffusion.timesteps} reverse steps x 2 CFG forwards each)...")
    amp = (torch.amp.autocast("cuda") if device.type == "cuda"
           else contextlib.nullcontext())
    with amp:
        prob = diffusion.sample_to_prob(model, xs, w=guidance_w, progress=True)
    prob = prob.float().cpu()                                   # [N, 1, H, W]
    # prob = sigmoid(-sdf_pixels / 2), so prob >= 0.5 <=> sdf <= 0,
    # i.e. exactly the SDF zero-level-set mask -- no need to resample.
    pred_mask = (prob >= 0.5).float()

    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n), squeeze=False)
    fig.suptitle(f"SDF-DDPM next-day fire predictions (guidance w={guidance_w})",
                 fontsize=14)
    titles = ["Today's fire (input)", "Predicted probability",
              "Predicted mask", "Ground truth (tomorrow)"]
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=12)
    for r in range(n):
        axes[r, 0].imshow(xs[r, -1].cpu().numpy(), cmap="magma", vmin=0, vmax=1)
        axes[r, 1].imshow(prob[r, 0].numpy(),      cmap="magma", vmin=0, vmax=1)
        axes[r, 2].imshow(pred_mask[r, 0].numpy(), cmap="magma", vmin=0, vmax=1)
        axes[r, 3].imshow(ys[r].numpy(),           cmap="magma", vmin=0, vmax=1)
        for c in range(4):
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {os.path.abspath(save_path)}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_to_keep = FEATURE_SETS[args.feature_set]
    in_channels      = CHANNELS[args.feature_set]

    print("Loading checkpoint...")
    state, sdf_mean, sdf_std, sdf_trunc = load_checkpoint(
        args.ckpt_dir, args.ckpt_name, device)
    print(f"SDF stats from checkpoint: mean={sdf_mean:.4f} std={sdf_std:.4f} "
          f"trunc={sdf_trunc} | recovery threshold={-sdf_mean/sdf_std:.4f}")

    model = Unet(in_ch=1, cond_ch=in_channels, output_ch=1,
                 model_ch=128, channel_mult=(1, 2, 2, 4)).to(device)
    model.load_state_dict(state)
    model.eval()

    diffusion = Diffusion(timesteps=1000, noise_schedule="sigmoid",
                          sdf_mean=sdf_mean, sdf_std=sdf_std, sdf_trunc=sdf_trunc)

    print("Loading test data (2021) through FireSpreadDataset...")
    test_dataset = FireSpreadDataset(
        data_dir=args.data_dir, included_fire_years=TEST_YEARS,
        n_leading_observations=1, n_leading_observations_test_adjustment=5,
        crop_side_length=128, load_from_hdf5=True, is_train=False,
        remove_duplicate_features=False, features_to_keep=features_to_keep,
        stats_years=STATS_YEARS,
    )

    print(f"Scanning for {args.num_examples} crops containing fire...")
    examples = find_examples(test_dataset, args.crop, args.num_examples,
                             min_fire_pixels=args.min_fire_pixels)
    if not examples:
        raise RuntimeError("No crops with enough fire pixels were found. "
                           "Lower --min_fire_pixels and try again.")

    save_path = os.path.join(args.ckpt_dir, "ddpm_predictions.png")
    visualize(model, diffusion, examples, device, args.guidance_w, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, required=True,
                        help="Root of the HDF5 dataset (contains 2018/ 2019/ 2021/ ...)")
    parser.add_argument("--ckpt_dir",  type=str, required=True,
                        help="Directory containing model checkpoints")
    parser.add_argument("--ckpt_name", type=str, default="fire_ddpm_last.pt",
                        help="Checkpoint filename (fire_ddpm_last.pt has the SDF stats)")
    parser.add_argument("--feature_set", type=str, default="vegetation",
                        choices=["vegetation", "multi", "all"])
    parser.add_argument("--num_examples", type=int, default=4,
                        help="Rows in the output figure")
    parser.add_argument("--crop", type=int, default=128,
                        help="Center-crop size (128 = training crop size)")
    parser.add_argument("--guidance_w", type=float, default=2.0,
                        help="Classifier-free guidance weight (match training/eval)")
    parser.add_argument("--min_fire_pixels", type=int, default=20,
                        help="Minimum fire pixels for a crop to be shown")
    main(parser.parse_args())
