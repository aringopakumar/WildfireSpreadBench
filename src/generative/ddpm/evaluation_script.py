"""
src/generative/ddpm/evaluation_script.py

Standalone DDPM evaluation. Rewritten to use the SAME data path as training
(FireSpreadDataset), so eval inputs are standardized identically to training.

Previous version used a hand-rolled dataset that:
  - globbed "*.h5" (real files are ".hdf5")  -> empty loader
  - read raw f['imagery'] with .tif channel indices, unstandardized
  - hardcoded cond_ch=6 while training uses 7 (vegetation)
All of that made standalone eval numbers inconsistent with training. Fixed.
"""

import os, sys, argparse, torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddpm.unet import Unet
from ddpm.diffusion_proc import Diffusion

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataloader.FireSpreadDataset import FireSpreadDataset
from evaluation.unified_eval import evaluate_ddpm

import wandb

FEATURE_SETS = {
    "vegetation": [0, 1, 2, 3, 4, 38, 39],
    "multi": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
              27, 28, 29, 30, 31, 32, 38, 39],
    "all": None,
}
CHANNELS = {"vegetation": 7, "multi": 34, "all": 40}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, required=True,
                        help="Root of HDF5 dataset (contains 2018/, 2019/, ...)")
    parser.add_argument("--ckpt_dir",  type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="fire_ddpm_last.pt")
    parser.add_argument("--feature_set", type=str, default="vegetation",
                        choices=["vegetation", "multi", "all"])
    args = parser.parse_args()

    features_to_keep = FEATURE_SETS[args.feature_set]
    IN_CHANNELS      = CHANNELS[args.feature_set]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(entity="ram-algoverse", project="WildfireSpreadBench", config={
        "task": "evaluation",
        "model": f"DDPM (Unet, cond_ch={IN_CHANNELS}, model_ch=128, CFG)",
        "feature_set": args.feature_set,
    })

    # Same stats_years as training; 2-year combo that exists in the stats table.
    test_dataset = FireSpreadDataset(
        data_dir=args.data_dir,
        included_fire_years=[2021],
        n_leading_observations=1,
        n_leading_observations_test_adjustment=5,
        crop_side_length=128,
        load_from_hdf5=True,
        is_train=False,
        remove_duplicate_features=False,
        features_to_keep=features_to_keep,
        stats_years=[2018, 2019],
    )
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Unet(in_ch=1, cond_ch=IN_CHANNELS, output_ch=1,
                 model_ch=128, channel_mult=(1, 2, 2, 4)).to(device)
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    diffusion = Diffusion(timesteps=1000, noise_schedule="sigmoid")

    # NOTE: evaluate_ddpm's predict_fn must squeeze the time dim, since
    # FireSpreadDataset yields x of shape [B, T, C, H, W]. The wrapper in
    # unified_eval.evaluate_ddpm does NOT do this. See Bug 4 fix below.
    evaluate_ddpm(model, diffusion, loader, device, epoch=None, wandb_log=True)
    wandb.finish()
