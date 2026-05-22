"""
src/train_generative.py

Single entry point for training all generative models.

Usage:
  python src/train_generative.py --model unet_bce --data_dir /path/to/hdf5 --ckpt_dir /path/to/ckpts --feature_set vegetation --epochs 200 --eval_every 10
  python src/train_generative.py --model flow --data_dir /path/to/hdf5 --ckpt_dir /path/to/ckpts --feature_set vegetation --epochs 200 --eval_every 10
  python src/train_generative.py --model ddpm --data_dir /path/to/hdf5 --ckpt_dir /path/to/ckpts --feature_set vegetation --epochs 200 --eval_every 10
"""

import argparse, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["unet_bce", "flow", "ddpm"])
    parser.add_argument("--data_dir",    type=str, required=True)
    parser.add_argument("--ckpt_dir",    type=str, required=True)
    parser.add_argument("--feature_set", type=str, default="vegetation",
                        choices=["vegetation", "multi", "all"])
    parser.add_argument("--epochs",      type=int, default=200)
    parser.add_argument("--eval_every",  type=int, default=10)
    args = parser.parse_args()

    # Build sys.argv for the child module's argparse
    sys.argv = [
        args.model,
        "--data_dir",    args.data_dir,
        "--ckpt_dir",    args.ckpt_dir,
        "--feature_set", args.feature_set,
        "--epochs",      str(args.epochs),
        "--eval_every",  str(args.eval_every),
    ]

    if args.model == "unet_bce":
        import src.generative.unet_bce as m
    elif args.model == "flow":
        import src.generative.flow_matching as m
    elif args.model == "ddpm":
        import src.generative.ddpm.training_script as m

if __name__ == "__main__":
    main()
