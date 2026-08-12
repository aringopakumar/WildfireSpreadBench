"""
src/train_generative.py

Single entry point for training all generative models.

Usage:
  python src/train_generative.py --model unet_bce --data_dir /path/to/hdf5 --ckpt_dir /path/to/ckpts --feature_set vegetation --epochs 200 --eval_every 10
  python src/train_generative.py --model flow    --data_dir /path/to/hdf5 --ckpt_dir /path/to/ckpts --feature_set vegetation --epochs 200 --eval_every 10

Note: each child module exposes a `main(args)` function. This dispatcher parses
the shared CLI and calls it directly, rather than relying on the child's
`__main__` block (which does not run on import).
"""

import argparse
import os
import sys

# The generative modules import each other relative to src/ (e.g.
# "from dataloader...", "from evaluation..."), so put src/ on the path
# regardless of the working directory this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["unet_bce", "flow"])
    parser.add_argument("--data_dir",    type=str, required=True)
    parser.add_argument("--ckpt_dir",    type=str, required=True)
    parser.add_argument("--feature_set", type=str, default="vegetation",
                        choices=["vegetation", "multi", "all"])
    parser.add_argument("--epochs",      type=int, default=200)
    parser.add_argument("--eval_every",  type=int, default=10)
    args = parser.parse_args()

    if args.model == "unet_bce":
        from generative.unet_bce import main as run
    elif args.model == "flow":
        from generative.flow_matching import main as run

    run(args)


if __name__ == "__main__":
    main()
