# Generative Models

This directory contains the generative model implementations
for WildfireSpreadBench:

- unet_bce.py        — BCE segmentation U-Net (discriminative)
- flow_matching.py   — Conditional Flow Matching (generative)

Both use FireSpreadDataset from src/dataloader/ for data loading
and share the unified evaluation harness at
src/evaluation/unified_eval.py.

Run via: python src/train_generative.py --model [unet_bce|flow]
