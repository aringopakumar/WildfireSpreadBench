"""Classify a folder of Lightning .ckpt files by model architecture.

Reads each checkpoint on CPU, inspects state_dict key prefixes and saved
hyper_parameters, and prints a one-line summary so we can pick the right
ConvLSTM checkpoint out of a folder that mixes several models.
"""

from __future__ import annotations

import argparse
import glob
import os

import torch


def detect_arch(state_dict_keys: list[str], hparams: dict) -> str:
    keys = " ".join(state_dict_keys[:200]).lower()
    if "convlstm" in keys:
        return "convlstm"
    if "temporal_encoder" in keys or "up_blocks" in keys or "down_blocks" in keys:
        return "utae"
    if "segmentation_head" in keys or ("encoder" in keys and "decoder" in keys):
        return "unet_smp"
    if hparams.get("required_img_size") is not None:
        return "convlstm?"  # ConvLSTM is the only model here that sets this
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.ckpt")))
    print(f"Found {len(paths)} checkpoints in {args.dir}\n")
    rows = []
    for p in paths:
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(p):45s}  LOAD FAILED: {e}")
            continue
        sd = ck.get("state_dict", {})
        hp = ck.get("hyper_parameters", {}) or {}
        arch = detect_arch(list(sd.keys()), hp)
        n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        epoch = ck.get("epoch", "?")
        # Pull the monitored val_loss if Lightning saved it in callbacks.
        val_loss = None
        for cb_state in (ck.get("callbacks", {}) or {}).values():
            if isinstance(cb_state, dict) and "best_model_score" in cb_state:
                bms = cb_state.get("best_model_score")
                if bms is not None:
                    val_loss = float(bms)
        rows.append((arch, os.path.basename(p), n_params, epoch, val_loss,
                     hp.get("n_channels"), hp.get("loss_function"),
                     hp.get("required_img_size")))

    rows.sort(key=lambda r: (r[0], r[1]))
    print(f"{'arch':10s} {'file':42s} {'params':>10s} {'ep':>4s} {'val':>8s} "
          f"{'nch':>4s} {'loss':>8s}  req_size")
    print("-" * 110)
    for arch, name, np_, ep, vl, nch, loss, req in rows:
        vl_s = f"{vl:.4f}" if isinstance(vl, float) else "  -  "
        print(f"{arch:10s} {name:42s} {np_:>10d} {str(ep):>4s} {vl_s:>8s} "
              f"{str(nch):>4s} {str(loss):>8s}  {req}")


if __name__ == "__main__":
    main()
