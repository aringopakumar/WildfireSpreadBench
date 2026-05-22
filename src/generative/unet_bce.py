"""
Wildfire next-day spread prediction as pure conditional binary segmentation.

Replaces the prior flow-matching setup. The old pipeline had:
  - A velocity head supervised by weighted MSE.
  - An auxiliary BCE head supervised by class-weighted BCE.
  - 20-step Euler integration at inference that computed `x`, then threw it
    away and read probabilities off the BCE head.

Because the flow target was a t-independent constant (`x1 - x0_fire`) with no
noise, the velocity regression provided no generative benefit — only extra
load on the shared backbone. The predictions being scored came entirely
from the BCE head.

This file strips the flow machinery and keeps the backbone. One loss, one
head, one forward pass at inference. All gradient signal now goes into the
classifier that actually produces the AP score.

Changes vs. flow_matching.py:
  - `VectorFieldNet` → `FireSegmentationUNet`; no timestep embedding, no t_proj
    in ResBlocks, `enc1` takes in_channels instead of 1+in_channels, single
    output head (no velocity head).
  - `flow_matching_loss` → `segmentation_loss` (plain BCE with pos_weight).
  - `integrate_flow` → `predict` (one forward pass).
  - `visualize_predictions` now calls `predict` and takes `ckpt_dir` as an
    argument (was hardcoded). Renders in grayscale.
  - Checkpoint filenames: wildfire_flow_* → wildfire_seg_*.

Datasets, augmentations, optimizer, scheduler, W&B project, and
hyperparameters are unchanged.
"""

import os, sys, argparse, time, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import wandb

sys.path.insert(0, 'src')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.FireSpreadDataset import FireSpreadDataset
from evaluation.unified_eval import evaluate_model

# ---------------------------------------------------------------------------
# Constants (default; overridden in __main__ based on --feature_set)
# ---------------------------------------------------------------------------
IN_CHANNELS = 7


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """
    Residual block, pre-activation style. No timestep conditioning — this is a
    plain segmentation backbone, not a flow model.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class FireSegmentationUNet(nn.Module):
    """
    Conditional U-Net for next-day fire segmentation.

    Layout (base_channels C = 96):
      Encoder: 3 levels, C → 2C → 4C, with strided-conv downsampling
      Bottleneck: 4C with two ResBlocks
      Decoder:  mirrors encoder with skip connections + multi-scale cond injection

    Input: x0 (6 ch) — current fire mask + 5 env features.
    Output: a single-channel logit map (pre-sigmoid).

    Multi-scale conditioning: x0 is avg-pooled to 1/2, 1/4, 1/8 res and
    concatenated alongside each encoder stage and decoder skip, giving the
    net direct access to env features at every spatial scale.
    """
    def __init__(self, in_channels=IN_CHANNELS, base_channels=96):
        super().__init__()
        C = base_channels

        # ---- Encoder ----
        # Level 1 — full res. Takes x0 directly (no x_t to concat with).
        self.enc1    = nn.Conv2d(in_channels, C, 3, padding=1)
        self.res1_1  = ResBlock(C)
        self.res1_2  = ResBlock(C)
        self.down1   = nn.Conv2d(C, C, 4, stride=2, padding=1)

        # Level 2 — 1/2 res
        self.enc2    = nn.Conv2d(C + in_channels, C * 2, 3, padding=1)
        self.res2_1  = ResBlock(C * 2)
        self.res2_2  = ResBlock(C * 2)
        self.down2   = nn.Conv2d(C * 2, C * 2, 4, stride=2, padding=1)

        # Level 3 — 1/4 res
        self.enc3    = nn.Conv2d(C * 2 + in_channels, C * 4, 3, padding=1)
        self.res3_1  = ResBlock(C * 4)
        self.res3_2  = ResBlock(C * 4)
        self.down3   = nn.Conv2d(C * 4, C * 4, 4, stride=2, padding=1)

        # ---- Bottleneck — 1/8 res ----
        self.enc4    = nn.Conv2d(C * 4 + in_channels, C * 4, 3, padding=1)
        self.res4_1  = ResBlock(C * 4)
        self.res4_2  = ResBlock(C * 4)

        # ---- Decoder ----
        # Level 3
        self.up3     = nn.ConvTranspose2d(C * 4, C * 4, 4, stride=2, padding=1)
        self.dec3    = nn.Conv2d(C * 8 + in_channels, C * 4, 3, padding=1)
        self.resd3_1 = ResBlock(C * 4)
        self.resd3_2 = ResBlock(C * 4)

        # Level 2
        self.up2     = nn.ConvTranspose2d(C * 4, C * 2, 4, stride=2, padding=1)
        self.dec2    = nn.Conv2d(C * 4 + in_channels, C * 2, 3, padding=1)
        self.resd2_1 = ResBlock(C * 2)
        self.resd2_2 = ResBlock(C * 2)

        # Level 1
        self.up1     = nn.ConvTranspose2d(C * 2, C, 4, stride=2, padding=1)
        self.dec1    = nn.Conv2d(C * 2 + in_channels, C, 3, padding=1)
        self.resd1_1 = ResBlock(C)
        self.resd1_2 = ResBlock(C)

        # ---- Output head (single) ----
        self.out_norm = nn.GroupNorm(8, C)
        self.out_head = nn.Conv2d(C, 1, 1)

    def forward(self, x0):
        # Multi-scale conditioning
        x0_half    = F.avg_pool2d(x0, 2)
        x0_quarter = F.avg_pool2d(x0_half, 2)
        x0_eighth  = F.avg_pool2d(x0_quarter, 2)

        # Encoder
        h1 = self.enc1(x0)
        h1 = self.res1_2(self.res1_1(h1))

        h2 = self.enc2(torch.cat([self.down1(h1), x0_half], dim=1))
        h2 = self.res2_2(self.res2_1(h2))

        h3 = self.enc3(torch.cat([self.down2(h2), x0_quarter], dim=1))
        h3 = self.res3_2(self.res3_1(h3))

        # Bottleneck
        h4 = self.enc4(torch.cat([self.down3(h3), x0_eighth], dim=1))
        h4 = self.res4_2(self.res4_1(h4))

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(h4), h3, x0_quarter], dim=1))
        d3 = self.resd3_2(self.resd3_1(d3))

        d2 = self.dec2(torch.cat([self.up2(d3), h2, x0_half], dim=1))
        d2 = self.resd2_2(self.resd2_1(d2))

        d1 = self.dec1(torch.cat([self.up1(d2), h1, x0], dim=1))
        d1 = self.resd1_2(self.resd1_1(d1))

        feat = F.silu(self.out_norm(d1))
        return self.out_head(feat)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def segmentation_loss(model, x0, x1, pos_weight_value=50.0):
    """
    Pixel-wise binary cross-entropy with positive-class reweighting.

    Fire pixels are rare (~1% of a typical crop), so unweighted BCE would
    drive the model to the trivial "no fire anywhere" solution. `pos_weight`
    multiplies the loss contribution of positive pixels, correcting the
    effective class balance in the gradient.
    """
    logits = model(x0)
    pos_w  = torch.tensor([pos_weight_value], device=x0.device)
    return F.binary_cross_entropy_with_logits(logits, x1, pos_weight=pos_w)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, x0, device="cpu"):
    """
    Single forward pass. Returns (binary prediction, probability map).
    """
    model.eval()
    x0     = x0.to(device)
    logits = model(x0)
    prob   = torch.sigmoid(logits)
    binary = (prob >= 0.5).float()
    return binary, prob


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_predictions(model, eval_loader, device, ckpt_dir,
                          epoch=None, num_examples=4):
    """
    Save a grid of (current fire | predicted fire | ground truth) panels for
    the first `num_examples` samples from the eval loader, and log it to W&B.

    Rendered in grayscale: black = 0 (no fire / low probability),
    white = 1 (fire / high probability).

    The eval loader is shuffled with a fixed seed, so the same samples appear
    across epochs — useful for visually tracking improvement of one prediction
    over training.
    """
    import matplotlib.pyplot as plt

    tag = f"epoch_{epoch}" if epoch is not None else "final"
    model.eval()

    fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))
    fig.suptitle(f"Wildfire Predictions ({tag})", fontsize=14)

    col_titles = ["Current Fire (Input)", "Predicted Fire", "Ground Truth"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12)

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= num_examples:
                break

            x, y = batch
            x0 = x[:, 0, :, :, :].to(device)        # [B, C, H, W]
            x1 = y.unsqueeze(1).float().to(device)   # [B, 1, H, W]
            _, pred_prob = predict(model, x0, device=device)

            # Current fire mask (first channel of x0)
            current_fire = x0[0, 0].cpu().numpy()
            # Predicted probability
            predicted    = pred_prob[0, 0].cpu().numpy()
            # Ground truth
            ground_truth = x1[0, 0].cpu().numpy()

            axes[i, 0].imshow(current_fire, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(predicted, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].axis('off')

    plt.tight_layout()

    save_path = os.path.join(ckpt_dir, f"predictions_{tag}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    wandb.log({f"predictions/{tag}": wandb.Image(save_path), "epoch": epoch})
    print(f"Saved prediction visualization to {save_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_evaluation(model, eval_loader, device, epoch=None):
    tag = f"epoch {epoch}" if epoch is not None else "final"
    print(f"\nEvaluating ({tag}) on {len(eval_loader)} test samples...")
    model.eval()
    all_targets, all_scores = [], []

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            x, y = batch
            x0 = x[:, 0, :, :, :].to(device)        # [B, C, H, W]
            x1 = y.unsqueeze(1).float().to(device)   # [B, 1, H, W]
            _, pred_prob = predict(model, x0, device=device)
            all_targets.append(x1.cpu().numpy().flatten())
            all_scores.append(pred_prob.cpu().numpy().flatten())
            if step % 50 == 0:
                print(f"  {step}/{len(eval_loader)} batches done...")

    ap = average_precision_score(
        np.concatenate(all_targets),
        np.concatenate(all_scores),
    )
    print(f"\n=========================================")
    print(f"  AP ({tag}):          {ap:.4f}")
    print(f"  Persistence baseline: 0.1930")
    print(f"  UNet target:          ~0.3330")
    print(f"=========================================\n")
    return ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str, required=True)
    parser.add_argument('--ckpt_dir',   type=str, required=True)
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=5,
                        help="Run full eval every N epochs (0 = end only)")
    parser.add_argument("--feature_set", type=str, default="vegetation",
                        choices=["vegetation", "multi", "all"])
    args = parser.parse_args()

    FEATURE_SETS = {
        "vegetation": [0, 1, 2, 3, 4, 38, 39],
        "multi": [0,1,2,3,4,5,6,7,8,9,11,12,13,14,
                  16,17,18,19,20,21,22,23,24,25,26,
                  27,28,29,30,31,32,38,39],
        "all": None,
    }
    CHANNELS = {"vegetation": 7, "multi": 34, "all": 40}
    features_to_keep = FEATURE_SETS[args.feature_set]
    IN_CHANNELS = CHANNELS[args.feature_set]

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = dict(
        architecture     = "FireSegmentationUNet (base_channels=96)",
        loss             = "BCEWithLogits (pos_weight)",
        epochs           = args.epochs,
        batch_size       = 16,
        learning_rate    = 2e-4,
        weight_decay     = 1e-4,
        pos_weight_value = 50.0,
        feature_set      = args.feature_set,
        in_channels      = IN_CHANNELS,
    )

    wandb.init(
        entity  = "ram-algoverse",
        project = "wildfire-flow",
        config  = cfg,
    )

    # ---- Data ----
    print(f"Loading data (IN_CHANNELS={IN_CHANNELS})...")
    train_dataset = FireSpreadDataset(
        data_dir=args.data_dir,
        included_fire_years=[2018, 2019, 2020],
        n_leading_observations=1,
        crop_side_length=128,
        load_from_hdf5=True,
        is_train=True,
        remove_duplicate_features=False,
        features_to_keep=features_to_keep,
        stats_years=[2018, 2019, 2020],
    )
    train_loader  = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                               shuffle=True, num_workers=2, pin_memory=True)

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
        stats_years=[2018, 2019, 2020],
    )
    eval_loader   = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ---- Model ----
    model = FireSegmentationUNet(in_channels=IN_CHANNELS, base_channels=96).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params/1e6:.1f}M")
    wandb.config.update({"n_params_M": round(n_params / 1e6, 1)})

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    global_step = 0
    start_epoch = 1

    # ---- Resume (note: old flow-matching checkpoints are incompatible) ----
    ckpt_path = os.path.join(args.ckpt_dir, "wildfire_seg_last.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            global_step = checkpoint.get('global_step', 0)
            start_epoch = checkpoint['epoch'] + 1
        else:
            model.load_state_dict(checkpoint)
            start_epoch = 1
        print(f"Resumed from epoch {start_epoch - 1}")

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for step, batch in enumerate(train_loader):
            x, y = batch
            x0 = x[:, 0, :, :, :].to(device)        # [B, C, H, W]
            x1 = y.unsqueeze(1).float().to(device)   # [B, 1, H, W]

            loss = segmentation_loss(
                model, x0, x1,
                pos_weight_value=cfg["pos_weight_value"],
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1

            wandb.log({
                "train/step_loss": loss.item(),
                "global_step":     global_step,
            })

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | loss={loss.item():.4f}")

        scheduler.step()
        n_steps = len(train_loader)
        dur     = time.time() - start

        wandb.log({
            "epoch":                    epoch,
            "train/epoch_loss":         epoch_loss / n_steps,
            "train/lr":                 scheduler.get_last_lr()[0],
            "train/epoch_duration_sec": dur,
        })
        print(f"--> Epoch {epoch} | avg loss={epoch_loss/n_steps:.4f} | {dur:.1f}s")

        torch.save({
            'epoch':       epoch,
            'global_step': global_step,
            'model':       model.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'scheduler':   scheduler.state_dict(),
        }, os.path.join(args.ckpt_dir, "wildfire_seg_last.pt"))

        # Periodic eval + visualization
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            model.eval()
            def predict_fn(x0):
                x0 = x0[:, 0, :, :, :]  # [B, C, H, W]
                return torch.sigmoid(model(x0))
            evaluate_model(predict_fn, eval_loader, device,
                           model_name="BCE-UNet", epoch=epoch, wandb_log=True)
            visualize_predictions(model, eval_loader, device,
                                  ckpt_dir=args.ckpt_dir, epoch=epoch)
            model.train()

    # ---- Final eval + visualization ----
    print("\nTraining complete. Running final evaluation...")
    model.eval()
    def predict_fn(x0):
        x0 = x0[:, 0, :, :, :]  # [B, C, H, W]
        return torch.sigmoid(model(x0))
    evaluate_model(predict_fn, eval_loader, device,
                   model_name="BCE-UNet", epoch=None, wandb_log=True)
    visualize_predictions(model, eval_loader, device,
                          ckpt_dir=args.ckpt_dir)
    torch.save(model.state_dict(),
               os.path.join(args.ckpt_dir, "wildfire_seg_best.pt"))
    wandb.finish()