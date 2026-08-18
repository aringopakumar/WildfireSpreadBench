"""
Wildfire next-day spread prediction as pure conditional binary segmentation.

Patches vs. previous version
----------------------------
- Training body wrapped in `def main(args)` so src/train_custom.py can
  call it. A standalone `__main__` block is retained for direct runs.
- Dataset year tuples corrected to WildfireSpreadTS fold 0:
  train = [2018, 2019], test = [2021], stats_years = [2018, 2019].
  (The previous [2018, 2019, 2020] raised KeyError in
  get_means_stds_missing_values, which only has 2-year stat tuples, and
  did not correspond to any paper fold.)

Architecture and hyperparameters are otherwise unchanged.
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

IN_CHANNELS = 7

FEATURE_SETS = {
    "vegetation": [0, 1, 2, 3, 4, 38, 39],
    "multi": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
              27, 28, 29, 30, 31, 32, 38, 39],
    "all": None,
}
CHANNELS = {"vegetation": 7, "multi": 34, "all": 40}

# WildfireSpreadTS fold 0 (suppl. Table 5): train (2018,2019), val 2020, test 2021.
TRAIN_YEARS = [2018, 2019]
TEST_YEARS  = [2021]
STATS_YEARS = [2018, 2019]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Residual block, pre-activation style. No timestep conditioning."""
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
    """Conditional U-Net for next-day fire segmentation (base_channels C=96)."""
    def __init__(self, in_channels=IN_CHANNELS, base_channels=96):
        super().__init__()
        C = base_channels

        self.enc1    = nn.Conv2d(in_channels, C, 3, padding=1)
        self.res1_1  = ResBlock(C)
        self.res1_2  = ResBlock(C)
        self.down1   = nn.Conv2d(C, C, 4, stride=2, padding=1)

        self.enc2    = nn.Conv2d(C + in_channels, C * 2, 3, padding=1)
        self.res2_1  = ResBlock(C * 2)
        self.res2_2  = ResBlock(C * 2)
        self.down2   = nn.Conv2d(C * 2, C * 2, 4, stride=2, padding=1)

        self.enc3    = nn.Conv2d(C * 2 + in_channels, C * 4, 3, padding=1)
        self.res3_1  = ResBlock(C * 4)
        self.res3_2  = ResBlock(C * 4)
        self.down3   = nn.Conv2d(C * 4, C * 4, 4, stride=2, padding=1)

        self.enc4    = nn.Conv2d(C * 4 + in_channels, C * 4, 3, padding=1)
        self.res4_1  = ResBlock(C * 4)
        self.res4_2  = ResBlock(C * 4)

        self.up3     = nn.ConvTranspose2d(C * 4, C * 4, 4, stride=2, padding=1)
        self.dec3    = nn.Conv2d(C * 8 + in_channels, C * 4, 3, padding=1)
        self.resd3_1 = ResBlock(C * 4)
        self.resd3_2 = ResBlock(C * 4)

        self.up2     = nn.ConvTranspose2d(C * 4, C * 2, 4, stride=2, padding=1)
        self.dec2    = nn.Conv2d(C * 4 + in_channels, C * 2, 3, padding=1)
        self.resd2_1 = ResBlock(C * 2)
        self.resd2_2 = ResBlock(C * 2)

        self.up1     = nn.ConvTranspose2d(C * 2, C, 4, stride=2, padding=1)
        self.dec1    = nn.Conv2d(C * 2 + in_channels, C, 3, padding=1)
        self.resd1_1 = ResBlock(C)
        self.resd1_2 = ResBlock(C)

        self.out_norm = nn.GroupNorm(8, C)
        self.out_head = nn.Conv2d(C, 1, 1)

    def forward(self, x0):
        x0_half    = F.avg_pool2d(x0, 2)
        x0_quarter = F.avg_pool2d(x0_half, 2)
        x0_eighth  = F.avg_pool2d(x0_quarter, 2)

        h1 = self.enc1(x0)
        h1 = self.res1_2(self.res1_1(h1))

        h2 = self.enc2(torch.cat([self.down1(h1), x0_half], dim=1))
        h2 = self.res2_2(self.res2_1(h2))

        h3 = self.enc3(torch.cat([self.down2(h2), x0_quarter], dim=1))
        h3 = self.res3_2(self.res3_1(h3))

        h4 = self.enc4(torch.cat([self.down3(h3), x0_eighth], dim=1))
        h4 = self.res4_2(self.res4_1(h4))

        d3 = self.dec3(torch.cat([self.up3(h4), h3, x0_quarter], dim=1))
        d3 = self.resd3_2(self.resd3_1(d3))

        d2 = self.dec2(torch.cat([self.up2(d3), h2, x0_half], dim=1))
        d2 = self.resd2_2(self.resd2_1(d2))

        d1 = self.dec1(torch.cat([self.up1(d2), h1, x0], dim=1))
        d1 = self.resd1_2(self.resd1_1(d1))

        feat = F.silu(self.out_norm(d1))
        return self.out_head(feat)


# ---------------------------------------------------------------------------
# Loss / inference / viz
# ---------------------------------------------------------------------------
def segmentation_loss(model, x0, x1, pos_weight_value=50.0):
    """Pixel-wise BCE with positive-class reweighting (fire pixels ~0.17%)."""
    logits = model(x0)
    pos_w  = torch.tensor([pos_weight_value], device=x0.device)
    return F.binary_cross_entropy_with_logits(logits, x1, pos_weight=pos_w)


@torch.no_grad()
def predict(model, x0, device="cpu"):
    model.eval()
    x0     = x0.to(device)
    logits = model(x0)
    prob   = torch.sigmoid(logits)
    binary = (prob >= 0.5).float()
    return binary, prob


def visualize_predictions(model, eval_loader, device, ckpt_dir,
                          epoch=None, num_examples=4):
    import matplotlib.pyplot as plt
    tag = f"epoch_{epoch}" if epoch is not None else "final"
    model.eval()

    fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))
    fig.suptitle(f"Wildfire Predictions ({tag})", fontsize=14)
    for col, title in enumerate(["Current Fire (Input)", "Predicted Fire", "Ground Truth"]):
        axes[0, col].set_title(title, fontsize=12)

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= num_examples:
                break
            x, y = batch
            x0 = x[:, 0, :, :, :].to(device)
            x1 = y.unsqueeze(1).float().to(device)
            _, pred_prob = predict(model, x0, device=device)
            axes[i, 0].imshow(x0[0, 0].cpu().numpy(),        cmap='gray', vmin=0, vmax=1); axes[i, 0].axis('off')
            axes[i, 1].imshow(pred_prob[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1); axes[i, 1].axis('off')
            axes[i, 2].imshow(x1[0, 0].cpu().numpy(),        cmap='gray', vmin=0, vmax=1); axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(ckpt_dir, f"predictions_{tag}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    wandb.log({f"predictions/{tag}": wandb.Image(save_path), "epoch": epoch})
    print(f"Saved prediction visualization to {save_path}")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def main(args):
    features_to_keep = FEATURE_SETS[args.feature_set]
    in_channels      = CHANNELS[args.feature_set]

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
        in_channels      = in_channels,
        fold             = "fold0 (train 2018+2019, test 2021)",
    )

    wandb.init(entity="ram-algoverse", project="WildfireSpreadBench", config=cfg)

    print(f"Loading data (in_channels={in_channels})...")
    train_dataset = FireSpreadDataset(
        data_dir=args.data_dir, included_fire_years=TRAIN_YEARS,
        n_leading_observations=1, crop_side_length=128, load_from_hdf5=True,
        is_train=True, remove_duplicate_features=False,
        features_to_keep=features_to_keep, stats_years=STATS_YEARS,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)

    test_dataset = FireSpreadDataset(
        data_dir=args.data_dir, included_fire_years=TEST_YEARS,
        n_leading_observations=1, n_leading_observations_test_adjustment=5,
        crop_side_length=128, load_from_hdf5=True, is_train=False,
        remove_duplicate_features=False, features_to_keep=features_to_keep,
        stats_years=STATS_YEARS,
    )
    eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = FireSegmentationUNet(in_channels=in_channels, base_channels=96).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params/1e6:.1f}M")
    wandb.config.update({"n_params_M": round(n_params / 1e6, 1)})

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    global_step = 0
    start_epoch = 1
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
        print(f"Resumed from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for step, batch in enumerate(train_loader):
            x, y = batch
            x0 = x[:, 0, :, :, :].to(device)
            x1 = y.unsqueeze(1).float().to(device)
            loss = segmentation_loss(model, x0, x1, pos_weight_value=cfg["pos_weight_value"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss  += loss.item()
            global_step += 1
            wandb.log({"train/step_loss": loss.item(), "global_step": global_step})
            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | loss={loss.item():.4f}")

        scheduler.step()
        n_steps = len(train_loader)
        dur     = time.time() - start
        wandb.log({"epoch": epoch, "train/epoch_loss": epoch_loss / n_steps,
                   "train/lr": scheduler.get_last_lr()[0], "train/epoch_duration_sec": dur})
        print(f"--> Epoch {epoch} | avg loss={epoch_loss/n_steps:.4f} | {dur:.1f}s")

        torch.save({'epoch': epoch, 'global_step': global_step,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                   os.path.join(args.ckpt_dir, "wildfire_seg_last.pt"))

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            model.eval()
            def predict_fn(x0):
                x0 = x0[:, 0, :, :, :]
                return torch.sigmoid(model(x0))
            evaluate_model(predict_fn, eval_loader, device,
                           model_name="BCE-UNet", epoch=epoch, wandb_log=True)
            visualize_predictions(model, eval_loader, device, ckpt_dir=args.ckpt_dir, epoch=epoch)
            model.train()

    print("\nTraining complete. Running final evaluation...")
    model.eval()
    def predict_fn(x0):
        x0 = x0[:, 0, :, :, :]
        return torch.sigmoid(model(x0))
    evaluate_model(predict_fn, eval_loader, device,
                   model_name="BCE-UNet", epoch=None, wandb_log=True)
    visualize_predictions(model, eval_loader, device, ckpt_dir=args.ckpt_dir)
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "wildfire_seg_best.pt"))
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str, required=True)
    parser.add_argument('--ckpt_dir',   type=str, required=True)
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument("--feature_set", type=str, default="vegetation",
                        choices=["vegetation", "multi", "all"])
    main(parser.parse_args())
