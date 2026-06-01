"""
src/generative/flow_matching.py

Pure Conditional Flow Matching for next-day wildfire spread prediction.

Patches vs. previous version
----------------------------
- Training body wrapped in `def main(args)` so src/train_generative.py can
  call it; standalone `__main__` retained.
- Dataset year tuples corrected to WildfireSpreadTS fold 0:
  train = [2018, 2019], test = [2021], stats_years = [2018, 2019].

Architecture: VectorFieldNet learns a velocity field mapping current fire
state x0_fire toward next-day fire state x1 along the straight-line path
x_t = (1-t) x0_fire + t x1, with target velocity v* = x1 - x0_fire.

Reference: Lipman et al., "Flow Matching for Generative Modeling" (2022).
"""

import os, sys, argparse, time, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

TRAIN_YEARS = [2018, 2019]
TEST_YEARS  = [2021]
STATS_YEARS = [2018, 2019]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Residual block with sinusoidal timestep conditioning."""
    def __init__(self, channels, t_dim):
        super().__init__()
        self.norm1  = nn.GroupNorm(8, channels)
        self.conv1  = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2  = nn.GroupNorm(8, channels)
        self.conv2  = nn.Conv2d(channels, channels, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, channels)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class VectorFieldNet(nn.Module):
    """Timestep-conditioned UNet velocity field predictor (C=128, 4 levels)."""
    def __init__(self, in_channels=IN_CHANNELS, base_channels=128, t_dim=256):
        super().__init__()
        C          = base_channels
        self.t_dim = t_dim

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim),
        )

        self.enc1   = nn.Conv2d(1 + in_channels, C, 3, padding=1)
        self.res1_1 = ResBlock(C, t_dim);     self.res1_2 = ResBlock(C, t_dim)

        self.down1  = nn.Conv2d(C, C, 4, stride=2, padding=1)
        self.enc2   = nn.Conv2d(C + in_channels, C * 2, 3, padding=1)
        self.res2_1 = ResBlock(C * 2, t_dim); self.res2_2 = ResBlock(C * 2, t_dim)

        self.down2  = nn.Conv2d(C * 2, C * 2, 4, stride=2, padding=1)
        self.enc3   = nn.Conv2d(C * 2 + in_channels, C * 4, 3, padding=1)
        self.res3_1 = ResBlock(C * 4, t_dim); self.res3_2 = ResBlock(C * 4, t_dim)

        self.down3  = nn.Conv2d(C * 4, C * 4, 4, stride=2, padding=1)
        self.enc4   = nn.Conv2d(C * 4 + in_channels, C * 8, 3, padding=1)
        self.res4_1 = ResBlock(C * 8, t_dim); self.res4_2 = ResBlock(C * 8, t_dim)

        self.up3    = nn.ConvTranspose2d(C * 8, C * 4, 4, stride=2, padding=1)
        self.dec3   = nn.Conv2d(C * 8 + in_channels, C * 4, 3, padding=1)
        self.resd3_1= ResBlock(C * 4, t_dim); self.resd3_2= ResBlock(C * 4, t_dim)

        self.up2    = nn.ConvTranspose2d(C * 4, C * 2, 4, stride=2, padding=1)
        self.dec2   = nn.Conv2d(C * 4 + in_channels, C * 2, 3, padding=1)
        self.resd2_1= ResBlock(C * 2, t_dim); self.resd2_2= ResBlock(C * 2, t_dim)

        self.up1    = nn.ConvTranspose2d(C * 2, C, 4, stride=2, padding=1)
        self.dec1   = nn.Conv2d(C * 2 + in_channels, C, 3, padding=1)
        self.resd1_1= ResBlock(C, t_dim);     self.resd1_2= ResBlock(C, t_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, C), nn.SiLU(), nn.Conv2d(C, 1, 3, padding=1),
        )

    def sinusoidal_embedding(self, t):
        half  = self.t_dim // 2
        denom = max(1, half - 1)
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / denom)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x_t, x0, t):
        t_emb = self.t_mlp(self.sinusoidal_embedding(t))

        x0_half    = F.avg_pool2d(x0, 2)
        x0_quarter = F.avg_pool2d(x0_half, 2)
        x0_eighth  = F.avg_pool2d(x0_quarter, 2)

        h1 = self.enc1(torch.cat([x_t, x0], dim=1))
        h1 = self.res1_2(self.res1_1(h1, t_emb), t_emb)

        h2 = self.enc2(torch.cat([self.down1(h1), x0_half], dim=1))
        h2 = self.res2_2(self.res2_1(h2, t_emb), t_emb)

        h3 = self.enc3(torch.cat([self.down2(h2), x0_quarter], dim=1))
        h3 = self.res3_2(self.res3_1(h3, t_emb), t_emb)

        h4 = self.enc4(torch.cat([self.down3(h3), x0_eighth], dim=1))
        h4 = self.res4_2(self.res4_1(h4, t_emb), t_emb)

        d3 = self.dec3(torch.cat([self.up3(h4), h3, x0_quarter], dim=1))
        d3 = self.resd3_2(self.resd3_1(d3, t_emb), t_emb)

        d2 = self.dec2(torch.cat([self.up2(d3), h2, x0_half], dim=1))
        d2 = self.resd2_2(self.resd2_1(d2, t_emb), t_emb)

        d1 = self.dec1(torch.cat([self.up1(d2), h1, x0], dim=1))
        d1 = self.resd1_2(self.resd1_1(d1, t_emb), t_emb)

        return self.out(d1)


# ---------------------------------------------------------------------------
# Loss / inference / viz
# ---------------------------------------------------------------------------
def flow_matching_loss(model, x0, x1, pos_weight_value=50.0):
    """Weighted MSE on the predicted velocity field v* = x1 - x0_fire."""
    B, device = x0.shape[0], x0.device
    x0_fire   = x0[:, :1]
    t   = torch.rand(B, device=device)
    t_b = t.view(B, 1, 1, 1)
    x_t      = (1 - t_b) * x0_fire + t_b * x1
    target_v = x1 - x0_fire
    pred_v     = model(x_t, x0, t)
    weight_map = 1.0 + (pos_weight_value - 1.0) * x1
    return (weight_map * (pred_v - target_v) ** 2).mean()


@torch.no_grad()
def integrate_flow(model, x0, n_steps=50, threshold=0.5, device="cpu"):
    """Euler integration of the velocity field from t=0 to t=1."""
    model.eval()
    x0 = x0.to(device)
    x  = x0[:, :1].clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i * dt
        t     = torch.full((x.shape[0],), t_val, device=device)
        v     = model(x, x0, t)
        x     = x + dt * v
    prob   = torch.clamp(x, 0.0, 1.0)
    binary = (prob >= threshold).float()
    return binary, prob


def visualize_predictions(model, eval_loader, device, ckpt_dir, epoch=None, num_examples=4):
    import matplotlib.pyplot as plt
    tag = f"epoch_{epoch}" if epoch is not None else "final"
    model.eval()
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))
    fig.suptitle(f"Flow Matching Predictions ({tag})", fontsize=14)
    for col, title in enumerate(["Current Fire (Input)", "Predicted Fire", "Ground Truth"]):
        axes[0, col].set_title(title, fontsize=12)
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= num_examples:
                break
            x, y = batch
            x0 = x[:, 0, :, :, :].to(device)
            x1 = y.unsqueeze(1).float().to(device)
            _, pred_prob = integrate_flow(model, x0, n_steps=50, device=device)
            axes[i, 0].imshow(x0[0, 0].cpu().numpy(),        cmap="gray", vmin=0, vmax=1)
            axes[i, 1].imshow(pred_prob[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(x1[0, 0].cpu().numpy(),        cmap="gray", vmin=0, vmax=1)
            for col in range(3):
                axes[i, col].axis("off")
    plt.tight_layout()
    save_path = os.path.join(ckpt_dir, f"flow_predictions_{tag}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    wandb.log({f"predictions/flow_{tag}": wandb.Image(save_path), "epoch": epoch})
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
        architecture      = "VectorFieldNet (base_channels=128, t_dim=256)",
        loss              = "Weighted MSE on velocity field",
        epochs            = args.epochs,
        batch_size        = 16,
        learning_rate     = 1e-4,
        weight_decay      = 1e-4,
        pos_weight_value  = 50.0,
        n_steps_inference = 50,
        feature_set       = args.feature_set,
        in_channels       = in_channels,
        fold              = "fold0 (train 2018+2019, test 2021)",
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
                              shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = FireSpreadDataset(
        data_dir=args.data_dir, included_fire_years=TEST_YEARS,
        n_leading_observations=1, n_leading_observations_test_adjustment=5,
        crop_side_length=128, load_from_hdf5=True, is_train=False,
        remove_duplicate_features=False, features_to_keep=features_to_keep,
        stats_years=STATS_YEARS,
    )
    eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VectorFieldNet(in_channels=in_channels, base_channels=128, t_dim=256).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M")
    wandb.config.update({"n_params_M": round(n_params / 1e6, 1)})

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    global_step = 0
    start_epoch = 1
    ckpt_path = os.path.join(args.ckpt_dir, "wildfire_flow_last.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            global_step = ckpt.get("global_step", 0)
            start_epoch = ckpt["epoch"] + 1
        else:
            model.load_state_dict(ckpt)
        print(f"Resumed from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for step, batch in enumerate(train_loader):
            x, y = batch
            x0 = x[:, 0, :, :, :].to(device)
            x1 = y.unsqueeze(1).float().to(device)
            loss = flow_matching_loss(model, x0, x1, pos_weight_value=cfg["pos_weight_value"])
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
        dur = time.time() - start
        wandb.log({"epoch": epoch, "train/epoch_loss": epoch_loss / len(train_loader),
                   "train/lr": scheduler.get_last_lr()[0], "train/epoch_duration_sec": dur})
        print(f"--> Epoch {epoch} | avg loss={epoch_loss/len(train_loader):.4f} | {dur:.1f}s")

        torch.save({"epoch": epoch, "global_step": global_step,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()}, ckpt_path)

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            def predict_fn(x0):
                x0 = x0[:, 0, :, :, :]
                _, prob = integrate_flow(model, x0, n_steps=50, device=device)
                return prob
            evaluate_model(predict_fn, eval_loader, device,
                           model_name="FlowMatching", epoch=epoch, wandb_log=True)
            visualize_predictions(model, eval_loader, device, ckpt_dir=args.ckpt_dir, epoch=epoch)
            model.train()

    print("\nTraining complete. Running final evaluation...")
    def predict_fn(x0):
        x0 = x0[:, 0, :, :, :]
        _, prob = integrate_flow(model, x0, n_steps=50, device=device)
        return prob
    evaluate_model(predict_fn, eval_loader, device,
                   model_name="FlowMatching", epoch=None, wandb_log=True)
    visualize_predictions(model, eval_loader, device, ckpt_dir=args.ckpt_dir)
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "wildfire_flow_best.pt"))
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--ckpt_dir",   type=str, required=True)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--feature_set", type=str, default="vegetation",
                        choices=["vegetation", "multi", "all"])
    main(parser.parse_args())
