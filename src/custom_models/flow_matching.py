"""
src/custom_models/flow_matching.py

Conditional Flow Matching for next-day wildfire spread prediction,
reformulated in the FlowSDF style (Bogensperger et al., IJCV 2025) and adapted
for the extreme spatial sparsity of wildfire masks (~1% positive pixels).

Why this reformulation
----------------------
The original version learned a velocity field mapping the *current* fire mask
x0_fire toward the *next-day* binary mask x1, integrating from x = x0_fire. That
collapsed into over-prediction "blobs" (measured: mean velocity ~0.42,
active-pixel fraction ~0.98 at t=0; AP fell to base rate). Two structural causes:

  1. Starting integration at the current mask makes the regression target the
     displacement x1 - x0_fire, which on sparse fire is overwhelmingly "+1"
     (new fire appears) -> the field is biased positive by construction.
  2. A binary {0,1} mask is a pathological target for a continuous velocity
     field: 99% zeros with sparse +/-1 spikes.

FlowSDF fixes the formulation and is the standard, citable approach for flow
matching on segmentation:

  * START FROM NOISE. x0 ~ N(0, I); the conditioning (current fire + covariates)
    enters through the network, not as the integration start. This removes the
    dominant positive bias.
  * SDF TARGET. The next-day mask -> truncated Signed Distance Function
    (negative inside fire, positive outside, 0 at the boundary). Dense, smooth,
    well-posed; thresholded masks distort along boundaries rather than punching
    random holes.

Sparsity adaptation (this codebase)
-----------------------------------
At ~1% fire, a raw SDF is dominated by the positive background plateau, so its
mean is strongly positive and the velocity target inherits that bias. We address
this with two fixed, inference-safe choices:

  * FIXED TRAIN-SET NORMALIZATION. The SDF mean/std are estimated ONCE over the
    training masks and applied identically at train and test time (exactly how
    inputs are standardized via stats_years). Target becomes ~zero-mean / unit
    scale. The mask boundary (raw SDF = 0) maps to the fixed normalized
    threshold thr = -mean/std, so mask recovery is deterministic and exact.
  * TIGHT TRUNCATION (SDF_TRUNC=3). Shrinks the background plateau, reducing the
    residual positive skew that standardization cannot remove (mean+var only).

Residual note: a small positive skew (~+0.1) remains because the SDF
distribution is right-skewed at this sparsity; this is an inherent, reportable
property of SDF flow matching on very sparse targets, not a tunable bug. It is a
mild constant offset the network learns around, not the structural sign
imbalance that caused the original blobs.

Inference: integrate noise -> normalized SDF, recover mask by thresholding at
thr = -SDF_MEAN/SDF_STD; a bounded map gives a pseudo-probability for AP.

Reference: Lipman et al., "Flow Matching for Generative Modeling" (2022);
Bogensperger et al., "FlowSDF: Flow Matching for Medical Image Segmentation
Using Distance Transforms," IJCV 2025 (arXiv:2405.18087).
"""

import os, sys, argparse, time, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt
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

SDF_TRUNC = 3.0   # tight truncation (pixels): shrinks background plateau / skew


# ---------------------------------------------------------------------------
# SDF transforms + fixed normalization
# ---------------------------------------------------------------------------
def mask_to_sdf(mask, trunc=SDF_TRUNC):
    """(B,1,H,W) binary mask -> truncated signed distance function (pixel units).

    Negative inside fire, positive outside, 0 at the boundary, clipped to
    [-trunc, trunc]. Empty -> +trunc constant; full -> -trunc constant.
    """
    m = mask.detach().cpu().numpy()
    out = np.empty_like(m, dtype=np.float32)
    for b in range(m.shape[0]):
        binm = (m[b, 0] > 0.5)
        if binm.any() and (~binm).any():
            sdf = distance_transform_edt(~binm) - distance_transform_edt(binm)
        elif binm.all():
            sdf = -np.full(binm.shape, trunc, dtype=np.float32)
        else:
            sdf = np.full(binm.shape, trunc, dtype=np.float32)
        out[b, 0] = np.clip(sdf, -trunc, trunc)
    return torch.from_numpy(out).to(mask.device)


def estimate_sdf_stats(dataset, max_samples=400, trunc=SDF_TRUNC):
    """Estimate fixed SDF mean/std over the training masks (run once)."""
    vals = []
    n = min(len(dataset), max_samples)
    step = max(1, len(dataset) // n)
    for i in range(0, len(dataset), step):
        _, y = dataset[i]
        y = torch.as_tensor(y).float().view(1, 1, *y.shape[-2:])
        vals.append(mask_to_sdf(y, trunc=trunc).flatten())
        if len(vals) >= n:
            break
    allv = torch.cat(vals)
    mean = allv.mean().item()
    std  = allv.std().item() + 1e-6
    return mean, std


def normalize_sdf(sdf, mean, std):
    return (sdf - mean) / std


def denormalize_sdf(sdf_n, mean, std):
    return sdf_n * std + mean


def sdf_to_prob(sdf_pixels, scale=2.0):
    """Bounded SDF (pixel units) -> pseudo-probability in (0,1) for AP."""
    return torch.sigmoid(-sdf_pixels / scale)


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
    """Timestep-conditioned UNet velocity field predictor (C=128, 4 levels).

    x_t (starting from noise) is concatenated with the conditioning x0 (current
    fire + covariates) at every scale; x0 rides along purely as conditioning.
    """
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
def flow_matching_loss(model, x0, x1, sdf_mean, sdf_std):
    """Plain MSE on the velocity field, FlowSDF-style with fixed normalization.

    Path: x_t = (1-t)*noise + t*sdf_norm, target v* = sdf_norm - noise.
    Conditioning x0 enters through the network.
    """
    B, device = x0.shape[0], x0.device
    sdf_norm = normalize_sdf(mask_to_sdf(x1), sdf_mean, sdf_std)

    noise = torch.randn_like(sdf_norm)
    t   = torch.rand(B, device=device)
    t_b = t.view(B, 1, 1, 1)

    x_t      = (1 - t_b) * noise + t_b * sdf_norm
    target_v = sdf_norm - noise
    pred_v   = model(x_t, x0, t)
    return F.mse_loss(pred_v, target_v)


@torch.no_grad()
def integrate_flow(model, x0, sdf_mean, sdf_std, n_steps=50, device="cpu",
                   return_diag=False):
    """Euler integration from noise (t=0) to the predicted normalized SDF (t=1).

    Recovers the mask via the fixed threshold thr = -mean/std (raw SDF = 0).
    Returns (binary_mask, pseudo_prob).
    """
    model.eval()
    x0 = x0.to(device)
    B, _, H, W = x0.shape
    x  = torch.randn(B, 1, H, W, device=device)
    dt = 1.0 / n_steps

    diag = None
    for i in range(n_steps):
        t_val = i * dt
        t     = torch.full((B,), t_val, device=device)
        v     = model(x, x0, t)
        if i == 0 and return_diag:
            diag = {
                "diag/mean_velocity":     v.mean().item(),
                "diag/abs_mean_velocity": v.abs().mean().item(),
            }
        x = x + dt * v

    thr    = -sdf_mean / sdf_std                 # normalized location of raw SDF=0
    binary = (x <= thr).float()
    prob   = sdf_to_prob(denormalize_sdf(x, sdf_mean, sdf_std))
    if return_diag:
        diag["diag/pred_fire_frac"] = binary.mean().item()
        return binary, prob, diag
    return binary, prob


def visualize_predictions(model, eval_loader, device, ckpt_dir, sdf_mean, sdf_std,
                          n_steps=50, epoch=None, num_examples=4):
    import matplotlib.pyplot as plt
    tag = f"epoch_{epoch}" if epoch is not None else "final"
    model.eval()
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))
    fig.suptitle(f"FlowSDF Predictions ({tag})", fontsize=14)
    for col, title in enumerate(["Current Fire (Input)", "Predicted Fire", "Ground Truth"]):
        axes[0, col].set_title(title, fontsize=12)
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= num_examples:
                break
            x, y = batch
            x0 = x[:, 0, :, :, :].to(device)
            x1 = y.unsqueeze(1).float().to(device)
            _, pred_prob = integrate_flow(model, x0, sdf_mean, sdf_std,
                                          n_steps=n_steps, device=device)
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


@torch.no_grad()
def log_field_diagnostics(model, eval_loader, device, sdf_mean, sdf_std,
                          n_steps, epoch, max_batches=32):
    model.eval()
    acc, n = {}, 0
    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break
        x, _ = batch
        x0 = x[:, 0, :, :, :].to(device)
        _, _, diag = integrate_flow(model, x0, sdf_mean, sdf_std,
                                    n_steps=n_steps, device=device, return_diag=True)
        for k, v in diag.items():
            acc[k] = acc.get(k, 0.0) + v
        n += 1
    if n:
        log = {k: v / n for k, v in acc.items()}
        log["epoch"] = epoch
        wandb.log(log)
        print("Field diagnostics: " +
              ", ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in log.items()
                        if k != "epoch"))


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
        loss              = "FlowSDF: MSE on velocity field, noise->normalized SDF",
        epochs            = args.epochs,
        batch_size        = 16,
        learning_rate     = args.lr,
        weight_decay      = 1e-4,
        warmup_steps      = args.warmup_steps,
        grad_clip         = args.grad_clip,
        sdf_trunc         = SDF_TRUNC,
        n_steps_inference = args.n_steps,
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

    # Fixed SDF normalization stats over training masks (computed once).
    print("Estimating fixed SDF normalization stats over training masks...")
    sdf_mean, sdf_std = estimate_sdf_stats(train_dataset, trunc=SDF_TRUNC)
    print(f"SDF stats: mean={sdf_mean:.4f} std={sdf_std:.4f} "
          f"| recovery threshold={-sdf_mean/sdf_std:.4f}")
    wandb.config.update({"sdf_mean": round(sdf_mean, 4), "sdf_std": round(sdf_std, 4),
                         "sdf_recover_thr": round(-sdf_mean / sdf_std, 4)})

    model = VectorFieldNet(in_channels=in_channels, base_channels=128, t_dim=256).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M")
    wandb.config.update({"n_params_M": round(n_params / 1e6, 1)})

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])

    # Per-step linear warmup -> cosine decay, keyed off the global step so that
    # resuming picks up the schedule at the right place. This replaces the old
    # per-epoch cosine: the previous run showed a loss explosion early in
    # training (a high LR meeting the noise-start velocity field), and warmup is
    # the standard remedy. total_steps must be in *steps*, not epochs.
    total_steps  = max(1, args.epochs * len(train_loader))
    warmup_steps = max(0, args.warmup_steps)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps          # linear warmup from ~0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + np.cos(np.pi * progress)) # cosine decay to 0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
            sdf_mean = ckpt.get("sdf_mean", sdf_mean)
            sdf_std  = ckpt.get("sdf_std", sdf_std)
        else:
            model.load_state_dict(ckpt)
        print(f"Resumed from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_skipped  = 0
        running    = None          # running mean of loss for spike detection
        start = time.time()
        for step, batch in enumerate(train_loader):
            x, y = batch
            x0 = x[:, 0, :, :, :].to(device)
            x1 = y.unsqueeze(1).float().to(device)
            loss = flow_matching_loss(model, x0, x1, sdf_mean, sdf_std)
            loss_val = loss.item()

            # Spike guard: skip the optimizer step on a non-finite loss or one
            # that is wildly above the running mean (a single pathological batch
            # corrupting the weights is exactly what derailed the prior run).
            spike = (not np.isfinite(loss_val)) or \
                    (running is not None and loss_val > 5.0 * running)
            if spike:
                optimizer.zero_grad(set_to_none=True)
                n_skipped += 1
                global_step += 1
                scheduler.step()
                wandb.log({"train/step_loss": loss_val, "train/skipped": 1,
                           "global_step": global_step})
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()                       # per-step warmup+cosine

            running = loss_val if running is None else 0.98 * running + 0.02 * loss_val
            epoch_loss  += loss_val
            global_step += 1
            wandb.log({"train/step_loss": loss_val, "train/lr": scheduler.get_last_lr()[0],
                       "global_step": global_step})
            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | loss={loss_val:.4f}")

        dur = time.time() - start
        denom = max(1, len(train_loader) - n_skipped)
        wandb.log({"epoch": epoch, "train/epoch_loss": epoch_loss / denom,
                   "train/lr": scheduler.get_last_lr()[0],
                   "train/epoch_skipped": n_skipped, "train/epoch_duration_sec": dur})
        print(f"--> Epoch {epoch} | avg loss={epoch_loss/denom:.4f} | "
              f"skipped={n_skipped} | {dur:.1f}s")

        torch.save({"epoch": epoch, "global_step": global_step,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "sdf_mean": sdf_mean, "sdf_std": sdf_std}, ckpt_path)

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            def predict_fn(x0):
                x0 = x0[:, 0, :, :, :]
                _, prob = integrate_flow(model, x0, sdf_mean, sdf_std,
                                         n_steps=args.n_steps, device=device)
                return prob
            evaluate_model(predict_fn, eval_loader, device,
                           model_name="FlowMatching", epoch=epoch, wandb_log=True)
            log_field_diagnostics(model, eval_loader, device, sdf_mean, sdf_std,
                                  args.n_steps, epoch)
            visualize_predictions(model, eval_loader, device, ckpt_dir=args.ckpt_dir,
                                  sdf_mean=sdf_mean, sdf_std=sdf_std,
                                  n_steps=args.n_steps, epoch=epoch)
            model.train()

    print("\nTraining complete. Running final evaluation...")
    def predict_fn(x0):
        x0 = x0[:, 0, :, :, :]
        _, prob = integrate_flow(model, x0, sdf_mean, sdf_std,
                                 n_steps=args.n_steps, device=device)
        return prob
    evaluate_model(predict_fn, eval_loader, device,
                   model_name="FlowMatching", epoch=None, wandb_log=True)
    log_field_diagnostics(model, eval_loader, device, sdf_mean, sdf_std,
                          args.n_steps, epoch=None)
    visualize_predictions(model, eval_loader, device, ckpt_dir=args.ckpt_dir,
                          sdf_mean=sdf_mean, sdf_std=sdf_std, n_steps=args.n_steps)
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
    parser.add_argument("--n_steps", type=int, default=50,
                        help="Euler integration steps at inference (noise -> SDF).")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Peak learning rate (lowered from 1e-4 for stability).")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Linear LR warmup steps before cosine decay.")
    parser.add_argument("--grad_clip", type=float, default=0.5,
                        help="Max gradient norm (tightened from 1.0).")
    main(parser.parse_args())
