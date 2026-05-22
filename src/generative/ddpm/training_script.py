"""
src/model/ddpm/training_script.py

DDPM training script for WildfireSpreadBench.

Changes vs. original:
  - Added W&B logging (step-level and epoch-level)
  - Added checkpoint resume (saves epoch, optimizer, scheduler state)
  - Added --eval_every flag for periodic evaluation during training
  - Removed hardcoded Lambda lab paths — all paths come from argparse
  - Integrated unified_eval.evaluate_ddpm() for consistent benchmark metrics
  - AMP scaler state is saved/restored in the checkpoint
  - Cosine LR scheduler added (matches BCE U-Net training setup)
"""

import os, argparse, time, torch, sys, numpy as np
sys.path.insert(0, 'src')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddpm.unet import Unet
from ddpm.diffusion_proc import Diffusion
from dataloader.FireSpreadDataset import FireSpreadDataset
from torch.utils.data import DataLoader
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.unified_eval import evaluate_model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, loader, eval_loader, diffusion, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler("cuda")

    global_step = 0
    start_epoch = 1

    # ---- Resume from checkpoint ----
    ckpt_path = os.path.join(args.ckpt_dir, "fire_ddpm_last.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            global_step = ckpt.get("global_step", 0)
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from epoch {start_epoch - 1} (global_step={global_step})")
        else:
            # Legacy checkpoint: weights-only dict
            model.load_state_dict(ckpt)
            print("Loaded weights-only checkpoint (no optimizer state).")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            x, y = batch
            x0 = x[:, 0, :, :, :].to(device)        # [B, C, H, W]
            x1 = y.unsqueeze(1).float().to(device)   # [B, 1, H, W]
            images      = x1
            cond_images = x0

            # Classifier-free guidance: randomly drop conditioning 20% of steps
            mask = (torch.rand(images.shape[0]) > 0.2).int().to(device)
            t    = torch.randint(0, 1000, (images.shape[0],), device=device).long()

            with torch.amp.autocast("cuda"):
                loss = diffusion.train_losses(model, images, t, cond_images, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            global_step += 1

            wandb.log({
                "train/step_loss": loss.item(),
                "global_step":     global_step,
            })

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        dur      = time.time() - start_time

        wandb.log({
            "epoch":                    epoch,
            "train/epoch_loss":         avg_loss,
            "train/lr":                 scheduler.get_last_lr()[0],
            "train/epoch_duration_sec": dur,
        })
        print(f"--> Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Time: {dur:.2f}s")

        # ---- Save full checkpoint ----
        torch.save({
            "epoch":       epoch,
            "global_step": global_step,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "scaler":      scaler.state_dict(),
        }, ckpt_path)

        # ---- Periodic eval ----
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            model.eval()
            def predict_fn(x0):
                x0 = x0[:, 0, :, :, :]  # [B, C, H, W]
                preds = diffusion.p_sample_loop(model, (x0.shape[0], 1, x0.shape[2], x0.shape[3]), x0)
                final = np.array(preds[-1])
                prob = torch.tensor(final).clamp(0.0, 1.0)
                return prob.to(device)
            results = evaluate_model(
                predict_fn, eval_loader, device,
                model_name="DDPM", epoch=epoch, wandb_log=True,
            )
            model.train()

    # ---- Final eval ----
    print("\nTraining complete. Running final evaluation...")
    model.eval()
    def predict_fn(x0):
        x0 = x0[:, 0, :, :, :]  # [B, C, H, W]
        preds = diffusion.p_sample_loop(model, (x0.shape[0], 1, x0.shape[2], x0.shape[3]), x0)
        final = np.array(preds[-1])
        prob = torch.tensor(final).clamp(0.0, 1.0)
        return prob.to(device)
    results = evaluate_model(
        predict_fn, eval_loader, device,
        model_name="DDPM", epoch=None, wandb_log=True,
    )

    # Save best weights separately (weights-only for easy loading)
    torch.save(
        model.state_dict(),
        os.path.join(args.ckpt_dir, "fire_ddpm_best.pt"),
    )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Root of HDF5 dataset (contains 2018/, 2019/, ...)")
    parser.add_argument("--ckpt_dir",   type=str, required=True,
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Run full eval every N epochs. 0 = end only.")
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
        architecture     = "DDPM (Unet, model_ch=128, CFG)",
        noise_schedule   = "sigmoid",
        timesteps        = 1000,
        epochs           = args.epochs,
        batch_size       = 32,
        learning_rate    = 1e-4,
        weight_decay     = 0.01,
        cfg_dropout_rate = 0.2,
        feature_set      = args.feature_set,
        in_channels      = IN_CHANNELS,
    )

    wandb.init(
        entity  = "ram-algoverse",
        project = "WildfireSpreadBench",
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
    train_loader  = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True,
    )

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
    eval_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ---- Model ----
    print(f"Initializing DDPM U-Net on {device}...")
    model = Unet(
        in_ch=1, cond_ch=IN_CHANNELS, output_ch=1,
        model_ch=128, channel_mult=(1, 2, 2, 4),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M")
    wandb.config.update({"n_params_M": round(n_params / 1e6, 1)})

    diffusion = Diffusion(timesteps=1000, noise_schedule="sigmoid")

    train(model, train_loader, eval_loader, diffusion, device, args)
    wandb.finish()