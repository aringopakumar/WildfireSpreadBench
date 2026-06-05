"""
src/generative/ddpm/training_script.py

DDPM training script for WildfireSpreadBench.

Patches vs. previous version
----------------------------
- Training body wrapped in `def main(args)` so src/train_generative.py can call it.
- Dataset year tuples = WildfireSpreadTS fold 0: train [2018,2019], test [2021].
- SDF TARGET (sparsity fix): the diffusion target is the normalized SDF of the
  next-day mask, using the SAME fixed train-set normalization as the flow model
  (src/generative/flow_matching.py). The previous binary-mask target collapsed
  to the background-dominated trivial solution (AP ~ base rate) under ~0.1%
  fire sparsity. SDF stats (mean/std) are estimated once over the training masks,
  passed into Diffusion, and stored in the checkpoint for exact resume/eval.
- Sampling recovery goes through diffusion.sample_to_prob (normalized SDF ->
  pixel SDF -> [0,1] probability via flow's sdf_to_prob).
- Periodic eval capped to --eval_samples batches; --guidance_w exposed.

Otherwise unchanged: W&B logging, checkpoint resume, AMP, cosine LR, CFG.
"""

import os, argparse, time, torch, sys, numpy as np
sys.path.insert(0, 'src')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddpm.unet import Unet
from ddpm.diffusion_proc import Diffusion, estimate_sdf_stats, SDF_TRUNC
from dataloader.FireSpreadDataset import FireSpreadDataset
from torch.utils.data import DataLoader
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.unified_eval import evaluate_model

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


def make_ddpm_predict_fn(model, diffusion, device, guidance_w):
    """predict_fn returning a [0,1] prob map; recovery via sample_to_prob."""
    def predict_fn(x0):
        x0 = x0[:, 0, :, :, :]                      # [B,T,C,H,W] -> [B,C,H,W]
        prob = diffusion.sample_to_prob(model, x0, w=guidance_w, progress=False)
        return prob.to(device)
    return predict_fn


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model, loader, eval_loader, diffusion, device, args):
    guidance_w   = getattr(args, "guidance_w", 2.0)
    eval_samples = getattr(args, "eval_samples", 64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler("cuda")

    global_step = 0
    start_epoch = 1

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
            model.load_state_dict(ckpt)
            print("Loaded weights-only checkpoint (no optimizer state).")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            x, y = batch
            x0 = x[:, 0, :, :, :].to(device)         # [B, C, H, W] conditioning
            x1 = y.unsqueeze(1).float().to(device)    # [B, 1, H, W] mask in {0,1}
            images      = x1
            cond_images = x0

            mask = (torch.rand(images.shape[0]) > 0.2).int().to(device)
            t    = torch.randint(0, 1000, (images.shape[0],), device=device).long()

            with torch.amp.autocast("cuda"):
                # train_losses converts the {0,1} mask to a normalized SDF target.
                loss = diffusion.train_losses(model, images, t, cond_images, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            global_step += 1
            wandb.log({"train/step_loss": loss.item(), "global_step": global_step})
            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        dur      = time.time() - start_time
        wandb.log({"epoch": epoch, "train/epoch_loss": avg_loss,
                   "train/lr": scheduler.get_last_lr()[0],
                   "train/epoch_duration_sec": dur})
        print(f"--> Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Time: {dur:.2f}s")

        torch.save({"epoch": epoch, "global_step": global_step,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "sdf_mean": diffusion.sdf_mean, "sdf_std": diffusion.sdf_std,
                    "sdf_trunc": diffusion.sdf_trunc},
                   ckpt_path)

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            model.eval()
            predict_fn = make_ddpm_predict_fn(model, diffusion, device, guidance_w)
            evaluate_model(predict_fn, eval_loader, device,
                           model_name="DDPM", epoch=epoch, wandb_log=True,
                           max_batches=eval_samples)
            model.train()

    print("\nTraining complete. Running final evaluation (full test set)...")
    model.eval()
    predict_fn = make_ddpm_predict_fn(model, diffusion, device, guidance_w)
    results = evaluate_model(predict_fn, eval_loader, device,
                             model_name="DDPM", epoch=None, wandb_log=True)
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "fire_ddpm_best.pt"))
    return results


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def main(args):
    features_to_keep = FEATURE_SETS[args.feature_set]
    IN_CHANNELS      = CHANNELS[args.feature_set]

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    guidance_w = getattr(args, "guidance_w", 2.0)

    cfg = dict(
        architecture     = "DDPM (Unet, model_ch=128, CFG) + SDF target",
        noise_schedule   = "sigmoid",
        timesteps        = 1000,
        epochs           = args.epochs,
        batch_size       = 32,
        learning_rate    = 1e-4,
        weight_decay     = 0.01,
        cfg_dropout_rate = 0.2,
        guidance_w       = guidance_w,
        sdf_trunc        = SDF_TRUNC,
        feature_set      = args.feature_set,
        in_channels      = IN_CHANNELS,
        fold             = "fold0 (train 2018+2019, test 2021)",
        target           = "normalized SDF (shared with flow_matching.py)",
    )

    wandb.init(entity="ram-algoverse", project="WildfireSpreadBench", config=cfg)

    print(f"Loading data (IN_CHANNELS={IN_CHANNELS})...")
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

    # Fixed SDF normalization stats over training masks (shared recipe with flow).
    print("Estimating fixed SDF normalization stats over training masks...")
    sdf_mean, sdf_std = estimate_sdf_stats(train_dataset, trunc=SDF_TRUNC)
    print(f"SDF stats: mean={sdf_mean:.4f} std={sdf_std:.4f} "
          f"| recovery threshold={-sdf_mean/sdf_std:.4f}")
    wandb.config.update({"sdf_mean": round(sdf_mean, 4), "sdf_std": round(sdf_std, 4),
                         "sdf_recover_thr": round(-sdf_mean / sdf_std, 4)})

    print(f"Initializing DDPM U-Net on {device}...")
    model = Unet(in_ch=1, cond_ch=IN_CHANNELS, output_ch=1,
                 model_ch=128, channel_mult=(1, 2, 2, 4)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M")
    wandb.config.update({"n_params_M": round(n_params / 1e6, 1)})

    diffusion = Diffusion(timesteps=1000, noise_schedule="sigmoid",
                          sdf_mean=sdf_mean, sdf_std=sdf_std, sdf_trunc=SDF_TRUNC)

    train(model, train_loader, eval_loader, diffusion, device, args)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--ckpt_dir",   type=str, required=True)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Run eval every N epochs. 0 = end only.")
    parser.add_argument("--eval_samples", type=int, default=64,
                        help="Cap periodic eval to this many test batches. Final uses all.")
    parser.add_argument("--guidance_w", type=float, default=2.0,
                        help="Classifier-free guidance weight at sampling.")
    parser.add_argument("--feature_set", type=str, default="vegetation",
                        choices=["vegetation", "multi", "all"])
    main(parser.parse_args())
