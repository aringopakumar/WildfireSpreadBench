"""
src/evaluation/unified_eval.py

Unified evaluation harness for WildfireSpreadBench.

Every model in the benchmark (BCE U-Net, Flow Matching, DDPM, ConvLSTM,
Logistic Regression) funnels through evaluate_model(). This guarantees
all numbers in the results tables come from identical data, identical
thresholds, and identical metric implementations.

Patches vs. previous version
-----------------------------
- evaluate_model gains an optional `max_batches` cap. DDPM's 1000-step reverse
  process makes full-test periodic eval expensive; capping the number of test
  batches during training keeps it affordable. Final eval passes max_batches=None
  to use the entire test set. Set-once, explicit, and logged via model_name tag.
- evaluate_ddpm now recovers the probability map via diffusion.sample_to_prob,
  which maps the [-1,1] sampler output to [0,1]. The previous wrapper clamped
  the raw sampler output to [0,1], discarding half the range and miscalibrating
  AP. (The matching [0,1]->[-1,1] scaling of the training target is in
  diffusion.train_losses.)
- The convenience wrappers squeeze the leading time dim (FireSpreadDataset
  yields x of shape [B, T, C, H, W]; models expect [B, C, H, W]).

predict_fn signature
--------------------
    predict_fn(x0: Tensor) -> Tensor
    Input:  x0  — conditioning tensor already on `device`. As yielded by the
                  loader this is [B, T, C, H, W]; wrappers squeeze T internally.
    Output: probability map in [0, 1], shape [B, 1, H, W]

Metrics
-------
    AP, F1, Precision, Recall, IoU. AP is the primary, threshold-free metric.

Baselines (WildfireSpreadTS, 12-fold mean, "All" feature set)
-------------------------------------------------------------
    Persistence:         AP ~ 0.193
    Logistic Regression: AP ~ 0.279
    ResNet18 U-Net:      AP ~ 0.328
    ConvLSTM:            AP ~ 0.306
    UTAE (vegetation):   AP ~ 0.372
  NOTE: these are 12-fold means. Single-fold (fold 0, test=2021) numbers
  run noticeably higher because 2021 is an easy test year (see suppl. Fig 2).
"""

import numpy as np
import torch
import wandb
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_model(
    predict_fn,
    eval_loader,
    device,
    model_name: str = "model",
    epoch=None,
    threshold: float = 0.5,
    wandb_log: bool = True,
    verbose: bool = True,
    max_batches=None,
) -> dict:
    """
    Run inference over the eval loader and compute all benchmark metrics.

    eval_loader yields (x0, x1). x0 = conditioning (as stored by the dataset,
    i.e. possibly [B, T, C, H, W]); x1 = binary target. predict_fn is
    responsible for any reshaping its model needs.

    max_batches: if set, evaluate only the first `max_batches` batches. Used to
    keep DDPM periodic eval affordable; final eval should pass None.

    Returns dict with keys: ap, f1, precision, recall, iou (all Python floats).
    """
    tag = f"epoch_{epoch}" if epoch is not None else "final"

    all_targets = []
    all_scores  = []

    with torch.no_grad():
        for step, (x0, x1) in enumerate(eval_loader):
            if max_batches is not None and step >= max_batches:
                break
            x0 = x0.to(device)
            x1 = x1.to(device)

            prob = predict_fn(x0)          # [B, 1, H, W], values in [0, 1]

            all_targets.append(x1.cpu().numpy().flatten())
            all_scores.append(prob.cpu().numpy().flatten())

            if verbose and step % 50 == 0:
                print(f"  [{model_name}] {step}/{len(eval_loader)} batches...")

    targets_flat = np.concatenate(all_targets)
    scores_flat  = np.concatenate(all_scores)
    binary_flat  = (scores_flat >= threshold).astype(np.float32)

    ap = float(average_precision_score(targets_flat, scores_flat))

    p, r, f1, _ = precision_recall_fscore_support(
        targets_flat, binary_flat,
        labels=[1], average=None, zero_division=0,
    )
    precision = float(p[0])
    recall    = float(r[0])
    f1_score  = float(f1[0])

    tp = float(np.sum((binary_flat == 1) & (targets_flat == 1)))
    fp = float(np.sum((binary_flat == 1) & (targets_flat == 0)))
    fn = float(np.sum((binary_flat == 0) & (targets_flat == 1)))
    iou = tp / (tp + fp + fn + 1e-8)

    results = dict(ap=ap, f1=f1_score, precision=precision, recall=recall, iou=iou)

    if verbose:
        _print_results(model_name, tag, results, threshold)
    if wandb_log:
        _log_to_wandb(model_name, tag, epoch, results)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_results(model_name: str, tag: str, results: dict, threshold: float):
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Model:     {model_name}  ({tag})")
    print(f"  Threshold: {threshold}")
    print(f"  ---")
    print(f"  AP        (threshold-free): {results['ap']:.4f}")
    print(f"  F1        (thr={threshold}):       {results['f1']:.4f}")
    print(f"  Precision (thr={threshold}):       {results['precision']:.4f}")
    print(f"  Recall    (thr={threshold}):       {results['recall']:.4f}")
    print(f"  IoU       (thr={threshold}):       {results['iou']:.4f}")
    print(f"  ---")
    print(f"  Persistence baseline AP (12-fold mean): 0.193")
    print(f"  ResNet18 U-Net AP (12-fold mean):       ~0.328")
    print(f"  (Single-fold test=2021 runs higher; see suppl. Fig 2.)")
    print(f"{sep}\n")


def _log_to_wandb(model_name: str, tag: str, epoch, results: dict):
    prefix = f"eval/{model_name}"
    log_dict = {f"{prefix}/{k}": v for k, v in results.items()}
    if epoch is not None:
        log_dict["epoch"] = epoch
    wandb.log(log_dict)


# ---------------------------------------------------------------------------
# Convenience wrappers for each model family
# ---------------------------------------------------------------------------
# Each wrapper builds a predict_fn that squeezes the leading time dim
# (FireSpreadDataset yields [B, T, C, H, W]) before calling its model.

@torch.no_grad()
def evaluate_bce_unet(model, eval_loader, device, epoch=None, wandb_log=True):
    """BCE Segmentation U-Net (FireSegmentationUNet)."""
    model.eval()
    def predict_fn(x0):
        x0 = x0[:, 0, :, :, :]          # [B, T, C, H, W] -> [B, C, H, W]
        return torch.sigmoid(model(x0))
    return evaluate_model(
        predict_fn=predict_fn, eval_loader=eval_loader, device=device,
        model_name="BCE-UNet", epoch=epoch, wandb_log=wandb_log,
    )


@torch.no_grad()
def evaluate_flow(model, eval_loader, device, n_steps=50, epoch=None, wandb_log=True):
    """Pure Flow Matching (VectorFieldNet)."""
    from src.generative.flow_matching import integrate_flow, estimate_sdf_stats
    model.eval()
    # Flow matching needs SDF normalization stats; recover from the loader's
    # dataset so standalone eval matches training-time recovery.
    sdf_mean, sdf_std = estimate_sdf_stats(eval_loader.dataset)
    def predict_fn(x0):
        x0 = x0[:, 0, :, :, :]          # [B, T, C, H, W] -> [B, C, H, W]
        _, prob = integrate_flow(model, x0, sdf_mean, sdf_std, n_steps=n_steps, device=device)
        return prob
    return evaluate_model(
        predict_fn=predict_fn, eval_loader=eval_loader, device=device,
        model_name="FlowMatching", epoch=epoch, wandb_log=wandb_log,
    )


@torch.no_grad()
def evaluate_ddpm(model, diffusion, eval_loader, device, epoch=None,
                  wandb_log=True, guidance_w=2.0, max_batches=None):
    """Classifier-Free DDPM (Unet + Diffusion).

    Recovery via diffusion.sample_to_prob: the sampler runs in [-1,1] and this
    maps the final state to a [0,1] probability map (single source of truth).
    """
    model.eval()
    def predict_fn(x0):
        x0 = x0[:, 0, :, :, :]          # [B, T, C, H, W] -> [B, C, H, W]
        return diffusion.sample_to_prob(model, x0, w=guidance_w, progress=False).to(device)
    return evaluate_model(
        predict_fn=predict_fn, eval_loader=eval_loader, device=device,
        model_name="DDPM", epoch=epoch, wandb_log=wandb_log, max_batches=max_batches,
    )


# ---------------------------------------------------------------------------
# Lightning model evaluation (discriminative baselines)
# ---------------------------------------------------------------------------

def evaluate_lightning_model(
    model, datamodule, device, model_name: str, wandb_log: bool = True,
) -> dict:
    """
    Evaluate a PyTorch Lightning BaseModel subclass through the unified harness,
    giving discriminative models identical metric computation to generative ones.
    """
    model.to(device)
    model.eval()

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    def predict_fn(x0):
        x0 = x0[:, 0, :, :, :]   # [B, C, H, W]
        logits = model(x0)
        prob = torch.sigmoid(logits)
        if prob.dim() == 3:
            prob = prob.unsqueeze(1)
        return prob

    return evaluate_model(
        predict_fn=predict_fn, eval_loader=test_loader, device=device,
        model_name=model_name, epoch=None, wandb_log=wandb_log,
    )
