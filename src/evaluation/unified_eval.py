"""
src/evaluation/unified_eval.py

Unified evaluation harness for WildfireSpreadBench.

Every model in the benchmark (BCE U-Net, Flow Matching, DDPM, ConvLSTM,
Logistic Regression) funnels through evaluate_model(). This guarantees
all numbers in the results tables come from identical data, identical
thresholds, and identical metric implementations.

Usage
-----
Wrap your model's inference in a predict_fn, then call evaluate_model:

    # BCE U-Net
    def predict_fn(x0):
        logits = model(x0)
        return torch.sigmoid(logits)          # → [B, 1, H, W] in [0,1]

    # DDPM
    def predict_fn(x0):
        preds = diffusion.p_sample_loop(model, x0.shape, x0)
        final = torch.tensor(preds[-1]).clamp(0, 1)
        return final

    # Flow Matching
    def predict_fn(x0):
        _, prob = integrate_flow(model, x0, n_steps=20, device=device)
        return prob

    results = evaluate_model(
        predict_fn   = predict_fn,
        eval_loader  = eval_loader,
        device       = device,
        model_name   = "BCE-UNet",
        epoch        = 10,           # omit or pass None for final eval
        threshold    = 0.5,
        wandb_log    = True,
    )

predict_fn signature
--------------------
    predict_fn(x0: Tensor) -> Tensor

    Input:  x0  — conditioning tensor already on `device`, shape [B, C, H, W]
    Output: probability map in [0, 1], shape [B, 1, H, W]

    The function must NOT call .to(device) internally — x0 is already there.
    The function must NOT call model.eval() — evaluate_model() handles that.

Metrics
-------
    AP        — Average Precision (area under PR curve), threshold-free.
                Primary ranking metric for the benchmark.
    F1        — 2·P·R / (P+R) at threshold=0.5. Harmonic mean of P and R.
    Precision — TP / (TP + FP) at threshold=0.5.
    Recall    — TP / (TP + FN) at threshold=0.5.
    IoU       — TP / (TP + FP + FN) at threshold=0.5. Also called Jaccard index.

Baselines (from WildfireSpreadTS paper, "All" feature set)
-----------------------------------------------------------
    Persistence:         AP ≈ 0.1930
    Logistic Regression: AP ≈ 0.2490
    ConvLSTM:            AP ≈ 0.2800
    ResNet18 U-Net:      AP ≈ 0.3330
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
) -> dict:
    """
    Run inference over the full eval loader and compute all benchmark metrics.

    Parameters
    ----------
    predict_fn : callable
        Wraps model inference. See module docstring for exact signature.
    eval_loader : DataLoader
        Yields (x0, x1) batches. x0 = conditioning, x1 = binary target.
        Both are CPU tensors; evaluate_model moves them to `device`.
    device : torch.device
    model_name : str
        Used in W&B keys and print output. E.g. "BCE-UNet", "DDPM", "Flow".
    epoch : int or None
        If provided, logged as a periodic checkpoint eval.
        If None, treated as the final evaluation.
    threshold : float
        Decision threshold for binary metrics (F1, Precision, Recall, IoU).
        AP is always threshold-free.
    wandb_log : bool
        Whether to push metrics to the active W&B run.
    verbose : bool
        Whether to print the results table.

    Returns
    -------
    dict with keys: ap, f1, precision, recall, iou
    All values are Python floats.
    """
    tag = f"epoch_{epoch}" if epoch is not None else "final"

    all_targets = []   # list of flat numpy arrays (binary)
    all_scores  = []   # list of flat numpy arrays (probabilities)

    with torch.no_grad():
        for step, (x0, x1) in enumerate(eval_loader):
            x0 = x0.to(device)
            x1 = x1.to(device)

            prob = predict_fn(x0)          # [B, 1, H, W], values in [0, 1]

            all_targets.append(x1.cpu().numpy().flatten())
            all_scores.append(prob.cpu().numpy().flatten())

            if verbose and step % 50 == 0:
                print(f"  [{model_name}] {step}/{len(eval_loader)} batches...")

    targets_flat = np.concatenate(all_targets)   # shape [N]
    scores_flat  = np.concatenate(all_scores)    # shape [N]
    binary_flat  = (scores_flat >= threshold).astype(np.float32)

    # --- Threshold-free ---
    ap = float(average_precision_score(targets_flat, scores_flat))

    # --- Threshold-dependent ---
    # precision_recall_fscore_support returns per-class values;
    # we take the positive class (index 1).
    p, r, f1, _ = precision_recall_fscore_support(
        targets_flat, binary_flat,
        labels=[1], average=None, zero_division=0,
    )
    precision = float(p[0])
    recall    = float(r[0])
    f1_score  = float(f1[0])

    # IoU (Jaccard) — computed from confusion matrix components
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
    print(f"  Persistence baseline AP:  0.1930")
    print(f"  ResNet18 U-Net AP target: ~0.3330")
    print(f"{sep}\n")


def _log_to_wandb(model_name: str, tag: str, epoch, results: dict):
    """
    Push metrics to W&B under a consistent key schema:
        eval/<model_name>/<metric>

    This keeps all models' metrics in the same W&B run without key collisions,
    making side-by-side comparison charts trivial to build.
    """
    prefix = f"eval/{model_name}"
    log_dict = {f"{prefix}/{k}": v for k, v in results.items()}
    if epoch is not None:
        log_dict["epoch"] = epoch
    wandb.log(log_dict)


# ---------------------------------------------------------------------------
# Convenience wrappers for each model family
# ---------------------------------------------------------------------------
# These thin wrappers let training scripts call evaluate_model without
# knowing the harness internals. Each one constructs the predict_fn for
# its model type and delegates to evaluate_model.

@torch.no_grad()
def evaluate_bce_unet(model, eval_loader, device, epoch=None, wandb_log=True):
    """BCE Segmentation U-Net (FireSegmentationUNet)."""
    model.eval()
    def predict_fn(x0):
        return torch.sigmoid(model(x0))

    return evaluate_model(
        predict_fn=predict_fn,
        eval_loader=eval_loader,
        device=device,
        model_name="BCE-UNet",
        epoch=epoch,
        wandb_log=wandb_log,
    )


@torch.no_grad()
def evaluate_flow(model, eval_loader, device, n_steps=50, epoch=None, wandb_log=True):
    """Pure Flow Matching (VectorFieldNet)."""
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
    from flow_matching_pure import integrate_flow
    model.eval()
    def predict_fn(x0):
        _, prob = integrate_flow(model, x0, n_steps=n_steps, device=device)
        return prob

    return evaluate_model(
        predict_fn=predict_fn,
        eval_loader=eval_loader,
        device=device,
        model_name="FlowMatching",
        epoch=epoch,
        wandb_log=wandb_log,
    )


@torch.no_grad()
def evaluate_ddpm(model, diffusion, eval_loader, device, epoch=None, wandb_log=True):
    """Classifier-Free DDPM (Unet + Diffusion)."""
    model.eval()
    def predict_fn(x0):
        preds = diffusion.p_sample_loop(model, x0.shape, x0)
        # preds is a list of numpy arrays, one per timestep; last = final sample
        final = np.array(preds[-1])                       # [B, 1, H, W]
        prob  = torch.tensor(final).clamp(0.0, 1.0)
        return prob.to(device)

    return evaluate_model(
        predict_fn=predict_fn,
        eval_loader=eval_loader,
        device=device,
        model_name="DDPM",
        epoch=epoch,
        wandb_log=wandb_log,
    )


# ---------------------------------------------------------------------------
# Lightning model evaluation (discriminative baselines)
# ---------------------------------------------------------------------------

def evaluate_lightning_model(
    model,
    datamodule,
    device,
    model_name: str,
    wandb_log: bool = True,
) -> dict:
    """
    Evaluate a PyTorch Lightning BaseModel subclass through the unified harness.

    This gives discriminative models (SMPModel, ConvLSTM, UTAE, etc.) identical
    metric computation to generative models, enabling apples-to-apples comparison.

    Parameters
    ----------
    model : BaseModel subclass, already loaded with weights.
    datamodule : FireSpreadDataModule, already configured.
    device : torch.device
    model_name : str
        Used in W&B keys, e.g. "SMPModel", "ConvLSTMLightning".
    wandb_log : bool
        Whether to push metrics to the active W&B run.

    Returns
    -------
    dict with keys: ap, f1, precision, recall, iou
    """
    model.to(device)
    model.eval()

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    def predict_fn(x0):
        # x0 arrives as [B, T, C, H, W] from FireSpreadDataset
        x0 = x0[:, 0, :, :, :]   # [B, C, H, W]
        logits = model(x0)        # BaseModel.forward returns logits
        prob = torch.sigmoid(logits)
        if prob.dim() == 3:
            prob = prob.unsqueeze(1)  # ensure [B, 1, H, W]
        return prob

    return evaluate_model(
        predict_fn=predict_fn,
        eval_loader=test_loader,
        device=device,
        model_name=model_name,
        epoch=None,
        wandb_log=wandb_log,
    )