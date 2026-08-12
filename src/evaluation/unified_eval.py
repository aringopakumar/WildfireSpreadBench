"""
src/evaluation/unified_eval.py

Unified evaluation harness for WildfireSpreadBench.

Every model in the benchmark funnels through the same metric code, so all
numbers in the results tables come from identical data, thresholds, and metric
implementations.

The discriminative Lightning baselines (ResNet U-Net, ConvLSTM, UTAE,
Logistic Regression, Persistence) are evaluated through
evaluate_lightning_module(), which reuses each model's own get_pred_and_gt
path so temporal flattening, doy features, and tiled inference all match the
training-time code.

The generative models (BCE U-Net, Flow Matching, DDPM) are evaluated through
evaluate_model(), a model-agnostic harness: pass any predict_fn(x0) ->
probability map. evaluate_bce_unet / evaluate_flow / evaluate_ddpm are thin
wrappers that build the appropriate predict_fn.

predict_fn signature
--------------------
    predict_fn(x0: Tensor) -> Tensor
    Input:  x0  — conditioning tensor already on `device`. As yielded by the
                  loader this is [B, T, C, H, W]; predict_fn is responsible for
                  any reshaping (e.g. squeezing the leading time dim).
    Output: probability map in [0, 1], shape [B, 1, H, W]

max_batches: optional cap on the number of eval batches, useful for keeping
periodic in-training eval affordable. Final eval should pass None to use the
entire test set.

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

import torch
import torchmetrics
import wandb

# Number of thresholds used for the streaming AveragePrecision estimate. Using a
# fixed grid keeps memory bounded (a small per-threshold confusion matrix)
# instead of retaining every pixel score across the whole test set.
_AP_THRESHOLDS = 200


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

    f1_metric = torchmetrics.F1Score("binary", threshold=threshold).to(device)
    precision_metric = torchmetrics.Precision("binary", threshold=threshold).to(device)
    recall_metric = torchmetrics.Recall("binary", threshold=threshold).to(device)
    iou_metric = torchmetrics.JaccardIndex("binary", threshold=threshold).to(device)
    ap_metric = torchmetrics.AveragePrecision("binary", thresholds=_AP_THRESHOLDS).to(device)

    with torch.no_grad():
        for step, (x0, x1) in enumerate(eval_loader):
            if max_batches is not None and step >= max_batches:
                break
            x0 = x0.to(device)
            x1 = x1.to(device)

            prob = predict_fn(x0).flatten()   # values in [0, 1]
            target = x1.flatten().long()

            f1_metric.update(prob, target)
            precision_metric.update(prob, target)
            recall_metric.update(prob, target)
            iou_metric.update(prob, target)
            ap_metric.update(prob, target)

            if verbose and step % 50 == 0:
                print(f"  [{model_name}] {step}/{len(eval_loader)} batches...")

    results = dict(
        ap=float(ap_metric.compute()),
        f1=float(f1_metric.compute()),
        precision=float(precision_metric.compute()),
        recall=float(recall_metric.compute()),
        iou=float(iou_metric.compute()),
    )

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
# Generative model evaluation
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
    # Imported lazily: generative.flow_matching imports evaluate_model from this
    # module, so a top-level import here would be circular.
    from generative.flow_matching import integrate_flow, estimate_sdf_stats

    model.eval()
    # Flow matching needs SDF normalization stats; recover from the loader's
    # dataset so standalone eval matches training-time recovery.
    sdf_mean, sdf_std = estimate_sdf_stats(eval_loader.dataset)

    def predict_fn(x0):
        x0 = x0[:, 0, :, :, :]          # [B, T, C, H, W] -> [B, C, H, W]
        _, prob = integrate_flow(model, x0, sdf_mean, sdf_std,
                                 n_steps=n_steps, device=device)
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

def _move_batch_to_device(batch, device):
    if len(batch) == 3:
        x, y, doys = batch
        return x.to(device), y.to(device), doys.to(device)
    x, y = batch
    return x.to(device), y.to(device)


def evaluate_lightning_module(
    pl_module,
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
    Evaluate a PyTorch Lightning BaseModel subclass through the unified harness.

    Uses get_pred_and_gt so temporal flattening, doy features, and tiled
    inference all match the training-time code path.

    Metrics are accumulated incrementally with torchmetrics rather than by
    concatenating every pixel across the test set. The previous implementation
    retained all scores/targets in host RAM and then ran sklearn's
    average_precision_score (which sorts the full array), which OOM-crashed at
    the aggregation step on large test sets. The streaming approach uses bounded
    memory and matches BaseModel.test_step's metric definitions.
    """
    tag = f"epoch_{epoch}" if epoch is not None else "final"

    pl_module.eval()

    f1_metric = torchmetrics.F1Score("binary", threshold=threshold).to(device)
    precision_metric = torchmetrics.Precision("binary", threshold=threshold).to(device)
    recall_metric = torchmetrics.Recall("binary", threshold=threshold).to(device)
    iou_metric = torchmetrics.JaccardIndex("binary", threshold=threshold).to(device)
    ap_metric = torchmetrics.AveragePrecision("binary", thresholds=_AP_THRESHOLDS).to(device)

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            if max_batches is not None and step >= max_batches:
                break

            batch = _move_batch_to_device(batch, device)
            y_hat, y = pl_module.get_pred_and_gt(batch)
            prob = torch.sigmoid(y_hat).flatten()
            target = y.flatten().long()

            f1_metric.update(prob, target)
            precision_metric.update(prob, target)
            recall_metric.update(prob, target)
            iou_metric.update(prob, target)
            ap_metric.update(prob, target)

            if verbose and step % 50 == 0:
                print(f"  [{model_name}] {step}/{len(eval_loader)} batches...")

    results = dict(
        ap=float(ap_metric.compute()),
        f1=float(f1_metric.compute()),
        precision=float(precision_metric.compute()),
        recall=float(recall_metric.compute()),
        iou=float(iou_metric.compute()),
    )

    if verbose:
        _print_results(model_name, tag, results, threshold)
    if wandb_log:
        _log_to_wandb(model_name, tag, epoch, results)

    return results


def evaluate_lightning_model(
    model, datamodule, device, model_name: str, wandb_log: bool = True,
) -> dict:
    """
    Evaluate a PyTorch Lightning BaseModel subclass through the unified harness,
    giving discriminative models identical metric computation to generative ones.
    """
    model.to(device)

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    return evaluate_lightning_module(
        pl_module=model,
        eval_loader=test_loader,
        device=device,
        model_name=model_name,
        epoch=None,
        wandb_log=wandb_log,
    )
