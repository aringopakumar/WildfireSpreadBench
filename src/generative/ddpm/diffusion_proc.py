'''
The code is modified from
https://github.com/tatakai1/classifier_free_ddim,

Diffusion model is based on "CLASSIFIER-FREE DIFFUSION GUIDANCE"
https://arxiv.org/abs/2207.12598,

SDF reformulation (sparsity fix)
--------------------------------
Predicting the binary next-day fire mask directly (data in {0,1}) is degenerate
under the extreme sparsity of this task (~0.1% positive pixels): the denoising
MSE is dominated by trivial background, training loss collapses toward zero
without using the conditioning, and AP falls to the base rate. This was observed
directly (DDPM AP ~0.001 at epoch 20).

Fix, consistent with the flow-matching model: the diffusion target is the
*normalized signed distance function* (SDF) of the next-day mask, using the SAME
helpers and fixed train-set normalization as src/generative/flow_matching.py
(mask_to_sdf, estimate_sdf_stats, normalize_sdf, sdf_to_prob, SDF_TRUNC). The SDF
is dense and smooth, so the denoising target carries signal everywhere and the
model must use the conditioning. Recovery: reverse-diffuse to a normalized SDF,
denormalize to pixel units, threshold at 0 for the mask, and map to a [0,1]
pseudo-probability via sdf_to_prob for AP.

Important interaction with clip_denoised
----------------------------------------
Standard DDPM clips the predicted x0 to [-1, 1] (valid for [-1,1] image data).
A normalized SDF is NOT in [-1, 1] (it can reach ~-trunc/std, e.g. -9). Clipping
to [-1, 1] would destroy the field. We therefore clip to the *actual* normalized
SDF bounds, derived from the fixed (mean, std, trunc): a pixel-unit SDF lives in
[-trunc, +trunc], so the normalized value lives in
    [(-trunc - mean)/std, (+trunc - mean)/std].
These bounds are computed at construction time.
'''

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

# Reuse the EXACT SDF machinery from the flow-matching model so the two
# generative baselines are provably identical in their SDF handling.
from generative.flow_matching import (
    mask_to_sdf, estimate_sdf_stats, normalize_sdf, denormalize_sdf,
    sdf_to_prob, SDF_TRUNC,
)


# Helper functions for beta schedules
def linear_noise_schedule(timesteps):
    scale = 1000 / timesteps
    return torch.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=torch.float64)

def sigmoid_noise_schedule(timesteps):
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas)/(betas.max()-betas.min())*(0.02-betas.min())/10
    return betas

def cosine_noise_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion:
    def __init__(
        self,
        timesteps=1000,
        noise_schedule='linear',
        sdf_mean=0.0,
        sdf_std=1.0,
        sdf_trunc=SDF_TRUNC,
    ):
        self.timesteps = timesteps

        # SDF normalization stats (fixed, from training masks) + clamp bounds.
        self.sdf_mean = float(sdf_mean)
        self.sdf_std  = float(sdf_std)
        self.sdf_trunc = float(sdf_trunc)
        # Valid normalized-SDF range used for clip_denoised (see module docstring).
        self.clip_lo = (-self.sdf_trunc - self.sdf_mean) / self.sdf_std
        self.clip_hi = ( self.sdf_trunc - self.sdf_mean) / self.sdf_std
        # Normalized location of the raw SDF zero level set (mask boundary).
        self.recover_thr = -self.sdf_mean / self.sdf_std

        if noise_schedule == 'linear':
            betas = linear_noise_schedule(timesteps)
        elif noise_schedule == 'cosine':
            betas = cosine_noise_schedule(timesteps)
        elif noise_schedule == 'sigmoid':
            betas = sigmoid_noise_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta schedule {noise_schedule}')

        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    # ---- SDF target construction (single source of truth via flow helpers) ----
    def mask_to_target(self, x1_mask):
        """Binary mask {0,1} -> normalized SDF target for the diffusion process."""
        return normalize_sdf(mask_to_sdf(x1_mask, trunc=self.sdf_trunc),
                             self.sdf_mean, self.sdf_std)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t, t, cond_img, w, clip_denoised=True):
        mask_cond = torch.ones(x_t.shape[0], device=x_t.device)
        pred_noise_cond = model(x_t, t, cond_img, mask_cond)
        mask_uncond = torch.zeros(x_t.shape[0], device=x_t.device)
        pred_noise_uncond = model(x_t, t, cond_img, mask_uncond)
        pred_noise = (1 + w) * pred_noise_cond - w * pred_noise_uncond

        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            # Clip to the valid NORMALIZED-SDF range, not [-1,1]: the SDF target
            # is unit-scale but can reach ~ -trunc/std, so a [-1,1] clip would
            # destroy the field. Bounds derived from fixed (mean, std, trunc).
            x_recon = torch.clamp(x_recon, min=self.clip_lo, max=self.clip_hi)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond_img, w, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, cond_img, w, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond_img, w=2, clip_denoised=True, progress=True):
        batch_size = shape[0]
        device = next(model.parameters()).device
        img = torch.randn(shape, device=device)
        imgs = []
        iterator = reversed(range(0, self.timesteps))
        if progress:
            iterator = tqdm(iterator, desc='sampling loop time step', total=self.timesteps)
        for i in iterator:
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), cond_img, w, clip_denoised)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample_to_prob(self, model, cond_img, w=2, clip_denoised=True, progress=False):
        """Reverse-diffuse to a normalized SDF, then recover a [0,1] prob map.

        normalized SDF -> denormalize to pixel units -> sdf_to_prob (sigmoid of
        -sdf/scale). Single source of truth for inference recovery, using the
        same flow-matching helpers so DDPM and flow are directly comparable.
        """
        if cond_img.dim() == 4:
            B, _, H, W = cond_img.shape
        else:
            B, H, W = cond_img.shape[0], cond_img.shape[-2], cond_img.shape[-1]
        device = next(model.parameters()).device
        shape = (B, 1, H, W)
        imgs = self.p_sample_loop(model, shape, cond_img, w=w,
                                  clip_denoised=clip_denoised, progress=progress)
        sdf_norm = torch.tensor(np.array(imgs[-1]), device=device)   # normalized SDF
        sdf_pixels = denormalize_sdf(sdf_norm, self.sdf_mean, self.sdf_std)
        return sdf_to_prob(sdf_pixels)                               # [0,1]

    @torch.no_grad()
    def sample_to_mask(self, model, cond_img, w=2, clip_denoised=True, progress=False):
        """Like sample_to_prob but returns the hard binary mask (SDF <= 0)."""
        if cond_img.dim() == 4:
            B, _, H, W = cond_img.shape
        else:
            B, H, W = cond_img.shape[0], cond_img.shape[-2], cond_img.shape[-1]
        device = next(model.parameters()).device
        shape = (B, 1, H, W)
        imgs = self.p_sample_loop(model, shape, cond_img, w=w,
                                  clip_denoised=clip_denoised, progress=progress)
        sdf_norm = torch.tensor(np.array(imgs[-1]), device=device)
        return (sdf_norm <= self.recover_thr).float()

    @torch.no_grad
    def sample(self, model, image_size, cond_img, batch_size=8, channels=1, w=2, clip_denoised=True):
        return self.p_sample_loop(model, (batch_size, channels, image_size, image_size), cond_img, w, clip_denoised)

    # compute train losses
    def train_losses(self, model, x_start_mask, t, cond_img, mask_c):
        # x_start_mask is the binary {0,1} next-day mask; convert to normalized SDF.
        x_start = self.mask_to_target(x_start_mask)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, cond_img, mask_c)
        loss = F.mse_loss(noise, predicted_noise)
        return loss
