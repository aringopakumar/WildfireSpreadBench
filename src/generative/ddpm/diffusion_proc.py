'''
The code is modified from
https://github.com/tatakai1/classifier_free_ddim,

Diffusion model is based on "CLASSIFIER-FREE DIFFUSION GUIDANCE"
https://arxiv.org/abs/2207.12598,

Patch (data-range correctness)
------------------------------
Standard DDPM assumes data in [-1, 1]: the forward process and the sampler's
clip_denoised step both presume a zero-centered [-1, 1] signal. The wildfire
masks, however, are binary {0, 1} (FireSpreadDataset yields y = (y>0).long(),
which training casts to float). Training x_start in [0, 1] while the sampler
clamps reconstructions to [-1, 1] is a train/inference mismatch that
miscalibrates the output probability map and depresses AP.

Fix: centralize the scaling in this class so no call site can drift.
  - _scale_in:  [0,1] -> [-1,1], applied to x_start inside train_losses.
  - _scale_out: [-1,1] -> [0,1], applied when recovering a probability map.
  - sample_to_prob: runs the reverse process and returns a [0,1] map directly,
    so training/eval code never hand-rolls recovery (which previously did
    clamp(0,1) and silently discarded the [-1,0) half of the sampler range).
The existing clip to [-1, 1] in p_mean_variance is now correct, because the
model is trained on [-1, 1] data.
'''

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

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
    ):
        self.timesteps = timesteps
        
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
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
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

    # ---- Data-range scaling (centralized; see module docstring) ----
    @staticmethod
    def _scale_in(x01):
        """Binary/[0,1] data -> [-1,1] for the diffusion process."""
        return x01 * 2.0 - 1.0

    @staticmethod
    def _scale_out(x_pm1):
        """[-1,1] sampler output -> [0,1] probability map."""
        return ((x_pm1 + 1.0) / 2.0).clamp(0.0, 1.0)

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward diffusion : q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # mean and variance of q(x_t | x_0)
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # mean and variance of diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: reverse of q_sample
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t) 
    def p_mean_variance(self, model, x_t, t, cond_img, w, clip_denoised=True):
        # Conditional pass
        mask_cond = torch.ones(x_t.shape[0], device=x_t.device)
        pred_noise_cond = model(x_t, t, cond_img, mask_cond)
        
        # Unconditional pass
        mask_uncond = torch.zeros(x_t.shape[0], device=x_t.device)
        pred_noise_uncond = model(x_t, t, cond_img, mask_uncond)
        
        # CFG combination
        pred_noise = (1 + w) * pred_noise_cond - w * pred_noise_uncond
        
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            # Correct now that the model is trained on [-1, 1] data.
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
    
    # denoise step: sample x_{t-1} from x_t and pred noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond_img, w, clip_denoised=True):
        # pred mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, cond_img, w, clip_denoised=clip_denoised)
        
        noise = torch.randn_like(x_t)
        # no noise when t = 0 
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # denoise : reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond_img, w=2, clip_denoised=True, progress=True):
        batch_size = shape[0]
        device = next(model.parameters()).device
        
        # start from pure noise
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
        """Run the reverse process and return a [0,1] probability map [B,1,H,W].

        Single source of truth for inference recovery: the final sampler state
        lives in [-1,1]; this maps it back to [0,1]. Eval/training code should
        call this instead of hand-rolling clamp(0,1) on the raw sampler output.
        """
        B, _, H, W = cond_img.shape if cond_img.dim() == 4 else (cond_img.shape[0], 1, cond_img.shape[-2], cond_img.shape[-1])
        device = next(model.parameters()).device
        shape = (B, 1, H, W)
        imgs = self.p_sample_loop(model, shape, cond_img, w=w,
                                  clip_denoised=clip_denoised, progress=progress)
        final = torch.tensor(np.array(imgs[-1]), device=device)   # [-1,1]
        return self._scale_out(final)                              # [0,1]

    # sample new images
    @torch.no_grad
    def sample(self, model, image_size, cond_img, batch_size=8, channels=1, w=2, clip_denoised=True):
        # Changed default channels to 1 (since you are predicting 1 channel fire maps)
        return self.p_sample_loop(model, (batch_size, channels, image_size, image_size), cond_img, w, clip_denoised)
    
    # compute train losses
    def train_losses(self, model, x_start, t, cond_img, mask_c):
        # Scale binary/[0,1] target into [-1,1] so training matches the sampler.
        x_start = self._scale_in(x_start)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        
        predicted_noise = model(x_noisy, t, cond_img, mask_c)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss
