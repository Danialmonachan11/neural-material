"""
Diffusion model implementation with DDPM and DDIM schedulers.
Core training and sampling logic for denoising diffusion probabilistic models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
from tqdm import tqdm


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Model (DDPM) noise scheduler.

    Implements the forward diffusion process (adding noise) and
    reverse sampling process (denoising).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        clip_sample: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample

        # Define beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == 'scaled_linear':
            # Stable Diffusion uses this
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)

        Args:
            original_samples: (B, C, H, W) clean samples
            noise: (B, C, H, W) Gaussian noise
            timesteps: (B,) timesteps
        Returns:
            (B, C, H, W) noisy samples
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)

        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Reverse diffusion step: p(x_{t-1} | x_t)

        Args:
            model_output: (B, C, H, W) predicted noise
            timestep: current timestep
            sample: (B, C, H, W) current noisy sample
            generator: random number generator
        Returns:
            (B, C, H, W) denoised sample at t-1
        """
        t = timestep

        # Predict x_0 from x_t and predicted noise
        alpha_prod_t = self.alphas_cumprod[t].to(sample.device)
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # Compute coefficients for q(x_{t-1} | x_t, x_0)
        pred_sample_coef = self.posterior_mean_coef1[t].to(sample.device)
        current_sample_coef = self.posterior_mean_coef2[t].to(sample.device)

        # Compute mean of q(x_{t-1} | x_t, x_0)
        pred_prev_sample = pred_sample_coef * pred_original_sample + current_sample_coef * sample

        # Add noise
        variance = 0
        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            variance = (self.posterior_variance[t] ** 0.5).to(sample.device) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample


class DDIMScheduler:
    """
    Denoising Diffusion Implicit Model (DDIM) scheduler.

    Faster sampling with fewer steps while maintaining quality.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        clip_sample: bool = True,
        eta: float = 0.0,  # 0 = deterministic, 1 = DDPM
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.eta = eta

        # Define beta schedule (same as DDPM)
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == 'scaled_linear':
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Standard deviation of q(x_{t-1} | x_t, x_0)
        self.final_alpha_cumprod = torch.tensor(1.0)

    def set_timesteps(self, num_inference_steps: int):
        """Set the discrete timesteps used for inference."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(self.timesteps)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        DDIM reverse step.

        Args:
            model_output: (B, C, H, W) predicted noise
            timestep: current timestep
            sample: (B, C, H, W) current sample
        Returns:
            (B, C, H, W) previous sample
        """
        # Get current and previous alpha_cumprod
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)

        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        if prev_timestep < 0:
            alpha_prod_t_prev = self.final_alpha_cumprod.to(sample.device)
        else:
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device)

        beta_prod_t = 1 - alpha_prod_t

        # Predict x_0
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # Compute variance
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = self.eta * variance ** 0.5

        # Direction pointing to x_t
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * model_output

        # Compute x_{t-1}
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        if self.eta > 0:
            noise = torch.randn_like(model_output)
            pred_prev_sample = pred_prev_sample + std_dev_t * noise

        return pred_prev_sample

    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        alpha_prod_t = self.alphas_cumprod[timestep]
        if prev_timestep < 0:
            alpha_prod_t_prev = self.final_alpha_cumprod
        else:
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Same as DDPM."""
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps]).to(original_samples.device)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.alphas_cumprod[timesteps]).to(original_samples.device)

        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


class DiffusionModel(nn.Module):
    """
    Main diffusion model wrapper combining UNet and scheduler.
    """

    def __init__(
        self,
        unet: nn.Module,
        scheduler_type: str = 'ddpm',
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.unet = unet
        self.num_train_timesteps = num_train_timesteps

        if scheduler_type == 'ddpm':
            self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        elif scheduler_type == 'ddim':
            self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
        else:
            raise ValueError(f'Unknown scheduler: {scheduler_type}')

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        encoder_hidden_states: torch.Tensor,
        height: int = 64,
        width: int = 64,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate samples using the diffusion model.

        Args:
            batch_size: number of samples to generate
            encoder_hidden_states: (B, seq_len, dim) conditioning
            height: latent height
            width: latent width
            num_inference_steps: number of denoising steps
            guidance_scale: classifier-free guidance scale
            generator: random number generator
        Returns:
            (B, C, H, W) generated samples
        """
        device = next(self.unet.parameters()).device

        # Start from pure noise
        latents = torch.randn(
            (batch_size, 4, height, width),
            generator=generator,
            device=device,
        )

        # Set timesteps
        if isinstance(self.scheduler, DDIMScheduler):
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
        else:
            timesteps = torch.linspace(
                self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long
            )

        # Denoising loop
        for t in tqdm(timesteps, desc='Sampling'):
            # Expand timestep
            timestep = t.to(device).long()
            timestep_batch = timestep.repeat(batch_size)

            # Predict noise
            noise_pred = self.unet(
                latents,
                timestep_batch,
                encoder_hidden_states=encoder_hidden_states,
            )

            # Perform guidance
            if guidance_scale > 1.0:
                noise_pred_uncond = self.unet(
                    latents,
                    timestep_batch,
                    encoder_hidden_states=torch.zeros_like(encoder_hidden_states),
                )
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # Denoise
            latents = self.scheduler.step(noise_pred, int(t), latents, generator=generator)

        return latents

    def training_step(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single training step for diffusion model.

        Args:
            latents: (B, C, H, W) clean latent samples
            encoder_hidden_states: (B, seq_len, dim) conditioning
        Returns:
            loss, predicted_noise
        """
        batch_size = latents.shape[0]
        device = latents.device

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=device
        ).long()

        # Sample noise
        noise = torch.randn_like(latents)

        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states)

        # Compute loss (simple MSE)
        loss = F.mse_loss(noise_pred, noise)

        return loss, noise_pred
