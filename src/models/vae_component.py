"""VAE Encoder / Decoder component wrapper with latent scaling and posterior inspection."""
from typing import Optional, Tuple
import torch
import torch.nn as nn


class VAEComponent(nn.Module):
    """Wrapper for AutoencoderKL providing latent encoding, scaling, decoding, and posterior stats."""

    def __init__(self, vae: nn.Module, scaling_factor: Optional[float] = None):
        super().__init__()
        self.vae = vae
        # Stable Diffusion 1.5 default scaling factor is 0.18215
        if scaling_factor is not None:
            self.scaling_factor = float(scaling_factor)
        elif hasattr(vae, "config") and hasattr(vae.config, "scaling_factor"):
            self.scaling_factor = float(vae.config.scaling_factor)
        else:
            self.scaling_factor = 0.18215

    @property
    def latent_channels(self) -> int:
        if hasattr(self.vae, "config") and hasattr(self.vae.config, "latent_channels"):
            return self.vae.config.latent_channels
        return 4

    def encode(self, pixel_values: torch.Tensor, sample_mode: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode normalized image tensors into scaled latent space.
        
        Args:
            pixel_values: Image tensor (B, C, H, W) normalized to [-1, 1]
            sample_mode: True to sample from Gaussian posterior, False to use mean (mode)
            
        Returns:
            Tuple of (scaled_latents, posterior_mean, posterior_logvar)
        """
        # AutoencoderKL expects input in the same dtype as VAE weights
        vae_dtype = next(self.vae.parameters()).dtype if len(list(self.vae.parameters())) > 0 else pixel_values.dtype
        pixel_values_typed = pixel_values.to(dtype=vae_dtype)

        posterior = self.vae.encode(pixel_values_typed).latent_dist
        if sample_mode:
            latents = posterior.sample()
        else:
            latents = posterior.mode()

        scaled_latents = latents * self.scaling_factor
        mean = posterior.mean
        logvar = posterior.logvar

        return scaled_latents, mean, logvar

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode scaled latents back to image pixel space.
        
        Args:
            latents: Scaled latent tensor (B, 4, H/8, W/8)
            
        Returns:
            Reconstructed image tensor (B, 3, H, W) in range [-1, 1]
        """
        vae_dtype = next(self.vae.parameters()).dtype if len(list(self.vae.parameters())) > 0 else latents.dtype
        unscaled_latents = (latents / self.scaling_factor).to(dtype=vae_dtype)
        decoded = self.vae.decode(unscaled_latents).sample
        return decoded
