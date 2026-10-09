"""Forward diffusion process module."""
from typing import Optional, Tuple
import torch
import torch.nn as nn

from src.models.scheduler_component import SchedulerComponent


class ForwardDiffusion:
    """Manages noise injection and timestep sampling for the forward diffusion process."""

    def __init__(self, scheduler: SchedulerComponent):
        self.scheduler = scheduler

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample random discrete timesteps uniformly."""
        num_timesteps = self.scheduler.num_train_timesteps
        return torch.randint(
            0,
            num_timesteps,
            (batch_size,),
            device=device,
            generator=generator,
            dtype=torch.long,
        )

    def sample_noise(
        self,
        latents: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample standard Gaussian noise matching the shape, device, and dtype of latents."""
        return torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )

    def forward_noise(
        self,
        latents: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute the forward process: sample timesteps & noise, and construct noisy latents.
        
        Returns:
            Tuple of (noisy_latents, noise, timesteps)
        """
        batch_size = latents.shape[0]
        device = latents.device

        if timesteps is None:
            timesteps = self.sample_timesteps(batch_size, device, generator=generator)

        if noise is None:
            noise = self.sample_noise(latents, generator=generator)

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        return noisy_latents, noise, timesteps
