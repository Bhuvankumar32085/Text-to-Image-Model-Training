"""Noise Scheduler component wrapper handling forward diffusion and velocity targets."""
from typing import Any, Optional, Tuple
import torch


class SchedulerComponent:
    """Wrapper around Hugging Face Diffusers noise scheduler."""

    def __init__(self, scheduler: Any):
        self.scheduler = scheduler

    @property
    def num_train_timesteps(self) -> int:
        return getattr(self.scheduler.config, "num_train_timesteps", 1000)

    @property
    def prediction_type(self) -> str:
        return getattr(self.scheduler.config, "prediction_type", "epsilon")

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random discrete timesteps uniformly from [0, num_train_timesteps)."""
        return torch.randint(
            0,
            self.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0) adding Gaussian noise at timesteps."""
        return self.scheduler.add_noise(original_samples, noise, timesteps)

    def get_target(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the ground-truth training target according to scheduler prediction_type.
        
        - 'epsilon': target is the added Gaussian noise.
        - 'v_prediction': target is velocity = alpha_t * noise - sigma_t * original_samples.
        - 'sample': target is the clean original_samples.
        """
        pred_type = self.prediction_type
        if pred_type == "epsilon":
            return noise
        elif pred_type == "v_prediction":
            if hasattr(self.scheduler, "get_velocity"):
                return self.scheduler.get_velocity(original_samples, noise, timesteps)
            else:
                # Manual calculation of velocity if method is missing
                alphas_cumprod = self.scheduler.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
                alpha_t = alphas_cumprod[timesteps].sqrt()
                sigma_t = (1 - alphas_cumprod[timesteps]).sqrt()
                while len(alpha_t.shape) < len(original_samples.shape):
                    alpha_t = alpha_t.unsqueeze(-1)
                    sigma_t = sigma_t.unsqueeze(-1)
                return alpha_t * noise - sigma_t * original_samples
        elif pred_type == "sample":
            return original_samples
        else:
            raise ValueError(f"Unknown prediction_type: {pred_type}")
