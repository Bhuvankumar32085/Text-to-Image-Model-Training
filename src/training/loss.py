"""Diffusion loss computation with support for MSE, Huber, and SNR weighting."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """Computes diffusion loss between model prediction and ground-truth target."""

    def __init__(
        self,
        loss_type: str = "mse",
        snr_gamma: Optional[float] = None,
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.snr_gamma = snr_gamma

    def forward(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        alphas_cumprod: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute training loss in float32 for numerical stability.
        
        Args:
            model_pred: Predicted noise or velocity from U-Net (B, C, H, W)
            target: Ground-truth target (noise or velocity) (B, C, H, W)
            timesteps: Diffusion timesteps (B,)
            alphas_cumprod: Scheduler cumulative alphas for SNR weighting
            
        Returns:
            Scalar loss tensor
        """
        # Always compute loss in float32 to prevent fp16 underflow/overflow
        pred_f32 = model_pred.float()
        target_f32 = target.float()

        if self.loss_type == "mse":
            loss = F.mse_loss(pred_f32, target_f32, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred_f32, target_f32, reduction="none")
        elif self.loss_type in ["huber", "smooth_l1"]:
            loss = F.smooth_l1_loss(pred_f32, target_f32, reduction="none")
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        # Mean over spatial dimensions (B, C, H, W) -> (B,)
        loss = loss.mean(dim=list(range(1, len(loss.shape))))

        # Apply Min-SNR weighting if configured
        if self.snr_gamma is not None and timesteps is not None and alphas_cumprod is not None:
            alphas = alphas_cumprod.to(device=timesteps.device)[timesteps]
            snr = alphas / (1.0 - alphas)
            gamma_tensor = torch.tensor(self.snr_gamma, device=snr.device)
            snr_weight = torch.stack([snr, gamma_tensor], dim=1).min(dim=1)[0] / snr
            loss = loss * snr_weight

        return loss.mean()
