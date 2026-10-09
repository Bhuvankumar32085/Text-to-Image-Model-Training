"""Conditional U-Net component wrapper with memory optimization and attention configurations."""
import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class UNetComponent(nn.Module):
    """Wrapper for UNet2DConditionModel in Stable Diffusion 1.5."""

    def __init__(
        self,
        unet: nn.Module,
        gradient_checkpointing: bool = True,
        enable_sdpa: bool = True,
        enable_xformers: bool = False,
    ):
        super().__init__()
        self.unet = unet

        # Configure gradient checkpointing to reduce VRAM during backward pass
        if gradient_checkpointing:
            if hasattr(self.unet, "enable_gradient_checkpointing"):
                self.unet.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled on U-Net.")

        # Configure memory efficient attention
        if enable_xformers:
            try:
                if hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
                    self.unet.enable_xformers_memory_efficient_attention()
                    logger.info("xFormers memory-efficient attention enabled on U-Net.")
            except Exception as e:
                logger.warning(f"Failed to enable xFormers: {e}. Falling back to standard/SDPA attention.")

        if enable_sdpa and not enable_xformers:
            # PyTorch 2.0+ native Scaled Dot Product Attention (FlashAttention / Memory-Efficient)
            if hasattr(self.unet, "set_default_attn_processor"):
                try:
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    self.unet.set_attn_processor(AttnProcessor2_0())
                    logger.info("PyTorch 2.x SDPA attention processor enabled on U-Net.")
                except Exception:
                    pass

    @property
    def in_channels(self) -> int:
        return getattr(self.unet.config, "in_channels", 4)

    @property
    def out_channels(self) -> int:
        return getattr(self.unet.config, "out_channels", 4)

    @property
    def cross_attention_dim(self) -> int:
        return getattr(self.unet.config, "cross_attention_dim", 768)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[dict] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """Forward pass predicting noise or velocity from noisy latents.
        
        Args:
            sample: (B, in_channels, H/8, W/8) noisy latents
            timestep: (B,) diffusion timesteps
            encoder_hidden_states: (B, 77, 768) text conditioning
            
        Returns:
            predicted_noise_or_velocity: (B, out_channels, H/8, W/8)
        """
        unet_param = next(self.unet.parameters(), None)
        if unet_param is not None:
            target_dtype = unet_param.dtype
            if sample.dtype != target_dtype:
                sample = sample.to(dtype=target_dtype)
            if encoder_hidden_states is not None and encoder_hidden_states.dtype != target_dtype:
                encoder_hidden_states = encoder_hidden_states.to(dtype=target_dtype)

        out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict,
        )
        if return_dict and hasattr(out, "sample"):
            return out.sample
        return out[0] if isinstance(out, (tuple, list)) else out

