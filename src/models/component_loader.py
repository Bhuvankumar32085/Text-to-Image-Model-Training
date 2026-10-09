"""Pretrained component loader with strict compatibility checks and trainability enforcement."""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.config.schema import AppConfig

logger = logging.getLogger(__name__)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Map string to torch dtype."""
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(dtype_str.lower(), torch.float32)


def verify_model_compatibility(
    tokenizer: Any,
    text_encoder: nn.Module,
    vae: nn.Module,
    unet: nn.Module,
    scheduler: Any,
) -> Dict[str, Any]:
    """Verify architectural compatibility between all loaded Stable Diffusion 1.5 components.
    
    Raises:
        ValueError: If component dimensions or interfaces mismatch.
    """
    report = {}

    # 1. Text Encoder hidden size vs U-Net cross_attention_dim
    text_hidden_size = getattr(text_encoder.config, "hidden_size", 768)
    unet_cross_attn_dim = getattr(unet.config, "cross_attention_dim", 768)
    if text_hidden_size != unet_cross_attn_dim:
        raise ValueError(
            f"Compatibility error: Text encoder hidden_size ({text_hidden_size}) does not match "
            f"U-Net cross_attention_dim ({unet_cross_attn_dim})."
        )
    report["cross_attention_dim"] = unet_cross_attn_dim

    # 2. VAE latent channels vs U-Net input channels
    vae_latent_channels = getattr(vae.config, "latent_channels", 4)
    unet_in_channels = getattr(unet.config, "in_channels", 4)
    if vae_latent_channels != unet_in_channels:
        raise ValueError(
            f"Compatibility error: VAE latent_channels ({vae_latent_channels}) does not match "
            f"U-Net in_channels ({unet_in_channels})."
        )
    report["latent_channels"] = vae_latent_channels

    # 3. Parameter counts
    def count_params(module: nn.Module) -> Tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    unet_total, unet_trainable = count_params(unet)
    te_total, te_trainable = count_params(text_encoder)
    vae_total, vae_trainable = count_params(vae)

    report["unet_params"] = {"total": unet_total, "trainable": unet_trainable}
    report["text_encoder_params"] = {"total": te_total, "trainable": te_trainable}
    report["vae_params"] = {"total": vae_total, "trainable": vae_trainable}
    report["scheduler_prediction_type"] = getattr(scheduler.config, "prediction_type", "epsilon")

    logger.info(
        f"Compatibility verified: U-Net ({unet_total / 1e6:.1f}M params), "
        f"Text Encoder ({te_total / 1e6:.1f}M params), VAE ({vae_total / 1e6:.1f}M params). "
        f"Scheduler target: {report['scheduler_prediction_type']}."
    )
    return report


def apply_trainability_policy(
    config: AppConfig,
    unet: nn.Module,
    text_encoder: nn.Module,
    vae: nn.Module,
) -> None:
    """Enforce parameter freezing and training policy as specified in configuration."""
    train_unet = config.fine_tuning.train_unet and ("unet" in config.fine_tuning.trainable_components)
    train_te = config.fine_tuning.train_text_encoder and ("text_encoder" in config.fine_tuning.trainable_components)
    train_vae = config.fine_tuning.train_vae and ("vae" in config.fine_tuning.trainable_components)

    # U-Net
    unet.requires_grad_(train_unet)
    if train_unet:
        unet.train()
    else:
        unet.eval()

    # Text Encoder
    text_encoder.requires_grad_(train_te)
    if train_te:
        text_encoder.train()
    else:
        text_encoder.eval()

    # VAE
    vae.requires_grad_(train_vae)
    if train_vae:
        vae.train()
    else:
        vae.eval()

    logger.info(
        f"Applied trainability policy: U-Net trainable={train_unet}, "
        f"Text Encoder trainable={train_te}, VAE trainable={train_vae}"
    )


def load_model_components(config: AppConfig) -> Tuple[Any, nn.Module, nn.Module, nn.Module, Any]:
    """Load tokenizer, text_encoder, vae, unet, and scheduler using Hugging Face Diffusers & Transformers.
    
    Returns:
        (tokenizer, text_encoder, vae, unet, scheduler)
    """
    from transformers import CLIPTokenizer, CLIPTextModel
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

    model_id = config.model.local_pretrained_model_dir or config.model.model_identifier
    revision = config.model.revision
    cache_dir = config.model.cache_dir
    local_files_only = config.model.local_files_only
    subfolders = config.model.subfolder_overrides
    torch_dtype = get_torch_dtype(config.model.dtype)

    logger.info(f"Loading Stable Diffusion 1.5 components from: {model_id} (revision: {revision})")

    # 1. Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder=subfolders.get("tokenizer", "tokenizer"),
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    # 2. Text Encoder
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder=subfolders.get("text_encoder", "text_encoder"),
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )

    # 3. VAE
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder=subfolders.get("vae", "vae"),
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )

    # 4. U-Net
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder=subfolders.get("unet", "unet"),
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        torch_dtype=torch.float32 if config.fine_tuning.train_unet else torch_dtype,
    )

    # 5. Noise Scheduler
    scheduler = DDPMScheduler.from_pretrained(
        model_id,
        subfolder=subfolders.get("scheduler", "scheduler"),
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    # Apply freezing policy
    apply_trainability_policy(config, unet, text_encoder, vae)

    # Strict compatibility verification
    if config.model.strict_compatibility_check:
        verify_model_compatibility(tokenizer, text_encoder, vae, unet, scheduler)

    return tokenizer, text_encoder, vae, unet, scheduler
