"""Optimizer and Learning Rate Scheduler construction."""
import logging
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler

from src.config.schema import AppConfig
from src.models.pipeline import StableDiffusionFineTuningPipeline

logger = logging.getLogger(__name__)


def create_optimizer_and_scheduler(
    config: AppConfig,
    pipeline: StableDiffusionFineTuningPipeline,
    total_training_steps: int,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Create optimizer and LR scheduler respecting component trainability and parameter groups.
    
    Returns:
        (optimizer, lr_scheduler)
    """
    train_cfg = config.training
    ft_cfg = config.fine_tuning

    params_to_optimize: List[Dict[str, Any]] = []

    # 1. U-Net parameter group
    if ft_cfg.train_unet and "unet" in ft_cfg.trainable_components:
        unet_params = [p for p in pipeline.unet.parameters() if p.requires_grad]
        if unet_params:
            unet_lr = train_cfg.component_learning_rates.get("unet", train_cfg.learning_rate)
            params_to_optimize.append({
                "params": unet_params,
                "lr": unet_lr,
                "weight_decay": train_cfg.optimizer_params.get("weight_decay", 0.01),
                "name": "unet",
            })
            logger.info(f"Added U-Net to optimizer: {len(unet_params)} parameter tensors, LR={unet_lr}")

    # 2. Text Encoder parameter group
    if ft_cfg.train_text_encoder and "text_encoder" in ft_cfg.trainable_components:
        te_params = [p for p in pipeline.text_encoder.parameters() if p.requires_grad]
        if te_params:
            te_lr = train_cfg.component_learning_rates.get("text_encoder", train_cfg.learning_rate)
            params_to_optimize.append({
                "params": te_params,
                "lr": te_lr,
                "weight_decay": train_cfg.optimizer_params.get("weight_decay", 0.01),
                "name": "text_encoder",
            })
            logger.info(f"Added Text Encoder to optimizer: {len(te_params)} parameter tensors, LR={te_lr}")

    # 3. VAE parameter group
    if ft_cfg.train_vae and "vae" in ft_cfg.trainable_components:
        vae_params = [p for p in pipeline.vae.parameters() if p.requires_grad]
        if vae_params:
            vae_lr = train_cfg.component_learning_rates.get("vae", train_cfg.learning_rate)
            params_to_optimize.append({
                "params": vae_params,
                "lr": vae_lr,
                "weight_decay": train_cfg.optimizer_params.get("weight_decay", 0.01),
                "name": "vae",
            })
            logger.info(f"Added VAE to optimizer: {len(vae_params)} parameter tensors, LR={vae_lr}")

    if not params_to_optimize:
        raise ValueError(
            "No trainable parameters found! Ensure at least one component has requires_grad=True "
            "and is listed in fine_tuning.trainable_components."
        )

    # 4. Optimizer instantiation
    opt_name = train_cfg.optimizer.lower()
    opt_params = train_cfg.optimizer_params
    betas = tuple(opt_params.get("betas", [0.9, 0.999]))
    eps = float(opt_params.get("eps", 1e-8))

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=train_cfg.learning_rate,
            betas=betas,
            eps=eps,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=train_cfg.learning_rate,
            betas=betas,
            eps=eps,
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=train_cfg.learning_rate,
            momentum=opt_params.get("momentum", 0.9),
            weight_decay=opt_params.get("weight_decay", 0.01),
        )
    elif opt_name == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            params_to_optimize,
            lr=train_cfg.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=train_cfg.optimizer_params.get("weight_decay", 0.01),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # 5. LR Scheduler instantiation
    lr_scheduler = get_scheduler(
        name=train_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_cfg.lr_warmup_steps,
        num_training_steps=total_training_steps,
    )

    return optimizer, lr_scheduler
