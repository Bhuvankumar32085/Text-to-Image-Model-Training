"""Validation loop implementation calculating validation loss and generating preview images."""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.config.schema import AppConfig
from src.models.pipeline import StableDiffusionFineTuningPipeline
from src.training.forward_diffusion import ForwardDiffusion
from src.training.loss import DiffusionLoss

logger = logging.getLogger(__name__)


class DiffusionValidator:
    """Evaluates diffusion models on held-out validation sets."""

    def __init__(
        self,
        config: AppConfig,
        pipeline: StableDiffusionFineTuningPipeline,
        forward_diffusion: ForwardDiffusion,
        loss_fn: DiffusionLoss,
        device: torch.device,
    ):
        self.config = config
        self.pipeline = pipeline
        self.forward_diffusion = forward_diffusion
        self.loss_fn = loss_fn
        self.device = device

    @torch.no_grad()
    def evaluate_loss(self, dataloader: DataLoader) -> Tuple[float, int, float]:
        """Compute average diffusion loss over validation dataloader.
        
        Returns:
            Tuple of (average_val_loss, total_samples, elapsed_time_sec)
        """
        self.pipeline.eval()
        start_time = time.perf_counter()

        total_loss = 0.0
        total_batches = 0
        total_samples = 0

        mixed_precision = self.config.training.mixed_precision.lower()
        use_amp = mixed_precision in ["fp16", "bf16"] and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        unet_dtype = next(self.pipeline.unet.parameters()).dtype

        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device) if batch["input_ids"] is not None else None
            attention_mask = batch["attention_mask"].to(self.device) if batch.get("attention_mask") is not None else None

            # 1. VAE encode (mode / mean for deterministic validation)
            latents, _, _ = self.pipeline.vae.encode(pixel_values, sample_mode=False)

            # 2. Text encode
            encoder_hidden_states = self.pipeline.text_encoder(input_ids, attention_mask=attention_mask)

            # 3. Sample timesteps & noise deterministically or uniformly
            noisy_latents, noise, timesteps = self.forward_diffusion.forward_noise(latents)

            # 4. Target
            target = self.pipeline.scheduler.get_target(latents, noise, timesteps)

            # 5. U-Net prediction under autocast or matched dtype
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    model_pred = self.pipeline.unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                    )
            else:
                model_pred = self.pipeline.unet(
                    sample=noisy_latents.to(dtype=unet_dtype),
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(dtype=unet_dtype),
                )

            loss = self.loss_fn(model_pred, target, timesteps)
            total_loss += loss.item()
            total_batches += 1
            total_samples += pixel_values.shape[0]

        elapsed = time.perf_counter() - start_time
        avg_loss = (total_loss / total_batches) if total_batches > 0 else float("nan")

        return avg_loss, total_samples, elapsed

    @torch.no_grad()
    def generate_previews(
        self,
        prompts: Optional[List[str]] = None,
        seed: int = 42,
    ) -> List[Image.Image]:
        """Generate visual preview samples using fixed prompts."""
        self.pipeline.eval()
        val_prompts = prompts or self.config.validation.fixed_prompts
        if not val_prompts:
            return []

        images = self.pipeline.generate(
            prompt=val_prompts,
            num_inference_steps=self.config.validation.num_inference_steps,
            guidance_scale=self.config.validation.guidance_scale,
            height=self.config.dataset.target_resolution[0],
            width=self.config.dataset.target_resolution[1],
            seed=seed,
        )
        return images
