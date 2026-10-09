"""Single training step execution with mixed precision, gradient accumulation, and tensor diagnostics."""
import logging
import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.config.schema import AppConfig
from src.models.pipeline import StableDiffusionFineTuningPipeline
from src.training.forward_diffusion import ForwardDiffusion
from src.training.loss import DiffusionLoss

logger = logging.getLogger(__name__)


class TrainStepExecutor:
    """Executes single forward, loss, backward, and optimization steps."""

    def __init__(
        self,
        config: AppConfig,
        pipeline: StableDiffusionFineTuningPipeline,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        loss_fn: DiffusionLoss,
        forward_diffusion: ForwardDiffusion,
        device: torch.device,
    ):
        self.config = config
        self.pipeline = pipeline
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.forward_diffusion = forward_diffusion
        self.device = device

        self.mixed_precision = config.training.mixed_precision.lower()
        self.use_amp = self.mixed_precision in ["fp16", "bf16"] and device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16

        self.scaler = torch.amp.GradScaler("cuda") if (self.use_amp and self.mixed_precision == "fp16") else None
        self.grad_accum_steps = config.training.gradient_accumulation_steps
        self.max_grad_norm = config.training.max_grad_norm

    def execute_step(
        self,
        batch: Dict[str, Any],
        step_idx: int,
        collect_diagnostics: bool = False,
    ) -> Tuple[float, Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
        """Execute a single batch step.
        
        Returns:
            Tuple of (loss_value, timing_metrics, optional_diagnostic_tensors)
        """
        start_time = time.perf_counter()
        diag_tensors: Dict[str, torch.Tensor] = {} if collect_diagnostics else None

        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device) if batch["input_ids"] is not None else None
        attention_mask = batch["attention_mask"].to(self.device) if batch.get("attention_mask") is not None else None

        if collect_diagnostics:
            diag_tensors["input_pixel_values"] = pixel_values.detach()
            if input_ids is not None:
                diag_tensors["input_ids"] = input_ids.detach()

        # 1. Encode images to VAE latent space (frozen or trainable)
        with torch.set_grad_enabled(self.config.fine_tuning.train_vae):
            latents, mean, logvar = self.pipeline.vae.encode(pixel_values, sample_mode=True)

        if collect_diagnostics:
            diag_tensors["clean_latents"] = latents.detach()
            diag_tensors["vae_mean"] = mean.detach()
            diag_tensors["vae_logvar"] = logvar.detach()

        # 2. Encode text captions (frozen or trainable)
        with torch.set_grad_enabled(self.config.fine_tuning.train_text_encoder):
            encoder_hidden_states = self.pipeline.text_encoder(input_ids, attention_mask=attention_mask)

        if collect_diagnostics:
            diag_tensors["text_embeddings"] = encoder_hidden_states.detach()

        # 3. Forward diffusion: sample noise and timesteps
        noisy_latents, noise, timesteps = self.forward_diffusion.forward_noise(latents)

        if collect_diagnostics:
            diag_tensors["noisy_latents"] = noisy_latents.detach()
            diag_tensors["sampled_noise"] = noise.detach()
            diag_tensors["sampled_timesteps"] = timesteps.detach()

        # 4. Ground-truth target calculation
        target = self.pipeline.scheduler.get_target(latents, noise, timesteps)
        if collect_diagnostics:
            diag_tensors["target"] = target.detach()

        # 5. U-Net Forward Pass (under AMP autocast if enabled)
        fwd_start = time.perf_counter()
        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                model_pred = self.pipeline.unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                )
                loss = self.loss_fn(model_pred, target, timesteps)
        else:
            model_pred = self.pipeline.unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
            )
            loss = self.loss_fn(model_pred, target, timesteps)

        fwd_time = time.perf_counter() - fwd_start

        if collect_diagnostics:
            diag_tensors["unet_prediction"] = model_pred.detach()
            diag_tensors["batch_loss"] = loss.detach()

        # Check for NaN / Inf
        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError(f"Encountered NaN or Inf loss at step {step_idx}: loss={loss.item()}")

        # 6. Backward Pass with Gradient Accumulation
        scaled_loss = loss / self.grad_accum_steps
        bwd_start = time.perf_counter()
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        bwd_time = time.perf_counter() - bwd_start

        # 7. Optimizer update when accumulation boundary is reached
        opt_time = 0.0
        grad_norm = 0.0
        param_norm = 0.0

        if (step_idx + 1) % self.grad_accum_steps == 0:
            opt_start = time.perf_counter()
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Compute and clip gradient norm
            trainable_params = [p for p in self.pipeline.parameters() if p.requires_grad]
            if trainable_params:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm).item()
                param_norm = torch.norm(torch.stack([torch.norm(p.detach(), 2) for p in trainable_params]), 2).item()

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.optimizer.zero_grad(set_to_none=True)
            opt_time = time.perf_counter() - opt_start

        total_time = time.perf_counter() - start_time

        metrics = {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "fwd_time": fwd_time,
            "bwd_time": bwd_time,
            "opt_time": opt_time,
            "step_time": total_time,
        }

        return loss.item(), metrics, diag_tensors
