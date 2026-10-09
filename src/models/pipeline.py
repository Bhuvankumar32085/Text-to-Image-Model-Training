"""Unified Stable Diffusion Fine-Tuning Pipeline encapsulating all components and inference."""
import logging
from pathlib import Path
from typing import Any, List, Optional, Union
from PIL import Image
import torch
import torch.nn as nn

from src.config.schema import AppConfig
from src.models.component_loader import get_torch_dtype
from src.models.scheduler_component import SchedulerComponent
from src.models.text_encoder_component import TextEncoderComponent
from src.models.unet_component import UNetComponent
from src.models.vae_component import VAEComponent
from src.utils.image_utils import tensor_to_pil

logger = logging.getLogger(__name__)


class StableDiffusionFineTuningPipeline(nn.Module):
    """Encapsulates tokenizer, text encoder, VAE, U-Net, and scheduler."""

    def __init__(
        self,
        config: AppConfig,
        tokenizer: Any,
        text_encoder: nn.Module,
        vae: nn.Module,
        unet: nn.Module,
        scheduler: Any,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.text_encoder_raw = text_encoder
        self.vae_raw = vae
        self.unet_raw = unet
        self.scheduler_raw = scheduler

        self.text_encoder = TextEncoderComponent(text_encoder)
        self.vae = VAEComponent(vae)
        self.unet = UNetComponent(
            unet,
            gradient_checkpointing=config.training.gradient_checkpointing,
            enable_sdpa=config.training.enable_sdpa,
            enable_xformers=config.training.enable_xformers_memory_efficient_attention,
        )
        self.scheduler = SchedulerComponent(scheduler)

    def to_device(self, device: torch.device) -> "StableDiffusionFineTuningPipeline":
        """Move components to target device with appropriate precision."""
        self.to(device)
        return self

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        scheduler_type: Optional[str] = None,
    ) -> List[Image.Image]:
        """Generate images via Classifier-Free Guidance (CFG) denoising loop."""
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype

        if isinstance(prompt, str):
            prompts = [prompt] * num_images_per_prompt
        else:
            prompts = prompt

        batch_size = len(prompts)

        if negative_prompt is None:
            negative_prompts = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompts = [negative_prompt] * batch_size
        else:
            negative_prompts = negative_prompt

        # Set generator seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        # 1. Tokenize text prompts
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = self.text_encoder(text_input_ids)

        # 2. Tokenize negative prompts for Classifier-Free Guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            uncond_inputs = self.tokenizer(
                negative_prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_inputs.input_ids.to(device)
            negative_prompt_embeds = self.text_encoder(uncond_input_ids)

            # Concatenate unconditional and conditional text embeddings
            text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        else:
            text_embeddings = prompt_embeds

        text_embeddings = text_embeddings.to(dtype=dtype)

        # 3. Setup inference scheduler
        from diffusers import (
            DDPMScheduler,
            DPMSolverMultistepScheduler,
            EulerDiscreteScheduler,
            PNDMScheduler,
        )

        sched_name = scheduler_type or self.config.inference.scheduler_type
        if sched_name == "DPMSolverMultistepScheduler":
            inference_scheduler = DPMSolverMultistepScheduler.from_config(self.scheduler_raw.config)
        elif sched_name == "EulerDiscreteScheduler":
            inference_scheduler = EulerDiscreteScheduler.from_config(self.scheduler_raw.config)
        elif sched_name == "PNDMScheduler":
            inference_scheduler = PNDMScheduler.from_config(self.scheduler_raw.config)
        else:
            inference_scheduler = DDPMScheduler.from_config(self.scheduler_raw.config)

        inference_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = inference_scheduler.timesteps

        # 4. Prepare initial random Gaussian latents
        latent_channels = self.vae.latent_channels
        latent_height = height // 8
        latent_width = width // 8
        shape = (batch_size, latent_channels, latent_height, latent_width)

        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        # Scale initial noise by scheduler's init_noise_sigma if supported
        if hasattr(inference_scheduler, "init_noise_sigma"):
            latents = latents * inference_scheduler.init_noise_sigma

        # 5. Denoising loop
        for t in timesteps:
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if hasattr(inference_scheduler, "scale_model_input"):
                latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)

            # Predict noise residual
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
            )

            # Perform CFG guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

        # 6. Decode latents with VAE
        images_tensor = self.vae.decode(latents)  # (B, 3, H, W) in [-1, 1]

        # Convert to PIL images
        images: List[Image.Image] = []
        for i in range(images_tensor.shape[0]):
            img_pil = tensor_to_pil(images_tensor[i], normalization="neg_one_to_one")
            images.append(img_pil)

        return images

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Export full deployable pipeline in official Hugging Face Diffusers format."""
        from diffusers import StableDiffusionPipeline

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        hf_pipeline = StableDiffusionPipeline(
            vae=self.vae_raw,
            text_encoder=self.text_encoder_raw,
            tokenizer=self.tokenizer,
            unet=self.unet_raw,
            scheduler=self.scheduler_raw,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        hf_pipeline.save_pretrained(save_dir)
        logger.info(f"Saved complete fine-tuned pipeline to {save_dir}")
