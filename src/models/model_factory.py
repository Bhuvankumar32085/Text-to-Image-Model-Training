"""Model factory and registry for instantiating diffusion pipelines."""
import logging
from typing import Callable, Dict
import torch.nn as nn

from src.config.schema import AppConfig
from src.models.component_loader import load_model_components
from src.models.pipeline import StableDiffusionFineTuningPipeline

logger = logging.getLogger(__name__)


class ModelFactory:
    """Registry pattern for supported diffusion model families."""

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, model_family: str):
        """Decorator to register a pipeline builder for a model family."""
        def decorator(fn: Callable):
            cls._registry[model_family.lower()] = fn
            return fn
        return decorator

    @classmethod
    def build_pipeline(cls, config: AppConfig) -> StableDiffusionFineTuningPipeline:
        """Build and return pipeline for configured model family."""
        family = config.model.model_family.lower()
        if family not in cls._registry:
            supported = list(cls._registry.keys())
            raise ValueError(
                f"Unsupported model_family '{config.model.model_family}'. "
                f"Supported families in registry: {supported}"
            )
        builder = cls._registry[family]
        return builder(config)


@ModelFactory.register("stable_diffusion")
def build_stable_diffusion_pipeline(config: AppConfig) -> StableDiffusionFineTuningPipeline:
    """Builder for Stable Diffusion 1.5 pipeline."""
    tokenizer, text_encoder, vae, unet, scheduler = load_model_components(config)
    pipeline = StableDiffusionFineTuningPipeline(
        config=config,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
    )
    return pipeline
