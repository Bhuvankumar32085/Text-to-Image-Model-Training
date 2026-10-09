"""Configuration validation logic to enforce constraints, bounds, and mutual compatibility."""
from pathlib import Path
from typing import List
from src.config.schema import AppConfig


class ConfigurationError(ValueError):
    """Custom exception raised when configuration validation fails."""
    pass


def validate_config(config: AppConfig) -> List[str]:
    """Validate configuration object and return a list of warnings.
    
    Raises:
        ConfigurationError: If critical validation constraints are violated.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # 1. Project Validation
    if not config.project.name.strip():
        errors.append("project.name must not be empty.")
    if not config.project.experiment_name.strip():
        errors.append("project.experiment_name must not be empty.")
    if config.project.seed < 0:
        errors.append("project.seed must be a non-negative integer.")

    # 2. Model Validation
    if config.model.model_family != "stable_diffusion":
        errors.append(
            f"Unsupported model_family: '{config.model.model_family}'. Currently only 'stable_diffusion' (SD 1.5) is supported."
        )
    if not config.model.model_identifier.strip():
        errors.append("model.model_identifier must not be empty.")
    if config.model.dtype not in ["float32", "float16", "bfloat16"]:
        errors.append(f"Invalid model.dtype '{config.model.dtype}'. Choose from ['float32', 'float16', 'bfloat16'].")

    # 3. Dataset Validation
    if config.dataset.dataset_source not in ["local", "huggingface", "custom"]:
        errors.append(f"Invalid dataset.dataset_source '{config.dataset.dataset_source}'.")
    if len(config.dataset.target_resolution) != 2:
        errors.append("dataset.target_resolution must be a list of two integers: [height, width].")
    else:
        h, w = config.dataset.target_resolution
        if h <= 0 or w <= 0:
            errors.append(f"target_resolution dimensions must be positive, got [{h}, {w}].")
        if h % 8 != 0 or w % 8 != 0:
            errors.append(
                f"target_resolution [{h}, {w}] must be divisible by 8 because SD 1.5 VAE has downsample factor 8."
            )
    if config.dataset.aspect_ratio_policy not in ["center_crop", "random_crop", "resize"]:
        errors.append(f"Invalid aspect_ratio_policy '{config.dataset.aspect_ratio_policy}'.")
    if config.dataset.image_normalization not in ["neg_one_to_one", "zero_to_one"]:
        errors.append(f"Invalid image_normalization '{config.dataset.image_normalization}'.")
    if config.dataset.min_caption_length < 0 or config.dataset.max_caption_length <= config.dataset.min_caption_length:
        errors.append("Invalid caption length range: ensure 0 <= min_caption_length < max_caption_length.")
    if config.dataset.corrupt_image_handling not in ["skip", "error"]:
        errors.append(f"Invalid corrupt_image_handling '{config.dataset.corrupt_image_handling}'.")

    # 4. Splits Validation
    train_f = config.splits.train_fraction
    test_f = config.splits.test_fraction
    val_f = config.splits.validation_fraction_of_train
    
    if not (0.0 < train_f <= 1.0):
        errors.append(f"splits.train_fraction must be in (0, 1], got {train_f}.")
    if not (0.0 <= test_f < 1.0):
        errors.append(f"splits.test_fraction must be in [0, 1), got {test_f}.")
    if train_f + test_f > 1.0 + 1e-6:
        errors.append(f"Sum of train_fraction ({train_f}) and test_fraction ({test_f}) cannot exceed 1.0.")
    if not (0.0 <= val_f < 1.0):
        errors.append(f"splits.validation_fraction_of_train must be in [0, 1), got {val_f}.")

    # 5. Fine-Tuning Policy Validation
    if config.fine_tuning.mode != "full_finetune":
        warnings.append(
            f"fine_tuning.mode is '{config.fine_tuning.mode}'. The default standard mode is 'full_finetune'."
        )
    if not config.fine_tuning.trainable_components:
        errors.append("fine_tuning.trainable_components cannot be empty.")
    for comp in config.fine_tuning.trainable_components:
        if comp not in ["unet", "text_encoder", "vae"]:
            errors.append(f"Unknown trainable component '{comp}'. Choose from ['unet', 'text_encoder', 'vae'].")

    # 6. Training Validation
    if config.training.epochs <= 0:
        errors.append(f"training.epochs must be positive, got {config.training.epochs}.")
    if config.training.max_train_steps <= 0:
        errors.append(f"training.max_train_steps must be positive, got {config.training.max_train_steps}.")
    if config.training.train_batch_size <= 0:
        errors.append(f"training.train_batch_size must be positive, got {config.training.train_batch_size}.")
    if config.training.gradient_accumulation_steps <= 0:
        errors.append(f"training.gradient_accumulation_steps must be positive, got {config.training.gradient_accumulation_steps}.")
    if config.training.learning_rate <= 0:
        errors.append(f"training.learning_rate must be positive, got {config.training.learning_rate}.")
    if config.training.optimizer not in ["adamw", "adam", "sgd", "adafactor"]:
        errors.append(f"Unsupported optimizer '{config.training.optimizer}'.")
    if config.training.mixed_precision not in ["no", "fp16", "bf16"]:
        errors.append(f"Invalid mixed_precision '{config.training.mixed_precision}'. Choose from ['no', 'fp16', 'bf16'].")
    if config.training.max_grad_norm <= 0:
        errors.append("training.max_grad_norm must be positive.")

    # 7. Device Validation
    if config.device.type not in ["auto", "cuda", "cpu"]:
        errors.append(f"Invalid device.type '{config.device.type}'. Choose from ['auto', 'cuda', 'cpu'].")

    # 8. Inference Validation
    if config.inference.height % 8 != 0 or config.inference.width % 8 != 0:
        errors.append(f"Inference dimensions [{config.inference.height}, {config.inference.width}] must be divisible by 8.")
    if config.inference.num_inference_steps <= 0:
        errors.append("inference.num_inference_steps must be positive.")
    if config.inference.guidance_scale < 1.0:
        warnings.append(f"inference.guidance_scale is {config.inference.guidance_scale} (< 1.0 means no classifier-free guidance).")

    if errors:
        error_msg = "Configuration validation failed with the following errors:\n" + "\n".join(f"  - {err}" for err in errors)
        raise ConfigurationError(error_msg)
        
    return warnings
