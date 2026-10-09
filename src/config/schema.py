"""Dataclass schema definitions for all configuration sections."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class ProjectConfig:
    name: str = "text-to-image-diffusion"
    experiment_name: str = "sd15_finetune"
    seed: int = 42
    project_root: str = "."
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    overwrite: bool = False
    resume: bool = False
    experiment_metadata: Dict[str, Any] = field(default_factory=lambda: {
        "description": "Config-driven fine-tuning of Stable Diffusion 1.5",
        "author": "Researcher",
        "tags": ["stable_diffusion", "sd15", "unet_finetune"]
    })


@dataclass
class ModelConfig:
    model_family: str = "stable_diffusion"
    model_identifier: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    revision: str = "main"
    local_pretrained_model_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    dtype: str = "float16"  # "float32", "float16", "bfloat16"
    local_files_only: bool = False
    strict_compatibility_check: bool = True
    subfolder_overrides: Dict[str, str] = field(default_factory=lambda: {
        "tokenizer": "tokenizer",
        "text_encoder": "text_encoder",
        "vae": "vae",
        "unet": "unet",
        "scheduler": "scheduler"
    })


@dataclass
class DatasetConfig:
    dataset_source: str = "local"  # "local", "huggingface", "custom"
    dataset_identifier: Optional[str] = None
    dataset_download_dest: str = "data/raw/downloaded_dataset"
    image_dir: str = "data/raw/sample_dataset"
    caption_metadata_path: str = "data/raw/sample_dataset/metadata.csv"
    image_column: str = "image_path"
    caption_column: str = "caption"
    group_column: Optional[str] = "group_id"
    supported_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
    target_resolution: List[int] = field(default_factory=lambda: [512, 512])
    aspect_ratio_policy: str = "center_crop"  # "center_crop", "random_crop", "resize"
    image_normalization: str = "neg_one_to_one"  # "neg_one_to_one" or "zero_to_one"
    corrupt_image_handling: str = "skip"  # "skip", "error"
    duplicate_detection: bool = True
    min_caption_length: int = 3
    max_caption_length: int = 500
    caption_filtering: Dict[str, Any] = field(default_factory=lambda: {
        "drop_empty": True,
        "strip_whitespace": True
    })
    dataset_cache_dir: str = "data/processed"
    max_samples_debug: Optional[int] = None


@dataclass
class SplitsConfig:
    train_fraction: float = 0.90
    test_fraction: float = 0.10
    validation_fraction_of_train: float = 0.10
    split_seed: int = 42
    group_aware_split: bool = True
    manifest_dir: str = "data/splits"
    save_manifests: bool = True
    overwrite_manifests: bool = False


@dataclass
class FineTuningConfig:
    mode: str = "full_finetune"  # "full_finetune", "partial", "lora"
    train_unet: bool = True
    train_text_encoder: bool = False
    train_vae: bool = False
    trainable_components: List[str] = field(default_factory=lambda: ["unet"])
    use_lora: bool = False


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.001


@dataclass
class TrainingConfig:
    epochs: int = 5
    max_train_steps: int = 500
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.0e-5
    component_learning_rates: Dict[str, float] = field(default_factory=lambda: {
        "unet": 1.0e-5,
        "text_encoder": 5.0e-6
    })
    optimizer: str = "adamw"
    optimizer_params: Dict[str, Any] = field(default_factory=lambda: {
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.01
    })
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 50
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = True
    enable_xformers_memory_efficient_attention: bool = False
    enable_sdpa: bool = True
    dataloader_num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    shuffle: bool = True
    checkpoint_frequency_steps: int = 100
    checkpoint_frequency_epochs: int = 1
    save_best_checkpoint: bool = True
    max_checkpoints_to_keep: int = 3
    validation_frequency_steps: int = 50
    logging_frequency_steps: int = 10
    tensor_stats_frequency_steps: int = 50
    image_preview_frequency_steps: int = 100
    seed: int = 42
    deterministic: bool = True
    resume_from_checkpoint: Optional[str] = None
    gpu_memory_warning_threshold_gb: float = 5.5
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class DeviceConfig:
    type: str = "auto"  # "auto", "cuda", "cpu"
    device_index: int = 0
    allow_cpu_fallback: bool = False
    require_cuda: bool = False
    memory_check: bool = True


@dataclass
class ValidationConfig:
    validation_batch_size: int = 1
    validation_steps: int = 50
    fixed_prompts: List[str] = field(default_factory=lambda: [
        "A serene mountain landscape with a crystal clear blue lake and pine trees at sunrise",
        "A futuristic neon cyberpunk cityscape with towering skyscrapers and rain-slicked streets"
    ])
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    num_preview_images: int = 2
    metrics: List[str] = field(default_factory=lambda: ["diffusion_loss"])
    save_side_by_side: bool = True
    report_output_dir: str = "outputs/reports"


@dataclass
class TestConfig:
    test_batch_size: int = 1
    test_prompts: List[str] = field(default_factory=lambda: [
        "A cozy wooden cabin glowing warmly in a snowy forest at dusk",
        "A cute playful golden retriever puppy sitting on lush green grass in summer sunlight"
    ])
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 2
    seed: int = 123
    metrics: List[str] = field(default_factory=lambda: ["diffusion_loss"])
    output_dir: str = "outputs/test_results"


@dataclass
class LoggingConfig:
    console: bool = True
    file: bool = True
    jsonl: bool = True
    csv: bool = True
    tensorboard: bool = True
    log_frequency: int = 10
    tensor_stats_frequency: int = 50
    image_preview_frequency: int = 100
    save_input_image_samples: bool = True
    save_clean_latent_stats: bool = True
    save_noisy_latent_stats: bool = True
    save_noise_stats: bool = True
    save_pred_noise_stats: bool = True
    save_loss_breakdown: bool = True
    save_gradient_norms: bool = True
    save_param_norms: bool = True
    save_gpu_memory_metrics: bool = True
    save_inference_timings: bool = True
    nan_inf_detection: bool = True
    log_component_shapes_and_dtypes: bool = True
    debug_mode: bool = False
    max_stored_diagnostic_tensors: int = 10


@dataclass
class InferenceConfig:
    checkpoint_path: Optional[str] = None
    prompt: str = "A cinematic photo of a cozy mountain cabin in autumn mist, 8k resolution"
    negative_prompt: str = "blurry, low quality, distorted, deformed, ugly, artifacts"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: int = 42
    scheduler_type: str = "DPMSolverMultistepScheduler"
    output_format: str = "png"
    output_dir: str = "outputs/inference"
    safety_checker: bool = False


@dataclass
class DownloadConfig:
    dataset_provider: str = "huggingface"
    dataset_identifier: Optional[str] = "lambdalabs/pokemon-blip-captions"
    auth_token_env_var: str = "HF_TOKEN"
    download_dir: str = "data/raw/pokemon_captions"
    max_samples: int = 100
    workers: int = 2
    retries: int = 3
    timeout_seconds: int = 60
    resume_download: bool = True
    verify_checksums: bool = False


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    test: TestConfig = field(default_factory=TestConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
