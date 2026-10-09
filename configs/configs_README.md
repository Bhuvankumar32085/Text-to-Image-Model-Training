# Configuration Reference Guide (`configs/config.yaml`)

This document details every section, key, allowed value, and constraint in the single-source configuration file `configs/config.yaml`.

---

## 1. `project`
- `name` (*str*): The project identifier.
- `experiment_name` (*str*): Name prefix for this experiment run.
- `seed` (*int*): Global random seed for PyTorch, NumPy, and Python standard random library.
- `project_root` (*str*): Root directory path.
- `log_dir` (*str*): Directory where structured experiment logs and metrics are stored.
- `checkpoint_dir` (*str*): Directory where model checkpoints and optimizer states are saved.
- `output_dir` (*str*): Directory for preview generations, final test images, and reports.
- `overwrite` (*bool*): Whether to overwrite previous outputs with the same experiment name.
- `resume` (*bool*): Whether to resume training from the latest checkpoint in `checkpoint_dir`.
- `experiment_metadata` (*dict*): Arbitrary metadata (description, tags, author).

---

## 2. `model`
- `model_family` (*str*): Model family name (default: `"stable_diffusion"`).
- `model_identifier` (*str*): Hugging Face repo ID (e.g. `"stable-diffusion-v1-5/stable-diffusion-v1-5"`) or local folder path.
- `revision` (*str*): Specific branch, tag, or commit hash (e.g., `"main"` or `"fp16"`).
- `local_pretrained_model_dir` (*str | null*): Path to locally downloaded model weights.
- `cache_dir` (*str | null*): Custom Hugging Face cache directory.
- `dtype` (*str*): Precision for frozen model components (`"float32"`, `"float16"`, `"bfloat16"`).
- `local_files_only` (*bool*): When `true`, prevents network calls to Hugging Face.
- `strict_compatibility_check` (*bool*): Validates channel dimensions, cross-attention dims, and scheduler config before training.

---

## 3. `dataset`
- `dataset_source` (*str*): `"local"`, `"huggingface"`, or `"custom"`.
- `image_dir` (*str*): Directory containing training images.
- `caption_metadata_path` (*str*): Path to CSV/JSONL containing image paths and captions.
- `image_column` (*str*): Name of the column containing image paths or filenames.
- `caption_column` (*str*): Name of the column containing caption strings.
- `group_column` (*str | null*): Name of column representing subject/video groups for data-leakage prevention.
- `supported_extensions` (*list[str]*): Accepted image file extensions.
- `target_resolution` (*list[int]*): `[height, width]`, e.g. `[512, 512]`.
- `aspect_ratio_policy` (*str*): `"center_crop"`, `"random_crop"`, or `"resize"`.
- `image_normalization` (*str*): `"neg_one_to_one"` (maps $[0, 255] \to [-1.0, 1.0]$) or `"zero_to_one"`.
- `corrupt_image_handling` (*str*): `"skip"` (logs warning and ignores) or `"error"` (raises immediate exception).
- `duplicate_detection` (*bool*): Identifies duplicate images via MD5 hashing.
- `min_caption_length` (*int*): Minimum character length for valid captions.
- `max_caption_length` (*int*): Maximum character length before truncation.

---

## 4. `splits`
- `train_fraction` (*float*): Fraction of dataset allocated to the train+val pool (e.g., `0.90`).
- `test_fraction` (*float*): Fraction held out strictly for final evaluation (e.g., `0.10`).
- `validation_fraction_of_train` (*float*): Fraction of train pool used for validation (e.g., `0.10` yields 81% train, 9% val, 10% test).
- `split_seed` (*int*): Deterministic seed for dataset splitting.
- `group_aware_split` (*bool*): Keeps images with the same `group_id` strictly in the same split.

---

## 5. `fine_tuning`
- `mode` (*str*): `"full_finetune"`.
- `train_unet` (*bool*): Enables gradient computation and optimizer updates for the U-Net.
- `train_text_encoder` (*bool*): Enables training for CLIP Text Encoder (frozen by default).
- `train_vae` (*bool*): Enables training for VAE (frozen by default).
- `trainable_components` (*list[str]*): List of component names to train (e.g., `["unet"]`).

---

## 6. `training`
- `epochs` (*int*): Number of complete dataset passes.
- `max_train_steps` (*int*): Hard cap on training steps.
- `train_batch_size` (*int*): Per-device batch size (keep at 1 for $\le 6$ GB VRAM).
- `gradient_accumulation_steps` (*int*): Steps over which gradients accumulate before optimizer update.
- `learning_rate` (*float*): Base learning rate for trainable components (e.g., `1.0e-5`).
- `component_learning_rates` (*dict*): Specific learning rates for individual components.
- `optimizer` (*str*): `"adamw"`, `"adam"`, `"sgd"`, `"adafactor"`.
- `lr_scheduler` (*str*): `"cosine"`, `"linear"`, `"cosine_with_restarts"`, `"constant"`.
- `lr_warmup_steps` (*int*): Number of linear warmup steps.
- `max_grad_norm` (*float*): Maximum gradient norm for clipping.
- `mixed_precision` (*str*): `"fp16"`, `"bf16"`, or `"no"`.
- `gradient_checkpointing` (*bool*): Trades compute for VRAM by recomputing activations during backward pass.
- `enable_sdpa` (*bool*): Uses PyTorch 2.x Scaled Dot Product Attention.
- `checkpoint_frequency_steps` (*int*): Save checkpoint every $N$ steps.
- `save_best_checkpoint` (*bool*): Automatically tracks and saves the checkpoint with lowest validation loss.
- `max_checkpoints_to_keep` (*int*): Pruning threshold for periodic checkpoints.

---

## 7. `device`
- `type` (*str*): `"auto"` (prefers CUDA if available, falls back to CPU), `"cuda"` (strict CUDA requirement), or `"cpu"`.
- `device_index` (*int*): CUDA GPU index (e.g., 0).
- `allow_cpu_fallback` (*bool*): When `false`, halts with an informative error if CUDA is requested but unavailable.

---

## 8. `validation` & `test`
- `validation_batch_size` (*int*): Batch size during validation loss computation.
- `fixed_prompts` (*list[str]*): Prompts evaluated at each validation interval.
- `num_inference_steps` (*int*): Denoising steps for preview generations.
- `guidance_scale` (*float*): Classifier-Free Guidance (CFG) scale (typical: 7.5).
- `test_prompts` (*list[str]*): Held-out evaluation prompts for final test suite.

---

## 9. `logging`
- `console` / `file` / `jsonl` / `csv` / `tensorboard` (*bool*): Enable/disable individual log outputs.
- `log_frequency` (*int*): Metric logging step frequency.
- `tensor_stats_frequency` (*int*): Intermediate tensor statistical recording frequency.
- `nan_inf_detection` (*bool*): Checks for NaN/Inf values across forward and backward passes.
- `debug_mode` (*bool*): Verbose logging and tensor dumps.

---

## 10. `inference`
- `prompt` (*str*): Text description to generate.
- `negative_prompt` (*str*): Undesired features to guide away from.
- `height` / `width` (*int*): Generated image dimensions (default: 512x512).
- `scheduler_type` (*str*): Inference sampler (`"DPMSolverMultistepScheduler"`, `"EulerDiscreteScheduler"`, `"DDPMScheduler"`).
