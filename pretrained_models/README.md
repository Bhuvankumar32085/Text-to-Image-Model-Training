# Pretrained Models Directory

This directory stores metadata, offline cache instructions, and component specifications for compatible pretrained text-to-image diffusion models.

## Supported Model Families

### 1. Stable Diffusion 1.5 (`stable_diffusion_15`)
- **Official Hugging Face Identifier**: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- **Architecture**: Latent Diffusion Model (LDM) with 2D Cross-Attention U-Net backbone (860M U-Net parameters).
- **Text Conditioning**: Frozen CLIP ViT-L/14 text encoder (768-dimensional token embeddings).
- **Autoencoder**: Frozen AutoencoderKL (VAE) with 8x spatial downsampling and 4 latent channels.
- **Noise Scheduler**: DDPMScheduler (training) and DPMSolverMultistepScheduler / EulerDiscreteScheduler (inference).

> **Architectural Note**: Stable Diffusion 1.5 utilizes a convolutional Residual U-Net with spatial cross-attention layers, **not** a Diffusion Transformer (DiT). Do not configure or treat this pipeline as a Transformer-based DiT model.

## Offline / Air-Gapped Usage

To pre-download and cache model weights locally:
```bash
python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5')"
```

Or clone the Hugging Face repository directly:
```bash
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 pretrained_models/stable_diffusion_15/weights
```

In `configs/config.yaml`, set:
```yaml
model:
  pretrained_model_name_or_path: "pretrained_models/stable_diffusion_15/weights"
  local_files_only: true
```
