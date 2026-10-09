# Stable Diffusion 1.5 Pretrained Checkpoint

## Architecture Specification
- **Model**: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- **U-Net**: Conv2D + Cross-Attention residual backbone with 860M parameters.
- **Latent Resolution**: $64 \times 64$ for $512 \times 512$ pixel inputs ($8\times$ spatial compression).
- **Latent Scaling Factor**: $0.18215$ applied to latent samples from the VAE encoder.
- **Text Conditioning**: CLIP ViT-L/14 embedding sequence of length $77 \times 768$.

## Parameter Trainability Policy
By default in full fine-tuning mode:
- **U-Net**: Trainable (`train_unet: true`)
- **Text Encoder**: Frozen (`train_text_encoder: false`)
- **VAE**: Frozen (`train_vae: false`)

To adjust component trainability, update `configs/config.yaml` under `fine_tuning`.
