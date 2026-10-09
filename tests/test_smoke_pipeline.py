"""End-to-end synthetic smoke test verifying the complete training, evaluation, and inference workflow on CPU."""
from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.config.schema import AppConfig
from src.data.collator import TextToImageCollator
from src.data.dataset import TextToImageDataset
from src.data.transforms import ImageTransformPipeline
from src.models.pipeline import StableDiffusionFineTuningPipeline
from src.models.scheduler_component import SchedulerComponent
from src.models.text_encoder_component import TextEncoderComponent
from src.models.unet_component import UNetComponent
from src.models.vae_component import VAEComponent
from src.training.checkpoint_manager import CheckpointManager
from src.training.forward_diffusion import ForwardDiffusion
from src.training.loss import DiffusionLoss
from src.training.optimizer import create_optimizer_and_scheduler
from src.training.train_step import TrainStepExecutor


class MockUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.config = type("Config", (), {"in_channels": 4, "out_channels": 4, "cross_attention_dim": 768})()

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        # Apply conv and add dummy interaction with text embeddings
        out = self.conv(sample) + encoder_hidden_states[:, :4, :4].mean() * 0.0
        return type("Output", (), {"sample": out})()


class MockVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"latent_channels": 4, "scaling_factor": 0.18215})()

    def encode(self, x):
        # (B, 3, 64, 64) -> (B, 4, 8, 8)
        B = x.shape[0]
        latent = torch.zeros(B, 4, x.shape[2] // 8, x.shape[3] // 8, device=x.device, dtype=x.dtype)
        dist = type("Dist", (), {"sample": lambda generator=None: latent, "mode": lambda: latent, "mean": latent, "logvar": latent})()
        return type("EncodeOutput", (), {"latent_dist": dist})()

    def decode(self, latents):
        # (B, 4, 8, 8) -> (B, 3, 64, 64)
        B = latents.shape[0]
        img = torch.zeros(B, 3, latents.shape[2] * 8, latents.shape[3] * 8, device=latents.device, dtype=latents.dtype)
        return type("DecodeOutput", (), {"sample": img})()


class MockTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 768})()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        B = input_ids.shape[0]
        hidden = torch.randn(B, 77, 768, device=input_ids.device)
        return type("Output", (), {"last_hidden_state": hidden})()


def test_full_cpu_synthetic_pipeline(tmp_path):
    """Smoke test running full training step, checkpointing, and inference with mock components on CPU."""
    device = torch.device("cpu")

    # 1. Configuration
    config = AppConfig()
    config.training.mixed_precision = "no"
    config.training.train_batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.dataset.target_resolution = [64, 64]
    config.fine_tuning.train_unet = True
    config.fine_tuning.train_text_encoder = False
    config.fine_tuning.train_vae = False
    config.fine_tuning.trainable_components = ["unet"]

    # 2. Mock Components
    mock_tokenizer = MagicMock()
    mock_tokenizer.model_max_length = 77
    mock_tokenizer.side_effect = lambda texts, **kwargs: type("Tokens", (), {
        "input_ids": torch.randint(0, 1000, (len(texts), 77)),
        "attention_mask": torch.ones(len(texts), 77),
    })()

    mock_scheduler = MagicMock()
    mock_scheduler.config.num_train_timesteps = 1000
    mock_scheduler.config.prediction_type = "epsilon"
    mock_scheduler.add_noise.side_effect = lambda x, n, t: x + n

    raw_unet = MockUNet()
    raw_vae = MockVAE()
    raw_te = MockTextEncoder()

    pipeline = StableDiffusionFineTuningPipeline(
        config=config,
        tokenizer=mock_tokenizer,
        text_encoder=raw_te,
        vae=raw_vae,
        unet=raw_unet,
        scheduler=mock_scheduler,
    )
    pipeline.to_device(device)

    # 3. Synthetic Dataset & DataLoader
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(4):
        img = Image.new("RGB", (64, 64), color=(50, 100, 150))
        img.save(img_dir / f"img_{i}.png")

    records = [{"image_path": str(img_dir / f"img_{i}.png"), "caption": f"Sample caption {i}"} for i in range(4)]
    transform = ImageTransformPipeline(target_resolution=(64, 64))
    dataset = TextToImageDataset(records, transform=transform)

    collator = TextToImageCollator(tokenizer=mock_tokenizer, max_length=77)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collator)

    # 4. Training Step Execution
    optimizer, lr_scheduler = create_optimizer_and_scheduler(config, pipeline, total_training_steps=5)
    loss_fn = DiffusionLoss(loss_type="mse")
    fwd_diff = ForwardDiffusion(pipeline.scheduler)

    step_executor = TrainStepExecutor(
        config=config,
        pipeline=pipeline,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        forward_diffusion=fwd_diff,
        device=device,
    )

    batch = next(iter(loader))
    loss_val, metrics, diag_tensors = step_executor.execute_step(batch, step_idx=0, collect_diagnostics=True)

    assert loss_val >= 0.0
    assert "unet_prediction" in diag_tensors
    assert "clean_latents" in diag_tensors

    # 5. Checkpoint Save and Restore
    ckpt_mgr = CheckpointManager(config, checkpoint_dir=tmp_path / "checkpoints")
    save_path = ckpt_mgr.save_checkpoint(
        pipeline=pipeline,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=None,
        global_step=1,
        epoch=0,
        val_loss=loss_val,
        is_best=True,
    )

    assert save_path.exists()
    assert (save_path / "unet.pt").exists()

    # Restore checkpoint
    reloaded_meta = ckpt_mgr.load_checkpoint(save_path, pipeline, optimizer, lr_scheduler)
    assert reloaded_meta["global_step"] == 1
