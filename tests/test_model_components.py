"""Unit tests for model component freezing, trainability policies, and optimizer parameter groups."""
import pytest
import torch
import torch.nn as nn

from src.config.schema import AppConfig
from src.models.component_loader import apply_trainability_policy
from src.models.pipeline import StableDiffusionFineTuningPipeline
from src.training.optimizer import create_optimizer_and_scheduler


class DummyModule(nn.Module):
    def __init__(self, in_features=4, out_features=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.config = type("Config", (), {"latent_channels": 4, "in_channels": 4, "cross_attention_dim": 768, "hidden_size": 768})()

    def forward(self, *args, **kwargs):
        return self.linear(torch.zeros(1, 4))


def test_trainability_policy_freezing():
    """Verify that only configured components have requires_grad=True."""
    config = AppConfig()
    config.fine_tuning.train_unet = True
    config.fine_tuning.train_text_encoder = False
    config.fine_tuning.train_vae = False
    config.fine_tuning.trainable_components = ["unet"]

    unet = DummyModule()
    text_encoder = DummyModule()
    vae = DummyModule()

    apply_trainability_policy(config, unet, text_encoder, vae)

    assert all(p.requires_grad for p in unet.parameters())
    assert not any(p.requires_grad for p in text_encoder.parameters())
    assert not any(p.requires_grad for p in vae.parameters())


def test_optimizer_parameter_groups():
    """Verify optimizer only receives trainable parameters with component-specific learning rates."""
    config = AppConfig()
    config.training.learning_rate = 1.0e-5
    config.training.component_learning_rates = {"unet": 2.0e-5, "text_encoder": 5.0e-6}
    config.fine_tuning.train_unet = True
    config.fine_tuning.train_text_encoder = False
    config.fine_tuning.train_vae = False
    config.fine_tuning.trainable_components = ["unet"]

    unet = DummyModule()
    text_encoder = DummyModule()
    vae = DummyModule()

    apply_trainability_policy(config, unet, text_encoder, vae)

    mock_pipeline = type(
        "MockPipeline",
        (),
        {
            "unet": DummyModule(),
            "text_encoder": DummyModule(),
            "vae": DummyModule(),
        },
    )()
    mock_pipeline.unet.requires_grad_(True)
    mock_pipeline.text_encoder.requires_grad_(False)
    mock_pipeline.vae.requires_grad_(False)

    optimizer, scheduler = create_optimizer_and_scheduler(config, mock_pipeline, total_training_steps=100)

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["name"] == "unet"
    assert optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]) == 2.0e-5
