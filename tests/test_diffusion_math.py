"""Unit tests for diffusion mathematical formulas, noise addition, and target calculation."""
from unittest.mock import MagicMock
import pytest
import torch

from src.models.scheduler_component import SchedulerComponent
from src.training.forward_diffusion import ForwardDiffusion
from src.training.loss import DiffusionLoss


def test_scheduler_epsilon_and_v_prediction_targets():
    """Verify ground-truth target calculation for epsilon and v_prediction."""
    mock_scheduler = MagicMock()
    mock_scheduler.config.prediction_type = "epsilon"
    mock_scheduler.config.num_train_timesteps = 1000

    comp = SchedulerComponent(mock_scheduler)

    latents = torch.randn(2, 4, 64, 64)
    noise = torch.randn(2, 4, 64, 64)
    timesteps = torch.tensor([100, 500])

    # 1. Epsilon prediction target
    target_eps = comp.get_target(latents, noise, timesteps)
    assert torch.equal(target_eps, noise)

    # 2. V-prediction target
    mock_scheduler.config.prediction_type = "v_prediction"
    expected_velocity = torch.randn(2, 4, 64, 64)
    mock_scheduler.get_velocity.return_value = expected_velocity

    target_v = comp.get_target(latents, noise, timesteps)
    assert torch.equal(target_v, expected_velocity)


def test_diffusion_loss_float32_computation():
    """Verify that diffusion loss operates correctly in float32 without error."""
    loss_fn = DiffusionLoss(loss_type="mse")

    # Inputs in float16
    pred_f16 = torch.randn(2, 4, 32, 32, dtype=torch.float16)
    target_f16 = torch.randn(2, 4, 32, 32, dtype=torch.float16)

    loss = loss_fn(pred_f16, target_f16)
    assert isinstance(loss, torch.Tensor)
    assert loss.dtype == torch.float32
    assert loss.item() >= 0.0


def test_forward_diffusion_noise_sampling():
    """Verify forward noise addition output shapes and device matching."""
    mock_scheduler = MagicMock()
    mock_scheduler.config.num_train_timesteps = 1000
    mock_scheduler.add_noise.side_effect = lambda x, n, t: x + n

    comp = SchedulerComponent(mock_scheduler)
    fwd = ForwardDiffusion(comp)

    latents = torch.randn(4, 4, 16, 16)
    noisy_latents, noise, timesteps = fwd.forward_noise(latents)

    assert noisy_latents.shape == latents.shape
    assert noise.shape == latents.shape
    assert timesteps.shape == (4,)
    assert timesteps.dtype == torch.long
    assert (timesteps >= 0).all() and (timesteps < 1000).all()
