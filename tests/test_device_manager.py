"""Unit tests for hardware device management and error handling."""
from unittest.mock import patch
import pytest
import torch

from src.config.schema import AppConfig
from src.training.device_manager import DeviceManager


def test_cpu_device_selection():
    """Verify that cpu selection creates a CPU torch device."""
    config = AppConfig()
    config.device.type = "cpu"

    dev_mgr = DeviceManager(config)
    assert dev_mgr.device.type == "cpu"


def test_cuda_required_failure():
    """Verify that requesting CUDA when unavailable raises RuntimeError."""
    config = AppConfig()
    config.device.type = "cuda"

    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError) as exc_info:
            _ = DeviceManager(config)
        assert "No CUDA GPU was detected" in str(exc_info.value)


def test_auto_fallback_to_cpu():
    """Verify that auto mode falls back to CPU when CUDA is absent."""
    config = AppConfig()
    config.device.type = "auto"
    config.device.allow_cpu_fallback = True

    with patch("torch.cuda.is_available", return_value=False):
        dev_mgr = DeviceManager(config)
        assert dev_mgr.device.type == "cpu"
