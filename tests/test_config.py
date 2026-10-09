"""Unit tests for configuration schema, loader, overrides, and validation rules."""
import pytest
from src.config.loader import load_config, save_config_snapshot
from src.config.schema import AppConfig
from src.config.validation import ConfigurationError, validate_config


def test_default_config_loading(tmp_path):
    """Test loading default config.yaml."""
    config = load_config("configs/config.yaml")
    assert isinstance(config, AppConfig)
    assert config.project.name == "text-to-image-diffusion"
    assert config.model.model_family == "stable_diffusion"
    assert config.dataset.target_resolution == [512, 512]
    assert config.fine_tuning.train_unet is True
    assert config.fine_tuning.train_text_encoder is False
    assert config.fine_tuning.train_vae is False


def test_config_overrides():
    """Test applying deep dictionary overrides to configuration."""
    overrides = {
        "training": {
            "learning_rate": 5.0e-5,
            "train_batch_size": 2,
        },
        "project": {
            "experiment_name": "custom_override_exp"
        }
    }
    config = load_config("configs/config.yaml", overrides=overrides)
    assert config.training.learning_rate == 5.0e-5
    assert config.training.train_batch_size == 2
    assert config.project.experiment_name == "custom_override_exp"


def test_invalid_resolution_validation():
    """Test validation failure on resolutions not divisible by 8."""
    config = load_config("configs/config.yaml")
    config.dataset.target_resolution = [500, 500]  # Not divisible by 8
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "divisible by 8" in str(exc_info.value)


def test_invalid_split_fractions():
    """Test validation failure when split fractions exceed 1.0."""
    config = load_config("configs/config.yaml")
    config.splits.train_fraction = 0.8
    config.splits.test_fraction = 0.3  # Sum = 1.1 > 1.0
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "Sum of train_fraction" in str(exc_info.value)


def test_invalid_model_family():
    """Test validation failure on unsupported model family."""
    config = load_config("configs/config.yaml")
    config.model.model_family = "transformer_dit"
    with pytest.raises(ConfigurationError) as exc_info:
        validate_config(config)
    assert "Unsupported model_family" in str(exc_info.value)


def test_config_snapshot_persistence(tmp_path):
    """Test saving and re-loading exact configuration snapshot."""
    config = load_config("configs/config.yaml")
    snapshot_path = tmp_path / "snapshot.yaml"
    save_config_snapshot(config, snapshot_path)
    assert snapshot_path.exists()

    reloaded = load_config(snapshot_path)
    assert reloaded.project.name == config.project.name
    assert reloaded.training.learning_rate == config.training.learning_rate
