"""Configuration loading, dictionary merging, and snapshot persistence."""
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from src.config.schema import (
    AppConfig,
    ProjectConfig,
    ModelConfig,
    DatasetConfig,
    SplitsConfig,
    FineTuningConfig,
    EarlyStoppingConfig,
    TrainingConfig,
    DeviceConfig,
    ValidationConfig,
    TestConfig,
    LoggingConfig,
    InferenceConfig,
    DownloadConfig,
)
from src.config.validation import validate_config


def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary."""
    result = copy.deepcopy(base_dict)
    for k, v in update_dict.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _dict_to_dataclass(config_dict: Dict[str, Any]) -> AppConfig:
    """Instantiate strongly typed AppConfig from raw dictionary."""
    proj = ProjectConfig(**config_dict.get("project", {}))
    model = ModelConfig(**config_dict.get("model", {}))
    dataset = DatasetConfig(**config_dict.get("dataset", {}))
    splits = SplitsConfig(**config_dict.get("splits", {}))
    
    ft_data = config_dict.get("fine_tuning", {})
    ft = FineTuningConfig(**ft_data)
    
    train_data = copy.deepcopy(config_dict.get("training", {}))
    early_stopping_data = train_data.pop("early_stopping", {})
    early_stopping = EarlyStoppingConfig(**early_stopping_data)
    training = TrainingConfig(early_stopping=early_stopping, **train_data)
    
    device = DeviceConfig(**config_dict.get("device", {}))
    validation = ValidationConfig(**config_dict.get("validation", {}))
    test = TestConfig(**config_dict.get("test", {}))
    logging = LoggingConfig(**config_dict.get("logging", {}))
    inference = InferenceConfig(**config_dict.get("inference", {}))
    download = DownloadConfig(**config_dict.get("download", {}))
    
    return AppConfig(
        project=proj,
        model=model,
        dataset=dataset,
        splits=splits,
        fine_tuning=ft,
        training=training,
        device=device,
        validation=validation,
        test=test,
        logging=logging,
        inference=inference,
        download=download,
    )


def _dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass tree to pure dictionary."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    else:
        return obj


def load_config(
    config_path: Union[str, Path] = "configs/config.yaml",
    overrides: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> AppConfig:
    """Load configuration from YAML file, apply optional overrides, and validate."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
        
    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}
        
    if overrides:
        raw_dict = _deep_update(raw_dict, overrides)
        
    config = _dict_to_dataclass(raw_dict)
    
    if validate:
        validate_config(config)
        
    return config


def save_config_snapshot(config: Union[AppConfig, Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Save an exact configuration snapshot to YAML for experiment reproducibility."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(config, AppConfig):
        data = _dataclass_to_dict(config)
    else:
        data = config
        
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
