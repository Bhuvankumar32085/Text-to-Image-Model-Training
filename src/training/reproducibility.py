"""Environment reproducibility and deterministic execution configuration."""
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict
import torch

from src.config.schema import AppConfig
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def configure_reproducibility(config: AppConfig) -> Dict[str, Any]:
    """Configure deterministic random states and gather environment fingerprint."""
    seed = config.training.seed if hasattr(config, "training") else 42
    deterministic = config.training.deterministic if hasattr(config, "training") else True

    set_seed(seed=seed, deterministic=deterministic)

    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "random_seed": seed,
        "deterministic": deterministic,
    }

    if torch.cuda.is_available():
        env_info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    logger.info(f"Configured reproducibility: Seed={seed}, Deterministic={deterministic}")
    return env_info


def save_environment_info(env_info: Dict[str, Any], output_path: str) -> None:
    """Save environment fingerprint to JSON."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)
