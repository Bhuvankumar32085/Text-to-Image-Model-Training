import os
import logging
from typing import Dict, Any, Tuple
import torch

# Prevent PyTorch CUDA memory fragmentation on consumer GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.config.schema import AppConfig
from src.utils.device_utils import get_device_info

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages PyTorch hardware devices and guarantees execution consistency."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.device, self.device_info = self._setup_device()

    def _setup_device(self) -> Tuple[torch.device, Dict[str, Any]]:
        dev_cfg = self.config.device
        req_type = dev_cfg.type.lower()
        dev_idx = dev_cfg.device_index
        cuda_is_avail = torch.cuda.is_available()

        if req_type == "cuda":
            if not cuda_is_avail:
                raise RuntimeError(
                    "Configuration explicitly requested CUDA ('device.type: cuda'), "
                    "but torch.cuda.is_available() is False. No CUDA GPU was detected."
                )
            if dev_idx >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested CUDA device_index {dev_idx}, but only "
                    f"{torch.cuda.device_count()} GPU(s) are available."
                )
            device = torch.device(f"cuda:{dev_idx}")
            torch.cuda.set_device(device)

        elif req_type == "cpu":
            device = torch.device("cpu")
            logger.warning(
                "Training configured explicitly on CPU ('device.type: cpu'). "
                "Full-model diffusion training on CPU will be EXTREMELY SLOW."
            )

        elif req_type == "auto":
            if cuda_is_avail:
                device = torch.device(f"cuda:{dev_idx}")
                torch.cuda.set_device(device)
            else:
                if not dev_cfg.allow_cpu_fallback and dev_cfg.require_cuda:
                    raise RuntimeError("Device set to auto with require_cuda=True, but CUDA is unavailable.")
                device = torch.device("cpu")
                logger.warning("CUDA not detected. Automatically falling back to CPU.")
        else:
            raise ValueError(f"Unknown device type '{req_type}'.")

        info = get_device_info(dev_idx if device.type == "cuda" else 0)
        logger.info(
            f"Hardware selected: {device.type.upper()} "
            f"(Name: {info['selected_device_name']}, Total VRAM: {info['total_memory_gb']} GB, "
            f"PyTorch: {info['torch_version']}, CUDA: {info['cuda_version']})"
        )

        return device, info

    def check_memory_threshold(self) -> None:
        """Check if current VRAM usage exceeds configured safety threshold."""
        if self.device.type == "cuda" and self.config.device.memory_check:
            alloc_gb = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            thresh = self.config.training.gpu_memory_warning_threshold_gb
            if alloc_gb > thresh:
                logger.warning(
                    f"High VRAM warning: Allocated memory ({alloc_gb:.2f} GB) exceeds "
                    f"warning threshold ({thresh:.2f} GB)."
                )
