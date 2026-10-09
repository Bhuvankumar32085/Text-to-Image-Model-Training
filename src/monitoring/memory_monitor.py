"""VRAM and System RAM usage monitoring."""
from typing import Any, Dict
import torch


class MemoryMonitor:
    """Monitors GPU VRAM allocations and system RAM."""

    @staticmethod
    def get_memory_snapshot(device: torch.device) -> Dict[str, Any]:
        """Collect current memory allocation statistics."""
        metrics = {
            "cuda_allocated_mb": 0.0,
            "cuda_reserved_mb": 0.0,
            "cuda_max_allocated_mb": 0.0,
            "cuda_max_reserved_mb": 0.0,
        }

        if device.type == "cuda" and torch.cuda.is_available():
            dev_idx = device.index if device.index is not None else 0
            allocated = torch.cuda.memory_allocated(dev_idx) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(dev_idx) / (1024 ** 2)
            max_alloc = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 2)
            max_res = torch.cuda.max_memory_reserved(dev_idx) / (1024 ** 2)

            metrics["cuda_allocated_mb"] = round(allocated, 2)
            metrics["cuda_reserved_mb"] = round(reserved, 2)
            metrics["cuda_max_allocated_mb"] = round(max_alloc, 2)
            metrics["cuda_max_reserved_mb"] = round(max_res, 2)

        return metrics

    @staticmethod
    def reset_peak_stats(device: torch.device) -> None:
        """Reset peak memory tracking."""
        if device.type == "cuda" and torch.cuda.is_available():
            dev_idx = device.index if device.index is not None else 0
            torch.cuda.reset_peak_memory_stats(dev_idx)
