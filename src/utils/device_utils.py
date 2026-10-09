"""Device inspection and hardware management utilities."""
import os
import sys
from typing import Dict, Any, Tuple


def check_cuda_availability() -> Tuple[bool, str]:
    """Check if CUDA is available in the current environment."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            return True, f"CUDA Available: {device_count} device(s) found. Device 0: {device_name}"
        return False, "CUDA is not available in PyTorch. Running on CPU."
    except ImportError:
        return False, "PyTorch is not installed."


def get_device_info(device_index: int = 0) -> Dict[str, Any]:
    """Return comprehensive hardware and PyTorch device information."""
    info = {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "selected_device_name": "CPU",
        "total_memory_gb": None,
        "free_memory_gb": None,
        "allocated_memory_gb": None,
        "reserved_memory_gb": None,
        "compute_capability": None,
    }
    
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            
            if 0 <= device_index < info["device_count"]:
                props = torch.cuda.get_device_properties(device_index)
                info["selected_device_name"] = props.name
                info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
                info["compute_capability"] = f"{props.major}.{props.minor}"
                
                allocated = torch.cuda.memory_allocated(device_index)
                reserved = torch.cuda.memory_reserved(device_index)
                info["allocated_memory_gb"] = round(allocated / (1024 ** 3), 3)
                info["reserved_memory_gb"] = round(reserved / (1024 ** 3), 3)
                info["free_memory_gb"] = round((props.total_memory - reserved) / (1024 ** 3), 2)
    except ImportError:
        pass
        
    return info
