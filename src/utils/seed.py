"""Random seed setting for full reproducibility across Python, NumPy, and PyTorch."""
import os
import random
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set seeds across Python standard library, NumPy, and PyTorch if available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # If PyTorch >= 1.8 and CUDA
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
    except ImportError:
        pass
        
    return seed
