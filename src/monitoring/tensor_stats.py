"""Statistical summaries and numerical behavior tracking for intermediate model tensors."""
from typing import Any, Dict, Optional, Union
import numpy as np
import torch


class TensorStatsTracker:
    """Computes numerical statistics (min, max, mean, std, NaN/Inf counts, quantiles) for diagnostic tensors."""

    @staticmethod
    def compute_stats(name: str, tensor: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Compute statistical summary of a tensor."""
        if isinstance(tensor, torch.Tensor):
            t = tensor.detach().cpu().float()
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
            device = str(tensor.device)
            t_flat = t.view(-1)
        elif isinstance(tensor, np.ndarray):
            t_flat = torch.from_numpy(tensor).float().view(-1)
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
            device = "cpu"
        else:
            return {"name": name, "error": f"Unsupported type {type(tensor)}"}

        nan_count = int(torch.isnan(t_flat).sum().item())
        inf_count = int(torch.isinf(t_flat).sum().item())

        # Filter finite values for mean/std/quantiles
        finite_mask = torch.isfinite(t_flat)
        if finite_mask.sum() > 0:
            finite_vals = t_flat[finite_mask]
            min_val = float(finite_vals.min().item())
            max_val = float(finite_vals.max().item())
            mean_val = float(finite_vals.mean().item())
            std_val = float(finite_vals.std().item()) if len(finite_vals) > 1 else 0.0

            # Quantiles (25%, 50%, 75%)
            try:
                q25 = float(torch.quantile(finite_vals, 0.25).item())
                q50 = float(torch.quantile(finite_vals, 0.50).item())
                q75 = float(torch.quantile(finite_vals, 0.75).item())
            except Exception:
                q25, q50, q75 = None, None, None
        else:
            min_val, max_val, mean_val, std_val = None, None, None, None
            q25, q50, q75 = None, None, None

        return {
            "tensor_name": name,
            "shape": str(shape),
            "dtype": dtype,
            "device": device,
            "num_elements": len(t_flat),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "q25": q25,
            "median_q50": q50,
            "q75": q75,
        }
