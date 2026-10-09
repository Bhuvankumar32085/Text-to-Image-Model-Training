"""Unit tests for TensorStatsTracker numerical metrics and NaN/Inf detection."""
import numpy as np
import pytest
import torch

from src.monitoring.tensor_stats import TensorStatsTracker


def test_tensor_stats_calculation():
    """Verify statistical metrics calculation for normal tensor."""
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    stats = TensorStatsTracker.compute_stats("sample_tensor", tensor)

    assert stats["tensor_name"] == "sample_tensor"
    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0
    assert stats["min"] == 1.0
    assert stats["max"] == 10.0
    assert stats["mean"] == 5.5
    assert stats["median_q50"] == 5.5


def test_nan_and_inf_detection():
    """Verify that NaNs and Infs are accurately detected and counted."""
    arr = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 2.0])
    stats = TensorStatsTracker.compute_stats("faulty_tensor", arr)

    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 2
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
