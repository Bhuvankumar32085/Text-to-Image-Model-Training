"""Unit tests for image transformation pipeline and normalization routines."""
import numpy as np
from PIL import Image
import pytest
import torch

from src.data.transforms import ImageTransformPipeline
from src.utils.image_utils import pil_to_tensor, tensor_to_pil


def test_transform_resolution_and_range():
    """Verify target resolution and [-1, 1] normalization."""
    img = Image.new("RGB", (800, 600), color=(200, 100, 50))
    pipeline = ImageTransformPipeline(
        target_resolution=(512, 512),
        aspect_ratio_policy="center_crop",
        normalization="neg_one_to_one",
    )

    tensor = pipeline(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 512, 512)
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0


def test_aspect_ratio_policies():
    """Verify center_crop and resize policies produce exact dimensions."""
    img = Image.new("RGB", (400, 800), color=(50, 150, 250))

    for policy in ["center_crop", "random_crop", "resize"]:
        pipeline = ImageTransformPipeline(
            target_resolution=(256, 256),
            aspect_ratio_policy=policy,
            normalization="zero_to_one",
        )
        t = pipeline(img)
        assert t.shape == (3, 256, 256)
        assert t.min() >= 0.0
        assert t.max() <= 1.0


def test_tensor_pil_roundtrip():
    """Verify roundtrip conversion between PIL and PyTorch tensor."""
    img_orig = Image.new("RGB", (64, 64), color=(120, 80, 200))
    tensor = pil_to_tensor(img_orig, normalization="neg_one_to_one")
    assert tensor.shape == (3, 64, 64)

    img_recovered = tensor_to_pil(tensor, normalization="neg_one_to_one")
    assert img_recovered.size == (64, 64)

    orig_arr = np.array(img_orig)
    rec_arr = np.array(img_recovered)
    # Differences due to float rounding should be <= 1 pixel intensity
    assert np.max(np.abs(orig_arr.astype(int) - rec_arr.astype(int))) <= 1
