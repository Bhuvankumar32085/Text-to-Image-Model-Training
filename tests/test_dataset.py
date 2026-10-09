"""Unit tests for TextToImageDataset loading and corrupt image handling."""
import csv
from pathlib import Path
from PIL import Image
import pytest
import torch

from src.data.dataset import TextToImageDataset
from src.data.transforms import ImageTransformPipeline


@pytest.fixture
def sample_data(tmp_path):
    """Create a temporary dataset with valid and corrupt images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create 3 valid images
    valid_paths = []
    for i in range(3):
        p = img_dir / f"img_{i}.png"
        img = Image.new("RGB", (256, 256), color=(i * 50, 100, 150))
        img.save(p)
        valid_paths.append(p)

    # Create 1 corrupt image
    corrupt_path = img_dir / "corrupt.png"
    with open(corrupt_path, "wb") as f:
        f.write(b"not a valid png file header")

    manifest_file = tmp_path / "metadata.csv"
    with open(manifest_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "caption", "group_id"])
        writer.writeheader()
        writer.writerow({"image_path": str(valid_paths[0]), "caption": "A blue sky with fluffy white clouds", "group_id": "g1"})
        writer.writerow({"image_path": str(valid_paths[1]), "caption": "A green forest in summer", "group_id": "g1"})
        writer.writerow({"image_path": str(valid_paths[2]), "caption": "Hi", "group_id": "g2"})  # Too short (< 3 chars)
        writer.writerow({"image_path": str(corrupt_path), "caption": "A corrupt image record", "group_id": "g2"})

    return manifest_file, corrupt_path


def test_dataset_loading_and_filtering(sample_data):
    """Test loading records and filtering short captions."""
    manifest_file, _ = sample_data
    transform = ImageTransformPipeline(target_resolution=(128, 128))

    dataset = TextToImageDataset(
        manifest_path_or_records=manifest_file,
        transform=transform,
        min_caption_length=5,  # Filters out "Hi"
        corrupt_image_handling="skip",
    )

    # Out of 4 records: 1 too short ("Hi"), 3 remaining (2 valid + 1 corrupt)
    assert len(dataset) == 3

    item0 = dataset[0]
    assert "pixel_values" in item0
    assert "caption" in item0
    assert isinstance(item0["pixel_values"], torch.Tensor)
    assert item0["pixel_values"].shape == (3, 128, 128)
    assert item0["caption"] == "A blue sky with fluffy white clouds"


def test_corrupt_image_error_mode(sample_data):
    """Test corrupt image raises IOError when corrupt_image_handling is 'error'."""
    manifest_file, corrupt_path = sample_data
    transform = ImageTransformPipeline(target_resolution=(128, 128))

    dataset = TextToImageDataset(
        manifest_path_or_records=[{"image_path": str(corrupt_path), "caption": "Valid caption text"}],
        transform=transform,
        corrupt_image_handling="error",
    )

    with pytest.raises(IOError):
        _ = dataset[0]
