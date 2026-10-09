"""Dataset download script supporting Hugging Face datasets and URL downloads with sample caps."""
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.config.schema import AppConfig
from src.utils.image_utils import save_image_safely

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def download_huggingface_dataset(
    dataset_id: str,
    download_dir: Path,
    max_samples: int = 100,
    auth_token: Optional[str] = None,
) -> Path:
    """Download a dataset from Hugging Face Datasets hub and export as image files + metadata.csv."""
    logger.info(f"Connecting to Hugging Face Hub for dataset '{dataset_id}' (max_samples={max_samples})...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' package is required. Install via: pip install datasets")

    download_dir.mkdir(parents=True, exist_ok=True)
    images_dir = download_dir / "images"
    images_dir.mkdir(exist_ok=True)
    metadata_file = download_dir / "metadata.csv"

    dataset = load_dataset(dataset_id, split="train", token=auth_token)
    total_available = len(dataset)
    num_to_download = min(total_available, max_samples) if max_samples else total_available
    logger.info(f"Found {total_available} samples. Downloading {num_to_download} samples...")

    records = []
    for i in range(num_to_download):
        item = dataset[i]
        # Identify image and caption fields
        img_obj = item.get("image") or item.get("img")
        caption = item.get("text") or item.get("caption") or item.get("description", "")

        if img_obj is None:
            continue

        filename = f"image_{i:05d}.png"
        img_path = images_dir / filename

        if isinstance(img_obj, Image.Image):
            save_image_safely(img_obj.convert("RGB"), img_path)
        else:
            logger.warning(f"Skipping sample {i}: unsupported image object format {type(img_obj)}")
            continue

        records.append({
            "image_path": f"images/{filename}",
            "caption": str(caption).strip(),
            "sample_index": i,
        })

        if (i + 1) % 25 == 0 or (i + 1) == num_to_download:
            logger.info(f"Downloaded and saved {i + 1}/{num_to_download} samples...")

    # Write metadata.csv
    with open(metadata_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "caption", "sample_index"])
        writer.writeheader()
        writer.writerows(records)

    # Write download manifest report
    report = {
        "dataset_provider": "huggingface",
        "dataset_identifier": dataset_id,
        "download_directory": str(download_dir.resolve()),
        "samples_downloaded": len(records),
        "metadata_csv": str(metadata_file.resolve()),
    }
    with open(download_dir / "download_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Successfully downloaded {len(records)} samples to {download_dir}")
    logger.info(f"Metadata written to {metadata_file}")
    return metadata_file


def main():
    parser = argparse.ArgumentParser(description="Download image-caption dataset based on config.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--dataset_id", type=str, default=None, help="Hugging Face dataset identifier.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to download.")
    parser.add_argument("--output_dir", type=str, default=None, help="Destination directory.")
    args = parser.parse_args()

    config: AppConfig = load_config(args.config)

    provider = config.download.dataset_provider
    dataset_id = args.dataset_id or config.download.dataset_identifier
    download_dir = Path(args.output_dir or config.download.download_dir)
    max_samples = args.max_samples or config.download.max_samples

    auth_token = os.environ.get(config.download.auth_token_env_var)

    if provider == "huggingface":
        if not dataset_id:
            raise ValueError("No dataset_identifier provided for Hugging Face download.")
        download_huggingface_dataset(
            dataset_id=dataset_id,
            download_dir=download_dir,
            max_samples=max_samples,
            auth_token=auth_token,
        )
    elif provider == "local":
        logger.info(f"Dataset provider is 'local'. Checking existing local data at {config.dataset.image_dir}...")
        if not Path(config.dataset.caption_metadata_path).exists():
            logger.error(f"Local metadata CSV not found at {config.dataset.caption_metadata_path}")
        else:
            logger.info("Local dataset verified successfully.")
    else:
        raise ValueError(f"Unsupported download provider: {provider}")


if __name__ == "__main__":
    main()
