"""Dataset validation tool inspecting image integrity, caption lengths, and aspect ratio distributions."""
import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.utils.image_utils import load_image_safely

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def validate_dataset_manifest(manifest_csv: str) -> None:
    """Perform in-depth validation and summary statistics on a dataset manifest."""
    path = Path(manifest_csv)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    logger.info(f"Inspecting manifest: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    total = len(records)
    logger.info(f"Total samples in manifest: {total}")

    resolutions = Counter()
    aspect_ratios = Counter()
    caption_lengths = []
    corrupt_files = []

    for idx, row in enumerate(records):
        img_path = Path(row.get("image_path", ""))
        caption = row.get("caption", "")

        caption_lengths.append(len(caption.strip()))

        if not img_path.exists():
            corrupt_files.append((str(img_path), "File does not exist"))
            continue

        img = load_image_safely(img_path)
        if img is None:
            corrupt_files.append((str(img_path), "Corrupt / unreadable PIL image"))
            continue

        resolutions[f"{img.width}x{img.height}"] += 1
        ar = round(img.width / img.height, 2)
        aspect_ratios[ar] += 1

    logger.info(f"--- VALIDATION REPORT FOR {path.name} ---")
    logger.info(f"Valid Images: {total - len(corrupt_files)} / {total}")
    if corrupt_files:
        logger.warning(f"Corrupt or missing files: {len(corrupt_files)}")
        for p, reason in corrupt_files[:5]:
            logger.warning(f"  - {p}: {reason}")

    logger.info("Top Image Resolutions:")
    for res, cnt in resolutions.most_common(5):
        logger.info(f"  {res}: {cnt} images ({cnt/total*100:.1f}%)")

    logger.info("Top Aspect Ratios (W/H):")
    for ar, cnt in aspect_ratios.most_common(5):
        logger.info(f"  {ar:.2f}: {cnt} images ({cnt/total*100:.1f}%)")

    if caption_lengths:
        logger.info(f"Caption Lengths: Min={min(caption_lengths)}, Max={max(caption_lengths)}, Mean={sum(caption_lengths)/len(caption_lengths):.1f} chars")


def main():
    parser = argparse.ArgumentParser(description="Validate dataset integrity and print summary statistics.")
    parser.add_argument("--manifest", type=str, default="data/splits/train_manifest.csv", help="Path to manifest CSV.")
    args = parser.parse_args()
    validate_dataset_manifest(args.manifest)


if __name__ == "__main__":
    main()
