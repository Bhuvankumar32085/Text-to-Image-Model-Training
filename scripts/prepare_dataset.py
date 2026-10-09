"""Dataset preparation script validating images, deduplicating, creating splits, and generating manifest reports."""
import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.config.schema import AppConfig
from src.data.splits import create_deterministic_splits, save_split_manifests
from src.utils.image_utils import load_image_safely

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def compute_image_hash(image_path: Path) -> Optional[str]:
    """Compute MD5 hash of image bytes for exact duplicate detection."""
    try:
        hasher = hashlib.md5()
        with open(image_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def prepare_dataset(config: AppConfig) -> Dict[str, Any]:
    """Execute complete dataset validation, duplicate detection, and deterministic split generation."""
    dataset_cfg = config.dataset
    splits_cfg = config.splits

    metadata_path = Path(dataset_cfg.caption_metadata_path)
    image_base_dir = Path(dataset_cfg.image_dir) if dataset_cfg.image_dir else metadata_path.parent

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path.resolve()}")

    logger.info(f"Reading dataset metadata from: {metadata_path}")
    logger.info(f"Resolving images relative to: {image_base_dir}")

    # Read raw records
    raw_records = []
    if metadata_path.suffix.lower() == ".csv":
        with open(metadata_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_records.append(row)
    elif metadata_path.suffix.lower() in [".json", ".jsonl"]:
        with open(metadata_path, "r", encoding="utf-8") as f:
            if metadata_path.suffix.lower() == ".jsonl":
                for line in f:
                    if line.strip():
                        raw_records.append(json.loads(line))
            else:
                data = json.load(f)
                raw_records = data if isinstance(data, list) else [data]

    total_records = len(raw_records)
    logger.info(f"Found {total_records} raw records in metadata file.")

    # Apply debug limit if specified
    if dataset_cfg.max_samples_debug is not None:
        raw_records = raw_records[: dataset_cfg.max_samples_debug]
        logger.info(f"Debug mode enabled: limited to {len(raw_records)} records.")

    valid_samples = []
    skipped_reasons = defaultdict(int)
    seen_hashes = {}
    duplicate_count = 0

    image_col = dataset_cfg.image_column
    caption_col = dataset_cfg.caption_column
    group_col = dataset_cfg.group_column

    for idx, row in enumerate(raw_records):
        img_rel = row.get(image_col)
        caption = row.get(caption_col, "")

        # 1. Check image path presence
        if not img_rel:
            skipped_reasons["missing_image_path_column"] += 1
            continue

        img_path = Path(img_rel)
        if not img_path.is_absolute():
            img_path = image_base_dir / img_path

        # 2. Check file existence
        if not img_path.exists():
            skipped_reasons["file_not_found"] += 1
            continue

        # 3. Check extension
        if img_path.suffix.lower() not in dataset_cfg.supported_extensions:
            skipped_reasons["unsupported_extension"] += 1
            continue

        # 4. Check image validity (PIL verify)
        img_pil = load_image_safely(img_path)
        if img_pil is None:
            skipped_reasons["corrupt_or_unreadable_image"] += 1
            continue

        # 5. Check caption validity
        caption_str = str(caption).strip()
        if len(caption_str) < dataset_cfg.min_caption_length:
            skipped_reasons["caption_too_short"] += 1
            continue

        if len(caption_str) > dataset_cfg.max_caption_length:
            caption_str = caption_str[: dataset_cfg.max_caption_length]

        # 6. Duplicate detection via MD5
        if dataset_cfg.duplicate_detection:
            img_hash = compute_image_hash(img_path)
            if img_hash in seen_hashes:
                duplicate_count += 1
                skipped_reasons["duplicate_image_hash"] += 1
                continue
            seen_hashes[img_hash] = img_path

        # Valid record
        valid_samples.append({
            "image_path": str(img_path.resolve()),
            "caption": caption_str,
            "group_id": row.get(group_col, f"group_{idx}") if group_col else f"group_{idx}",
            "original_width": img_pil.width,
            "original_height": img_pil.height,
        })

    logger.info(f"Validated {len(valid_samples)}/{total_records} samples. Skipped: {dict(skipped_reasons)}")

    if len(valid_samples) == 0:
        raise ValueError("No valid samples remained after preprocessing and validation!")

    # 7. Create deterministic train, val, and test splits
    train_recs, val_recs, test_recs = create_deterministic_splits(
        records=valid_samples,
        train_fraction=splits_cfg.train_fraction,
        test_fraction=splits_cfg.test_fraction,
        validation_fraction_of_train=splits_cfg.validation_fraction_of_train,
        seed=splits_cfg.split_seed,
        group_column=group_col if splits_cfg.group_aware_split else None,
    )

    # 8. Save split manifests
    manifest_paths = save_split_manifests(
        train_records=train_recs,
        val_records=val_recs,
        test_records=test_recs,
        output_dir=splits_cfg.manifest_dir,
    )

    # 9. Save dataset report
    processed_dir = Path(dataset_cfg.dataset_cache_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "total_source_records": total_records,
        "valid_samples_count": len(valid_samples),
        "skipped_summary": dict(skipped_reasons),
        "duplicates_filtered": duplicate_count,
        "train_samples_count": len(train_recs),
        "validation_samples_count": len(val_recs),
        "test_samples_count": len(test_recs),
        "target_resolution": dataset_cfg.target_resolution,
        "aspect_ratio_policy": dataset_cfg.aspect_ratio_policy,
        "manifest_files": manifest_paths,
    }

    report_path = processed_dir / "dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Dataset preparation complete! Report saved to {report_path}")
    logger.info(f"Manifests saved in {splits_cfg.manifest_dir}:")
    logger.info(f"  - Train: {len(train_recs)} samples -> {manifest_paths['train']}")
    logger.info(f"  - Val:   {len(val_recs)} samples -> {manifest_paths['val']}")
    logger.info(f"  - Test:  {len(test_recs)} samples -> {manifest_paths['test']}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate dataset and generate deterministic train/val/test splits.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    prepare_dataset(config)


if __name__ == "__main__":
    main()
