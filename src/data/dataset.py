"""PyTorch Dataset implementation for Image-Caption pairs."""
import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)


class TextToImageDataset:
    """Dataset for pairing images with text captions."""

    def __init__(
        self,
        manifest_path_or_records: Union[str, Path, List[Dict[str, Any]]],
        image_dir: Optional[Union[str, Path]] = None,
        image_column: str = "image_path",
        caption_column: str = "caption",
        group_column: Optional[str] = "group_id",
        transform: Optional[Callable] = None,
        corrupt_image_handling: str = "skip",
        min_caption_length: int = 3,
        max_caption_length: int = 500,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.image_column = image_column
        self.caption_column = caption_column
        self.group_column = group_column
        self.transform = transform
        self.corrupt_image_handling = corrupt_image_handling
        self.min_caption_length = min_caption_length
        self.max_caption_length = max_caption_length
        self.max_samples = max_samples

        self.samples: List[Dict[str, Any]] = []
        self._load_samples(manifest_path_or_records)

    def _load_samples(self, source: Union[str, Path, List[Dict[str, Any]]]) -> None:
        """Parse records from file or memory and validate basic fields."""
        raw_records: List[Dict[str, Any]] = []

        if isinstance(source, list):
            raw_records = source
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Manifest file not found: {path}")

            if path.suffix.lower() == ".csv":
                with open(path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        raw_records.append(row)
            elif path.suffix.lower() in [".json", ".jsonl"]:
                with open(path, "r", encoding="utf-8") as f:
                    if path.suffix.lower() == ".jsonl":
                        for line in f:
                            line = line.strip()
                            if line:
                                raw_records.append(json.loads(line))
                    else:
                        data = json.load(f)
                        raw_records = data if isinstance(data, list) else [data]
            else:
                raise ValueError(f"Unsupported manifest format '{path.suffix}'. Use CSV or JSONL.")

        valid_samples = []
        for idx, record in enumerate(raw_records):
            img_rel = record.get(self.image_column)
            caption = record.get(self.caption_column, "")

            if not img_rel:
                continue

            caption_str = str(caption).strip()
            if len(caption_str) < self.min_caption_length:
                continue

            if len(caption_str) > self.max_caption_length:
                caption_str = caption_str[: self.max_caption_length]

            # Resolve image path
            img_path = Path(img_rel)
            if not img_path.is_absolute() and self.image_dir is not None:
                img_path = self.image_dir / img_path

            valid_samples.append({
                "image_path": str(img_path),
                "caption": caption_str,
                "group_id": record.get(self.group_column, str(idx)) if self.group_column else str(idx),
                "original_record": record,
            })

            if self.max_samples is not None and len(valid_samples) >= self.max_samples:
                break

        self.samples = valid_samples
        if len(self.samples) == 0:
            logger.warning("Dataset initialized with 0 valid samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch and preprocess a single sample."""
        sample = self.samples[idx]
        img_path = sample["image_path"]
        caption = sample["caption"]

        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            if self.corrupt_image_handling == "error":
                raise IOError(f"Failed to load image at {img_path}: {e}")
            else:
                logger.warning(f"Corrupt or missing image {img_path}: {e}. Returning fallback dummy tensor.")
                img_rgb = Image.new("RGB", (512, 512), color=(128, 128, 128))

        if self.transform is not None:
            pixel_values = self.transform(img_rgb)
        else:
            import numpy as np
            import torch
            arr = np.array(img_rgb).astype(np.float32) / 255.0
            pixel_values = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0

        return {
            "pixel_values": pixel_values,
            "caption": caption,
            "image_path": img_path,
            "group_id": sample.get("group_id", str(idx)),
        }
