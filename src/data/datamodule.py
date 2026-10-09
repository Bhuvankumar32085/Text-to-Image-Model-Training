"""DataModule managing dataset initialization, splitting, and DataLoader instantiation."""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader, Dataset

from src.config.schema import AppConfig
from src.data.collator import TextToImageCollator
from src.data.dataset import TextToImageDataset
from src.data.splits import create_deterministic_splits, load_split_manifests, save_split_manifests
from src.data.transforms import ImageTransformPipeline

logger = logging.getLogger(__name__)


class DataModule:
    """Encapsulates data loading, transformation, and DataLoader construction."""

    def __init__(self, config: AppConfig, tokenizer: Any):
        self.config = config
        self.tokenizer = tokenizer

        # Image transform
        self.transform = ImageTransformPipeline(
            target_resolution=tuple(config.dataset.target_resolution),
            aspect_ratio_policy=config.dataset.aspect_ratio_policy,
            normalization=config.dataset.image_normalization,
        )

        # Batch collator
        self.collator = TextToImageCollator(
            tokenizer=self.tokenizer,
            max_length=getattr(self.tokenizer, "model_max_length", 77) if self.tokenizer else 77,
        )

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self) -> None:
        """Parse source dataset, create/load deterministic splits, and initialize datasets."""
        manifest_dir = Path(self.config.splits.manifest_dir)
        train_manifest = manifest_dir / "train_manifest.csv"
        val_manifest = manifest_dir / "val_manifest.csv"
        test_manifest = manifest_dir / "test_manifest.csv"

        use_existing = (
            not self.config.splits.overwrite_manifests
            and train_manifest.exists()
            and val_manifest.exists()
            and test_manifest.exists()
        )

        if use_existing:
            logger.info(f"Loading existing split manifests from {manifest_dir}")
            train_recs, val_recs, test_recs = load_split_manifests(manifest_dir)
        else:
            logger.info("Creating new deterministic splits from source dataset...")
            # Load all raw samples first
            raw_dataset = TextToImageDataset(
                manifest_path_or_records=self.config.dataset.caption_metadata_path,
                image_dir=self.config.dataset.image_dir,
                image_column=self.config.dataset.image_column,
                caption_column=self.config.dataset.caption_column,
                group_column=self.config.dataset.group_column if self.config.splits.group_aware_split else None,
                corrupt_image_handling=self.config.dataset.corrupt_image_handling,
                min_caption_length=self.config.dataset.min_caption_length,
                max_caption_length=self.config.dataset.max_caption_length,
                max_samples=self.config.dataset.max_samples_debug,
            )

            all_records = [s["original_record"] for s in raw_dataset.samples]
            train_recs, val_recs, test_recs = create_deterministic_splits(
                records=all_records,
                train_fraction=self.config.splits.train_fraction,
                test_fraction=self.config.splits.test_fraction,
                validation_fraction_of_train=self.config.splits.validation_fraction_of_train,
                seed=self.config.splits.split_seed,
                group_column=self.config.dataset.group_column if self.config.splits.group_aware_split else None,
            )

            if self.config.splits.save_manifests:
                save_split_manifests(train_recs, val_recs, test_recs, output_dir=manifest_dir)
                logger.info(f"Saved deterministic split manifests to {manifest_dir}")

        # Initialize split datasets
        self.train_dataset = TextToImageDataset(
            manifest_path_or_records=train_recs,
            image_dir=self.config.dataset.image_dir,
            image_column=self.config.dataset.image_column,
            caption_column=self.config.dataset.caption_column,
            transform=self.transform,
            corrupt_image_handling=self.config.dataset.corrupt_image_handling,
            min_caption_length=self.config.dataset.min_caption_length,
            max_caption_length=self.config.dataset.max_caption_length,
        )

        self.val_dataset = TextToImageDataset(
            manifest_path_or_records=val_recs,
            image_dir=self.config.dataset.image_dir,
            image_column=self.config.dataset.image_column,
            caption_column=self.config.dataset.caption_column,
            transform=self.transform,
            corrupt_image_handling=self.config.dataset.corrupt_image_handling,
            min_caption_length=self.config.dataset.min_caption_length,
            max_caption_length=self.config.dataset.max_caption_length,
        )

        self.test_dataset = TextToImageDataset(
            manifest_path_or_records=test_recs,
            image_dir=self.config.dataset.image_dir,
            image_column=self.config.dataset.image_column,
            caption_column=self.config.dataset.caption_column,
            transform=self.transform,
            corrupt_image_handling=self.config.dataset.corrupt_image_handling,
            min_caption_length=self.config.dataset.min_caption_length,
            max_caption_length=self.config.dataset.max_caption_length,
        )

        logger.info(
            f"Dataset splits initialized: Train={len(self.train_dataset)}, "
            f"Val={len(self.val_dataset)}, Test={len(self.test_dataset)}"
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.train_batch_size,
            shuffle=self.config.training.shuffle,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=self.config.training.pin_memory,
            persistent_workers=self.config.training.persistent_workers if self.config.training.dataloader_num_workers > 0 else False,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.validation.validation_batch_size,
            shuffle=False,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=self.config.training.pin_memory,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.test.test_batch_size,
            shuffle=False,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=self.config.training.pin_memory,
            collate_fn=self.collator,
        )
