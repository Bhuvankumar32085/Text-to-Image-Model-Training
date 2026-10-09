"""Visual and numerical inspection of dataset batches and preprocessed tensors."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.data.datamodule import DataModule
from src.models.component_loader import load_model_components
from src.utils.image_utils import make_image_grid, save_image_safely, tensor_to_pil

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def inspect_dataset(config_path: str = "configs/config.yaml", num_samples: int = 4) -> None:
    """Inspect dataset batches and save preview grids of preprocessed tensors."""
    config = load_config(config_path)
    logger.info(f"Loading tokenizer for inspection...")

    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        config.model.model_identifier,
        subfolder="tokenizer",
    )

    dm = DataModule(config, tokenizer=tokenizer)
    dm.setup()

    train_loader = dm.train_dataloader()
    logger.info(f"Train DataLoader initialized with {len(train_loader)} batches.")

    batch = next(iter(train_loader))
    pixel_values = batch["pixel_values"]
    input_ids = batch["input_ids"]
    captions = batch["captions"]

    logger.info("=== BATCH INSPECTION ===")
    logger.info(f"Image tensor batch shape: {pixel_values.shape} (dtype: {pixel_values.dtype})")
    logger.info(f"Pixel value range: min={pixel_values.min().item():.3f}, max={pixel_values.max().item():.3f}")
    logger.info(f"Token IDs shape: {input_ids.shape if input_ids is not None else None}")

    preview_images = []
    for i in range(min(num_samples, pixel_values.shape[0])):
        cap = captions[i]
        logger.info(f"Sample {i}: \"{cap}\"")
        img_pil = tensor_to_pil(pixel_values[i], normalization=config.dataset.image_normalization)
        preview_images.append(img_pil)

    if preview_images:
        out_path = Path("outputs") / "dataset_inspection_grid.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        grid = make_image_grid(preview_images)
        save_image_safely(grid, out_path)
        logger.info(f"Saved dataset inspection preview grid to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset batches and preprocessed images.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples to preview.")
    args = parser.parse_args()

    inspect_dataset(args.config, args.samples)


if __name__ == "__main__":
    main()
