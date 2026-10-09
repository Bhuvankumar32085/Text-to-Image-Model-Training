"""Validation CLI evaluating validation loss and preview images from a checkpoint."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.loader import load_config
from src.data.datamodule import DataModule
from src.evaluation.validator import DiffusionValidator
from src.models.model_factory import ModelFactory
from src.training.checkpoint_manager import CheckpointManager
from src.training.device_manager import DeviceManager
from src.training.forward_diffusion import ForwardDiffusion
from src.training.loss import DiffusionLoss
from src.utils.image_utils import save_image_safely

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate validation loss and generate previews on a trained checkpoint.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory (default: pretrained base).")
    parser.add_argument("--output_dir", type=str, default="outputs/validation_run", help="Output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    dev_mgr = DeviceManager(config)
    device = dev_mgr.device

    # Build pipeline
    pipeline = ModelFactory.build_pipeline(config)
    pipeline.to_device(device)

    # Restore weights
    if args.checkpoint is not None:
        ckpt_mgr = CheckpointManager(config, checkpoint_dir=Path(args.checkpoint).parent)
        ckpt_mgr.load_checkpoint(args.checkpoint, pipeline)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Setup dataloaders
    dm = DataModule(config, tokenizer=pipeline.tokenizer)
    dm.setup()
    val_loader = dm.val_dataloader()

    # Validator
    loss_fn = DiffusionLoss(loss_type="mse")
    fwd_diff = ForwardDiffusion(pipeline.scheduler)
    validator = DiffusionValidator(
        config=config,
        pipeline=pipeline,
        forward_diffusion=fwd_diff,
        loss_fn=loss_fn,
        device=device,
    )

    logger.info("Computing validation split diffusion loss...")
    val_loss, count, elapsed = validator.evaluate_loss(val_loader)
    logger.info(f"Validation Loss: {val_loss:.6f} ({count} samples evaluated in {elapsed:.2f}s)")

    # Previews
    logger.info("Generating fixed preview images...")
    previews = validator.generate_previews()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preview_records = []
    for idx, (img, prompt) in enumerate(zip(previews, config.validation.fixed_prompts)):
        path = out_dir / f"val_preview_{idx:02d}.png"
        save_image_safely(img, path)
        preview_records.append({"prompt": prompt, "file_path": str(path.resolve())})

    results = {
        "validation_loss": val_loss,
        "sample_count": count,
        "evaluation_time_sec": elapsed,
        "previews": preview_records,
    }

    with open(out_dir / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Validation complete! Results saved to {out_dir / 'validation_results.json'}")


if __name__ == "__main__":
    main()
