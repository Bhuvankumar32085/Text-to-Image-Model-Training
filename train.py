"""Main entry point for training and fine-tuning Stable Diffusion 1.5."""
import argparse
import logging
import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.loader import load_config
from src.data.datamodule import DataModule
from src.models.model_factory import ModelFactory
from src.monitoring.experiment_tracker import ExperimentTracker
from src.training.device_manager import DeviceManager
from src.training.trainer import DiffusionTrainer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion 1.5 on custom image-caption datasets.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml file.")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from specified checkpoint path or 'latest'.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override base learning rate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="Override mixed precision mode.")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"], help="Override hardware device.")
    args = parser.parse_args()

    # Build overrides dictionary
    overrides = {}
    if args.resume is not None:
        overrides.setdefault("training", {})["resume_from_checkpoint"] = args.resume
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        overrides.setdefault("training", {})["train_batch_size"] = args.batch_size
    if args.mixed_precision is not None:
        overrides.setdefault("training", {})["mixed_precision"] = args.mixed_precision
    if args.device is not None:
        overrides.setdefault("device", {})["type"] = args.device

    # Load validated configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config, overrides=overrides if overrides else None)

    # Initialize Hardware Device Manager
    device_manager = DeviceManager(config)

    # Instantiate Model Pipeline
    logger.info("Initializing Stable Diffusion model components...")
    pipeline = ModelFactory.build_pipeline(config)

    # Initialize DataModule
    logger.info("Setting up dataset and dataloaders...")
    datamodule = DataModule(config, tokenizer=pipeline.tokenizer)

    # Initialize Experiment Tracker
    experiment_tracker = ExperimentTracker(config)

    # Instantiate and Run Trainer
    trainer = DiffusionTrainer(
        config=config,
        pipeline=pipeline,
        datamodule=datamodule,
        device_manager=device_manager,
        experiment_tracker=experiment_tracker,
    )

    logger.info("Starting training run...")
    report_path = trainer.train()
    logger.info(f"Training completed successfully! Full diagnostic report at: {report_path}")


if __name__ == "__main__":
    main()
