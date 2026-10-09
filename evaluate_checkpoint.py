"""Standalone CLI tool to evaluate and benchmark any specific saved checkpoint."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.loader import load_config
from src.data.datamodule import DataModule
from src.evaluation.report_generator import generate_evaluation_report
from src.evaluation.tester import DiffusionTester
from src.evaluation.validator import DiffusionValidator
from src.models.model_factory import ModelFactory
from src.training.checkpoint_manager import CheckpointManager
from src.training.device_manager import DeviceManager
from src.training.forward_diffusion import ForwardDiffusion
from src.training.loss import DiffusionLoss

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a specific checkpoint directory across val and test sets.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (e.g., checkpoints/best_checkpoint).")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: looks inside checkpoint folder or configs/config.yaml).")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoint_evaluation", help="Destination directory for evaluation results.")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path.resolve()}")

    # Try loading config snapshot from checkpoint if not specified
    if args.config:
        config_path = args.config
    elif (ckpt_path / "config_snapshot.yaml").exists():
        config_path = str(ckpt_path / "config_snapshot.yaml")
    else:
        config_path = "configs/config.yaml"

    logger.info(f"Loading configuration from {config_path}...")
    config = load_config(config_path)

    dev_mgr = DeviceManager(config)
    device = dev_mgr.device

    # Build model pipeline
    pipeline = ModelFactory.build_pipeline(config)
    pipeline.to_device(device)

    # Load weights
    ckpt_mgr = CheckpointManager(config, checkpoint_dir=ckpt_path.parent)
    meta = ckpt_mgr.load_checkpoint(ckpt_path, pipeline)
    logger.info(f"Loaded checkpoint metadata: {meta}")

    # Datamodule
    dm = DataModule(config, tokenizer=pipeline.tokenizer)
    dm.setup()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    loss_fn = DiffusionLoss(loss_type="mse")
    fwd_diff = ForwardDiffusion(pipeline.scheduler)
    validator = DiffusionValidator(
        config=config,
        pipeline=pipeline,
        forward_diffusion=fwd_diff,
        loss_fn=loss_fn,
        device=device,
    )

    # Validation evaluation
    val_loss, val_count, _ = validator.evaluate_loss(val_loader)
    logger.info(f"Validation Split Loss: {val_loss:.6f} ({val_count} samples)")

    # Test evaluation
    tester = DiffusionTester(
        config=config,
        pipeline=pipeline,
        validator=validator,
        output_dir=args.output_dir,
    )
    test_results = tester.run_test_suite(test_dataloader=test_loader)
    test_results["validation_loss"] = val_loss

    report_file = generate_evaluation_report(test_results, output_path=f"{args.output_dir}/checkpoint_evaluation_report.md")
    logger.info(f"Evaluation complete! Full report written to: {report_file}")


if __name__ == "__main__":
    main()
