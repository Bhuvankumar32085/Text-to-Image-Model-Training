"""Test suite runner evaluating held-out test data and benchmark test prompts."""
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
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned checkpoint on the held-out test split.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory (default: pretrained base).")
    parser.add_argument("--output_dir", type=str, default="outputs/test_results", help="Directory for test results.")
    args = parser.parse_args()

    config = load_config(args.config)
    dev_mgr = DeviceManager(config)
    device = dev_mgr.device

    # Build pipeline
    pipeline = ModelFactory.build_pipeline(config)
    pipeline.to_device(device)

    # Restore checkpoint
    if args.checkpoint is not None:
        ckpt_mgr = CheckpointManager(config, checkpoint_dir=Path(args.checkpoint).parent)
        ckpt_mgr.load_checkpoint(args.checkpoint, pipeline)
        logger.info(f"Loaded checkpoint for test evaluation: {args.checkpoint}")

    # Setup datamodule
    dm = DataModule(config, tokenizer=pipeline.tokenizer)
    dm.setup()
    test_loader = dm.test_dataloader()

    # Validator & Tester
    loss_fn = DiffusionLoss(loss_type="mse")
    fwd_diff = ForwardDiffusion(pipeline.scheduler)
    validator = DiffusionValidator(
        config=config,
        pipeline=pipeline,
        forward_diffusion=fwd_diff,
        loss_fn=loss_fn,
        device=device,
    )

    tester = DiffusionTester(
        config=config,
        pipeline=pipeline,
        validator=validator,
        output_dir=args.output_dir,
    )

    logger.info("Executing comprehensive test suite on held-out test split...")
    results = tester.run_test_suite(test_dataloader=test_loader)

    # Generate Markdown Report
    report_file = generate_evaluation_report(results, output_path=f"{args.output_dir}/test_report.md")
    logger.info(f"Test suite finished. Report generated at: {report_file}")


if __name__ == "__main__":
    main()
