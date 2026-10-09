"""Test suite evaluator for held-out splits and final benchmark generation."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.config.schema import AppConfig
from src.evaluation.generation_metrics import compute_generation_metrics
from src.evaluation.validator import DiffusionValidator
from src.models.pipeline import StableDiffusionFineTuningPipeline
from src.utils.image_utils import save_image_safely

logger = logging.getLogger(__name__)


class DiffusionTester:
    """Runs final evaluation and benchmark image generation on held-out test splits."""

    def __init__(
        self,
        config: AppConfig,
        pipeline: StableDiffusionFineTuningPipeline,
        validator: DiffusionValidator,
        output_dir: Optional[str] = None,
    ):
        self.config = config
        self.pipeline = pipeline
        self.validator = validator
        self.output_dir = Path(output_dir or config.test.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_test_suite(
        self,
        test_dataloader: Optional[DataLoader] = None,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute test loss evaluation and test image generation suite."""
        test_report: Dict[str, Any] = {
            "experiment_name": self.config.project.experiment_name,
            "test_loss": None,
            "test_samples_count": 0,
            "evaluation_time_sec": None,
            "generated_images": [],
            "metrics": {},
        }

        # 1. Evaluate test split diffusion loss if dataloader provided
        if test_dataloader is not None and len(test_dataloader) > 0:
            logger.info("Computing diffusion loss on held-out test split...")
            avg_loss, count, elapsed = self.validator.evaluate_loss(test_dataloader)
            test_report["test_loss"] = avg_loss
            test_report["test_samples_count"] = count
            test_report["evaluation_time_sec"] = elapsed
            logger.info(f"Test split diffusion loss: {avg_loss:.4f} ({count} samples)")

        # 2. Generate benchmark test images
        eval_prompts = prompts or self.config.test.test_prompts
        if eval_prompts:
            logger.info(f"Generating benchmark images for {len(eval_prompts)} test prompt(s)...")
            images = self.pipeline.generate(
                prompt=eval_prompts,
                num_inference_steps=self.config.test.num_inference_steps,
                guidance_scale=self.config.test.guidance_scale,
                num_images_per_prompt=self.config.test.num_images_per_prompt,
                height=self.config.dataset.target_resolution[0],
                width=self.config.dataset.target_resolution[1],
                seed=self.config.test.seed,
            )

            # Save generated images
            saved_meta = []
            for idx, (img, prompt) in enumerate(zip(images, eval_prompts * self.config.test.num_images_per_prompt)):
                img_name = f"test_sample_{idx:03d}.png"
                img_path = self.output_dir / img_name
                save_image_safely(img, img_path)
                saved_meta.append({
                    "index": idx,
                    "prompt": prompt,
                    "file_path": str(img_path),
                    "seed": self.config.test.seed,
                    "steps": self.config.test.num_inference_steps,
                    "guidance_scale": self.config.test.guidance_scale,
                })

            test_report["generated_images"] = saved_meta

            # Compute quality metrics
            metrics = compute_generation_metrics(
                generated_images=images,
                prompts=eval_prompts * self.config.test.num_images_per_prompt,
                metrics_to_compute=self.config.test.metrics,
            )
            test_report["metrics"] = metrics

        # 3. Save report JSON
        report_file = self.output_dir / "test_metrics.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(test_report, f, indent=2)

        logger.info(f"Saved complete test report to {report_file}")
        return test_report
