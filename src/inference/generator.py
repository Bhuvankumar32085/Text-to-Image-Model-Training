"""High-level Image Generation engine with checkpoint loading and metadata output."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch

from src.config.schema import AppConfig
from src.models.model_factory import ModelFactory
from src.training.checkpoint_manager import CheckpointManager
from src.training.device_manager import DeviceManager
from src.utils.image_utils import save_image_safely
from src.utils.paths import get_timestamp_str

logger = logging.getLogger(__name__)


class ImageGenerator:
    """High-level generator for running text-to-image inference with Stable Diffusion 1.5."""

    def __init__(
        self,
        config: AppConfig,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config

        # Setup hardware
        if device is None:
            dev_mgr = DeviceManager(config)
            self.device = dev_mgr.device
        else:
            self.device = device

        logger.info(f"Initializing Stable Diffusion pipeline on {self.device}...")
        self.pipeline = ModelFactory.build_pipeline(config)
        self.pipeline.to_device(self.device)

        # Restore checkpoint weights if provided
        ckpt = checkpoint_path or config.inference.checkpoint_path
        if ckpt is not None:
            ckpt_mgr = CheckpointManager(config, checkpoint_dir=Path(ckpt).parent)
            ckpt_mgr.load_checkpoint(ckpt, self.pipeline)
            logger.info(f"Loaded fine-tuned weights from {ckpt}")

        self.pipeline.eval()

    def generate_images(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: Optional[int] = None,
        seed: Optional[int] = None,
        scheduler_type: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate images and save them with JSON metadata."""
        inf_cfg = self.config.inference
        h = height or inf_cfg.height
        w = width or inf_cfg.width
        steps = num_inference_steps or inf_cfg.num_inference_steps
        guidance = guidance_scale if guidance_scale is not None else inf_cfg.guidance_scale
        num_imgs = num_images_per_prompt or inf_cfg.num_images_per_prompt
        s = seed if seed is not None else inf_cfg.seed
        sched = scheduler_type or inf_cfg.scheduler_type
        out_dir = Path(output_dir or inf_cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Generating image(s) for prompt: \"{prompt}\" "
            f"[Resolution: {h}x{w}, Steps: {steps}, CFG: {guidance}, Seed: {s}, Scheduler: {sched}]"
        )

        images = self.pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt or inf_cfg.negative_prompt,
            height=h,
            width=w,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_imgs,
            seed=s,
            scheduler_type=sched,
        )

        timestamp = get_timestamp_str()
        results = []

        for idx, img in enumerate(images):
            filename = f"gen_{timestamp}_{idx:02d}.{inf_cfg.output_format}"
            filepath = out_dir / filename
            save_image_safely(img, filepath, format=inf_cfg.output_format.upper())

            meta = {
                "file_path": str(filepath.resolve()),
                "prompt": prompt if isinstance(prompt, str) else prompt[idx % len(prompt)],
                "negative_prompt": negative_prompt or inf_cfg.negative_prompt,
                "height": h,
                "width": w,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "seed": s,
                "scheduler_type": sched,
                "model_identifier": self.config.model.model_identifier,
                "timestamp": timestamp,
            }

            meta_filepath = out_dir / f"gen_{timestamp}_{idx:02d}_metadata.json"
            with open(meta_filepath, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            results.append({"image": img, "metadata": meta})
            logger.info(f"Saved generated image to {filepath}")

        return results
