"""Inference CLI for generating images with fine-tuned or pretrained Stable Diffusion 1.5."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.loader import load_config
from src.inference.generator import ImageGenerator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate images from text prompts using Stable Diffusion 1.5.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml file.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt to generate.")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt to guide generation away from.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned checkpoint folder.")
    parser.add_argument("--height", type=int, default=None, help="Image height (must be divisible by 8).")
    parser.add_argument("--width", type=int, default=None, help="Image width (must be divisible by 8).")
    parser.add_argument("--steps", type=int, default=None, help="Number of denoising inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Classifier-Free Guidance (CFG) scale.")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation.")
    parser.add_argument("--scheduler", type=str, default=None, help="Inference scheduler type (e.g. DPMSolverMultistepScheduler).")
    parser.add_argument("--output_dir", type=str, default=None, help="Destination directory for output images.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize generator
    generator = ImageGenerator(
        config=config,
        checkpoint_path=args.checkpoint,
    )

    # Prompt selection
    prompt = args.prompt or config.inference.prompt

    # Execute generation
    generator.generate_images(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        seed=args.seed,
        scheduler_type=args.scheduler,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
