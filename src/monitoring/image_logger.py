"""Image saving, comparison grid generation, and visual preview tracking."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from src.utils.image_utils import make_image_grid, save_image_safely

logger = logging.getLogger(__name__)


class ImageLogger:
    """Manages saving sample predictions, preview grids, and image metadata."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_previews(
        self,
        images: List[Image.Image],
        prompts: List[str],
        step: int,
        epoch: int,
        subfolder: str = "generated_previews",
        save_grid: bool = True,
    ) -> List[str]:
        """Save a batch of generated preview images with JSON metadata and optional combined grid."""
        save_dir = self.output_dir / subfolder / f"step_{step:06d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        metadata_list = []

        for idx, (img, prompt) in enumerate(zip(images, prompts)):
            filename = f"preview_e{epoch:02d}_s{step:06d}_idx{idx:02d}.png"
            filepath = save_dir / filename
            save_image_safely(img, filepath)
            saved_paths.append(str(filepath))

            metadata_list.append({
                "step": step,
                "epoch": epoch,
                "index": idx,
                "prompt": prompt,
                "file_path": str(filepath),
            })

        # Save metadata JSON
        with open(save_dir / "previews_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2)

        # Save combined grid
        if save_grid and len(images) > 1:
            grid = make_image_grid(images)
            grid_path = save_dir / "previews_grid.png"
            save_image_safely(grid, grid_path)
            saved_paths.append(str(grid_path))

        logger.info(f"Saved {len(images)} preview image(s) to {save_dir}")
        return saved_paths
