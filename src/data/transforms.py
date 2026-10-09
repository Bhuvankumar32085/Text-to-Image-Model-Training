"""Image transformation and preprocessing pipelines for Text-to-Image models."""
from typing import Tuple, Union
import numpy as np
from PIL import Image


class ImageTransformPipeline:
    """Preprocesses PIL images into normalized PyTorch tensors suitable for Stable Diffusion."""

    def __init__(
        self,
        target_resolution: Tuple[int, int] = (512, 512),
        aspect_ratio_policy: str = "center_crop",
        normalization: str = "neg_one_to_one",
    ):
        self.height, self.width = target_resolution
        self.aspect_ratio_policy = aspect_ratio_policy
        self.normalization = normalization

    def __call__(self, image: Image.Image):
        """Transform a PIL Image into a normalized tensor (C, H, W)."""
        import torch

        # Ensure 3-channel RGB
        img = image.convert("RGB")
        w, h = img.size

        # Apply aspect ratio policy
        if self.aspect_ratio_policy == "resize":
            img = img.resize((self.width, self.height), Image.Resampling.BICUBIC)
        elif self.aspect_ratio_policy == "center_crop":
            # Scale so smallest dimension matches target resolution, then center crop
            scale = max(self.width / w, self.height / h)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

            left = (new_w - self.width) // 2
            top = (new_h - self.height) // 2
            img = img.crop((left, top, left + self.width, top + self.height))
        elif self.aspect_ratio_policy == "random_crop":
            scale = max(self.width / w, self.height / h)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

            if new_w > self.width:
                left = np.random.randint(0, new_w - self.width + 1)
            else:
                left = 0
            if new_h > self.height:
                top = np.random.randint(0, new_h - self.height + 1)
            else:
                top = 0
            img = img.crop((left, top, left + self.width, top + self.height))
        else:
            raise ValueError(f"Unsupported aspect_ratio_policy: {self.aspect_ratio_policy}")

        # Convert to numpy array [0, 1]
        arr = np.array(img).astype(np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        # Apply normalization
        if self.normalization == "neg_one_to_one":
            tensor = tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        elif self.normalization == "zero_to_one":
            pass  # already in [0, 1]
        else:
            raise ValueError(f"Unsupported image_normalization: {self.normalization}")

        return tensor
