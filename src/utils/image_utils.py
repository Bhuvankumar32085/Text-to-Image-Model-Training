"""Image loading, saving, transformation, and grid composition utilities."""
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from PIL import Image


def load_image_safely(image_path: Union[str, Path]) -> Optional[Image.Image]:
    """Load an image using PIL, convert to RGB, and verify integrity.
    
    Returns None if the image is corrupt or cannot be opened.
    """
    path = Path(image_path)
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            img.verify()
        # Re-open after verify() since verify() closes the file pointer
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        return None


def save_image_safely(image: Image.Image, output_path: Union[str, Path], format: str = "PNG") -> bool:
    """Save PIL image safely, creating parent directories if necessary."""
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format=format)
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False


def make_image_grid(images: List[Image.Image], rows: int = None, cols: int = None) -> Image.Image:
    """Combine a list of PIL images into a single grid image."""
    if not images:
        raise ValueError("Cannot create grid from empty list of images.")
    
    num_images = len(images)
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    elif rows is None:
        rows = int(np.ceil(num_images / cols))
    elif cols is None:
        cols = int(np.ceil(num_images / rows))
        
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h), color=(255, 255, 255))
    
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
        # Resize image if it doesn't match the primary size
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.BICUBIC)
        r = i // cols
        c = i % cols
        grid.paste(img, box=(c * w, r * h))
        
    return grid


def tensor_to_pil(tensor, normalization: str = "neg_one_to_one") -> Image.Image:
    """Convert a PyTorch image tensor (C, H, W) or (1, C, H, W) to PIL Image.
    
    normalization:
        "neg_one_to_one": input in [-1.0, 1.0] -> [0, 255]
        "zero_to_one": input in [0.0, 1.0] -> [0, 255]
    """
    import torch
    if isinstance(tensor, torch.Tensor):
        t = tensor.detach().cpu().float()
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.dim() != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W), got shape {t.shape}")
        
        if normalization == "neg_one_to_one":
            t = (t / 2.0 + 0.5).clamp(0, 1)
        else:
            t = t.clamp(0, 1)
            
        t = t.permute(1, 2, 0).numpy()
        array = (t * 255.0).round().astype(np.uint8)
        return Image.fromarray(array)
    elif isinstance(tensor, np.ndarray):
        arr = tensor
        if normalization == "neg_one_to_one":
            arr = np.clip((arr / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def pil_to_tensor(image: Image.Image, normalization: str = "neg_one_to_one"):
    """Convert a PIL Image to a PyTorch tensor (C, H, W) normalized."""
    import torch
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1) # (C, H, W)
    if normalization == "neg_one_to_one":
        tensor = tensor * 2.0 - 1.0
    return tensor
