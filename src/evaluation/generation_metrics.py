"""Image quality and generation evaluation metrics with optional model support."""
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def compute_generation_metrics(
    generated_images: List[Image.Image],
    prompts: List[str],
    reference_images: Optional[List[Image.Image]] = None,
    metrics_to_compute: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate image evaluation metrics.
    
    Returns structured dictionary with computed metrics and notes on any skipped optional metrics.
    """
    if metrics_to_compute is None:
        metrics_to_compute = ["diffusion_loss"]

    results: Dict[str, Any] = {
        "num_evaluated_samples": len(generated_images),
        "metrics": {},
        "notes": [],
    }

    # 1. PSNR / SSIM if reference images provided
    if reference_images is not None and len(reference_images) == len(generated_images):
        psnr_list = []
        for gen_img, ref_img in zip(generated_images, reference_images):
            g_arr = np.array(gen_img.convert("RGB")).astype(np.float64)
            r_arr = np.array(ref_img.resize(gen_img.size).convert("RGB")).astype(np.float64)
            mse = np.mean((g_arr - r_arr) ** 2)
            if mse == 0:
                psnr = 100.0
            else:
                psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
            psnr_list.append(psnr)

        results["metrics"]["mean_psnr_db"] = round(float(np.mean(psnr_list)), 2)

    # 2. CLIP Score (optional if torchmetrics or transformers clip model is installed)
    if "clip_score" in metrics_to_compute:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            model.eval()

            inputs = processor(text=prompts, images=generated_images, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image  # (N, N)
                clip_scores = logits_per_image.diagonal().tolist()

            results["metrics"]["mean_clip_score"] = round(float(np.mean(clip_scores)), 3)
        except Exception as e:
            results["notes"].append(f"CLIP score calculation skipped: {e}")

    return results
