"""Batch collation and caption tokenization for Text-to-Image models."""
from typing import Any, Dict, List, Optional


class TextToImageCollator:
    """Collator function for PyTorch DataLoader that tokenizes captions and stacks image tensors."""

    def __init__(self, tokenizer: Any, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else getattr(tokenizer, "model_max_length", 77)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        # Collect images and captions
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        captions = [item["caption"] for item in batch]
        image_paths = [item.get("image_path", "") for item in batch]
        group_ids = [item.get("group_id", "") for item in batch]

        if self.tokenizer is not None:
            # Tokenize captions with padding and truncation
            text_inputs = self.tokenizer(
                captions,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids
            attention_mask = getattr(text_inputs, "attention_mask", None)
        else:
            input_ids = None
            attention_mask = None

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "captions": captions,
            "image_paths": image_paths,
            "group_ids": group_ids,
        }
