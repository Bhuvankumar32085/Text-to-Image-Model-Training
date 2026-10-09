"""Prompt sanitization, validation, and token inspection utilities."""
import logging
from typing import Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def validate_prompt_length(prompt: str, tokenizer: Any, max_length: int = 77) -> Tuple[bool, int]:
    """Check if tokenized prompt exceeds tokenizer's maximum sequence length."""
    if tokenizer is None:
        return True, len(prompt.split())

    tokens = tokenizer.encode(prompt)
    length = len(tokens)
    if length > max_length:
        logger.warning(
            f"Prompt length ({length} tokens) exceeds maximum sequence length ({max_length}). "
            f"Prompt will be truncated."
        )
        return False, length
    return True, length


def prepare_prompts(
    prompt: Union[str, List[str]],
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
) -> Tuple[List[str], List[str]]:
    """Format prompt and negative prompt lists matching batch size requirements."""
    if isinstance(prompt, str):
        prompt_list = [prompt.strip()] * num_images_per_prompt
    else:
        prompt_list = [p.strip() for p in prompt]

    if negative_prompt is None:
        neg_list = [""] * len(prompt_list)
    elif isinstance(negative_prompt, str):
        neg_list = [negative_prompt.strip()] * len(prompt_list)
    else:
        neg_list = [p.strip() for p in negative_prompt]

    return prompt_list, neg_list
