"""CLIP Text Encoder component wrapper."""
from typing import Optional
import torch
import torch.nn as nn


class TextEncoderComponent(nn.Module):
    """Wrapper for CLIPTextModel encoding text tokens into conditioning embeddings."""

    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.text_encoder = text_encoder

    @property
    def hidden_size(self) -> int:
        if hasattr(self.text_encoder, "config") and hasattr(self.text_encoder.config, "hidden_size"):
            return self.text_encoder.config.hidden_size
        return 768

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode token IDs into text conditioning embeddings.
        
        Args:
            input_ids: (B, sequence_length) token IDs
            attention_mask: Optional attention mask (B, sequence_length)
            
        Returns:
            encoder_hidden_states: (B, sequence_length, hidden_size)
        """
        encoder_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state
