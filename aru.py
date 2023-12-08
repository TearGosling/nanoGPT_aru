"""
Activated Rotary Units - some random dumbass idea I came up with inspired by rotary embeddings
It's like calculating rotary embeddings but as an activation function. I think.
NOTE(TG): What the fuck are you talking about? Fix this shit.
I've put it here in its own file for convenience - you know where to look.
"""

import torch
import torch.nn.functional as F
from torch import nn

# rotate half operation, taken from HF's rotate_half function in the LLaMA code:
# https://github.com/huggingface/transformers/blob/5e5fa0d88c293e6d5be2517b4f45680ba3bb5df2/src/transformers/models/llama/modeling_llama.py#L173
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class ActivatedRotaryUnit(nn.Module):
    def forward(self, x):
        # Split tensor into two ("gating itself")
        x1, x2 = torch.chunk(x, 2, dim=-1)
        # Cosine
        cos = x1 * x1.cos()
        # Sine with x2 rotated by half
        sin = rotate_half(x2) * x2.sin()
        # Concat and return
        return torch.cat([cos, sin], dim=-1)

class ReluSquared(nn.Module):
    """
    ReLU^2 function, just for comparison's sake.
    """
    def forward(self, x):
        return F.relu(x) ** 2
