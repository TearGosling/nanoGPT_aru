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
        #x1, x2 = torch.chunk(x, 2, dim=-1)
        # Cosine
        cos = x * x.cos()
        # Sine with x2 rotated by half
        sin = rotate_half(x) * x.sin()
        # Concat and return
        #return torch.cat([cos, sin], dim=-1)
        return cos * sin

class ReluSquared(nn.Module):
    """
    ReLU^2 function, just for comparison's sake.
    """
    def forward(self, x):
        return F.relu(x) ** 2

class PCubic(nn.Module):
    """
    PCubic. Let's see how this goes.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(-1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.tensor(1.0))
        self.d = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return (self.a * x ** 3) + (self.b * x ** 2) + (self.c * x) + self.d

