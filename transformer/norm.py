"""
Normalization layers for transformer architectures.

- **LayerNorm** (GPT-2): normalizes to zero mean and unit variance, then
  applies learnable scale (gamma) and shift (beta).
- **RMSNorm** (LLaMA/Qwen/Mistral): simpler — scales by 1/RMS(x) then
  multiplies by gamma.  No mean subtraction, no beta.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and shift.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input (e.g. 768 for GPT-2).
    eps : float
        Small constant for numerical stability in division.
    """

    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """Root Mean Square normalization.

    Simpler than LayerNorm: no mean subtraction, no beta parameter.
    Just scale by 1/RMS(x) then multiply by learnable gamma.

    Used by LLaMA, Qwen, Mistral.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input.
    eps : float
        Small constant for numerical stability.
    """

    def __init__(self, embed_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)
