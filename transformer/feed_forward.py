"""
Feed-Forward Networks for transformer architectures.

- **FeedForward** (GPT-2): standard two-linear FFN with GELU activation.
    FFN(x) = Linear₂(GELU(Linear₁(x)))

- **GatedFeedForward** (LLaMA/Qwen/Mistral): three-linear gated FFN (SwiGLU-style).
    GatedFFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Standard position-wise feed-forward network (GPT-2 style).

    Parameters
    ----------
    embed_dim : int
        Input and output dimensionality.
    intermediate_size : int
        Hidden layer size (typically 4 * embed_dim).
    drop_rate : float
        Dropout probability on the output.
    """

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int | None = None,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        hidden_dim = intermediate_size if intermediate_size is not None else 4 * embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFeedForward(nn.Module):
    """Gated FFN (SwiGLU-style) used by LLaMA, Qwen, Mistral.

    Architecture: output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Three linear layers instead of two.  The gate and up projections are
    multiplied element-wise (with SiLU on gate), then projected back down.

    Parameters
    ----------
    embed_dim : int
        Input and output dimensionality.
    intermediate_size : int
        Hidden layer size.
    drop_rate : float
        Dropout probability on the output.
    """

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, embed_dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)))
