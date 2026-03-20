"""
Rotary Position Embeddings (RoPE).

RoPE encodes position by rotating pairs of dimensions in Q and K tensors
using sinusoidal frequencies.  Because the dot-product of two rotated
vectors depends only on their *relative* distance, RoPE naturally captures
relative position — enabling better length generalization than learned
absolute embeddings.

Used by LLaMA, Qwen, Mistral, and most modern open-weight LLMs.

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
Position Embedding" (2021).
"""

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """Precompute and apply rotary embeddings to Q and K tensors.

    Parameters
    ----------
    head_dim : int
        Dimensionality of each attention head (must be even).
    max_seq_len : int
        Maximum sequence length to precompute frequencies for.
    theta : float
        Base frequency for the sinusoidal rotation (10000 default,
        500000 for LLaMA 3).
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        # Precompute frequency bands: theta_i = 1 / (theta^(2i/d))
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # (head_dim // 2,)

        # Precompute position * frequency table
        t = torch.arange(max_seq_len).float()
        angles = torch.outer(t, freqs)  # (max_seq_len, head_dim // 2)

        # Store cos/sin as buffers (not parameters — no gradient)
        self.register_buffer("cos", angles.cos(), persistent=False)  # (max_seq_len, head_dim//2)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to Q and K.

        Parameters
        ----------
        q, k : Tensor of shape (batch, num_heads, seq_len, head_dim)
        start_pos : int
            Starting position index (useful for KV-cache during generation).

        Returns
        -------
        (q_rotated, k_rotated) with the same shapes as input.
        """
        seq_len = q.shape[2]
        cos = self.cos[start_pos : start_pos + seq_len]  # (T, head_dim//2)
        sin = self.sin[start_pos : start_pos + seq_len]

        # Broadcast to (1, 1, T, head_dim//2) for batch + head dims
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rotated = self._rotate(q, cos, sin)
        k_rotated = self._rotate(k, cos, sin)
        return q_rotated, k_rotated

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate pairs of dimensions: [x0,x1] → [x0*cos - x1*sin, x0*sin + x1*cos]."""
        # Split into even and odd dimensions
        x1 = x[..., ::2]   # (B, nh, T, head_dim//2)
        x2 = x[..., 1::2]  # (B, nh, T, head_dim//2)

        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Interleave back: stack on last dim then flatten
        return torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(-2)
