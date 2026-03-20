"""
Multi-Head Causal Self-Attention with Grouped Query Attention (GQA) and RoPE.

Supports three attention patterns:
- **MHA** (Multi-Head Attention): num_kv_heads == num_heads (GPT-2)
- **GQA** (Grouped Query Attention): num_kv_heads < num_heads (LLaMA 3, Mistral)
- **MQA** (Multi-Query Attention): num_kv_heads == 1

When RoPE is provided, rotary position embeddings are applied to Q and K
instead of relying on learned absolute position embeddings.
"""

import torch
import torch.nn as nn

from .rope import RotaryPositionEmbedding


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with optional GQA and RoPE.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int
        Number of query attention heads.
    num_kv_heads : int or None
        Number of key/value heads. None or equal to num_heads = standard MHA.
        Less than num_heads = GQA (K,V heads are repeated to match Q).
    qkv_bias : bool
        Whether to use bias in Q, K, V projections.
    rope : RotaryPositionEmbedding or None
        If provided, apply rotary position embeddings to Q and K.
    drop_rate : float
        Dropout probability on attention weights and output.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        rope: RotaryPositionEmbedding | None = None,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = embed_dim // num_heads
        self.rope = rope

        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_kv_groups = num_heads // self.num_kv_heads  # how many Q heads per KV head

        if self.num_kv_heads == self.num_heads:
            # Standard MHA: single combined QKV projection
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
            self.q_proj = None
            self.kv_proj = None
        else:
            # GQA: separate Q and KV projections
            self.qkv_proj = None
            self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=qkv_bias)
            self.kv_proj = nn.Linear(embed_dim, 2 * self.num_kv_heads * self.head_dim, bias=qkv_bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(drop_rate)
        self.out_dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch_size, seq_len, embed_dim)

        Returns
        -------
        Tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, T, C = x.shape

        # ── Project to Q, K, V ──
        if self.qkv_proj is not None:
            # Standard MHA path
            qkv = self.qkv_proj(x)  # (B, T, 3*C)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            # GQA path: separate projections
            q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            kv = self.kv_proj(x)
            k, v = kv.chunk(2, dim=-1)
            k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ── Apply RoPE if available ──
        if self.rope is not None:
            q, k = self.rope(q, k)

        # ── Expand KV heads for GQA ──
        if self.num_kv_groups > 1:
            # (B, num_kv_heads, T, hd) → (B, num_heads, T, hd)
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # ── Scaled dot-product attention ──
        scale = self.head_dim ** 0.5
        attn_scores = (q @ k.transpose(-2, -1)) / scale  # (B, nh, T, T)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted combination
        context = attn_weights @ v  # (B, nh, T, hd)
        context = context.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_dropout(self.out_proj(context))
