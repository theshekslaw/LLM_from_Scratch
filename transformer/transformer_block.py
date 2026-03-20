"""
Transformer Block — one layer of a configurable transformer.

Each block applies two sub-layers with residual connections and
pre-norm style:

    x = x + Attention(Norm(x))
    x = x + FFN(Norm(x))

The specific norm, attention, and FFN variants are selected based on
the ModelConfig (LayerNorm vs RMSNorm, MHA vs GQA, standard vs gated FFN, etc.).
"""

from __future__ import annotations

import torch.nn as nn

from .attention import MultiHeadAttention
from .config import ModelConfig
from .feed_forward import FeedForward, GatedFeedForward
from .moe import MoEFeedForward
from .norm import LayerNorm, RMSNorm
from .rope import RotaryPositionEmbedding


class TransformerBlock(nn.Module):
    """Single transformer block driven by ModelConfig.

    Parameters
    ----------
    config : ModelConfig
        Full model configuration.
    rope : RotaryPositionEmbedding or None
        Shared RoPE module (created once in GPTModel, passed to every block).
    """

    def __init__(self, config: ModelConfig, rope: RotaryPositionEmbedding | None = None):
        super().__init__()

        # ── Normalization ──
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm
        self.norm1 = norm_cls(config.embed_dim)
        self.norm2 = norm_cls(config.embed_dim)

        # ── Attention ──
        self.attn = MultiHeadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.effective_num_kv_heads,
            qkv_bias=config.qkv_bias,
            rope=rope,
            drop_rate=config.drop_rate,
        )

        # ── Feed-forward ──
        intermediate = config.effective_intermediate_size
        if config.use_moe:
            self.ffn = MoEFeedForward(
                embed_dim=config.embed_dim,
                intermediate_size=intermediate,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                drop_rate=config.drop_rate,
            )
        elif config.ffn_type == "gated":
            self.ffn = GatedFeedForward(
                embed_dim=config.embed_dim,
                intermediate_size=intermediate,
                drop_rate=config.drop_rate,
            )
        else:
            self.ffn = FeedForward(
                embed_dim=config.embed_dim,
                intermediate_size=intermediate,
                drop_rate=config.drop_rate,
            )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
