"""
Full GPT Model — stacking all components into an autoregressive LM.

Accepts a ModelConfig to support multiple architectures:
GPT-2, LLaMA 3, Qwen3, Mistral, etc.

Architecture:
    Token IDs
      → Token Embedding (+ Positional Embedding if learned)
      → N × TransformerBlock (Attention + FFN with residuals)
      → Final Norm
      → Linear head (project to vocab_size for next-token prediction)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from embedder import TokenPositionEmbedding

from .config import ModelConfig
from .norm import LayerNorm, RMSNorm
from .rope import RotaryPositionEmbedding
from .transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """Configurable autoregressive language model.

    Parameters
    ----------
    config : ModelConfig
        Full model configuration specifying architecture variant.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── Embedding ──
        self.embedding = TokenPositionEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            max_seq_len=config.max_seq_len,
            drop_rate=config.drop_rate,
            pos_encoding=config.pos_encoding,
        )

        # ── RoPE (shared across all layers) ──
        if config.pos_encoding == "rope":
            self.rope = RotaryPositionEmbedding(
                head_dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
        else:
            self.rope = None

        # ── Transformer blocks ──
        self.blocks = nn.Sequential(
            *[TransformerBlock(config, rope=self.rope) for _ in range(config.num_layers)]
        )

        # ── Final normalization ──
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else LayerNorm
        self.final_norm = norm_cls(config.embed_dim)

        # ── Output head ──
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : Tensor of shape (batch_size, seq_len)

        Returns
        -------
        Tensor of shape (batch_size, seq_len, vocab_size)
            Logits for next-token prediction at each position.
        """
        x = self.embedding(token_ids)   # (B, T, C)
        x = self.blocks(x)              # (B, T, C)
        x = self.final_norm(x)          # (B, T, C)
        logits = self.lm_head(x)        # (B, T, vocab_size)
        return logits
