"""
Token + Positional Embedding layer.

Two modes of positional encoding:
- **"learned"** (GPT-2): adds a learned nn.Embedding for each position.
- **"rope"**: no positional embedding here — RoPE is applied inside
  the attention layers instead.
"""

import torch
import torch.nn as nn


class TokenPositionEmbedding(nn.Module):
    """Combined token + optional positional embedding layer.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    embed_dim : int
        Dimensionality of each embedding vector.
    max_seq_len : int
        Maximum context length the model supports.
    drop_rate : float
        Dropout probability applied after embedding.
    pos_encoding : str
        "learned" for absolute positional embeddings (GPT-2) or
        "rope" to skip (RoPE is applied in attention instead).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        drop_rate: float = 0.0,
        pos_encoding: str = "learned",
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = pos_encoding

        if pos_encoding == "learned":
            self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        else:
            self.position_embedding = None

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : Tensor of shape (batch_size, seq_len)

        Returns
        -------
        Tensor of shape (batch_size, seq_len, embed_dim)
        """
        tok_emb = self.token_embedding(token_ids)  # (B, T, C)

        if self.position_embedding is not None:
            positions = torch.arange(token_ids.shape[1], device=token_ids.device)
            tok_emb = tok_emb + self.position_embedding(positions)

        return self.dropout(tok_emb)
