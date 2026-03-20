"""
Mixture of Experts (MoE) feed-forward layer.

Instead of a single FFN, MoE uses multiple "expert" FFNs and a learned
router that selects the top-k experts for each token.  This allows
scaling model capacity without proportionally increasing compute —
each token only activates a fraction of the total parameters.

Used by Mixtral, Qwen-MoE, and other sparse models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Learned routing network that assigns tokens to experts.

    Parameters
    ----------
    embed_dim : int
        Input dimensionality.
    num_experts : int
        Total number of expert FFNs.
    top_k : int
        Number of experts activated per token.
    """

    def __init__(self, embed_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (batch_size * seq_len, embed_dim)

        Returns
        -------
        weights : Tensor of shape (batch_size * seq_len, top_k)
            Softmax-normalized weights for selected experts.
        indices : Tensor of shape (batch_size * seq_len, top_k)
            Indices of the selected experts.
        """
        logits = self.gate(x)  # (N, num_experts)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)
        return weights, indices


class MoEFeedForward(nn.Module):
    """Mixture-of-Experts feed-forward layer (drop-in FFN replacement).

    Parameters
    ----------
    embed_dim : int
        Input and output dimensionality.
    intermediate_size : int
        Hidden size for each expert FFN.
    num_experts : int
        Total number of expert FFNs.
    top_k : int
        Number of experts activated per token.
    drop_rate : float
        Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.router = Router(embed_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            _ExpertFFN(embed_dim, intermediate_size, drop_rate)
            for _ in range(num_experts)
        ])

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
        x_flat = x.view(-1, C)  # (N, C)

        weights, indices = self.router(x_flat)  # (N, top_k), (N, top_k)

        # Compute weighted sum of expert outputs
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Mask of tokens routed to this expert, across all top-k slots
            mask = (indices == i).any(dim=-1)  # (N,)
            if not mask.any():
                continue

            expert_input = x_flat[mask]  # (n_tokens, C)
            expert_output = expert(expert_input)  # (n_tokens, C)

            # Gather weights for this expert across top-k slots
            expert_weights = torch.where(indices[mask] == i, weights[mask], torch.zeros_like(weights[mask]))
            expert_weight = expert_weights.sum(dim=-1, keepdim=True)  # (n_tokens, 1)

            output[mask] += expert_weight * expert_output

        return output.view(B, T, C)


class _ExpertFFN(nn.Module):
    """Single expert: gated FFN (SwiGLU-style)."""

    def __init__(self, embed_dim: int, intermediate_size: int, drop_rate: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, embed_dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
