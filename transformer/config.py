"""
Model configuration dataclass.

A single config that can represent any supported architecture:
GPT-2, LLaMA 3, Qwen3, Mistral, etc.  Architecture differences
(positional encoding, normalization, FFN style, attention grouping)
are all controlled by fields in this dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Core dimensions
    vocab_size: int = 50257
    embed_dim: int = 768
    max_seq_len: int = 1024
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int | None = None  # None = MHA (same as num_heads). < num_heads = GQA
    drop_rate: float = 0.0
    qkv_bias: bool = False

    # Architecture switches
    pos_encoding: str = "learned"       # "learned" (GPT-2) or "rope" (LLaMA/Qwen/Mistral)
    rope_theta: float = 10_000.0        # RoPE base frequency
    norm_type: str = "layernorm"        # "layernorm" or "rmsnorm"
    activation: str = "gelu"            # "gelu" or "silu"
    ffn_type: str = "standard"          # "standard" (2 linears) or "gated" (3 linears, SwiGLU)
    intermediate_size: int | None = None  # FFN hidden dim. None = 4 * embed_dim

    # MoE
    use_moe: bool = False
    num_experts: int = 8
    top_k_experts: int = 2

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def effective_num_kv_heads(self) -> int:
        return self.num_kv_heads if self.num_kv_heads is not None else self.num_heads

    @property
    def effective_intermediate_size(self) -> int:
        return self.intermediate_size if self.intermediate_size is not None else 4 * self.embed_dim

    # ── Presets ──────────────────────────────────────────────

    @classmethod
    def gpt2_124m(cls) -> ModelConfig:
        """GPT-2 124M: learned pos, LayerNorm, GELU, standard FFN, MHA."""
        return cls(
            vocab_size=50257,
            embed_dim=768,
            max_seq_len=1024,
            num_layers=12,
            num_heads=12,
            qkv_bias=True,
            pos_encoding="learned",
            norm_type="layernorm",
            activation="gelu",
            ffn_type="standard",
        )

    @classmethod
    def llama3_8b(cls) -> ModelConfig:
        """LLaMA 3 8B: RoPE(500k), RMSNorm, SiLU, gated FFN, GQA(8 KV)."""
        return cls(
            vocab_size=128256,
            embed_dim=4096,
            max_seq_len=8192,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            pos_encoding="rope",
            rope_theta=500_000.0,
            norm_type="rmsnorm",
            activation="silu",
            ffn_type="gated",
            intermediate_size=14336,
        )

    @classmethod
    def qwen3_8b(cls) -> ModelConfig:
        """Qwen3 8B: RoPE, RMSNorm, SiLU, gated FFN, GQA."""
        return cls(
            vocab_size=151936,
            embed_dim=4096,
            max_seq_len=8192,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            pos_encoding="rope",
            rope_theta=1_000_000.0,
            norm_type="rmsnorm",
            activation="silu",
            ffn_type="gated",
            intermediate_size=12288,
        )

    @classmethod
    def mistral_7b(cls) -> ModelConfig:
        """Mistral 7B: RoPE, RMSNorm, SiLU, gated FFN, GQA."""
        return cls(
            vocab_size=32000,
            embed_dim=4096,
            max_seq_len=8192,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            pos_encoding="rope",
            rope_theta=10_000.0,
            norm_type="rmsnorm",
            activation="silu",
            ffn_type="gated",
            intermediate_size=14336,
        )
