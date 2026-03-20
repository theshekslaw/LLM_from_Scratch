"""
Load pretrained GPT-2 weights from HuggingFace into our GPTModel.

Handles the mapping from HuggingFace parameter names to our parameter
names, including transposing Conv1D weights (HF uses Conv1D, we use Linear).
"""

from __future__ import annotations

import torch

from transformer.config import ModelConfig
from transformer.model import GPTModel


# Map from our parameter names to HuggingFace GPT-2 parameter names
_HF_PARAM_MAP = {
    "embedding.token_embedding.weight": "wte.weight",
    "embedding.position_embedding.weight": "wpe.weight",
    "final_norm.gamma": "ln_f.weight",
    "final_norm.beta": "ln_f.bias",
}

# Per-block mappings (prefix: "blocks.{i}." → "h.{i}.")
_HF_BLOCK_MAP = {
    "norm1.gamma": "ln_1.weight",
    "norm1.beta": "ln_1.bias",
    "norm2.gamma": "ln_2.weight",
    "norm2.beta": "ln_2.bias",
    "attn.qkv_proj.weight": "attn.c_attn.weight",
    "attn.qkv_proj.bias": "attn.c_attn.bias",
    "attn.out_proj.weight": "attn.c_proj.weight",
    "attn.out_proj.bias": "attn.c_proj.bias",
    "ffn.net.0.weight": "mlp.c_fc.weight",
    "ffn.net.0.bias": "mlp.c_fc.bias",
    "ffn.net.2.weight": "mlp.c_proj.weight",
    "ffn.net.2.bias": "mlp.c_proj.bias",
}

# HF Conv1D stores weights transposed relative to nn.Linear
_TRANSPOSE_PARAMS = {
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
}


def load_gpt2_weights(model_name: str = "gpt2") -> GPTModel:
    """Download GPT-2 weights from HuggingFace and load into our GPTModel.

    Parameters
    ----------
    model_name : str
        HuggingFace model name: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl".

    Returns
    -------
    GPTModel
        Our model with pretrained weights loaded.
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        raise ImportError(
            "Install transformers to load GPT-2 weights: pip install transformers"
        )

    configs = {
        "gpt2": ModelConfig.gpt2_124m,
        "gpt2-medium": lambda: ModelConfig(
            vocab_size=50257, embed_dim=1024, max_seq_len=1024,
            num_layers=24, num_heads=16, qkv_bias=True,
            pos_encoding="learned", norm_type="layernorm",
            activation="gelu", ffn_type="standard",
        ),
        "gpt2-large": lambda: ModelConfig(
            vocab_size=50257, embed_dim=1280, max_seq_len=1024,
            num_layers=36, num_heads=20, qkv_bias=True,
            pos_encoding="learned", norm_type="layernorm",
            activation="gelu", ffn_type="standard",
        ),
        "gpt2-xl": lambda: ModelConfig(
            vocab_size=50257, embed_dim=1600, max_seq_len=1024,
            num_layers=48, num_heads=25, qkv_bias=True,
            pos_encoding="learned", norm_type="layernorm",
            activation="gelu", ffn_type="standard",
        ),
    }

    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(configs.keys())}")

    config = configs[model_name]()
    our_model = GPTModel(config)

    # Load HF model
    print(f"Downloading {model_name} from HuggingFace...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_state = hf_model.state_dict()

    # Build mapping and load
    our_state = our_model.state_dict()
    new_state = {}

    # Global params
    for our_key, hf_key in _HF_PARAM_MAP.items():
        if our_key in our_state:
            new_state[our_key] = hf_state[hf_key]

    # Per-block params
    for i in range(config.num_layers):
        for our_suffix, hf_suffix in _HF_BLOCK_MAP.items():
            our_key = f"blocks.{i}.{our_suffix}"
            hf_key = f"h.{i}.{hf_suffix}"

            if our_key not in our_state:
                continue

            param = hf_state[hf_key]
            if hf_suffix in _TRANSPOSE_PARAMS:
                param = param.t()
            new_state[our_key] = param

    # Weight tying: lm_head shares weights with token embedding
    new_state["lm_head.weight"] = hf_state["wte.weight"]

    # Verify all params are covered
    missing = set(our_state.keys()) - set(new_state.keys())
    if missing:
        print(f"Warning: missing parameters (not loaded): {missing}")

    our_model.load_state_dict(new_state, strict=True)
    print(f"Successfully loaded {model_name} weights ({sum(p.numel() for p in our_model.parameters()):,} parameters)")
    return our_model
