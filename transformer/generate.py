"""
Autoregressive text generation.

Given a prompt (as token IDs), generates new tokens one at a time by
sampling from the model's next-token distribution.
"""

import torch


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate token IDs autoregressively.

    Parameters
    ----------
    model : GPTModel
        The language model (in eval mode).
    prompt_ids : Tensor of shape (1, prompt_len)
        Token IDs for the prompt.
    max_new_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature (1.0 = neutral, <1 = sharper, >1 = flatter).
        Use 0 for greedy (argmax) decoding.
    top_k : int or None
        If set, only sample from the top-k most probable tokens.
    eos_token_id : int or None
        If set, stop generation when this token is produced.

    Returns
    -------
    Tensor of shape (1, prompt_len + generated_len)
        The full sequence including the prompt.
    """
    model.eval()
    ids = prompt_ids  # (1, T)
    max_seq_len = model.config.max_seq_len

    for _ in range(max_new_tokens):
        # Crop to max context length
        context = ids if ids.shape[1] <= max_seq_len else ids[:, -max_seq_len:]

        logits = model(context)                  # (1, T, vocab_size)
        next_logits = logits[:, -1, :]           # (1, vocab_size)

        if temperature == 0:
            # Greedy
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature

            if top_k is not None:
                # Zero out everything below the top-k threshold
                top_vals, _ = next_logits.topk(top_k, dim=-1)
                min_val = top_vals[:, -1:]
                next_logits = next_logits.where(
                    next_logits >= min_val, torch.full_like(next_logits, float("-inf"))
                )

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=1)

        if eos_token_id is not None and next_id.item() == eos_token_id:
            break

    return ids
