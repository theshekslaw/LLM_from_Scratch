"""
BPE Tokenizer — built on top of OpenAI's tiktoken library.

tiktoken is a fast byte-pair-encoding tokenizer used by GPT-2, GPT-3,
GPT-4, etc.  This module wraps it in a simple class so it can be
imported and reused across the project.

Supported encodings (models):
  - "gpt2"       : GPT-2          (vocab ~50k)
  - "cl100k_base": GPT-4 / ChatGPT (vocab ~100k)
  - "o200k_base" : GPT-4o          (vocab ~200k)
"""

import tiktoken


class BPETokenizer:
    """A BPE tokenizer backed by tiktoken.

    Parameters
    ----------
    encoding_name : str
        The tiktoken encoding to use.  Common choices:
        ``"gpt2"``, ``"cl100k_base"``, ``"o200k_base"``.

    Usage
    -----
    >>> tok = BPETokenizer("gpt2")
    >>> ids = tok.encode("hello world")
    >>> tok.decode(ids)
    'hello world'
    """

    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding_name = encoding_name
        self.encoder: tiktoken.Encoding = tiktoken.get_encoding(encoding_name)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        return self.encoder.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        return self.encoder.decode(ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_vocab_size(self) -> int:
        """Return the vocabulary size of the loaded encoding."""
        return self.encoder.n_vocab

    def get_token_string(self, token_id: int) -> str:
        """Return the decoded string for a single token ID."""
        return self.encoder.decode([token_id])

    def tokenize(self, text: str) -> list[str]:
        """Encode then decode each token individually — useful for
        inspecting how a string gets split into sub-word pieces."""
        ids = self.encode(text)
        return [self.get_token_string(tid) for tid in ids]

    def print_tokens(self, text: str) -> None:
        """Pretty-print the token breakdown of *text*."""
        ids = self.encode(text)
        tokens = self.tokenize(text)
        print(f"Text:       {text!r}")
        print(f"Num tokens: {len(ids)}")
        print(f"Token IDs:  {ids}")
        print("Tokens:    ", tokens)

    def __repr__(self) -> str:
        return f"BPETokenizer(encoding={self.encoding_name!r}, vocab_size={self.get_vocab_size()})"
