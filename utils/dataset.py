"""
Download the "The Verdict" training text from the LLMs-from-scratch repo.
"""

import os
import urllib.request
from pathlib import Path

_DEFAULT_PATH = Path("data/the-verdict.txt")
_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"


def download_verdict(dest: Path = _DEFAULT_PATH, force: bool = False) -> str:
    """Download the training text and return its contents.

    Parameters
    ----------
    dest : Path
        Where to save the file.
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    str
        The full text content.
    """
    if dest.exists() and not force:
        return dest.read_text(encoding="utf-8")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_URL) as response:
        text = response.read().decode("utf-8")
    dest.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    text = download_verdict()
    print(f"Downloaded {len(text)} characters.")
    print(text[:500])
