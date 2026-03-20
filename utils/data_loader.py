"""
Sliding-window dataset and dataloader creation for next-token prediction.

The dataset creates overlapping windows of token IDs:
- input  = tokens[i : i + max_seq_len]
- target = tokens[i+1 : i + max_seq_len + 1]

The stride controls how much windows overlap (stride < max_seq_len → overlap).
"""

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Sliding-window dataset for causal language modeling.

    Parameters
    ----------
    token_ids : list[int]
        Full sequence of token IDs.
    max_seq_len : int
        Context window size.
    stride : int
        Step size between consecutive windows.
    """

    def __init__(self, token_ids: list[int], max_seq_len: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - max_seq_len, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_seq_len]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_seq_len + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloaders(
    text: str,
    tokenizer,
    max_seq_len: int,
    stride: int,
    train_ratio: float = 0.9,
    batch_size: int = 4,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Tokenize text and create train/val DataLoaders.

    Parameters
    ----------
    text : str
        Raw text corpus.
    tokenizer
        Tokenizer with an ``encode(text) -> list[int]`` method.
    max_seq_len : int
        Context window size.
    stride : int
        Step size between windows.
    train_ratio : float
        Fraction of tokens used for training (rest is validation).
    batch_size : int
        Batch size for both loaders.
    shuffle : bool
        Whether to shuffle the training set.

    Returns
    -------
    (train_loader, val_loader)
    """
    token_ids = tokenizer.encode(text)
    split = int(len(token_ids) * train_ratio)

    train_dataset = TextDataset(token_ids[:split], max_seq_len, stride)
    val_dataset = TextDataset(token_ids[split:], max_seq_len, stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader
