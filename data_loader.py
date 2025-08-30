from typing import Tuple
import torch


def get_synthetic_dataset(repeat: int = 10):
    text = (
        "Natural language processing is a significant part of machine learning use cases. "
        "It requires data and careful training. "
    ) * repeat
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, chars, stoi


def random_batch(data: torch.Tensor, block_size: int, batch_size: int):
    ix = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
