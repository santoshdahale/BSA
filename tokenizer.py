"""Minimal character-level tokenizer for quick experiments."""
from typing import List, Tuple


class CharTokenizer:
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for i, c in enumerate(vocab)}

    @classmethod
    def from_text(cls, text: str):
        vocab = sorted(list(set(text)))
        return cls(vocab)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos[i] for i in ids)

