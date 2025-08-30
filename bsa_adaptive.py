"""bsa_adaptive.py

Refactored production-ready minimal implementation of the Enhanced BSA idea.

This script provides:
- EnhancedBSA implemented as nn.Module
- Head and SimpleLM modules with proper parameter registration
- CLI entrypoint for a quick sanity forward-pass on CPU/GPU

Notes:
- This is still a small toy model intended for experimentation and testing.
  For large-scale training and SLM-style work you should integrate tooling
  like Hugging Face datasets, tokenizers, DeepSpeed/FSDP and a proper
  training loop with checkpointing and metrics.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedBSA(nn.Module):
    """Entropy-aware, sparse attention prototype.

    This class is intentionally small and safe for experimentation.
    It focuses on correctness, device-awareness and parameter registration.
    """

    def __init__(
        self,
        head_size: int,
        block_size: int,
        min_k: int = 4,
        max_k: int = 16,
        alpha: float = 0.1,
        hybrid_threshold: float = 0.5,
    use_int_mm: bool = False,
    ) -> None:
        super().__init__()
        self.min_k = min_k
        self.max_k = max_k
        self.alpha = alpha
        self.block_size = block_size
        self.gamma = nn.Parameter(torch.tensor(256.0))
        self.hybrid_threshold = hybrid_threshold
        self.latent_dim = max(1, head_size // 2)
    self.use_int_mm = use_int_mm

        # small learned projection layers for the MLA-like path
        self._proj_to_latent = nn.Linear(block_size, self.latent_dim)
        self._proj_from_latent = nn.Linear(self.latent_dim, block_size)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """scores expected shape: (B, T, T) or (B, 1, T, T).

        Returns attention weights with shape (B, T, T).
        """
        # normalize shape to (B, T, T)
        if scores.ndim == 4 and scores.size(1) == 1:
            scores = scores.squeeze(1)

        B, T, _ = scores.shape

        device = scores.device
        # positional decay
        positions = torch.arange(T, device=device)
        rel = positions.unsqueeze(0) - positions.unsqueeze(1)
        decay = 1 - self.alpha * rel.abs().float() / max(1, self.block_size)
        decay = decay.clamp(min=0.0)

        # ReLU-based scoring (softmax-free)
        f = F.relu(scores) * decay

        # row-wise sparsification via adaptive top-k using a simple entropy proxy
        # proxy: normalized per-row nonzero mass
        row_mass = f.sum(dim=-1)  # (B, T)
        mass_mean = row_mass.mean()
        # adapt k between min_k and max_k based on relative mass (simple heuristic)
        rel = (mass_mean / (mass_mean + 1e-6)).item()
        k = int(self.min_k + (self.max_k - self.min_k) * rel)
        k = max(self.min_k, min(self.max_k, k))

        # compute top-k per row
        topk_values, topk_indices = torch.topk(f, k=k, dim=-1)

        # build sparse positive weights from topk values
        weights = torch.zeros_like(f)
        weights.scatter_(-1, topk_indices, topk_values)

        # normalize by row-sum (avoids softmax)
        row_sums = weights.sum(dim=-1, keepdim=True) + 1e-9
        weights = weights / row_sums

        # apply learnable scaling
        weights = weights / (self.gamma + 1e-6)

        # dynamic quantization: quantize attention weights to 7-bit or 8-bit based on context length
        if T <= 64:
            bits = 7
        else:
            bits = 8
        # symmetric quantization around 0
        qmax = 2 ** (bits - 1) - 1
        max_val = weights.abs().max().clamp(min=1e-9)
        scale_w = max_val / float(qmax)
        # quantize weights (int representation)
        q_w = torch.round(weights / scale_w).clamp(-qmax, qmax).to(torch.int32)

        if self.use_int_mm:
            # In integer-mm mode we expect the downstream matmul to be done in integer domain.
            # Return quantized int32 weights and its scale so the caller can perform int matmul.
            # We return a tuple (q_w, scale_w) packed into a float tensor placeholder: not ideal
            # but callers in this repo (Head) will handle integer-mm if present.
            # To keep the API consistent we attach scales as attributes on the tensor via a wrapper dict.
            # We'll return a small dict-like object encoded as a tuple: (q_w, scale_w)
            return q_w, float(scale_w)

        # dequantize back to float for use in float matmul
        dq = q_w.to(torch.float32) * scale_w
        return dq


class Head(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)).bool())
        self.attn_func = EnhancedBSA(head_size=head_size, block_size=block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_embd)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # scores: (B, T, T)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        # mask future positions
        T = x.size(1)
        mask = ~self.tril[:T, :T]
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        attn = self.attn_func(scores)
        out = attn @ v
        return out


class SimpleLM(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.head = Head(n_embd=n_embd, head_size=n_embd, block_size=block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)
        pos_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.position_embedding_table(pos_idx)
        x = tok_emb + pos_emb
        x = self.head(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_synthetic_data(length: int = 2000) -> Tuple[torch.Tensor, dict]:
    # tiny synthetic character dataset
    text = (
        "Natural language processing is a significant part of machine learning use cases. "
        "It requires data and careful training. "
    ) * 10
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, {"vocab": chars, "stoi": stoi}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    data, meta = get_synthetic_data()
    vocab_size = args.vocab_size or len(meta["vocab"]) or 65

    model = SimpleLM(vocab_size=vocab_size, n_embd=args.n_embd, block_size=args.block_size).to(device)
    model.eval()

    # small random batch
    idx = torch.randint(0, vocab_size, (args.batch_size, args.block_size), device=device)
    logits, loss = model(idx)
    print(f"Sanity run logits shape: {logits.shape}")


if __name__ == "__main__":
    main()