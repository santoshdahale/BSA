"""Fine-tune SimpleLM on a small CodeSearchNet subset using GPT-2 tokenizer.

This script attempts to use Hugging Face `datasets` and `transformers` if installed.
If those libraries are not available it falls back to the repo's synthetic dataset.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset

from bsa_adaptive import SimpleLM, set_seed
from data_loader import get_synthetic_dataset, random_batch


def build_token_blocks(tokenizer, texts: List[str], block_size: int):
    ids = []
    for t in texts:
        enc = tokenizer.encode(t)
        if isinstance(enc, dict):
            # huggingface fast tokenizer may return dict including 'input_ids'
            enc = enc.get('input_ids', [])
        ids.extend(enc)
    # chunk into blocks
    total_len = len(ids)
    n_blocks = total_len // block_size
    if n_blocks == 0:
        return torch.tensor([], dtype=torch.long)
    ids = ids[: n_blocks * block_size]
    arr = torch.tensor(ids, dtype=torch.long).view(n_blocks, block_size)
    return arr


def detect_code_field(example: dict) -> str:
    # heuristics to pick a code-like string field from HF dataset example
    for k, v in example.items():
        if isinstance(v, str) and len(v) > 20:
            low = v.lower()
            if 'def ' in low or 'class ' in low or '\n' in v or '(' in v:
                return k
    # fallback to first string field
    for k, v in example.items():
        if isinstance(v, str):
            return k
    raise RuntimeError('No suitable text field found in example')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--use-hf", action="store_true", help="Use Hugging Face datasets and gpt2 tokenizer when available")
    args = parser.parse_args()

    set_seed(1337)
    device = torch.device(args.device)

    tokenizer = None
    texts = []
    vocab_size = None

    if args.use_hf:
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer

            print("Loading CodeSearchNet (python) small subset via datasets...")
            ds = load_dataset("code_search_net", "python", split=f"train[:{min(100, args.max_examples)}]")
            sample = ds[0]
            code_field = detect_code_field(sample)
            texts = [ex[code_field] for ex in ds]
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            vocab_size = tokenizer.vocab_size
            print(f"Loaded {len(texts)} examples, tokenizer vocab {vocab_size}, code field '{code_field}'")
        except Exception as e:
            print("HF pipeline unavailable or failed, falling back to synthetic dataset:", e)

    if tokenizer is None:
        # fallback to synthetic char data
        data, chars, stoi = get_synthetic_dataset(repeat=20)
        # convert numeric data to pseudo text strings for tokenizer.encode compatibility
        sample_text = ''.join(chars)

        class SimpleToken:
            def __init__(self, chars):
                self.chars = chars

            def encode(self, t):
                return [self.chars.index(c) for c in t if c in self.chars]

        tokenizer = SimpleToken(chars)
        texts = ["".join(chars) for _ in range(100)]
        vocab_size = len(chars)

    # Build blocks
    token_blocks = build_token_blocks(tokenizer, texts, args.block_size)
    if token_blocks.numel() == 0:
        print("Not enough tokens produced, exiting")
        return

    # prepare dataset and dataloader
    dataset = TensorDataset(token_blocks)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SimpleLM(vocab_size=vocab_size, n_embd=128, block_size=args.block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # small training loop
    model.train()
    start = time.time()
    for epoch in range(args.epochs):
        total_loss = 0.0
        steps = 0
        for i, (xb,) in enumerate(dl):
            if steps >= args.steps_per_epoch:
                break
            xb = xb.to(device)
            # targets are next token prediction shift by 1
            inputs = xb[:, :args.block_size]
            targets = xb[:, :args.block_size].clone()
            logits, loss = model(inputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        avg = total_loss / max(1, steps)
        print(f"Epoch {epoch+1} avg loss {avg:.4f}")
    print(f"Finished in {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
