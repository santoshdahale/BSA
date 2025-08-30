"""Training harness: small, configurable training loop with checkpointing and a smoke run."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from bsa_adaptive import SimpleLM, set_seed, get_synthetic_data
from data_loader import random_batch
try:
    from hf_tokenizer import HFWrapper
    HF_TOKENIZER_AVAILABLE = True
except Exception:
    HF_TOKENIZER_AVAILABLE = False


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str, step: int):
    payload = {"model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "step": step}
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optim_state"])
    return payload.get("step", 0)


def train_one_epoch(model, optimizer, data, device, batch_size, block_size, steps=50):
    model.train()
    losses = []
    for step in range(steps):
        xb, yb = random_batch(data, block_size=block_size, batch_size=batch_size)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    set_seed(1337)
    device = torch.device(args.device)

    data, meta = get_synthetic_data()
    vocab_size = len(meta["vocab"]) if isinstance(meta, dict) and "vocab" in meta else len(set(data.tolist()))

    tokenizer = None
    if args.use_hf and HF_TOKENIZER_AVAILABLE:
        tokenizer = HFWrapper(sample_text=' '.join(meta.get('vocab', [])))

    model = SimpleLM(vocab_size=vocab_size, n_embd=64, block_size=args.block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    start = time.time()
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, optimizer, data, device, args.batch_size, args.block_size, steps=20)
        ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_epoch{epoch+1}.pt")
        save_checkpoint(model, optimizer, ckpt_path, epoch+1)
        print(f"Epoch {epoch+1} loss: {loss:.4f} saved {ckpt_path}")
    print(f"Training finished in {time.time()-start:.2f}s")


if __name__ == "__main__":
    main()
