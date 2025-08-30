"""Robust finetune runner: supports HF dataset/tokenizer (optional), tensorboard logging, and checkpoints.

Falls back to local synthetic data when HF libraries are not installed or when use_hf is false.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
from torch.utils.data import DataLoader, TensorDataset

from bsa_adaptive import SimpleLM, set_seed
from data_loader import get_synthetic_dataset

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


def build_blocks_from_texts(tokenizer, texts, block_size):
    ids = []
    for t in texts:
        enc = tokenizer.encode(t)
        if isinstance(enc, dict):
            enc = enc.get('input_ids', [])
        ids.extend(enc)
    n_blocks = len(ids) // block_size
    if n_blocks == 0:
        return torch.tensor([], dtype=torch.long)
    arr = torch.tensor(ids[: n_blocks * block_size], dtype=torch.long).view(n_blocks, block_size)
    return arr


def save_ckpt(model, optim, path, step):
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "step": step}, path)


def run(config_path: Optional[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config_path)
    args = parser.parse_args()

    cfg = None
    # prefer YAML config
    try:
        import yaml
        cfg_path = args.config or 'configs/finetune_config.yaml'
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        # fallback to JSON if provided
        if args.config and os.path.exists(args.config):
            with open(args.config) as f:
                cfg = json.load(f)
    if not cfg:
        raise RuntimeError('No valid config found at configs/finetune_config.yaml or --config path')

    set_seed(cfg.get('seed', 1337))
    device = torch.device(cfg.get('device', 'cpu'))

    use_hf = cfg.get('use_hf', False) and HF_AVAILABLE
    tokenizer = None
    texts = []
    vocab_size = None

    if use_hf:
        ds_name = cfg.get('hf_dataset', 'code_search_net')
        subset = cfg.get('hf_subset', 'python')
        try:
            ds = load_dataset(ds_name, subset, split='train[:1%]')
            sample = ds[0]
            # pick a field with code/text
            code_field = next(k for k, v in sample.items() if isinstance(v, str))
            texts = [ex[code_field] for ex in ds]
            tokenizer = AutoTokenizer.from_pretrained(cfg.get('tokenizer_name', 'gpt2'))
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            vocab_size = tokenizer.vocab_size
        except Exception as e:
            print('HF path failed, falling back to synthetic:', e)
            use_hf = False

    if not use_hf:
        data, chars, stoi = get_synthetic_dataset(repeat=20)
        texts = [''.join(chars) for _ in range(100)]
        class SimpleToken:
            def __init__(self, chars):
                self.chars = chars

            def encode(self, t):
                return [self.chars.index(c) for c in t if c in self.chars]

        tokenizer = SimpleToken(chars)
        vocab_size = len(chars)

    blocks = build_blocks_from_texts(tokenizer, texts, block_size=cfg['model']['block_size'])
    if blocks.numel() == 0:
        raise RuntimeError('No token blocks created')

    ds = TensorDataset(blocks)
    dl = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True)

    # ensure numeric types
    cfg['training']['lr'] = float(cfg['training']['lr'])
    cfg['training']['batch_size'] = int(cfg['training']['batch_size'])
    cfg['training']['steps_per_epoch'] = int(cfg['training']['steps_per_epoch'])
    cfg['training']['epochs'] = int(cfg['training']['epochs'])

    model = SimpleLM(vocab_size=vocab_size, n_embd=int(cfg['model']['n_embd']), block_size=int(cfg['model']['block_size'])).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'])

    log_dir = cfg['logging'].get('log_dir', 'runs')
    use_tb = cfg['logging'].get('tb', True) and TENSORBOARD_AVAILABLE
    writer = SummaryWriter(log_dir) if use_tb else None

    model.train()
    step = 0
    for epoch in range(cfg['training']['epochs']):
        total = 0.0
        count = 0
        for i, (xb,) in enumerate(dl):
            if i >= cfg['training']['steps_per_epoch']:
                break
            xb = xb.to(device)
            logits, loss = model(xb, xb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += loss.item()
            count += 1
            step += 1
            if writer and step % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), step)
        avg = total / max(1, count)
        print(f'Epoch {epoch+1} avg loss {avg:.4f}')
        ckpt_dir = Path(cfg['training'].get('checkpoint_dir', 'checkpoints'))
        ckpt_dir.mkdir(exist_ok=True)
        save_ckpt(model, optim, ckpt_dir / f'ckpt_epoch{epoch+1}.pt', step)

    if writer:
        writer.close()


if __name__ == '__main__':
    run()
