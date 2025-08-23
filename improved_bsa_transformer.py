import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datasets import load_dataset
import numpy as np
import os

# Suppress symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load 20 Newsgroups or synthetic fallback
try:
    dataset = load_dataset("20_newsgroups", split="train")
    text = " ".join(dataset["text"][:1000])
except Exception as e:
    print(f"Failed to load 20_newsgroups: {e}. Using synthetic dataset.")
    text = """
    Natural language processing is a significant part of machine learning use cases, but it requires a lot of data and some deftly handled training.
    They’re arranged in six fields — polarity, tweet date, user, text, query, and ID. MultiDomain Sentiment Analysis Dataset: Includes a wide range of Amazon reviews.
    Dataset can be converted to binary labels based on star review, and some product categories have thousands of entries. Yelp Reviews: Restaurant rankings and reviews.
    """ * 10

# Character-level tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Prepare data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Hyperparameters
block_size = 64
batch_size = 16
n_embd = 64
n_head = 4
learning_rate = 3e-4
max_iters = 2000
eval_iters = 200

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Original ByteScale Attention
def byte_scale_attention(scores):
    f = torch.relu(scores)
    m = torch.max(f, dim=-1, keepdim=True)[0]
    epsilon = 1e-6
    weights = torch.floor(256 * f / (m + epsilon)).clamp(0, 255).to(torch.int8)
    weights = weights / 256.0
    return weights

# Improved ByteScale Attention
class ImprovedBSA:
    def __init__(self, head_size, block_size, top_k=8, alpha=0.1):
        self.top_k = top_k
        self.alpha = alpha
        self.block_size = block_size
        self.gamma = nn.Parameter(torch.tensor(256.0))

    def __call__(self, scores):
        b, h, t, _ = scores.shape
        pos_bias = torch.zeros(1, 1, t, t, device=scores.device)
        scores = scores + pos_bias
        positions = torch.arange(t, device=scores.device).unsqueeze(0) - torch.arange(t, device=scores.device).unsqueeze(1)
        decay = 1 - self.alpha * torch.abs(positions) / self.block_size
        decay = decay.unsqueeze(0).unsqueeze(0)
        f = torch.relu(scores) * decay
        sigma = torch.std(f, dim=-1, keepdim=True)
        m = torch.max(f, dim=-1, keepdim=True)[0]
        denom = torch.max(m, sigma) + 1e-6
        max_range = 127 if t <= 64 else 255
        topk_values, topk_indices = torch.topk(f, self.top_k, dim=-1)
        weights = torch.zeros_like(f)
        norm_weights = torch.floor(max_range * topk_values / denom).clamp(0, max_range).to(torch.int8)
        weights.scatter_(-1, topk_indices, norm_weights)
        weights = weights / self.gamma
        return weights

# Self-Attention Head
class Head(nn.Module):
    def __init__(self, head_size, attn_func):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = torch.tril(torch.ones(block_size, block_size))
        self.attn_func = attn_func

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.tril[:x.size(1), :x.size(1)] == 0, float('-inf'))
        if self.attn_func == 'softmax':
            attn = F.softmax(scores, dim=-1)
        elif self.attn_func == 'bsa_original':
            attn = byte_scale_attention(scores.unsqueeze(1)).squeeze(1)
        else:  # improved
            attn = self.attn_func(scores.unsqueeze(1)).squeeze(1)
        out = attn @ v
        return out

# Simple Language Model
class SimpleLM(nn.Module):
    def __init__(self, attn_type='softmax'):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        if attn_type == 'bsa_improved':
            self.head = Head(n_embd, ImprovedBSA(n_embd, block_size))
        else:
            self.head = Head(n_embd, attn_type)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.head(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

# Memory usage estimation
def estimate_memory(model, input_shape):
    total_params = sum(p.numel() for p in model.parameters())
    attn_weights_size = input_shape[0] * input_shape[1] * input_shape[1] * n_head
    if isinstance(model.head.attn_func, ImprovedBSA) and input_shape[1] <= 64:
        bytes_per_weight = 7/8
    else:
        bytes_per_weight = 1
    attn_memory = attn_weights_size * bytes_per_weight / 1024**2
    return attn_memory

# Training and evaluation
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(attn_type):
    model = SimpleLM(attn_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    input_shape = (batch_size, block_size)
    for iter in range(max_iters):
        if iter % eval_iters == 0:
            losses = estimate_loss(model)
            mem = estimate_memory(model, input_shape)
            print(f"{attn_type} step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, attn memory {mem:.4f} MB")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    X, Y = get_batch('val')
    logits, _ = model(X, Y)
    probs = F.softmax(logits.view(-1, vocab_size), dim=-1)
    perplexity = torch.exp(F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1)))
    mem = estimate_memory(model, input_shape)
    print(f"{attn_type} Final Val Perplexity: {perplexity.item():.4f}, Memory: {mem:.4f} MB")
    return model

# Run
if __name__ == "__main__":
    print("Testing Softmax (Baseline)")
    train_model('softmax')
    print("\nTesting Original BSA")
    train_model('bsa_original')
    print("\nTesting Improved BSA")
    train_model('bsa_improved')