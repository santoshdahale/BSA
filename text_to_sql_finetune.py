import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os

# Suppress symlink warning for Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Synthetic Text-to-SQL Dataset
TEXT_TO_SQL_DATA = [
    ("What are the names of employees with salary > 50000?", "SELECT name FROM employees WHERE salary > 50000;"),
    ("List all departments with more than 10 employees.", "SELECT dept_name FROM departments WHERE employee_count > 10;"),
    ("Find the total salary of employees in the IT department.", "SELECT SUM(salary) FROM employees WHERE dept = 'IT';"),
    ("Show employees hired after 2020.", "SELECT name FROM employees WHERE hire_year > 2020;"),
    ("Get the average age of employees in each department.", "SELECT dept, AVG(age) FROM employees GROUP BY dept;"),
] * 20  # 100 pairs

# Character-level tokenizer
chars = sorted(list(set(''.join([q + a for q, a in TEXT_TO_SQL_DATA])))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Prepare data
data = [encode(q + " => " + a) for q, a in TEXT_TO_SQL_DATA]
data = torch.tensor([item for sublist in data for item in sublist], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Hyperparameters
block_size = 128
batch_size = 16
n_embd = 64
n_head = 4
learning_rate = 1e-4
max_iters = 2000
eval_iters = 200

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Refined ByteScale Attention
class RefinedBSA:
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

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = torch.tril(torch.ones(block_size, block_size))
        self.attn_func = RefinedBSA(head_size, block_size)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.tril[:x.size(1), :x.size(1)] == 0, float('-inf'))
        attn = self.attn_func(scores.unsqueeze(1)).squeeze(1)
        out = attn @ v
        return out

class SimpleLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.head = Head(n_embd)
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

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

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

def train_model():
    model = SimpleLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_iters == 0:
            losses = estimate_loss(model)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model

# Test case
def test_sql_generation(model):
    test_question = "List employees with age < 30."
    encoded = encode(test_question + " => ")
    context = torch.tensor([encoded], dtype=torch.long)
    generated = model.generate(context, max_new_tokens=50)
    print(f"Input: {test_question}")
    print(f"Generated SQL: {decode(generated[0].tolist())}")

# Run
if __name__ == "__main__":
    model = train_model()
    test_sql_generation(model)