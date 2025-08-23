import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datasets import load_dataset
import os

# Suppress symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load 20 Newsgroups or use synthetic fallback
try:
    dataset = load_dataset("20_newsgroups", split="train")
    text = " ".join(dataset["text"][:1000])  # First 1000 documents
except Exception as e:
    print(f"Failed to load 20_newsgroups: {e}. Using synthetic dataset.")
    text = """
    Great Barrier Reef: The world's largest coral reef system, located off Queensland, Australia, spanning over 2,300 km with 2,900 reefs and vibrant marine life.
    Grand Canyon: A massive geological formation in Arizona, USA, carved by the Colorado River over 6 million years, up to 1.8 km deep and 446 km long.
    Aurora Borealis: A spectacular light display in the Arctic, caused by solar particles hitting atmospheric gases, best seen in Norway, Iceland, or Canada.
    Mount Everest: The highest peak on Earth at 8,848 m, in the Himalayas, Nepal/China border, a challenging climb with extreme weather conditions.
    Amazon Rainforest: The largest rainforest, spanning 5.5 million km² across Brazil and eight other countries, home to 400 billion trees and 2.5 million insect species.
    """ * 20  # 100 entries

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
block_size = 128
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
def test_wonder_generation(model):
    test_input = "Victoria Falls:"
    encoded = encode(test_input)
    context = torch.tensor([encoded], dtype=torch.long)
    generated = model.generate(context, max_new_tokens=100)
    print(f"Input: {test_input}")
    print(f"Generated Description: {decode(generated[0].tolist())}")

# Run
if __name__ == "__main__":
    model = train_model()
    test_wonder_generation(model)