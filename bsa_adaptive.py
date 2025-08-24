import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnhancedBSA:
    def __init__(self, head_size, block_size, min_k=4, max_k=16, alpha=0.1, hybrid_threshold=0.5):
        self.min_k = min_k
        self.max_k = max_k
        self.alpha = alpha
        self.block_size = block_size
        self.gamma = nn.Parameter(torch.tensor(256.0))  # Learnable scaling
        self.hybrid_threshold = hybrid_threshold
        self.latent_dim = head_size // 2  # MLA-inspired latent compression

    def __call__(self, scores):
        b, h, t, _ = scores.shape
        # Positional decay (inspired by DeepSeek's efficiency)
        pos_bias = torch.zeros(1, 1, t, t, device=scores.device)
        scores = scores + pos_bias
        positions = torch.arange(t, device=scores.device).unsqueeze(0) - torch.arange(t, device=scores.device).unsqueeze(1)
        decay = 1 - self.alpha * torch.abs(positions) / self.block_size
        decay = decay.unsqueeze(0).unsqueeze(0)

        # ReLU scoring
        f = torch.relu(scores) * decay

        # Entropy-based adaptive top-K
        probs = F.softmax(f, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        max_entropy = math.log(t)
        normalized_entropy = entropy / max_entropy
        avg_normalized_entropy = normalized_entropy.mean()

        # Hybrid: Switch to MLA-like compression for high-entropy contexts
        if avg_normalized_entropy > self.hybrid_threshold:
            # MLA-inspired: Compress to latent space (simplified)
            latent = F.linear(probs, torch.randn(t, self.latent_dim, device=probs.device))
            latent = F.relu(latent)
            weights = F.linear(latent, torch.randn(self.latent_dim, t, device=probs.device))
            weights = F.softmax(weights, dim=-1)
            return weights

        # Adaptive top-K based on entropy
        k = int(self.min_k + (self.max_k - self.min_k) * avg_normalized_entropy)
        k = max(self.min_k, min(self.max_k, k))

        # Entropy-weighted quantization (7-bit, 0-127)
        min_range = 15
        max_range = 127
        adapt_max_range = int(min_range + (max_range - min_range) * avg_normalized_entropy)
        sigma = torch.std(f, dim=-1, keepdim=True)
        m = torch.max(f, dim=-1, keepdim=True)[0]
        denom = torch.max(m, sigma) + 1e-6
        topk_values, topk_indices = torch.topk(f, k, dim=-1)
        denom_exp = denom.expand(-1, -1, -1, k)
        norm_weights = torch.floor(adapt_max_range * topk_values / denom_exp).clamp(0, adapt_max_range).to(torch.int8)
        weights = torch.zeros_like(f)
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
        self.attn_func = EnhancedBSA(head_size, block_size)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.tril[:x.size(1), :x.size(1)] == 0, float('-inf'))
        attn = self.attn_func(scores.unsqueeze(1)).squeeze(1)
        out = attn @ v
        return out

# Example usage (simplified model)
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
            return logits, None
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

# Hyperparameters (from your original setup)
vocab_size = 65  # Example vocab size
n_embd = 32
block_size = 32
batch_size = 4

# Instantiate and test
model = SimpleLM()
x = torch.randint(0, vocab_size, (batch_size, block_size))
logits, loss = model(x)
print(f"Output shape: {logits.shape}")