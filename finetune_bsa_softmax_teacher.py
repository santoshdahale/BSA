import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Assuming same tokenizer and data setup as above
# Hyperparameters
learning_rate = 3e-4
max_iters = 500
eval_iters = 50
distillation_alpha = 0.5  # Weight for CE vs. distillation loss
temperature = 2.0  # Softening for distillation

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Teacher Model (Softmax-based)
class SoftmaxHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = torch.tril(torch.ones(block_size, block_size))

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.tril[:x.size(1), :x.size(1)] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        return out

class TeacherLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.head = SoftmaxHead(n_embd)
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

# Training and Distillation
def train_teacher():
    model = TeacherLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_iters == 0:
            losses = estimate_loss(model)
            print(f"Teacher step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model

def distill_student(teacher):
    student = SimpleLM()  # Uses EnhancedBSA from above
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_iters == 0:
            losses = estimate_loss(student)
            print(f"Student step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        with torch.no_grad():
            teacher_logits, _ = teacher(xb)
        student_logits, _ = student(xb)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        dist_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
        ce_loss = F.cross_entropy(student_logits.view(-1, vocab_size), yb.view(-1))
        loss = distillation_alpha * ce_loss + (1 - distillation_alpha) * dist_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return student

@torch.no_grad()
def estimate_loss(model):
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

@torch.no_grad()
def estimate_perplexity(model, split='val'):
    data = val_data if split == 'val' else train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    X = torch.stack([data[i:i+block_size] for i in ix])
    Y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    logits, _ = model(X)
    perplexity = torch.exp(F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1)))
    return perplexity.item()

def estimate_memory(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params * 1 / 1024**2  # Approximate MB for INT8 (7-bit effective)

# Run
if __name__ == "__main__":
    # Synthetic data setup (from your original)
    text = """
    Natural language processing is a significant part of machine learning use cases, but it requires a lot of data and some deftly handled training.
    They’re arranged in six fields — polarity, tweet date, user, text, query, and ID. MultiDomain Sentiment Analysis Dataset: Includes a wide range of Amazon reviews.
    Dataset can be converted to binary labels based on star review, and some product categories have thousands of entries. Yelp Reviews: Restaurant rankings and reviews.
    """ * 10
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    teacher = train_teacher()
    teacher_mem = estimate_memory(teacher)
    teacher_perp = estimate_perplexity(teacher)
    print(f"Teacher (Softmax): Perplexity {teacher_perp:.4f}, Memory {teacher_mem:.4f} MB")

    student = distill_student(teacher)
    student_mem = estimate_memory(student)
    student_perp = estimate_perplexity(student)
    print(f"Student (BSA): Perplexity {student_perp:.4f}, Memory {student_mem:.4f} MB")

    mem_reduction = (1 - student_mem / teacher_mem) * 100
    print(f"Memory reduction: {mem_reduction:.2f}% (target ~29%)")