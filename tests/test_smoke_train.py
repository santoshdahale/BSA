import os
import sys
import torch

# ensure project root is on sys.path so tests can import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bsa_adaptive import SimpleLM, set_seed, get_synthetic_data
from train import save_checkpoint, load_checkpoint


def test_one_step_train(tmp_path):
    set_seed(0)
    data, meta = get_synthetic_data()
    vocab_size = len(meta["vocab"]) if isinstance(meta, dict) and "vocab" in meta else 32
    device = torch.device("cpu")
    model = SimpleLM(vocab_size=vocab_size, n_embd=32, block_size=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    xb = torch.randint(0, vocab_size, (2, 16), device=device)
    yb = xb.clone()
    model.train()
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, str(ckpt), step=1)
    step = load_checkpoint(model, optimizer, str(ckpt))
    assert step == 1
    # forward after load
    model.eval()
    with torch.no_grad():
        logits, _ = model(xb)
    assert logits.shape[0] == 2
