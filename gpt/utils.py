"""Training utilities: LR schedule, loss estimation, checkpoints, plots, sampling."""

import json
import math
import os
from dataclasses import asdict

import torch

from .config import GPTConfig
from .dataset import CharTokenizer, get_batch
from .model import GPT


def infer_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lr(step: int, max_iters: int, learning_rate: float, warmup_iters: int = 100) -> float:
    
    if step < warmup_iters:
        return learning_rate * step / warmup_iters
    
    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    #LR = 0.1LR + 0.9LR * c
    #decaying with cosine decay formula until it goes to 0 and only 10$ is left
    return learning_rate * 0.1 + coeff * learning_rate * 0.9 


@torch.no_grad()
def estimate_loss(
    model: GPT,
    splits: dict[str, torch.Tensor],
    config: GPTConfig,
    device: torch.device,
) -> dict[str, float]:
    #Average the model loss over `config.eval_iters` random batches per split.
    
    out: dict[str, float] = {}
    #eval starts
    model.eval()
    
    for split, data in splits.items():
        losses = torch.zeros(config.eval_iters)
        
        for k in range(config.eval_iters):
            X, Y = get_batch(data, config, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
            
        out[split] = losses.mean().item()
    #eval ends and back to training mode3
    model.train()
    return out


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    tokenizer: CharTokenizer,
    iteration: int,
    path: str | os.PathLike,
) -> None:
    
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "tokenizer": tokenizer.to_dict(),
        "iter": iteration,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | os.PathLike, device: torch.device | None = None
) -> tuple[GPT, GPTConfig, CharTokenizer, dict]:
    """Rebuild the model + tokenizer stored in a checkpoint directory/file path."""
    if device is None:
        device = infer_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**payload["config"])
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    model = GPT(config).to(device)
    model.load_state_dict(payload["model"])
    return model, config, tokenizer, payload


def generate_sample(
    model: GPT,
    tokenizer: CharTokenizer,
    config: GPTConfig,
    device: torch.device,
    prompt: str = "",
    max_new_tokens: int = 200,
) -> str:
    """Encode a prompt, sample `max_new_tokens` tokens and decode back to text."""
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = tokenizer.encode("\n") or [0]  # start from a newline, like nanoGPT
    index = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = model.generate(index, max_new_tokens)
    return tokenizer.decode(generated[0].tolist())


class LossHistory:
    """Train/val loss curve recorded at each evaluation point."""

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.train: list[float] = []
        self.val: list[float] = []

    def record(self, step: int, train_loss: float, val_loss: float) -> None:
        self.steps.append(step)
        self.train.append(train_loss)
        self.val.append(val_loss)

    def to_dict(self) -> dict[str, list]:
        return {"step": self.steps, "train_loss": self.train, "val_loss": self.val}

    def save(self, path: str | os.PathLike) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def plot_loss_curve(history: LossHistory, path: str | os.PathLike) -> bool:
    
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping loss curve plot")
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.steps, history.train, label="train loss", color="#1f77b4")
    ax.plot(history.steps, history.val, label="val loss", color="#d62728")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("MiniGPT training loss (TinyShakespeare)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True
