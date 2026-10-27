"""Train the MiniGPT model from the command line.

Usage:
    python train.py --data data/shakespeare.txt --max-iters 10000
"""

import argparse
import json
import time
from pathlib import Path

import torch

from gpt import (
    GPT,
    GPTConfig,
    LossHistory,
    build_tokenizer,
    encode_and_split,
    estimate_loss,
    generate_sample,
    get_batch,
    get_lr,
    plot_loss_curve,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch character-level GPT (PyTorch)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="Path to a training text file, or a folder of .txt files to combine.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory where model.pt and loss_history.json are written.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override the number of training iterations (default: config.max_iters).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. 'cuda' or 'cpu' (default: CUDA if available else CPU).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <checkpoint-dir>/model.pt (model, optimizer and loss history).",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Data first, config second: the corpus decides vocab_size, so count it
    # before building the model's configuration.
    tokenizer = build_tokenizer(args.data)
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    train_data, val_data = encode_and_split(tokenizer, config, args.data)
    print(
        f"Corpus: {args.data} | vocab {config.vocab_size} chars | "
        f"train {len(train_data):,} tokens | val {len(val_data):,} tokens"
    )

    model = GPT(config).to(device)
    print(f"Model parameters: {model.num_parameters():,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Optionally continue from the last checkpoint (model + optimizer + history).
    start_step = 0
    if args.resume:
        ckpt_path = checkpoint_dir / "model.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"Cannot resume - no checkpoint at {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("iter", 0))
        print(f"Resumed model + optimizer from {ckpt_path} (step {start_step})")

    history = LossHistory()
    if args.resume and (checkpoint_dir / "loss_history.json").exists():
        with open(checkpoint_dir / "loss_history.json", "r", encoding="utf-8") as f:
            prev = json.load(f)
        history.steps = list(prev["step"])
        history.train = list(prev["train_loss"])
        history.val = list(prev["val_loss"])
    last_eval_time = time.time()

    for step in range(start_step, config.max_iters):

        # Evaluate, print, checkpoint and record the curve at the eval interval.
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            now = time.time()
            dt = now - last_eval_time
            last_eval_time = now
            iters_since = step - history.steps[-1] if history.steps else step
            losses = estimate_loss(
                model, {"train": train_data, "val": val_data}, config, device
            )
            history.record(step, losses["train"], losses["val"])
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f} "
                f"({iters_since} iters in {dt:.1f}s)"
            )
            save_checkpoint(model, optimizer, config, tokenizer, step, checkpoint_dir / "model.pt")
            history.save(checkpoint_dir / "loss_history.json")

        # Learning rate warmup + cosine decay (Step 4).
        lr = get_lr(step, config.max_iters, config.learning_rate, config.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Standard minibatch update with gradient clipping (Step 5).
        xb, yb = get_batch(train_data, config, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

    # Final artifacts: checkpoint, history, loss curve plot and a sample.
    save_checkpoint(model, optimizer, config, tokenizer, config.max_iters, checkpoint_dir / "model.pt")
    history.save(checkpoint_dir / "loss_history.json")
    curve_path = checkpoint_dir / "loss_curve.png"
    if plot_loss_curve(history, curve_path):
        print(f"Loss curve saved to {curve_path}")

    sample = generate_sample(model, tokenizer, config, device)
    sample_path = checkpoint_dir / "sample.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample)
    print("\n--- sample generation after training ---")
    print(sample)
    print(f"\nDone. Checkpoint: {checkpoint_dir / 'model.pt'} | Sample: {sample_path}")


if __name__ == "__main__":
    train(parse_args())
