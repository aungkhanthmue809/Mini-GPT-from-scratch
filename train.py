"""Train the MiniGPT model from the command line.

Usage:
    python train.py --data data/shakespeare.txt --max-iters 10000
"""

import argparse
import json
from pathlib import Path

import torch

from gpt import (
    GPT,
    GPTConfig,
    LossHistory,
    build_tokenizer,
    encode_and_split,
    estimate_loss,
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
    #step 0 set device and intialise checkpoint dir
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    #step 1 set up tokeniser and config using dataset
    tokenizer = build_tokenizer(args.data) 
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    #just configuration for dynamic max iter as cli
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    
    #step 2 use tokeniser to on the dataset to encode and split it
    train_data, val_data = encode_and_split(tokenizer, config, args.data)
    print(
        f"Corpus: {args.data} | vocab {config.vocab_size} chars | "
        f"train {len(train_data):,} tokens | val {len(val_data):,} tokens"
    )

    #step 3 intialise model and its optimizer using model_config
    model = GPT(config).to(device)
    print(f"Model parameters: {model.num_parameters():,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    #step 3.1 reload data from pt file in case this is a resume
    start_step = 0
    if args.resume:
        ckpt_path = checkpoint_dir / "model.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"Cannot resume - no checkpoint at {ckpt_path}")
        
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        #soft access to allow absent step
        start_step = int(payload.get("iter", 0))
        
        print(f"Resumed model + optimizer from {ckpt_path} (step {start_step})")

    #step 3.2 check if loss history.json exists and load the past values
    history = LossHistory()
    if args.resume and (checkpoint_dir / "loss_history.json").exists():
        with open(checkpoint_dir / "loss_history.json", "r", encoding="utf-8") as f:
            prev = json.load(f)
        history.steps = list(prev["step"])
        history.train = list(prev["train_loss"])
        history.val = list(prev["val_loss"])

    # step 4 the main Training LOOP
    for step in range(start_step, config.max_iters):

        # Evaluate, print, checkpoint and record the curve at the eval interval.
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            
            losses = estimate_loss(
                model, {"train": train_data, "val": val_data}, config, device
            )

            history.record(step, losses["train"], losses["val"])
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            history.save(checkpoint_dir / "loss_history.json")
            
            
            save_checkpoint(model, optimizer, config, tokenizer, step, checkpoint_dir / "model.pt")
            

        # Learning rate warmup + cosine decay (Step 4).
        lr = get_lr(step, config.max_iters, config.learning_rate, config.warmup_iters)

        #single param group (AdamW(model.parameters())) -> index it directly
        #param_groups is a dict list; lr lives in group["lr"] and is read fresh by step()
        optimizer.param_groups[0]["lr"] = lr

        # Standard minibatch update with gradient clipping (Step 5).
        xb, yb = get_batch(train_data, config, device)
        
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

    # step 5 save checkpoint,loss history json and a matplot png
    save_checkpoint(model, optimizer, config, tokenizer, config.max_iters, checkpoint_dir / "model.pt")
    history.save(checkpoint_dir / "loss_history.json")
    
    curve_path = checkpoint_dir / "loss_curve.png"
    if plot_loss_curve(history, curve_path):
        print(f"Loss curve saved to {curve_path}")

    print(f"\nDone. Checkpoint: {checkpoint_dir / 'model.pt'}. Sample with: python generate.py")


if __name__ == "__main__":
    train(parse_args())
