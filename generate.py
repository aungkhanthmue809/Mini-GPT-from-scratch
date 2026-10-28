"""
Usage:
    python generate.py --checkpoint checkpoints/model.pt --prompt "To be or not" --max-new-tokens 500
"""

import argparse
from pathlib import Path
from gpt.utils import generate_sample, infer_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained MiniGPT checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model.pt",
        help="Path to a checkpoint file written by train.py.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="How many tokens (characters) to generate.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt text; if empty, generation starts from a newline character.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train a model first, e.g.: python train.py --data data/shakespeare.txt"
        )

    device = infer_device()
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    #left to check
    model, config, tokenizer, payload = load_checkpoint(args.checkpoint, device)
    
    print(f"Checkpoint trained for {payload.get('iter', '?')} iterations")

    print(f"Prompt: {args.prompt!r}" + ("" if args.prompt else " (starting from newline)"))
    sample = generate_sample(
        model, tokenizer, config, device,
        prompt=args.prompt, max_new_tokens=args.max_new_tokens,
    )

    print("--- generated text ---")
    print(sample)


if __name__ == "__main__":
    main()
