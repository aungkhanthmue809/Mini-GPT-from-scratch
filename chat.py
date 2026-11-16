#Usage: python chat.py --checkpoint checkpoints/model.pt --max-new-tokens 400
import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from gpt.utils import get_device, load_checkpoint

console = Console()

YOU_STYLE = "cyan"
GPT_STYLE = "magenta"

HELP_TEXT = """\
commands:
  /help   show this help
  /clear  wipe the conversation and start fresh
  /exit   leave the chat (also Ctrl+C at the prompt, Ctrl+D)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a trained MiniGPT checkpoint in the terminal."
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
        default=40,
        help="How many tokens (characters) the model may generate per turn.",
    )
    return parser.parse_args()


def stream_sample(
    model,
    tokenizer,
    device: torch.device,
    context: str,
    max_new_tokens: int,
):
    tokens = tokenizer.encode(context)
    if not tokens:
        tokens = tokenizer.encode("\n") or [0]  # start from a newline, like generate.py
    index = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        index = model.generate(index, 1)  # sample exactly one character
        yield tokenizer.decode(index[0, -1:].tolist())


def chat_turn(
    model,
    tokenizer,
    device: torch.device,
    history: str,
    message: str,
    max_new_tokens: int,
) -> str:
    text = ""
    with Live(
        Panel(Text("", no_wrap=False), title="minigpt", border_style=GPT_STYLE),
        console=console,
        refresh_per_second=20,
    ) as live:
        try:
            for char in stream_sample(model, tokenizer, device, history + message, max_new_tokens):
                text += char
                live.update(
                    Panel(Text(text, no_wrap=False), title="minigpt", border_style=GPT_STYLE)
                )
        except KeyboardInterrupt:
            pass  # keep whatever was generated so far
    return text


def main() -> None:
    args = parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train a model first, e.g.: python train.py --data data/shakespeare.txt"
        )

    device = get_device()
    console.print(f"Loading checkpoint: {args.checkpoint}")
    model, config, tokenizer, payload = load_checkpoint(args.checkpoint, device)

    console.print(
        Panel(
            Text(
                f"device: {device}\n"
                f"trained for: {payload.get('iter', '?')} iterations\n"
                f"context window: {config.block_size} characters\n"
                f"per-turn limit: {args.max_new_tokens} characters\n\n"
                "type /help for commands - this model continues text, it does not\n"
                "follow instructions (it was trained on novels, not conversations).",
                no_wrap=False,
            ),
            title="MiniGPT chat",
            border_style="bright_black",
        )
    )

    history = ""
    while True:
        console.print(Rule(style="dim"))
        try:
            message = Prompt.ask("[bold cyan]you[/]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not message:
            continue
        if message.startswith("/"):
            command = message.lower()
            if command in ("/exit", "/quit"):
                break
            if command == "/clear":
                history = ""
                console.print("[dim]conversation cleared[/]")
                continue
            console.print(HELP_TEXT)
            continue

        console.print()
        reply = chat_turn(model, tokenizer, device, history, message, args.max_new_tokens)
        history += message + reply  # the model sees plain text, not roles

    console.print("[dim]bye[/]")


if __name__ == "__main__":
    main()
