"""Data pipeline: corpus loading, character tokenizer, and batch sampling."""

import os

import torch

from .config import GPTConfig

#NOT A NN COMPONENT ANYMORE
class CharTokenizer:

    def __init__(self, chars: list[str]) -> None:
        self.chars: list[str] = chars
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size: int = len(chars)

    #edits the cls object to have a vocab
    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        """Convert text into token ids. Characters outside the vocab are skipped."""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, tokens: list[int]) -> str:
        """Convert a list of token ids back into text."""
        return "".join(self.itos[int(t)] for t in tokens)

    #tokeniser object -> dicttnary -> json on disk
    def to_dict(self) -> dict[str, list[str]]:
        return {"chars": self.chars}
    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "CharTokenizer":
        return cls(data["chars"])

def load_text(path: str | os.PathLike[str]) -> str:
    path = str(path)
    if os.path.isdir(path):
        texts: list[str] = []
        for filename in sorted(os.listdir(path)):
            if filename.lower().endswith(".txt" ):
                with open(os.path.join(path, filename), "r", encoding="utf-8-sig") as f:
                    text = f.read()
                    texts.append(text)
                    
        if not texts:
            raise FileNotFoundError(f"No .txt files found in directory: {path}")
        
        #join two books with a two enters
        return "\n\n".join(texts)
    
    elif ( os.path.isfile(path) ):
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Path does not exist or is not a file/directory: {path}")

def build_tokenizer(path: str | os.PathLike[str]) -> CharTokenizer:
    #Load the corpus (file or folder of .txt files) and build its tokenizer
    return CharTokenizer.from_text(load_text(path))

def encode_and_split(
    tokenizer: CharTokenizer, config: GPTConfig, path: str | os.PathLike[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    #Encode the corpus with the tokenizer and split into train/val tensors
    data = torch.tensor(tokenizer.encode(load_text(path)), dtype=torch.long)
    n = int( len(data) * config.train_split)
    return data[:n], data[n:]


def get_batch(
    data: torch.Tensor, config: GPTConfig, device: torch.device ) -> tuple[torch.Tensor, torch.Tensor]:

    indices = torch.randint(len(data) - config.block_size , (config.batch_size,))
    
    x = torch.stack([data[i : i + config.block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + 1 + config.block_size] for i in indices])
    
    return x.to(device), y.to(device)
