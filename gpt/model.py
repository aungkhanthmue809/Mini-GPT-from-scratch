import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class Head(nn.Module):
    """One head of scaled dot-product self-attention with causal masking."""

    def __init__(self, head_size: int, config: GPTConfig) -> None:
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # Lower-triangular mask (block_size x block_size). Constant, not learned,
        # so it is excluded from checkpoints via persistent=False.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)),
            persistent=False,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # Scale by 1/sqrt(head_size) so q·k dot products keep unit variance,
        # preventing the softmax from saturating early in training.
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # Causal mask: position i may only attend to positions <= i. T is re-sliced
        # because during generation the sequence can be shorter than block_size.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # (B, T, head_size)
        return wei @ v     # weighted average of the value vectors


class MultiHeadAttention(nn.Module):
    """Several attention heads run in parallel; concatenated outputs are
    projected back to the embedding dimension."""

    def __init__(self, num_heads: int, head_size: int, config: GPTConfig) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(num_heads)])
        # Projection layer that recomposes the concatenated outputs of all heads.
        self.proj = nn.Linear(head_size * num_heads, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [head(x) for head in self.heads]
        out = torch.cat(outs, dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> ReLU -> Linear -> Dropout.

    Every token position passes through the same network, expanding the
    per-token computation range beyond what attention alone provides.
    """

    def __init__(self, n_embd: int, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLUFeedForward(nn.Module):

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        hidden = int(4 * n_embd * 2 / 3)  # SwiGLU uses 2/3 scaling for equal param count
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def make_norm(config: GPTConfig) -> nn.Module:
    
    if config.use_rmsnorm:
        return RMSNorm(config.n_embd)
    
    return nn.LayerNorm(config.n_embd)


class Block(nn.Module):
    """Transformer block: pre-normalized residual wrappers around attention
    and feed-forward."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        head_size = config.n_embd // config.n_heads
        self.sa = MultiHeadAttention(config.n_heads, head_size, config)
        self.ffwd: nn.Module = (
            SwiGLUFeedForward(config.n_embd) if config.use_swiglu else FeedForward(config.n_embd, config)
        )
        self.ln1 = make_norm(config)
        self.ln2 = make_norm(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))    # residual around attention
        x = x + self.ffwd(self.ln2(x))  # residual around feed-forward
        return x


class GPT(nn.Module):
    """Character-level autoregressive transformer language model (decoder-only).

    Flow: token embeddings + learned positional embeddings -> N transformer
    blocks -> final norm -> linear projection to logits over the vocabulary,
    weight-tied with the token embedding table.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_f = make_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: input embedding and output projection share one matrix,
        # which reduces parameters and helps small models generalize.
        self.token_embedding_table.weight = self.lm_head.weight

        # Custom init instead of PyTorch defaults, applied recursively to every submodule.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Biases start at zero: the weights already break symmetry.
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """Total trainable parameter count (weight-tied tensors count once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, index: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits over the vocabulary; if targets are given, also the
        mean cross-entropy loss."""
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=index.device)
        )                                            # (T, n_embd)
        logits = tok_emb + pos_emb                   # (B, T, n_embd)
        logits = self.blocks(logits)
        logits = self.ln_f(logits)
        logits = self.lm_head(logits)                # (B, T, vocab_size)

        loss: torch.Tensor | None = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C), targets.view(B * T), reduction="mean"
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, index: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively sample `max_new_tokens` new tokens after `index`.

        At each step the model sees only the last block_size tokens, predicts the
        distribution for the next position, samples from it, and appends.
        """
        self.eval()
        block_size = self.config.block_size
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :]          # only the next-position logits
            probs = F.softmax(logits, dim=-1)  # normalize scores into probabilities
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            index = torch.cat((index, index_next), dim=1)
        return index
