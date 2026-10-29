from dataclasses import dataclass

@dataclass
class GPTConfig:
    # Model architecture
    vocab_size: int
    block_size: int = 64
    n_embd: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 8
    max_iters: int = 3000
    learning_rate: float = 3e-4
    eval_interval: int = 100
    eval_iters: int = 50
    warmup_iters: int = 100
    gradient_clip_norm: float = 1.0
    train_split: float = 0.9

    # Architecture options
    use_rmsnorm: bool = False
    use_swiglu: bool = False
    '''
    vocab_size: int
    block_size: int = 256      # context window: max tokens the model attends over
    n_embd: int = 384          # embedding dimension
    n_heads: int = 6           # number of parallel attention heads
    n_layers: int = 6          # number of stacked transformer blocks
    dropout: float = 0.2       # dropout probability (0 disables it)


    batch_size: int = 32    # sequences per gradient step
    max_iters: int = 20000     # total optimizer steps
    learning_rate: float = 3e-4
    eval_interval: int = 500   # evaluate / checkpoint every N steps
    eval_iters: int = 200      # batches averaged per loss estimate
    warmup_iters: int = 100    # linear LR warmup steps before cosine decay
    gradient_clip_norm: float = 1.0
    train_split: float = 0.9   # fraction of corpus used for training (rest is val)

    # --- Modern architecture switches (default off = original design) -----------
    use_rmsnorm: bool = False  # swap LayerNorm -> RMSNorm (LLaMA / Mistral / Gemma style)
    use_swiglu: bool = False   # swap ReLU feed-forward -> SwiGLU feed-forward
    '''