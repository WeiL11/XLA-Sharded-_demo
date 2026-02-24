from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the 12-layer target Llama-style model (~125M params)."""

    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12        # query heads
    n_kv_heads: int = 4      # GQA: 3 query heads per KV head
    vocab_size: int = 32_000
    max_seq_len: int = 2048
    mlp_hidden: int = 2048   # SwiGLU gate/up projection dimension
    dtype: str = "bfloat16"
    rope_theta: float = 10_000.0


@dataclass
class DraftConfig:
    """Configuration for the 2-layer draft model (~25M params). Same dims as target."""

    n_layers: int = 2
    d_model: int = 768        # must match ModelConfig (shared embed/lm_head dims)
    n_heads: int = 12
    n_kv_heads: int = 4
    vocab_size: int = 32_000  # must match ModelConfig
    max_seq_len: int = 2048
    mlp_hidden: int = 2048
    dtype: str = "bfloat16"
    rope_theta: float = 10_000.0
