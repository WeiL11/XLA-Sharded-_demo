"""
Llama-style Transformer that works with any compatible config (ModelConfig or DraftConfig).

No sharding logic here. The engine/ layer is responsible for device placement.
"""

import jax.numpy as jnp
from flax import nnx

from configs.model_config import ModelConfig
from models.kv_cache import KVCache
from models.layers import RMSNorm, TransformerBlock, precompute_rope_freqs


class Buffer(nnx.Variable):
    """
    Non-trainable precomputed buffer (e.g., RoPE frequency tables).

    Unlike nnx.Param, Buffer values are not included in gradient updates.
    In Flax 0.12+, JAX arrays stored as module attributes must be wrapped
    in a Variable subclass so NNX can track them correctly as data.
    """
    pass


class Transformer(nnx.Module):
    """
    Llama-style Transformer: token embed → N × TransformerBlock → RMSNorm → lm_head.

    The same class is used for both the 12-layer target model and the 2-layer draft model;
    the only difference is the config passed in.

    RoPE frequency tables are stored as Buffer variables (non-trainable, tracked data).
    Transformer layers are stored in an nnx.List (required in Flax 0.12+ for lists of modules).
    """

    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config

        self.embed   = nnx.Embed(config.vocab_size, config.d_model, rngs=rngs)
        # nnx.List is required in Flax 0.12+ for Python lists that contain modules/arrays.
        self.layers  = nnx.List([TransformerBlock(config, i, rngs=rngs) for i in range(config.n_layers)])
        self.norm    = RMSNorm(config.d_model, rngs=rngs)
        self.lm_head = nnx.Linear(config.d_model, config.vocab_size, use_bias=False, rngs=rngs)

        head_dim = config.d_model // config.n_heads
        rope_cos, rope_sin = precompute_rope_freqs(head_dim, config.max_seq_len, config.rope_theta)
        # Buffer wraps plain JAX arrays so NNX tracks them as data, not static Python values.
        self.rope_cos = Buffer(rope_cos)
        self.rope_sin = Buffer(rope_sin)

    def __call__(
        self,
        tokens: jnp.ndarray,
        kv_cache: KVCache,
        pos: int,
    ) -> tuple[jnp.ndarray, KVCache]:
        """
        Args:
            tokens:   (batch, seq_len) int32 token IDs
            kv_cache: KVCache for all layers
            pos:      starting position index for this forward pass
                      (0 for prefill; current length for single-token decode)

        Returns:
            logits:   (batch, seq_len, vocab_size)
            kv_cache: updated KVCache (functionally, via NamedTuple replacement)
        """
        # Embed and cast to bfloat16; all activations stay in bfloat16 throughout.
        x = self.embed(tokens).astype(jnp.bfloat16)  # (batch, seq_len, d_model)

        for layer in self.layers:
            x, kv_cache = layer(x, kv_cache, pos, self.rope_cos.value, self.rope_sin.value)

        x = self.norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits, kv_cache


def create_target_model(rngs: nnx.Rngs | None = None) -> Transformer:
    """Instantiate the 12-layer target model with default ModelConfig."""
    if rngs is None:
        rngs = nnx.Rngs(params=42)
    return Transformer(ModelConfig(), rngs=rngs)
