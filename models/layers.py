"""
Primitive building blocks: RMSNorm, RoPE, GQA attention, SwiGLU MLP, TransformerBlock.

All modules use Flax NNX. No sharding logic lives here — that belongs in engine/sharder.py.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from models.kv_cache import KVCache


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization (no re-centering bias)."""

    def __init__(self, d_model: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.scale = nnx.Param(jnp.ones(d_model))
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Upcast to float32 for numerical stability, then restore input dtype.
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        normed = (x_f32 / rms) * self.scale[...].astype(jnp.float32)
        return normed.astype(x.dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10_000.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute cosine and sine tables for RoPE.

    Returns:
        cos, sin — each shape (max_seq_len, head_dim // 2), dtype float32.
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = jnp.outer(t, freqs)          # (max_seq_len, head_dim // 2)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(
    x: jnp.ndarray,
    rope_cos: jnp.ndarray,
    rope_sin: jnp.ndarray,
    pos: int,
) -> jnp.ndarray:
    """
    Apply Rotary Position Embedding to a query or key tensor.

    Args:
        x:        (batch, seq_len, n_heads, head_dim)
        rope_cos: (max_seq_len, head_dim // 2)  — precomputed
        rope_sin: (max_seq_len, head_dim // 2)  — precomputed
        pos:      starting sequence position (Python int or JAX scalar)

    Returns:
        Rotated tensor, same shape and dtype as x.
    """
    seq_len = x.shape[1]
    half = x.shape[-1] // 2

    # lax.dynamic_slice is JIT-compatible even when pos is a JAX traced scalar.
    cos = jax.lax.dynamic_slice(rope_cos, (pos, 0), (seq_len, half))  # (seq_len, half)
    sin = jax.lax.dynamic_slice(rope_sin, (pos, 0), (seq_len, half))

    # Broadcast over batch and head dims: (1, seq_len, 1, half)
    cos = cos[None, :, None, :].astype(x.dtype)
    sin = sin[None, :, None, :].astype(x.dtype)

    x1 = x[..., :half]   # (batch, seq_len, n_heads, half)
    x2 = x[..., half:]

    # Complex rotation: (x1 + i*x2) * (cos + i*sin)
    return jnp.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1
    )


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLU(nnx.Module):
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, d_model: int, mlp_hidden: int, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(d_model, mlp_hidden, use_bias=False, rngs=rngs)
        self.up_proj   = nnx.Linear(d_model, mlp_hidden, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(mlp_hidden, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Grouped Query Attention
# ---------------------------------------------------------------------------

class GQAAttention(nnx.Module):
    """
    Grouped Query Attention with a static pre-allocated KV cache.

    n_kv_heads < n_heads: each KV head is shared by (n_heads // n_kv_heads) Q heads.
    The KV cache is read in full on each call; the causal mask excludes future
    and uninitialised positions.
    """

    def __init__(self, config, layer_idx: int, *, rngs: nnx.Rngs):
        self.layer_idx = layer_idx
        self.n_heads   = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim  = config.d_model // config.n_heads
        self.n_rep     = config.n_heads // config.n_kv_heads

        kv_dim = config.n_kv_heads * self.head_dim
        q_dim  = config.n_heads   * self.head_dim

        self.wq = nnx.Linear(config.d_model, q_dim,  use_bias=False, rngs=rngs)
        self.wk = nnx.Linear(config.d_model, kv_dim, use_bias=False, rngs=rngs)
        self.wv = nnx.Linear(config.d_model, kv_dim, use_bias=False, rngs=rngs)
        self.wo = nnx.Linear(q_dim, config.d_model,  use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        kv_cache: KVCache,
        pos: int,
        rope_cos: jnp.ndarray,
        rope_sin: jnp.ndarray,
    ) -> tuple[jnp.ndarray, KVCache]:
        batch, seq_len, _ = x.shape
        dtype = x.dtype

        q = self.wq(x).reshape(batch, seq_len, self.n_heads,    self.head_dim).astype(dtype)
        k = self.wk(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim).astype(dtype)
        v = self.wv(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim).astype(dtype)

        q = apply_rope(q, rope_cos, rope_sin, pos)
        k = apply_rope(k, rope_cos, rope_sin, pos)

        # Write new K, V into the cache at [pos, pos + seq_len).
        kv_cache = kv_cache.update(self.layer_idx, pos, k, v)

        # Read the full (max_seq_len) K and V — static shapes required for JIT.
        full_k, full_v = kv_cache.read(self.layer_idx)
        # full_k/v: (batch, max_seq_len, n_kv_heads, head_dim)

        # GQA: repeat each KV head to match the number of Q heads.
        full_k = jnp.repeat(full_k, self.n_rep, axis=2)  # → (batch, max_seq_len, n_heads, head_dim)
        full_v = jnp.repeat(full_v, self.n_rep, axis=2)

        # Scaled dot-product attention.
        scale  = self.head_dim ** -0.5
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, full_k) * scale  # (batch, n_heads, seq_len, max_seq_len)

        # Causal mask: query at (pos + i) may attend to keys at positions [0, pos + i].
        max_seq_len = kv_cache.data.shape[3]
        q_pos = jnp.arange(seq_len)      + pos   # (seq_len,)
        k_pos = jnp.arange(max_seq_len)           # (max_seq_len,)
        mask  = q_pos[:, None] >= k_pos[None, :]  # (seq_len, max_seq_len) bool
        mask  = mask[None, None, :, :]            # (1, 1, seq_len, max_seq_len)

        # Apply mask and softmax in float32 for stability.
        scores = jnp.where(mask, scores.astype(jnp.float32), jnp.finfo(jnp.float32).min)
        attn   = jax.nn.softmax(scores, axis=-1).astype(dtype)

        out = jnp.einsum("bhqk,bkhd->bqhd", attn, full_v)
        out = out.reshape(batch, seq_len, self.n_heads * self.head_dim)
        return self.wo(out), kv_cache


# ---------------------------------------------------------------------------
# Transformer Block (single layer)
# ---------------------------------------------------------------------------

class TransformerBlock(nnx.Module):
    """Pre-norm transformer layer: attn with residual, then MLP with residual."""

    def __init__(self, config, layer_idx: int, *, rngs: nnx.Rngs):
        self.attn  = GQAAttention(config, layer_idx, rngs=rngs)
        self.mlp   = SwiGLU(config.d_model, config.mlp_hidden, rngs=rngs)
        self.norm1 = RMSNorm(config.d_model, rngs=rngs)
        self.norm2 = RMSNorm(config.d_model, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        kv_cache: KVCache,
        pos: int,
        rope_cos: jnp.ndarray,
        rope_sin: jnp.ndarray,
    ) -> tuple[jnp.ndarray, KVCache]:
        attn_out, kv_cache = self.attn(self.norm1(x), kv_cache, pos, rope_cos, rope_sin)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, kv_cache
