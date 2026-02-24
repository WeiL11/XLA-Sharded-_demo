from typing import NamedTuple

import jax
import jax.numpy as jnp


class KVCache(NamedTuple):
    """
    Static pre-allocated KV cache for all transformer layers.

    Shape: (n_layers, 2, batch, max_seq_len, n_kv_heads, head_dim)
      - axis 1: 0 = K, 1 = V

    NamedTuple makes this a JAX pytree, so it can be used inside jax.jit
    and jax.lax.while_loop without any special handling.
    """

    data: jnp.ndarray

    @staticmethod
    def init(config, batch_size: int = 1) -> "KVCache":
        """Allocate a zero-filled cache for the given config."""
        head_dim = config.d_model // config.n_heads
        data = jnp.zeros(
            (config.n_layers, 2, batch_size, config.max_seq_len, config.n_kv_heads, head_dim),
            dtype=jnp.bfloat16,
        )
        return KVCache(data=data)

    def update(
        self,
        layer_idx: int,
        pos: int,
        new_k: jnp.ndarray,
        new_v: jnp.ndarray,
    ) -> "KVCache":
        """
        Write new K and V at positions [pos, pos + seq_len).

        Uses lax.dynamic_update_slice so traced pos values are JIT-safe
        in generation loops.
        """
        new_k = new_k.astype(self.data.dtype)
        new_v = new_v.astype(self.data.dtype)

        layer_k = self.data[layer_idx, 0]
        layer_v = self.data[layer_idx, 1]

        layer_k = jax.lax.dynamic_update_slice(layer_k, new_k, (0, pos, 0, 0))
        layer_v = jax.lax.dynamic_update_slice(layer_v, new_v, (0, pos, 0, 0))

        new_data = self.data.at[layer_idx, 0].set(layer_k)
        new_data = new_data.at[layer_idx, 1].set(layer_v)
        return KVCache(data=new_data)

    def read(self, layer_idx: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return the full K and V arrays for a layer (all max_seq_len positions).

        Callers are responsible for masking out-of-range positions via the
        causal mask in attention. Returning the full static shape keeps the
        cache compatible with jax.jit (no dynamic slicing on read).
        """
        k = self.data[layer_idx, 0]  # (batch, max_seq_len, n_kv_heads, head_dim)
        v = self.data[layer_idx, 1]
        return k, v
