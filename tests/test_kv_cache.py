import jax
import jax.numpy as jnp

from configs.model_config import ModelConfig
from models.kv_cache import KVCache


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=256,
        max_seq_len=32,
        mlp_hidden=128,
    )


def test_kv_cache_update_and_read():
    cfg = _tiny_config()
    cache = KVCache.init(cfg, batch_size=1)
    head_dim = cfg.d_model // cfg.n_heads

    new_k = jnp.ones((1, 3, cfg.n_kv_heads, head_dim), dtype=jnp.float32)
    new_v = jnp.ones((1, 3, cfg.n_kv_heads, head_dim), dtype=jnp.float32) * 2
    updated = cache.update(0, 4, new_k, new_v)
    k, v = updated.read(0)

    assert k.dtype == jnp.bfloat16
    assert v.dtype == jnp.bfloat16
    assert jnp.allclose(k[:, 4:7], new_k.astype(jnp.bfloat16))
    assert jnp.allclose(v[:, 4:7], new_v.astype(jnp.bfloat16))


def test_kv_cache_update_jittable_with_dynamic_pos():
    cfg = _tiny_config()
    cache = KVCache.init(cfg, batch_size=1)
    head_dim = cfg.d_model // cfg.n_heads
    new_k = jnp.ones((1, 1, cfg.n_kv_heads, head_dim), dtype=jnp.float32)
    new_v = jnp.ones((1, 1, cfg.n_kv_heads, head_dim), dtype=jnp.float32)

    @jax.jit
    def _fn(c, pos):
        return c.update(0, pos, new_k, new_v)

    updated = _fn(cache, jnp.asarray(5, dtype=jnp.int32))
    k, _ = updated.read(0)
    assert jnp.allclose(k[:, 5:6], new_k.astype(jnp.bfloat16))

