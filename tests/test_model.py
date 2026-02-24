"""
Phase 1 tests: forward pass shapes, KV-cache correctness, RoPE, and tokenizer.

Run all:        pytest tests/
Run this file:  pytest tests/test_model.py -v
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from configs.model_config import DraftConfig, ModelConfig
from models.draft_model import create_draft_model
from models.kv_cache import KVCache
from models.layers import GQAAttention, RMSNorm, SwiGLU, apply_rope, precompute_rope_freqs
from models.transformer import Transformer, create_target_model
from tokenizer.tokenizer import DummyTokenizer

BATCH = 1
SEQ_LEN = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def target_config():
    return ModelConfig()


@pytest.fixture
def draft_config():
    return DraftConfig()


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------


class TestKVCache:
    def test_init_shape(self, target_config):
        cache = KVCache.init(target_config, batch_size=BATCH)
        head_dim = target_config.d_model // target_config.n_heads
        expected = (
            target_config.n_layers,
            2,
            BATCH,
            target_config.max_seq_len,
            target_config.n_kv_heads,
            head_dim,
        )
        assert cache.data.shape == expected, f"got {cache.data.shape}"

    def test_init_dtype(self, target_config):
        cache = KVCache.init(target_config, batch_size=BATCH)
        assert cache.data.dtype == jnp.bfloat16

    def test_init_zeros(self, target_config):
        cache = KVCache.init(target_config, batch_size=BATCH)
        assert jnp.all(cache.data == 0)

    def test_update_writes_correct_slice(self, target_config):
        cache = KVCache.init(target_config, batch_size=BATCH)
        head_dim = target_config.d_model // target_config.n_heads
        new_k = jnp.ones((BATCH, SEQ_LEN, target_config.n_kv_heads, head_dim), dtype=jnp.bfloat16)
        new_v = jnp.ones((BATCH, SEQ_LEN, target_config.n_kv_heads, head_dim), dtype=jnp.bfloat16) * 2

        updated = cache.update(0, 0, new_k, new_v)
        k_read, v_read = updated.read(0)

        assert jnp.allclose(k_read[:, :SEQ_LEN, :, :], new_k)
        assert jnp.allclose(v_read[:, :SEQ_LEN, :, :], new_v)

    def test_update_leaves_rest_as_zeros(self, target_config):
        cache = KVCache.init(target_config, batch_size=BATCH)
        head_dim = target_config.d_model // target_config.n_heads
        new_k = jnp.ones((BATCH, SEQ_LEN, target_config.n_kv_heads, head_dim), dtype=jnp.bfloat16)
        new_v = jnp.ones_like(new_k)

        updated = cache.update(0, 0, new_k, new_v)
        k_read, v_read = updated.read(0)

        assert jnp.all(k_read[:, SEQ_LEN:, :, :] == 0)
        assert jnp.all(v_read[:, SEQ_LEN:, :, :] == 0)

    def test_update_at_offset(self, target_config):
        """Writing at pos=4 should leave positions 0-3 unchanged."""
        cache = KVCache.init(target_config, batch_size=BATCH)
        head_dim = target_config.d_model // target_config.n_heads
        new_k = jnp.ones((BATCH, SEQ_LEN, target_config.n_kv_heads, head_dim), dtype=jnp.bfloat16)
        new_v = jnp.ones_like(new_k)

        updated = cache.update(0, 4, new_k, new_v)
        k_read, _ = updated.read(0)

        assert jnp.all(k_read[:, :4, :, :] == 0)
        assert jnp.allclose(k_read[:, 4 : 4 + SEQ_LEN, :, :], new_k)

    def test_draft_cache_has_fewer_layers(self, target_config, draft_config):
        target_cache = KVCache.init(target_config)
        draft_cache = KVCache.init(draft_config)
        assert target_cache.data.shape[0] == 12
        assert draft_cache.data.shape[0] == 2


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class TestRoPE:
    def test_precompute_shape(self):
        head_dim, max_seq = 64, 128
        cos, sin = precompute_rope_freqs(head_dim, max_seq)
        assert cos.shape == (max_seq, head_dim // 2)
        assert sin.shape == (max_seq, head_dim // 2)

    def test_apply_rope_preserves_shape_and_dtype(self):
        cos, sin = precompute_rope_freqs(64, 128)
        x = jnp.ones((BATCH, SEQ_LEN, 12, 64), dtype=jnp.bfloat16)
        out = apply_rope(x, cos, sin, 0)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_rope_identity_at_position_zero(self):
        """At pos=0 all angles are 0, so cos=1 and sin=0 → RoPE is identity."""
        cos, sin = precompute_rope_freqs(64, 128)
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, 4, 64))
        out = apply_rope(x, cos, sin, 0)
        assert jnp.allclose(out, x, atol=1e-5)

    def test_rope_changes_values_at_nonzero_position(self):
        """Rotation at pos=5 must differ from identity (pos=0)."""
        cos, sin = precompute_rope_freqs(64, 128)
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 1, 4, 64))
        out0 = apply_rope(x, cos, sin, 0)
        out5 = apply_rope(x, cos, sin, 5)
        assert not jnp.allclose(out0, out5, atol=1e-5)

    def test_rope_different_positions_differ(self):
        """Two consecutive positions should produce different embeddings."""
        cos, sin = precompute_rope_freqs(64, 256)
        x = jax.random.normal(jax.random.PRNGKey(2), (1, 1, 4, 64))
        out1 = apply_rope(x, cos, sin, 10)
        out2 = apply_rope(x, cos, sin, 11)
        assert not jnp.allclose(out1, out2, atol=1e-5)


# ---------------------------------------------------------------------------
# Target model (12-layer)
# ---------------------------------------------------------------------------


class TestTargetModel:
    def test_forward_output_shape(self, target_config):
        rngs = nnx.Rngs(params=42)
        model = Transformer(target_config, rngs=rngs)
        cache = KVCache.init(target_config, batch_size=BATCH)
        tokens = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.int32)

        logits, new_cache = model(tokens, cache, pos=0)

        assert logits.shape == (BATCH, SEQ_LEN, target_config.vocab_size)
        assert new_cache.data.shape == cache.data.shape

    def test_forward_updates_cache(self, target_config):
        rngs = nnx.Rngs(params=42)
        model = Transformer(target_config, rngs=rngs)
        cache = KVCache.init(target_config, batch_size=BATCH)
        tokens = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.int32)

        _, new_cache = model(tokens, cache, pos=0)
        k, _ = new_cache.read(0)

        # Written positions must be non-zero (random weights on ones input).
        assert not jnp.all(k[:, :SEQ_LEN, :, :] == 0)
        # Unwritten positions must remain zero.
        assert jnp.all(k[:, SEQ_LEN:, :, :] == 0)

    def test_autoregressive_step_after_prefill(self, target_config):
        """Single-token decode step must accept pos=SEQ_LEN and return shape (B, 1, V)."""
        rngs = nnx.Rngs(params=42)
        model = Transformer(target_config, rngs=rngs)
        cache = KVCache.init(target_config, batch_size=BATCH)

        prompt = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.int32)
        _, cache = model(prompt, cache, pos=0)

        next_tok = jnp.array([[42]], dtype=jnp.int32)
        logits, _ = model(next_tok, cache, pos=SEQ_LEN)

        assert logits.shape == (BATCH, 1, target_config.vocab_size)

    def test_layer_count(self, target_config):
        model = create_target_model()
        assert len(model.layers) == target_config.n_layers

    def test_rope_tables_shape(self, target_config):
        model = create_target_model()
        head_dim = target_config.d_model // target_config.n_heads
        assert model.rope_cos.shape == (target_config.max_seq_len, head_dim // 2)
        assert model.rope_sin.shape == (target_config.max_seq_len, head_dim // 2)

    def test_different_seeds_give_different_logits(self, target_config):
        cache = KVCache.init(target_config, batch_size=BATCH)
        tokens = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.int32)

        m1 = Transformer(target_config, rngs=nnx.Rngs(params=1))
        m2 = Transformer(target_config, rngs=nnx.Rngs(params=2))

        logits1, _ = m1(tokens, cache, pos=0)
        logits2, _ = m2(tokens, cache, pos=0)

        assert not jnp.allclose(logits1, logits2)


# ---------------------------------------------------------------------------
# Draft model (2-layer)
# ---------------------------------------------------------------------------


class TestDraftModel:
    def test_forward_output_shape(self, draft_config):
        model = create_draft_model()
        cache = KVCache.init(draft_config, batch_size=BATCH)
        tokens = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.int32)

        logits, new_cache = model(tokens, cache, pos=0)

        assert logits.shape == (BATCH, SEQ_LEN, draft_config.vocab_size)
        assert new_cache.data.shape == cache.data.shape

    def test_layer_count(self, draft_config):
        model = create_draft_model()
        assert len(model.layers) == draft_config.n_layers

    def test_shared_vocab_size(self, target_config, draft_config):
        """Both models must share vocab_size so token IDs are interchangeable."""
        assert target_config.vocab_size == draft_config.vocab_size

    def test_shared_d_model(self, target_config, draft_config):
        """Both models must share d_model so embedding dims are compatible."""
        assert target_config.d_model == draft_config.d_model


# ---------------------------------------------------------------------------
# DummyTokenizer
# ---------------------------------------------------------------------------


class TestDummyTokenizer:
    def test_encode_starts_with_bos(self):
        tok = DummyTokenizer()
        ids = tok.encode("hello")
        assert ids[0] == tok.bos_id

    def test_encode_length(self):
        tok = DummyTokenizer()
        text = "hello"
        ids = tok.encode(text)
        assert len(ids) == 1 + len(text)  # BOS + one ID per character

    def test_encode_ids_in_vocab_range(self):
        tok = DummyTokenizer()
        for _ in range(5):
            ids = tok.encode("test string 123")
            assert all(0 <= i < tok.vocab_size for i in ids)

    def test_decode_excludes_special_tokens(self):
        tok = DummyTokenizer()
        out = tok.decode([tok.bos_id, 100, tok.eos_id, 200])
        assert str(tok.bos_id) not in out.split()
        assert "[100]" in out
        assert "[200]" in out

    def test_encode_is_deterministic(self):
        tok = DummyTokenizer()
        assert tok.encode("hello") == tok.encode("hello")
