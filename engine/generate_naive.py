"""Single-device baseline generation with a Python loop and jitted decode step."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from configs.model_config import ModelConfig
from models.kv_cache import KVCache
from models.transformer import Transformer
from tokenizer.tokenizer import DummyTokenizer


@dataclass
class GenerationResult:
    prompt_ids: list[int]
    generated_ids: list[int]
    all_ids: list[int]
    decoded_generated_text: str
    decoded_all_text: str


def _build_jitted_step(model: Transformer):
    """Build a jitted single-step forward function for autoregressive decode."""
    return jax.jit(lambda token, cache, pos: model(token, cache, pos))


def generate_naive(
    prompt: str,
    *,
    max_new_tokens: int = 200,
    seed: int = 42,
    config: ModelConfig | None = None,
    tokenizer: DummyTokenizer | None = None,
) -> GenerationResult:
    """
    Generate tokens with a Python loop baseline.

    Uses:
    - one prefill forward pass on the full prompt
    - jitted per-token decode step in a Python loop
    """
    if config is None:
        config = ModelConfig()
    if tokenizer is None:
        tokenizer = DummyTokenizer()
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")

    model = Transformer(config, rngs=nnx.Rngs(params=seed))
    kv_cache = KVCache.init(config, batch_size=1)
    step_fn = _build_jitted_step(model)

    prompt_ids = tokenizer.encode(prompt)
    prompt_arr = jnp.array([prompt_ids], dtype=jnp.int32)

    # Prefill prompt once so cache holds all prompt K/V states.
    logits, kv_cache = model(prompt_arr, kv_cache, pos=0)
    next_id = int(jnp.argmax(logits[0, -1]))

    all_ids = list(prompt_ids)
    generated_ids: list[int] = []

    cur_pos = len(prompt_ids)
    for _ in range(max_new_tokens):
        generated_ids.append(next_id)
        all_ids.append(next_id)

        if next_id == tokenizer.eos_id:
            break

        token_arr = jnp.array([[next_id]], dtype=jnp.int32)
        logits, kv_cache = step_fn(token_arr, kv_cache, jnp.asarray(cur_pos, dtype=jnp.int32))
        next_id = int(jnp.argmax(logits[0, -1]))
        cur_pos += 1

    return GenerationResult(
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        all_ids=all_ids,
        decoded_generated_text=tokenizer.decode(generated_ids),
        decoded_all_text=tokenizer.decode(all_ids),
    )

