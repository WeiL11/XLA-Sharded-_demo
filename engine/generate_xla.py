"""Single-device XLA generation baseline using jax.lax.while_loop."""

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


def generate_xla(
    prompt: str,
    *,
    max_new_tokens: int = 200,
    seed: int = 42,
    config: ModelConfig | None = None,
    tokenizer: DummyTokenizer | None = None,
) -> GenerationResult:
    """
    Generate tokens with an XLA-friendly while_loop decode core.

    Prefill remains a regular forward pass; decode loop is compiled.
    """
    if config is None:
        config = ModelConfig()
    if tokenizer is None:
        tokenizer = DummyTokenizer()
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")

    model = Transformer(config, rngs=nnx.Rngs(params=seed))
    kv_cache = KVCache.init(config, batch_size=1)

    prompt_ids = tokenizer.encode(prompt)
    prompt_arr = jnp.array([prompt_ids], dtype=jnp.int32)

    logits, kv_cache = model(prompt_arr, kv_cache, pos=0)
    first_next = jnp.argmax(logits[0, -1]).astype(jnp.int32)
    start_pos = jnp.asarray(len(prompt_ids), dtype=jnp.int32)

    @jax.jit
    def decode_loop(
        cache: KVCache,
        start_token: jnp.ndarray,
        pos: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        generated = jnp.zeros((max_new_tokens,), dtype=jnp.int32)

        def cond_fn(state):
            _, step, _, _, done = state
            return (step < max_new_tokens) & (~done)

        def body_fn(state):
            out, step, cur_tok, cur_cache, done = state
            out = out.at[step].set(cur_tok)
            now_done = done | (cur_tok == tokenizer.eos_id)

            def keep_branch(args):
                c, t = args
                return t, c

            def run_branch(args):
                c, t = args
                token_arr = t.reshape(1, 1).astype(jnp.int32)
                new_logits, new_cache = model(token_arr, c, pos + step)
                next_tok = jnp.argmax(new_logits[0, -1]).astype(jnp.int32)
                return next_tok, new_cache

            next_tok, next_cache = jax.lax.cond(
                now_done,
                keep_branch,
                run_branch,
                operand=(cur_cache, cur_tok),
            )
            return out, step + 1, next_tok, next_cache, now_done

        init = (generated, jnp.asarray(0, dtype=jnp.int32), start_token, cache, jnp.asarray(False))
        out, steps, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)
        return out, steps

    generated_arr, generated_count = decode_loop(kv_cache, first_next, start_pos)
    generated_arr.block_until_ready()

    n_gen = int(generated_count)
    generated_ids = [int(x) for x in generated_arr[:n_gen]]
    all_ids = list(prompt_ids) + generated_ids
    return GenerationResult(
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        all_ids=all_ids,
        decoded_generated_text=tokenizer.decode(generated_ids),
        decoded_all_text=tokenizer.decode(all_ids),
    )

