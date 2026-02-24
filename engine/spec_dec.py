"""Speculative decoding orchestration (draft -> verify -> accept/reject)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from configs.model_config import DraftConfig, ModelConfig
from models.kv_cache import KVCache
from models.transformer import Transformer
from tokenizer.tokenizer import DummyTokenizer


@dataclass
class SpeculativeResult:
    prompt_ids: list[int]
    generated_ids: list[int]
    all_ids: list[int]
    decoded_generated_text: str
    decoded_all_text: str
    rounds: int
    accepted_tokens: int
    proposed_tokens: int

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens


def _tiny_softmax_probs(logits: jnp.ndarray, token_id: int) -> tuple[float, jnp.ndarray]:
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    return float(probs[token_id]), probs


def _advance_cache(model: Transformer, cache: KVCache, tokens: list[int], pos: int) -> KVCache:
    cur_cache = cache
    cur_pos = pos
    for tok in tokens:
        tok_arr = jnp.array([[tok]], dtype=jnp.int32)
        _, cur_cache = model(tok_arr, cur_cache, pos=cur_pos)
        cur_pos += 1
    return cur_cache


def speculative_decode(
    prompt: str,
    *,
    max_new_tokens: int = 200,
    k: int = 5,
    seed: int = 42,
    target_config: ModelConfig | None = None,
    draft_config: DraftConfig | None = None,
    tokenizer: DummyTokenizer | None = None,
) -> SpeculativeResult:
    """
    Python-orchestrated speculative decoding with separate draft/target models.

    This keeps model forward passes in JAX while the accept/reject control logic
    remains explicit and easy to validate.
    """
    if target_config is None:
        target_config = ModelConfig()
    if draft_config is None:
        draft_config = DraftConfig()
    if tokenizer is None:
        tokenizer = DummyTokenizer()
    if k <= 0:
        raise ValueError("k must be > 0")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")

    if target_config.vocab_size != draft_config.vocab_size:
        raise ValueError("target and draft vocab_size must match")
    if target_config.d_model != draft_config.d_model:
        raise ValueError("target and draft d_model must match")

    target = Transformer(target_config, rngs=nnx.Rngs(params=seed))
    draft = Transformer(draft_config, rngs=nnx.Rngs(params=seed + 1))

    target_cache = KVCache.init(target_config, batch_size=1)
    draft_cache = KVCache.init(draft_config, batch_size=1)

    prompt_ids = tokenizer.encode(prompt)
    prompt_arr = jnp.array([prompt_ids], dtype=jnp.int32)
    target_logits, target_cache = target(prompt_arr, target_cache, pos=0)
    _, draft_cache = draft(prompt_arr, draft_cache, pos=0)

    last_token = int(jnp.argmax(target_logits[0, -1]))
    cur_pos = len(prompt_ids)
    generated: list[int] = []
    accepted_total = 0
    proposed_total = 0
    rounds = 0
    key = jax.random.PRNGKey(seed)

    while len(generated) < max_new_tokens:
        rounds += 1
        round_start_draft = draft_cache
        round_start_target = target_cache
        round_pos = cur_pos

        draft_tokens: list[int] = []
        draft_token_probs: list[float] = []
        draft_step_logits: list[jnp.ndarray] = []

        next_tok = last_token
        for _ in range(min(k, max_new_tokens - len(generated))):
            tok_arr = jnp.array([[next_tok]], dtype=jnp.int32)
            d_logits, draft_cache = draft(tok_arr, draft_cache, pos=cur_pos)
            prob, full_probs = _tiny_softmax_probs(d_logits[0, -1], next_tok)
            draft_tokens.append(next_tok)
            draft_token_probs.append(prob)
            draft_step_logits.append(full_probs)
            proposed_total += 1
            cur_pos += 1
            next_tok = int(jnp.argmax(d_logits[0, -1]))

        accepted_this_round = 0
        rejected = False
        bonus_token: int | None = None

        cur_pos = round_pos
        for i, proposed_tok in enumerate(draft_tokens):
            tok_arr = jnp.array([[proposed_tok]], dtype=jnp.int32)
            t_logits, target_cache = target(tok_arr, target_cache, pos=cur_pos)
            t_prob, target_probs = _tiny_softmax_probs(t_logits[0, -1], proposed_tok)
            d_prob = max(draft_token_probs[i], 1e-8)
            accept_prob = min(t_prob / d_prob, 1.0)

            key, sub = jax.random.split(key)
            u = float(jax.random.uniform(sub))
            if u <= accept_prob:
                accepted_this_round += 1
                cur_pos += 1
                continue

            rejected = True
            adjusted = jnp.maximum(target_probs - draft_step_logits[i], 0.0)
            denom = jnp.sum(adjusted)
            sampled_probs = jnp.where(
                denom > 0,
                adjusted / denom,
                target_probs,
            )
            key, sub2 = jax.random.split(key)
            bonus_token = int(jax.random.categorical(sub2, jnp.log(sampled_probs + 1e-8)))
            break

        accepted_tokens = draft_tokens[:accepted_this_round]
        accepted_total += accepted_this_round
        generated.extend(accepted_tokens)

        # Rebuild both caches from round start using accepted prefix only.
        draft_cache = _advance_cache(draft, round_start_draft, accepted_tokens, round_pos)
        target_cache = _advance_cache(target, round_start_target, accepted_tokens, round_pos)
        cur_pos = round_pos + accepted_this_round

        if len(generated) >= max_new_tokens:
            break

        if rejected and bonus_token is not None:
            generated.append(bonus_token)
            draft_cache = _advance_cache(draft, draft_cache, [bonus_token], cur_pos)
            target_cache = _advance_cache(target, target_cache, [bonus_token], cur_pos)
            cur_pos += 1
            last_token = bonus_token
        elif draft_tokens:
            last_token = draft_tokens[-1]
        else:
            break

        if generated and generated[-1] == tokenizer.eos_id:
            break

    all_ids = list(prompt_ids) + generated
    return SpeculativeResult(
        prompt_ids=prompt_ids,
        generated_ids=generated,
        all_ids=all_ids,
        decoded_generated_text=tokenizer.decode(generated),
        decoded_all_text=tokenizer.decode(all_ids),
        rounds=rounds,
        accepted_tokens=accepted_total,
        proposed_tokens=proposed_total,
    )

