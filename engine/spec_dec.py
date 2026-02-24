"""
Speculative decoding orchestration — outer round loop compiled with jax.lax.while_loop.

Draft inner loop:  jax.lax.fori_loop (k steps)
Target verify:     single parallel forward pass on all k draft tokens
Accept/reject:     vectorised rejection sampling with jnp.cumprod
Bonus token:       sampled via jax.lax.cond on the adjusted distribution
Cache rollback:    position counter reset only; stale entries are masked by causal attention
"""

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
    Speculative decoding with the outer round loop compiled via jax.lax.while_loop.

    Pipeline per round
    ------------------
    1. Draft fori_loop (k steps): generates k draft tokens + their probability distributions.
    2. Target parallel verify: one forward pass on all k draft tokens at once.
    3. Vectorised rejection sampling: accept the longest matching prefix.
    4. All-accepted branch  → last draft token becomes the new seed; advance pos by k.
       Some-rejected branch → sample bonus token from adjusted distribution; feed it
                              through both model caches; advance pos by n_accepted + 1.
    5. Repeat until EOS or max_new_tokens.
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

    vocab_size = target_config.vocab_size
    eos_id = tokenizer.eos_id

    target = Transformer(target_config, rngs=nnx.Rngs(params=seed))
    draft  = Transformer(draft_config,  rngs=nnx.Rngs(params=seed + 1))

    target_cache = KVCache.init(target_config, batch_size=1)
    draft_cache  = KVCache.init(draft_config,  batch_size=1)

    # Prefill both caches with the prompt.
    prompt_ids  = tokenizer.encode(prompt)
    prompt_arr  = jnp.array([prompt_ids], dtype=jnp.int32)
    t_logits, target_cache = target(prompt_arr, target_cache, pos=0)
    _,        draft_cache  = draft( prompt_arr, draft_cache,  pos=0)

    first_token = jnp.argmax(t_logits[0, -1]).astype(jnp.int32)
    start_pos   = jnp.asarray(len(prompt_ids), dtype=jnp.int32)
    prng_key    = jax.random.PRNGKey(seed)

    # ------------------------------------------------------------------
    # Compiled speculative loop.  Both models are captured by closure so
    # their parameters are treated as JIT constants (not loop state).
    # All loop state is a pytree of JAX arrays.
    # ------------------------------------------------------------------
    @jax.jit
    def _compiled_loop(
        d_cache: KVCache,
        t_cache: KVCache,
        first_tok: jnp.ndarray,
        start_p: jnp.ndarray,
        key: jnp.ndarray,
    ):
        out_buf  = jnp.zeros((max_new_tokens,), dtype=jnp.int32)
        init_state = (
            out_buf,
            jnp.asarray(0,     jnp.int32),   # n_generated
            first_tok,                         # last_token (seed for next draft round)
            start_p,                           # cur_pos
            d_cache,
            t_cache,
            key,
            jnp.asarray(False),               # done
            jnp.asarray(0,     jnp.int32),   # rounds
            jnp.asarray(0,     jnp.int32),   # accepted_total
            jnp.asarray(0,     jnp.int32),   # proposed_total
        )

        def cond_fn(state):
            _, n_gen, _, _, _, _, _, done, _, _, _ = state
            return (~done) & (n_gen < max_new_tokens)

        def body_fn(state):
            out_buf, n_gen, last_tok, cur_pos, d_cache, t_cache, key, done, rounds, acc_total, prop_total = state

            # --------------------------------------------------------------
            # 1. Draft phase — generate k tokens via fori_loop.
            #    draft_tokens[i] = token fed into the draft at step i.
            #    draft_probs[i]  = probability distribution output at step i.
            # --------------------------------------------------------------
            tok_buf   = jnp.zeros((k,),            dtype=jnp.int32)
            probs_buf = jnp.zeros((k, vocab_size), dtype=jnp.float32)

            def draft_step(i, ds):
                dc, pos, tok, tb, pb = ds
                logits, new_dc = draft(tok.reshape(1, 1).astype(jnp.int32), dc, pos=pos)
                probs    = jax.nn.softmax(logits[0, -1].astype(jnp.float32))
                next_tok = jnp.argmax(logits[0, -1]).astype(jnp.int32)
                return new_dc, pos + 1, next_tok, tb.at[i].set(tok), pb.at[i].set(probs)

            d_cache_new, _, _, draft_tokens, draft_probs = jax.lax.fori_loop(
                0, k, draft_step, (d_cache, cur_pos, last_tok, tok_buf, probs_buf)
            )

            # --------------------------------------------------------------
            # 2. Target parallel verify — one forward pass on all k tokens.
            #    Causal masking ensures target_probs[i] conditions only on
            #    positions 0..cur_pos+i, identical to running token-by-token.
            # --------------------------------------------------------------
            t_logits, t_cache_new = target(
                draft_tokens.reshape(1, k).astype(jnp.int32), t_cache, pos=cur_pos
            )
            target_probs = jax.nn.softmax(t_logits[0].astype(jnp.float32), axis=-1)  # (k, vocab)

            # --------------------------------------------------------------
            # 3. Vectorised rejection sampling.
            # --------------------------------------------------------------
            arange_k   = jnp.arange(k)
            d_tok_p    = draft_probs[ arange_k, draft_tokens]        # (k,)
            t_tok_p    = target_probs[arange_k, draft_tokens]        # (k,)
            accept_p   = jnp.minimum(t_tok_p / jnp.maximum(d_tok_p, 1e-8), 1.0)

            key, sub   = jax.random.split(key)
            uniform_s  = jax.random.uniform(sub, shape=(k,))
            accepted   = uniform_s <= accept_p                        # (k,) bool
            # n_accepted = length of the leading run of True values.
            n_accepted = jnp.sum(
                jnp.cumprod(accepted.astype(jnp.int32))
            ).astype(jnp.int32)
            all_accepted = n_accepted == k

            # --------------------------------------------------------------
            # 4. Write accepted tokens to output buffer.
            # --------------------------------------------------------------
            def write_one(i, buf):
                return jax.lax.cond(
                    (i < n_accepted) & (n_gen + i < max_new_tokens),
                    lambda b: b.at[n_gen + i].set(draft_tokens[i]),
                    lambda b: b,
                    operand=buf,
                )

            out_buf    = jax.lax.fori_loop(0, k, write_one, out_buf)
            n_gen_after = jnp.minimum(n_gen + n_accepted, max_new_tokens)

            # --------------------------------------------------------------
            # 5. Bonus token from the adjusted distribution.
            #    When all k accepted: sample from target at position k-1.
            #    When some rejected: sample from max(target - draft, 0) at
            #                        the first rejected position.
            # --------------------------------------------------------------
            bonus_idx = jnp.minimum(
                jnp.where(all_accepted, k - 1, n_accepted), k - 1
            )
            t_at  = jax.lax.dynamic_slice(target_probs, (bonus_idx, 0), (1, vocab_size))[0]
            d_at  = jax.lax.dynamic_slice(draft_probs,  (bonus_idx, 0), (1, vocab_size))[0]

            adj     = jnp.maximum(t_at - d_at, 0.0)
            adj_sum = jnp.sum(adj)
            adj     = jnp.where(adj_sum > 0, adj / adj_sum, t_at)

            key, sub2  = jax.random.split(key)
            bonus_tok  = jax.random.categorical(sub2, jnp.log(adj + 1e-8)).astype(jnp.int32)

            # --------------------------------------------------------------
            # 6. Branch: all accepted vs some rejected.
            #    Both branches receive the same operand pytree.
            # --------------------------------------------------------------
            def all_acc_fn(args):
                dc, tc, ng, pos, buf = args
                # No bonus token; last draft token seeds the next round.
                return dc, tc, draft_tokens[k - 1], pos + k, ng, buf

            def some_rej_fn(args):
                dc, tc, ng, pos, buf = args
                bpos      = pos + n_accepted
                bonus_arr = bonus_tok.reshape(1, 1).astype(jnp.int32)
                # Feed bonus token through both caches at the rejection position.
                _, new_dc = draft( bonus_arr, dc, pos=bpos)
                _, new_tc = target(bonus_arr, tc, pos=bpos)
                # Append bonus token to output if within budget.
                new_buf = jax.lax.cond(
                    ng < max_new_tokens,
                    lambda b: b.at[ng].set(bonus_tok),
                    lambda b: b,
                    operand=buf,
                )
                new_ng = jnp.minimum(ng + 1, max_new_tokens)
                return new_dc, new_tc, bonus_tok, bpos + 1, new_ng, new_buf

            d_cache_fin, t_cache_fin, last_tok_fin, cur_pos_fin, n_gen_fin, out_buf = (
                jax.lax.cond(
                    all_accepted,
                    all_acc_fn,
                    some_rej_fn,
                    operand=(d_cache_new, t_cache_new, n_gen_after, cur_pos, out_buf),
                )
            )

            # EOS check over the accepted slice and the bonus token.
            eos_in_accepted = jnp.any((draft_tokens == eos_id) & (arange_k < n_accepted))
            bonus_is_eos    = (~all_accepted) & (bonus_tok == eos_id)
            done_new        = done | eos_in_accepted | bonus_is_eos | (n_gen_fin >= max_new_tokens)

            return (
                out_buf, n_gen_fin, last_tok_fin, cur_pos_fin,
                d_cache_fin, t_cache_fin, key, done_new,
                rounds + 1, acc_total + n_accepted, prop_total + k,
            )

        final = jax.lax.while_loop(cond_fn, body_fn, init_state)
        out_buf, n_gen, _, _, _, _, _, _, rounds, acc_total, prop_total = final
        return out_buf, n_gen, rounds, acc_total, prop_total

    out_buf, n_gen, rounds, acc_total, prop_total = _compiled_loop(
        draft_cache, target_cache, first_token, start_pos, prng_key
    )
    out_buf.block_until_ready()

    n_gen      = int(n_gen)
    generated  = [int(x) for x in out_buf[:n_gen]]
    all_ids    = list(prompt_ids) + generated
    return SpeculativeResult(
        prompt_ids=prompt_ids,
        generated_ids=generated,
        all_ids=all_ids,
        decoded_generated_text=tokenizer.decode(generated),
        decoded_all_text=tokenizer.decode(all_ids),
        rounds=int(rounds),
        accepted_tokens=int(acc_total),
        proposed_tokens=int(prop_total),
    )
