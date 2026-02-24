# XLA-Sharded

Built a multi-stage JAX/Flax NNX inference system with tensor-parallel sharding, static KV-cache design, XLA while-loop decoding, speculative draft/target verification, and benchmark instrumentation.

XLA-Sharded is an educational ML systems project that shows how to evolve a transformer inference stack from a simple single-device baseline into a speculative decoding pipeline with sharding and benchmarking.

## What This Project Demonstrates

- Llama-style transformer implementation in Flax NNX (RoPE, RMSNorm, GQA, SwiGLU)
- Two-model speculative decoding architecture (draft + target)
- Static pre-allocated KV-cache compatible with JIT and looped decode
- Single-device generation in two styles:
- `naive`: Python loop with a jitted decode step
- `xla`: decode loop lowered into `jax.lax.while_loop`
- Sharding primitives and partition spec utilities for TP=2 setups
- Lightweight benchmark reporting for throughput and acceptance metrics

## Repository Structure

```text
configs/
  model_config.py           # Target + Draft config dataclasses

models/
  layers.py                 # RMSNorm, RoPE, GQA, SwiGLU, TransformerBlock
  transformer.py            # Main Transformer module
  draft_model.py            # Draft model factory
  kv_cache.py               # Static KV cache init/update/read

engine/
  generate_naive.py         # Phase 2 baseline generation
  generate_xla.py           # Phase 3 XLA while-loop generation
  spec_dec.py               # Phase 4 speculative decode orchestration
  sharder.py                # Mesh + NamedSharding helper utilities

benchmark/
  throughput.py             # Warmup/run benchmarking helpers
  report.py                 # Inline table formatter
  profiler.py               # Optional JAX trace context manager

tests/
  test_model.py             # Phase 1 core model tests
  test_generation.py        # Naive + XLA generation tests
  test_kv_cache.py          # KV cache correctness/JIT update tests
  test_sharding.py          # Sharding placement tests (requires >=2 devices)
  test_spec_dec.py          # Speculative decoding behavior tests

demo.py                     # CLI entry point for all modes
```

## How The Project Works (Detailed)

### 1. Model Core (Phase 1)

The target and draft models share the same transformer building blocks in `models/layers.py`:

- `RMSNorm`: pre-norm stabilization
- `precompute_rope_freqs` + `apply_rope`: positional rotation
- `GQAAttention`: grouped-query attention with separate KV head count
- `SwiGLU`: gated feed-forward block
- `TransformerBlock`: attention + MLP residual block

`models/transformer.py` composes these into a full decoder-only model:

1. token embedding
2. stacked transformer blocks
3. final RMSNorm
4. linear LM head to logits

### 2. Static KV Cache

`models/kv_cache.py` pre-allocates one fixed-shape cache tensor:

`(n_layers, 2, batch, max_seq_len, n_kv_heads, head_dim)`

- Axis `2` stores K and V.
- `update()` writes with `jax.lax.dynamic_update_slice`, so dynamic decode positions work under JIT.
- `read()` returns full layer K/V tensors; attention masking handles valid position boundaries.

This design avoids dynamic reallocation and keeps decode steps compatible with JAX compilation.

### 3. Generation Modes

The CLI (`demo.py`) supports four modes.

#### `naive`

Implemented in `engine/generate_naive.py`:

1. Encode prompt with `DummyTokenizer`
2. Prefill model once on full prompt (initial KV population)
3. Repeatedly decode one token at a time:
4. Run jitted one-step forward
5. Take argmax token
6. Stop at EOS or max token limit

This is easy to reason about, but still has Python-loop overhead.

#### `xla`

Implemented in `engine/generate_xla.py`:

1. Same prefill stage as naive mode
2. Decode loop body is placed inside a compiled `jax.lax.while_loop`
3. Loop state carries output buffer, step index, current token, and KV cache

This reduces Python dispatch overhead and is the stepping stone toward fully compiled pipelines.

#### `speculative`

Implemented in `engine/spec_dec.py`:

1. Draft model proposes `k` tokens
2. Target model verifies proposal token-by-token
3. Acceptance probability uses `min(target_prob / draft_prob, 1)`
4. Accept contiguous prefix until first rejection
5. On rejection, sample one bonus token from adjusted target distribution
6. Rebuild caches from round start using accepted prefix (rollback behavior)
7. Continue until EOS or max token count

The current orchestration is explicit Python control flow around JAX forwards; it is designed for clarity and correctness testing.

### 4. Sharding Utilities

`engine/sharder.py` includes:

- `create_mesh()` for model-axis mesh construction
- `ShardSpecs` for partition spec conventions
- `shard_array()` and `shard_kv_cache()` placement helpers
- `create_single_device_sharding()` for draft placement utilities

These utilities support Phase 3+ experiments and tests; they do not force sharding inside model code.

### 5. Benchmarks and Reporting

`benchmark/throughput.py` provides warmup + repeated run timing.
`benchmark/report.py` formats metrics into an inline table.
`benchmark/profiler.py` exposes an optional `jax.profiler` trace context manager.

`demo.py --mode compare` runs all modes and prints:

- tokens/sec
- TTFT (currently placeholder-friendly measurement path)
- total time
- acceptance rate and rounds (for speculative mode)

## Installation

```bash
pip install -e .
pip install -e ".[dev]"
```

## Running

```bash
python demo.py --mode naive --prompt "The meaning" --max-tokens 50
python demo.py --mode xla --prompt "The meaning" --max-tokens 50
python demo.py --mode speculative --prompt "The meaning" --max-tokens 50 --k 5
python demo.py --mode compare --prompt "The meaning" --max-tokens 50 --warmup 1 --runs 3
```

## Testing

```bash
pytest -q
```

Optional targeted runs:

```bash
pytest -q tests/test_model.py
pytest -q tests/test_generation.py
pytest -q tests/test_spec_dec.py
pytest -q tests/test_sharding.py
```

## Practical Notes

- Default weights are random initialization, so decoded outputs are token IDs rather than meaningful text.
- `DummyTokenizer` is intentionally simple for pipeline validation.
- Sharding tests are skipped automatically when fewer than 2 JAX devices are available.
