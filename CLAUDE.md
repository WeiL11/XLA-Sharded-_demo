# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**XLA-Sharded** is a multi-GPU speculative inference engine built with JAX, XLA, and Flax NNX. It demonstrates coordination of two independently compiled XLA computation graphs (draft + target model) for speculative decoding across 2 GPUs.

This project is currently in the **planning/implementation phase**. The plan documents live in `plan/`.

## Target File Structure

```
xla-sharded/
├── pyproject.toml
├── configs/model_config.py        # ModelConfig + DraftConfig dataclasses
├── models/
│   ├── layers.py                  # RMSNorm, RoPE, GQA-Attention, SwiGLU
│   ├── transformer.py             # Target: 12L Llama-style (Flax NNX)
│   ├── draft_model.py             # Draft: 2L Tiny-Transformer (Flax NNX)
│   └── kv_cache.py                # Static KV-Cache with sharding support
├── engine/
│   ├── sharder.py                 # Mesh, NamedSharding, weight placement
│   ├── generate_naive.py          # Baseline: 1-GPU, jit, Python for-loop
│   ├── generate_xla.py            # Optimized: jax.lax.while_loop
│   └── spec_dec.py                # Draft→Verify→Accept/Reject orchestrator
├── benchmark/
│   ├── profiler.py                # jax.profiler.trace(), HLO dump
│   ├── throughput.py              # Tokens/sec, TTFT, MBU, acceptance rate
│   └── report.py                  # Rich-formatted terminal output
├── tokenizer/tokenizer.py         # DummyTokenizer (V1) / Gemma SentencePiece (V2)
├── demo.py                        # CLI entry point (typer)
└── tests/
    ├── test_model.py
    ├── test_sharding.py
    ├── test_kv_cache.py
    ├── test_spec_dec.py
    └── test_generation.py
```

## Commands

```bash
# Run the demo
python demo.py --mode naive
python demo.py --mode xla
python demo.py --mode speculative
python demo.py --mode compare

# CLI options
python demo.py --prompt "The meaning" --max-tokens 200 --mode speculative --k 5 --seed 42 --warmup 1 --runs 5

# Run tests
pytest tests/
pytest tests/test_model.py         # single test file
pytest -x tests/                   # stop on first failure

# Simulate 2 devices on CPU (development without GPU)
XLA_FLAGS=--xla_force_host_platform_device_count=2 python demo.py --mode speculative

# XLA profiling (view with: tensorboard --logdir /tmp/xla_trace)
# See benchmark/profiler.py
```

## Architecture

### Separation of Concerns
- **`models/`**: Pure Flax NNX modules. No sharding logic. No generation loops.
- **`engine/`**: Sharding, device placement, generation loops, spec-dec orchestration.
- **`benchmark/`**: Profiling, timing, reporting. Never imported by engine or models.

### Two XLA Graphs
The core system compiles exactly two XLA programs:
1. **Draft model** (`jax.jit(draft_generate)`) — 2-layer Tiny-Transformer on GPU 0, unsharded, runs `jax.lax.fori_loop` to generate K=5 tokens
2. **Target model** (`jax.jit(target_verify)`) — 12-layer Llama, tensor-parallel across GPU 0+1, runs ONE forward pass to verify all 5 draft tokens

### Sharding Strategy (Tensor Parallelism, TP=2)
```python
mesh = Mesh(devices[:2], axis_names=("model",))
```
| Weight | PartitionSpec | Effect |
|--------|--------------|--------|
| `W_q, W_k, W_v` | `P(None, "model")` | Split heads across GPUs |
| `W_o, W_down` | `P("model", None)` | Gather (implicit all_reduce) |
| `W_gate, W_up` | `P(None, "model")` | Split intermediate dim |
| `embed_tokens` | `P("model", None)` | Vocab-parallel |
| `lm_head` | `P(None, "model")` | Split logit computation |
| RMSNorm, RoPE | `P()` | Replicated |

KV-cache sharding: `P(None, None, None, None, "model", None)` — heads dimension must match `W_k/W_v`.

Draft model: `SingleDeviceSharding(devices[0])` — all weights on GPU 0.

### Model Specs
- **Target**: 12 layers, d_model=768, 12Q/4KV heads (GQA), vocab=32k, ~125M params (~250 MB bf16, ~125 MB/GPU)
- **Draft**: 2 layers, same d_model/vocab/head dims, ~25M params (~50 MB bf16, GPU 0 only)

### Static KV-Cache
Shape: `(n_layers, 2, batch, max_seq_len, n_kv_heads, head_dim)` — pre-allocated with zeros. Updated via `.at[].set()` inside JIT. Rollback after rejection is just resetting the position counter; no zeroing needed.

### Speculative Decoding Loop (inside `jax.lax.while_loop`)
1. Draft generates K=5 tokens via `jax.lax.fori_loop`
2. Target verifies all 5 in one forward pass
3. Vectorized rejection sampling: `n_accepted = cumsum(uniform < min(target_p/draft_p, 1)).sum()`
4. Cache position reset to `n_accepted`; bonus token sampled from adjusted distribution
5. Repeat until EOS or max tokens

## Key Constraints

- **Static shapes required**: All arrays in `jax.lax.while_loop` / `fori_loop` must have static shapes. Pre-allocate and use position indices.
- **No Python control flow inside JIT**: Use `jax.lax.cond` instead of `if/else`, `jax.lax.fori_loop` instead of `for`.
- **Always call `jax.block_until_ready()`** after timed generations — JAX is async.
- **Always warmup**: First JIT call includes compilation. Run 1 warmup pass before benchmarking.
- **Sharding consistency**: KV-cache head dimension must match attention weight sharding spec or silent all-gathers degrade performance.

## Dependencies

```toml
dependencies = [
    "jax[cuda12]>=0.4.35",
    "flax>=0.10.0",
    "jaxlib>=0.4.35",
    "numpy",
    "rich>=13.0",
    "typer>=0.9",
]
# Optional: transformers + sentencepiece for Gemma tokenizer (V2)
# Optional: tensorboard + tensorboard-plugin-profile for XLA profiling
```

**Hardware**: 2× NVIDIA GPU ≥8 GB VRAM (CUDA 12.x + cuDNN 9+). CPU fallback: `XLA_FLAGS=--xla_force_host_platform_device_count=2`.

## Tokenizer

- **V1 (default)**: `DummyTokenizer` — hashes characters to integer IDs. No real vocabulary. For pipeline testing only.
- **V2 (optional)**: Gemma SentencePiece via `transformers`. Requires updating `vocab_size=256_000` in both `ModelConfig` and `DraftConfig`.
