# XLA-Sharded: Multi-GPU Speculative Inference Engine

## Final Plan v3 — All Decisions Locked

---

## 0. Locked Decisions

| # | Decision | Value | Rationale |
|---|----------|-------|-----------|
| 1 | Weights | **Random Init** | Proves pipeline correctness. Benchmarks valid regardless. |
| 2 | Tokenizer | **V1: Dummy integer IDs → V2: Gemma SentencePiece** | Decouples tokenizer from model dev. V1 ships fast. |
| 3 | Speculation K | **Fixed K=5** | Simple. Reliable. No adaptive complexity. |
| 4 | Draft Placement | **GPU 0 only, unsharded** | ~50 MB. Cross-device transfer of 5 token IDs is 20 bytes. |
| 5 | Profile Output | **Inline terminal: Tokens/sec, TTFT, Acceptance Rate** | No TensorBoard dependency. Instant feedback. |
| 6 | Flax API | **Flax NNX** | Modern. Stateful Python objects. No init/apply. |
| 7 | KV-Cache | **Static pre-allocated** | Required for `jax.lax.while_loop` (static shapes). |
| 8 | Sharding | **Tensor Parallelism (TP=2)** via `jax.sharding.NamedSharding` | Splits heads + MLP across 2 GPUs. |
| 9 | Draft Model | **Separate 2-layer Tiny-Transformer** | Two distinct XLA graphs. Same vocab. |
| 10 | Architecture | **Llama-style** (RoPE, RMSNorm, GQA, SwiGLU) | Modern. Demonstrates 2025 best practices. |

**Zero open questions remain.**

---

## 1. The Core Proof: Two XLA Graphs

This project demonstrates coordination of two independently compiled XLA programs:

```
XLA Graph #1 — Draft Model
  Compiled once via jax.jit(draft_generate)
  2-layer Tiny-Transformer
  Lives entirely on GPU 0 (unsharded)
  Owns its own KV-cache (~8 MB)
  Generates K=5 tokens autoregressively (jax.lax.fori_loop)
          │
          │ draft_tokens[0..5], draft_probs[0..5]
          │ (on-device transfer, zero host roundtrip)
          ▼
XLA Graph #2 — Target Model
  Compiled once via jax.jit(target_verify)
  12-layer Llama, tensor-parallel across GPU 0 + GPU 1
  Each GPU holds half of every weight matrix
  Owns its own KV-cache (~384 MB total, split across GPUs)
  ONE forward pass on all K=5 draft tokens (parallel verify)
          │
          ▼
  Rejection Sampling → Accept n ≤ 5 → Rollback caches → Loop
```

---

## 2. System Structure

```
xla-sharded/
├── pyproject.toml
├── README.md
│
├── configs/
│   └── model_config.py           # ModelConfig + DraftConfig dataclasses
│
├── models/
│   ├── layers.py                 # RMSNorm, RoPE, GQA-Attention, SwiGLU
│   ├── transformer.py            # Target: 12L Llama-style (Flax NNX)
│   ├── draft_model.py            # Draft:  2L Tiny-Transformer (Flax NNX)
│   └── kv_cache.py               # Static KV-Cache with sharding support
│
├── engine/
│   ├── sharder.py                # Mesh, NamedSharding, weight placement
│   ├── generate_naive.py         # Baseline: 1-GPU, jit, Python for-loop
│   ├── generate_xla.py           # Optimized: jax.lax.while_loop, zero overhead
│   └── spec_dec.py               # Draft→Verify→Accept/Reject orchestrator
│
├── benchmark/
│   ├── profiler.py               # jax.profiler.trace(), HLO text dump (optional)
│   ├── throughput.py             # Tokens/sec, TTFT, MBU, acceptance rate
│   └── report.py                 # Rich-formatted inline terminal output
│
├── tokenizer/
│   └── tokenizer.py              # V1: DummyTokenizer / V2: Gemma SentencePiece
│
├── demo.py                       # CLI entry point (typer)
│
└── tests/
    ├── test_model.py             # Forward pass shapes, dtype, output dims
    ├── test_sharding.py          # Weight placement verification
    ├── test_kv_cache.py          # Update + rollback correctness
    ├── test_spec_dec.py          # Acceptance logic, output equivalence
    └── test_generation.py        # Naive vs. XLA output match
```

---

## 3. Model Specifications

### Target Model (12-layer Llama)

```python
@dataclass
class ModelConfig:
    n_layers:    int   = 12
    d_model:     int   = 768
    n_heads:     int   = 12       # query heads
    n_kv_heads:  int   = 4        # GQA: 3 query heads per KV head
    vocab_size:  int   = 32_000
    max_seq_len: int   = 2048
    mlp_hidden:  int   = 2048     # SwiGLU: gate+up project to this dim
    dtype:       str   = "bfloat16"
    rope_theta:  float = 10_000.0
```

**Parameter count:** ~125M (bf16 → ~250 MB total, ~125 MB per GPU shard)

### Draft Model (2-layer Tiny-Transformer)

```python
@dataclass
class DraftConfig:
    n_layers:    int   = 2
    d_model:     int   = 768      # same as target (shares embed/lm_head dims)
    n_heads:     int   = 12
    n_kv_heads:  int   = 4
    vocab_size:  int   = 32_000   # MUST match target
    max_seq_len: int   = 2048
    mlp_hidden:  int   = 2048
    dtype:       str   = "bfloat16"
    rope_theta:  float = 10_000.0
```

**Parameter count:** ~25M (bf16 → ~50 MB, all on GPU 0)

Both models share `vocab_size=32,000` and `d_model=768` so the same embedding and lm_head dimensions are compatible with the shared tokenizer.

---

## 4. Sharding Map (Tensor Parallelism)

```
Mesh: 2 devices on axis "model"
  GPU 0 = mesh["model"][0]
  GPU 1 = mesh["model"][1]
```

| Weight | PartitionSpec | Effect |
|--------|--------------|--------|
| `W_q` (768 → 768) | `P(None, "model")` | Split heads: GPU0 gets heads 0-5, GPU1 gets 6-11 |
| `W_k` (768 → 256) | `P(None, "model")` | Split KV heads: GPU0 gets 0-1, GPU1 gets 2-3 |
| `W_v` (768 → 256) | `P(None, "model")` | Same as W_k |
| `W_o` (768 → 768) | `P("model", None)` | Gather (all_reduce after matmul) |
| `W_gate` (768 → 2048) | `P(None, "model")` | Split intermediate dim |
| `W_up` (768 → 2048) | `P(None, "model")` | Split intermediate dim |
| `W_down` (2048 → 768) | `P("model", None)` | Gather (all_reduce after matmul) |
| `embed_tokens` (32k → 768) | `P("model", None)` | Vocab-parallel |
| `lm_head` (768 → 32k) | `P(None, "model")` | Split logit computation |
| `RMSNorm` scales | `P()` | Replicated |
| `RoPE` freq table | `P()` | Replicated |

**KV-Cache sharding:** `P(None, None, None, None, "model", None)` — heads dimension split, matching W_k/W_v.

**Draft model:** No sharding. All weights on GPU 0 via `SingleDeviceSharding(devices[0])`.

---

## 5. KV-Cache Design

### Shape

```python
# Target cache — sharded across GPUs
target_cache = jnp.zeros(
    (n_layers, 2, batch, max_seq_len, n_kv_heads, head_dim),
    #  12      K/V  1      2048        4           64
    dtype=jnp.bfloat16
)
# After sharding: each GPU holds (12, 2, 1, 2048, 2, 64)

# Draft cache — unsharded, GPU 0 only
draft_cache = jnp.zeros(
    (2, 2, batch, max_seq_len, n_kv_heads, head_dim),
    #  2  K/V  1     2048        4          64
    dtype=jnp.bfloat16
)
```

### Operations

```python
# Write (inside jax.lax body)
cache = cache.at[layer, 0, :, pos, :, :].set(new_k)  # K
cache = cache.at[layer, 1, :, pos, :, :].set(new_v)  # V

# Read (attention)
k = cache[layer, 0, :, :pos+1, :, :]  # all K up to current position
v = cache[layer, 1, :, :pos+1, :, :]

# Rollback (after rejection) — just reset the position counter
# No need to zero stale entries; they'll be overwritten
cache_pos = n_accepted  # next draft round starts writing from here
```

---

## 6. Speculative Decoding Protocol

```
INIT:
  Prefill prompt through both draft and target models.
  Initialize both KV-caches with prompt's K/V states.
  Record TTFT (time from prompt submission to first token available).

LOOP (jax.lax.while_loop):
  ① DRAFT:   draft_model generates 5 tokens autoregressively
              using jax.lax.fori_loop(0, 5, draft_step, state)
              → draft_tokens[0..5], draft_probs[0..5]

  ② SEND:    draft_tokens transferred to target model
              (on-device, both models share GPU 0 address space;
               GPU 1's shard receives via implicit XLA data movement)

  ③ VERIFY:  target_model runs ONE forward pass on all 5 tokens
              XLA Graph #2 executes, with all_reduce across GPUs
              → target_probs[0..5]

  ④ ACCEPT:  Rejection sampling (vectorized, no Python branching):
              ratios = target_probs / max(draft_probs, 1e-8)
              accept_probs = min(ratios, 1.0)
              accepted = uniform_samples < accept_probs
              n_accepted = cumulative_product(accepted).sum()

  ⑤ OUTPUT:  Append accepted_tokens[0..n] to output buffer.
              Print to terminal.

  ⑥ ROLLBACK: Reset draft_cache_pos = prompt_len + total_accepted
               Reset target_cache_pos = same
               (stale entries will be overwritten next round)

  ⑦ BONUS:   Sample 1 extra token from:
              adjusted_dist = max(0, target_probs[n] - draft_probs[n])
              normalized and sampled. Append to output.

  CHECK:     If any accepted token == EOS or total_generated >= max_tokens → STOP
```

**Expected acceptance rate (random init):** Since both models are randomly initialized, draft and target distributions will diverge significantly. Expect ~1-2 accepted tokens per round on average. This is fine — the goal is to prove the coordination mechanism, not achieve high acceptance rates. With pretrained weights (V2), acceptance would be higher.

---

## 7. Benchmark Design

### Modes (selectable via CLI)

```
python demo.py --mode naive       # Single-GPU, Python loop, jit per step
python demo.py --mode xla         # Single-GPU, jax.lax.while_loop
python demo.py --mode speculative # Multi-GPU, spec-dec with K=5
python demo.py --mode compare     # Runs all 3, prints comparison table
```

### Metrics (printed inline after generation)

```
┌─────────────────────────────────────────────────────────────────┐
│  XLA-Sharded Benchmark Report                                   │
├──────────────────┬──────────┬──────────┬───────────────────────┤
│ Metric           │ Naive    │ XLA      │ Speculative (K=5)     │
├──────────────────┼──────────┼──────────┼───────────────────────┤
│ Tokens/sec       │ 42.3     │ 128.7    │ 187.4                 │
│ TTFT (ms)        │ 312      │ 289      │ 445                   │
│ Total time (s)   │ 4.73     │ 1.55     │ 1.07                  │
│ Tokens generated │ 200      │ 200      │ 200                   │
│ Accept rate      │ —        │ —        │ 3.2 / 5 (64%)         │
│ Rounds           │ —        │ —        │ 48                    │
│ Speedup vs naive │ 1.0×     │ 3.0×     │ 4.4×                  │
└──────────────────┴──────────┴──────────┴───────────────────────┘
```

### Measurement Rules

1. **Warmup:** Always run 1 generation pass before timing (first call compiles XLA graph).
2. **`jax.block_until_ready()`:** Called after every timed generation. JAX dispatches async — without this you measure dispatch, not compute.
3. **N=5 runs:** Average over 5 runs for stable numbers. Report mean ± std.
4. **TTFT:** Time from `generate()` call to first token being written to the output buffer.

---

## 8. Tokenizer Strategy

### V1: DummyTokenizer (ships with Phase 1)

```python
class DummyTokenizer:
    """Integer token IDs. No real vocabulary. For pipeline testing."""
    vocab_size: int = 32_000
    eos_id: int = 2
    bos_id: int = 1

    def encode(self, text: str) -> list[int]:
        # Just hash characters to token IDs
        return [self.bos_id] + [hash(c) % self.vocab_size for c in text]

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"[{i}]" for i in ids if i not in (self.bos_id, self.eos_id))
```

### V2: Gemma SentencePiece (Phase 5 polish)

```python
from transformers import AutoTokenizer

class GemmaTokenizer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained("google/gemma-2b")
        self.vocab_size = self.tok.vocab_size  # 256,000 — update ModelConfig!
        self.eos_id = self.tok.eos_token_id
        self.bos_id = self.tok.bos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)
```

**Note:** Switching to V2 requires updating `vocab_size` in both ModelConfig and DraftConfig. The tokenizer wrapper interface is identical, so nothing else changes.

---

## 9. Implementation Phases

### Phase 1: Models + Dummy Tokenizer (Days 1-3)
- [ ] `configs/model_config.py` — ModelConfig + DraftConfig dataclasses
- [ ] `models/layers.py` — RMSNorm, RoPE, GQA multi-head attention, SwiGLU MLP
- [ ] `models/transformer.py` — 12-layer target model (Flax NNX)
- [ ] `models/draft_model.py` — 2-layer draft model (Flax NNX)
- [ ] `models/kv_cache.py` — static pre-allocated cache, `.update()` + `.read()`
- [ ] `tokenizer/tokenizer.py` — DummyTokenizer (V1)
- [ ] Verify: forward pass runs, shapes are correct, dtypes are bf16

### Phase 2: Single-GPU Baseline (Days 4-5)
- [ ] `engine/generate_naive.py` — single-GPU, `@jax.jit` per step, Python for-loop
- [ ] Wire up: config → model → tokenizer → generate → decode → print
- [ ] Verify: generates 200 tokens, dummy-decoded output is printed

### Phase 3: Sharding + XLA Generation (Days 6-8)
- [ ] `engine/sharder.py` — `Mesh`, `NamedSharding`, `shard_params()`
- [ ] Apply sharding to target model weights per the sharding map (Section 4)
- [ ] `jax.debug.visualize_array_sharding()` — verify every weight
- [ ] Shard KV-cache heads dimension consistently
- [ ] Place draft model on GPU 0 via `SingleDeviceSharding`
- [ ] `engine/generate_xla.py` — `jax.lax.while_loop`, fully on-device
- [ ] Verify: output token sequence matches naive baseline exactly

### Phase 4: Speculative Decoding (Days 9-12)
- [ ] `engine/spec_dec.py` — the full protocol (Section 6)
  - [ ] Draft generation inner loop (`jax.lax.fori_loop`)
  - [ ] Parallel verification (single target forward pass on K tokens)
  - [ ] Vectorized rejection sampling (no Python branching)
  - [ ] KV-cache rollback (position counter reset)
  - [ ] Bonus token sampling from adjusted distribution
  - [ ] Outer loop (`jax.lax.while_loop`)
- [ ] Verify: with random init, tokens are generated, acceptance rate is printed

### Phase 5: Benchmark + CLI + Polish (Days 13-16)
- [ ] `benchmark/throughput.py` — timing harness with warmup + `block_until_ready`
- [ ] `benchmark/report.py` — Rich-formatted comparison table
- [ ] `benchmark/profiler.py` — optional HLO dump utility
- [ ] `demo.py` — typer CLI with `--mode {naive,xla,speculative,compare}`
- [ ] `tests/` — all 5 test files
- [ ] README.md — setup, architecture diagram reference, sample CLI output
- [ ] Optional: swap DummyTokenizer → Gemma SentencePiece (V2)

---

## 10. Dependencies

```toml
[project]
name = "xla-sharded"
requires-python = ">=3.10"
dependencies = [
    "jax[cuda12]>=0.4.35",
    "flax>=0.10.0",
    "jaxlib>=0.4.35",
    "numpy",
    "rich>=13.0",
    "typer>=0.9",
]

[project.optional-dependencies]
gemma = ["transformers", "sentencepiece"]   # V2 tokenizer
dev = ["pytest", "pytest-xdist"]
profile = ["tensorboard", "tensorboard-plugin-profile"]
```

### Hardware Requirements

- **Minimum:** 2× NVIDIA GPU with ≥8 GB VRAM each (RTX 3070+)
- **Recommended:** 2× A100/H100 with NVLink
- **Development fallback:** `XLA_FLAGS=--xla_force_host_platform_device_count=2` on CPU
- CUDA 12.x + cuDNN 9+

---

## 11. CLI Interface

```
$ python demo.py --help

 Usage: demo.py [OPTIONS]

 XLA-Sharded: Multi-GPU Speculative Inference Engine

╭─ Options ─────────────────────────────────────────────────────╮
│ --prompt        TEXT    Input prompt [default: "The meaning"]  │
│ --max-tokens    INT     Max tokens to generate [default: 200] │
│ --mode          TEXT    naive|xla|speculative|compare          │
│ --k             INT     Speculation length [default: 5]        │
│ --seed          INT     Random seed [default: 42]              │
│ --warmup        INT     Warmup runs [default: 1]               │
│ --runs          INT     Benchmark runs [default: 5]            │
│ --help                  Show this help message                  │
╰───────────────────────────────────────────────────────────────╯
```

---

## 12. Success Criteria

A completed project demonstrates all of the following:

| # | Criterion | Verified By |
|---|-----------|-------------|
| 1 | Flax NNX Transformer generates tokens | `test_model.py` |
| 2 | Target weights sharded TP=2 across GPUs | `test_sharding.py` + `visualize_array_sharding` |
| 3 | Draft model lives on GPU 0, unsharded | `test_sharding.py` |
| 4 | KV-cache statically allocated, shard-aware | `test_kv_cache.py` |
| 5 | `jax.lax.while_loop` generation matches naive | `test_generation.py` |
| 6 | Spec-dec orchestrates two XLA graphs | `test_spec_dec.py` |
| 7 | Rejection sampling + rollback works | `test_spec_dec.py` |
| 8 | CLI prints Tokens/sec, TTFT, Acceptance Rate | Manual run of `demo.py --mode compare` |
| 9 | Speculative mode is faster than naive mode | Benchmark report |
