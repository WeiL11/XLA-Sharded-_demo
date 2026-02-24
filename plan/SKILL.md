---
name: ml-systems
description: Use this skill for building high-performance ML inference or training systems with JAX/XLA, Flax NNX, multi-GPU sharding, speculative decoding, or any project requiring coordination of multiple XLA computation graphs across devices. Trigger whenever the user mentions JAX sharding, tensor parallelism, KV-cache, speculative decoding, XLA profiling, Flax NNX model architecture, jax.lax.while_loop generation, or multi-device inference engines. Also trigger for performance benchmarking of ML systems (tokens/sec, memory bandwidth utilization, arithmetic intensity). If the user is building any JAX-based ML system that touches distributed computation, device placement, or inference optimization, use this skill.
---

# ML Systems Skill — JAX/XLA Multi-GPU Inference Engineering

This skill encodes best practices for building production-grade ML inference systems with JAX, XLA, and Flax NNX. It covers model architecture, multi-device sharding, speculative decoding, static KV-cache management, and performance profiling.

## When to Use

- Building a Transformer inference engine in JAX
- Sharding model weights across multiple GPUs with `jax.sharding`
- Implementing speculative decoding (draft + target model coordination)
- Creating zero-Python-overhead generation loops with `jax.lax.while_loop`
- Profiling XLA traces, HLO graphs, or measuring tokens/sec and MBU
- Any project requiring two distinct XLA computation graphs to coordinate

---

## 1. Architecture Principles

### Separate Models from Engine from Benchmarks

The three concerns — model definition, execution strategy, and measurement — must live in separate modules. Models define math. The engine decides where and how that math runs. Benchmarks observe without interfering.

```
models/        → Pure Flax NNX modules. No sharding logic. No generation loops.
engine/        → Sharding, device placement, generation loops, spec-dec orchestration.
benchmark/     → Profiling, timing, reporting. Never imported by engine or models.
```

Why: A model that knows about its sharding is a model you can't test on a single CPU. An engine that contains model architecture is an engine you can't swap models in.

### Use Flax NNX, Not Linen

Flax NNX is the current recommended API. It uses regular Python objects with stateful semantics — no `init`/`apply` ceremony.

```python
from flax import nnx

class TransformerBlock(nnx.Module):
    def __init__(self, d_model, n_heads, rngs: nnx.Rngs):
        self.attn = MultiHeadAttention(d_model, n_heads, rngs=rngs)
        self.mlp = SwiGLU(d_model, rngs=rngs)
        self.norm1 = nnx.RMSNorm(d_model, rngs=rngs)
        self.norm2 = nnx.RMSNorm(d_model, rngs=rngs)

    def __call__(self, x, kv_cache, pos):
        h = x + self.attn(self.norm1(x), kv_cache, pos)
        out = h + self.mlp(self.norm2(h))
        return out
```

### Config Dataclass — Pin Everything

Every architectural decision should live in one config object. Never hardcode dims, layer counts, or precision scattered across files.

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4       # GQA: fewer KV heads than Q heads
    vocab_size: int = 32000
    max_seq_len: int = 2048
    mlp_ratio: float = 2.667  # SwiGLU: hidden = int(d_model * mlp_ratio * 2/3) * 2
    dtype: str = "bfloat16"
```

---

## 2. Static KV-Cache

The KV-cache must be pre-allocated with fixed shapes to work inside `jax.jit` and `jax.lax.while_loop`. JAX requires static shapes at trace time.

### Structure

```python
# Shape: (n_layers, 2, batch, max_seq_len, n_kv_heads, head_dim)
# The "2" is for K and V
kv_cache = jnp.zeros(
    (config.n_layers, 2, batch_size, config.max_seq_len, config.n_kv_heads, head_dim),
    dtype=jnp.bfloat16
)
```

### Update Pattern

Use `.at[].set()` for in-place updates inside JIT:

```python
def update_kv(cache, layer_idx, pos, new_k, new_v):
    cache = cache.at[layer_idx, 0, :, pos, :, :].set(new_k)
    cache = cache.at[layer_idx, 1, :, pos, :, :].set(new_v)
    return cache
```

### Sharding the Cache

The KV-cache must be sharded consistently with the attention weights. If you shard attention heads across the `"model"` axis, the cache's head dimension must also be sharded:

```python
kv_sharding = NamedSharding(mesh, P(None, None, None, None, "model", None))
#                                    layers K/V batch  seq   heads  dim
```

---

## 3. Distributed Sharding (Tensor Parallelism)

### Mesh Setup

```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = jax.devices()[:n_gpus]
mesh = Mesh(devices, axis_names=("model",))
```

### Sharding Strategy for Inference

For tensor-parallel inference on 2+ GPUs, shard weight matrices so that the most compute-heavy ops (attention projections, MLP) split work across devices:

| Weight | Sharding | Reasoning |
|--------|----------|-----------|
| `W_q, W_k, W_v` | `P(None, "model")` | Split attention heads across GPUs |
| `W_o` (attn output) | `P("model", None)` | Gather from split heads |
| `W_gate, W_up` (MLP) | `P(None, "model")` | Split intermediate dim |
| `W_down` (MLP) | `P("model", None)` | Gather back |
| `embed_tokens` | `P("model", None)` | Vocab-parallel embedding |
| `lm_head` | `P(None, "model")` | Split logit computation |

After each "gather" weight (W_o, W_down), XLA inserts an `all_reduce` automatically. You don't write it — the sharding spec implies it.

### Placing Weights

```python
import jax

def shard_params(params, mesh, shard_map):
    """Place each param array on the correct devices per shard_map."""
    def _shard(path, arr):
        spec = shard_map.get(path, P())  # default: replicated
        sharding = NamedSharding(mesh, spec)
        return jax.device_put(arr, sharding)
    return jax.tree.map_with_path(_shard, params)
```

### Verification

Always verify sharding visually during development:

```python
jax.debug.visualize_array_sharding(params['layers'][0]['attn']['wq'])
```

---

## 4. Speculative Decoding (Draft + Target)

Speculative decoding coordinates two models: a fast draft model proposes tokens, and the larger target model verifies them in parallel. This is the key technique for improving inference throughput without changing output quality.

### Architecture

- **Target model**: Full-size Transformer (e.g., 12 layers), sharded across GPUs
- **Draft model**: Tiny Transformer (e.g., 2 layers), typically on a single GPU
- Both models share the same vocabulary and tokenizer
- Each model has its own KV-cache

### The Spec-Dec Loop

```
repeat until done:
    1. Draft model generates K tokens autoregressively
    2. Target model runs ONE forward pass on all K draft tokens (parallel verify)
    3. Compare draft vs. target probabilities at each position
    4. Accept tokens while target agrees (rejection sampling)
    5. On first rejection: discard remaining draft tokens
    6. Sample one bonus token from adjusted target distribution
    7. Roll back draft model's KV-cache to the accepted prefix
```

### Implementation Constraints in JAX

The entire spec-dec loop should run inside `jax.lax.while_loop` for zero Python overhead. This means:

- No Python `if/else` — use `jax.lax.cond` or `jnp.where`
- No Python `for` — use `jax.lax.fori_loop` or `jax.lax.scan`
- All arrays must have static shapes (pad to max, use masks)
- The rejection/acceptance logic must be expressed as array operations

### Rejection Sampling

```python
def rejection_sample(draft_probs, target_probs, draft_tokens, key):
    """
    For each position i:
      - accept if uniform() < target_p(token_i) / draft_p(token_i)
      - reject otherwise; all subsequent tokens are also rejected
    """
    ratios = target_probs / jnp.maximum(draft_probs, 1e-8)
    accept_probs = jnp.minimum(ratios, 1.0)
    uniforms = jax.random.uniform(key, shape=accept_probs.shape)
    accepted = uniforms < accept_probs
    # First rejection kills everything after it
    accepted_cumulative = jnp.cumprod(accepted.astype(jnp.int32))
    n_accepted = jnp.sum(accepted_cumulative)
    return n_accepted
```

### KV-Cache Rollback

After rejection, the draft model's cache positions beyond `n_accepted` are stale. You don't need to zero them — just track a `cache_position` counter and overwrite on the next draft round.

---

## 5. Generation Loops

### Naive Baseline (Single GPU, Python Loop)

```python
@jax.jit
def forward_step(params, token, kv_cache, pos):
    logits, new_cache = model(params, token, kv_cache, pos)
    return logits, new_cache

def generate_naive(params, prompt_tokens, max_tokens):
    """Python loop — each iteration round-trips to Python."""
    kv_cache = init_cache()
    tokens = list(prompt_tokens)
    for i in range(max_tokens):
        logits, kv_cache = forward_step(params, tokens[-1], kv_cache, len(tokens)-1)
        next_token = int(jnp.argmax(logits[-1]))
        tokens.append(next_token)
        if next_token == eos_id:
            break
    return tokens
```

### XLA-Optimized (Zero Python Overhead)

```python
@jax.jit
def generate_xla(params, prompt_tokens, max_new_tokens):
    def cond_fn(state):
        tokens, pos, kv, done = state
        return (~done) & (pos < len(tokens))

    def body_fn(state):
        tokens, pos, kv, done = state
        logits, kv = model(params, tokens[pos-1:pos], kv, pos)
        next_tok = jnp.argmax(logits[-1], axis=-1)
        tokens = tokens.at[pos].set(next_tok)
        done = (next_tok == eos_id)
        return (tokens, pos + 1, kv, done)

    init = (padded_tokens, prompt_len, kv_cache, jnp.bool_(False))
    return jax.lax.while_loop(cond_fn, body_fn, init)
```

---

## 6. Profiling and Benchmarks

### Tokens/sec

```python
import time

def measure_throughput(generate_fn, params, prompt, n_runs=5):
    # Warmup (first run compiles the XLA graph)
    _ = generate_fn(params, prompt)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        output = generate_fn(params, prompt)
        jax.block_until_ready(output)  # Critical: JAX is async
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    tokens_generated = count_tokens(output) - len(prompt)
    avg_time = sum(times) / len(times)
    return tokens_generated / avg_time
```

The `jax.block_until_ready()` call is essential — without it you're measuring dispatch time, not compute time.

### Memory Bandwidth Utilization (MBU)

```
MBU = actual_bandwidth / peak_bandwidth

actual_bandwidth = bytes_moved_per_token / time_per_token
bytes_moved_per_token ≈ 2 * n_params (for bf16) + kv_cache_bytes_per_step
```

Query peak bandwidth via `nvidia-smi` or hardcode per GPU model:
- A100 80GB: 2039 GB/s
- H100 80GB: 3352 GB/s
- RTX 4090: 1008 GB/s

### XLA Profiling

```python
with jax.profiler.trace("/tmp/xla_trace"):
    output = generate_fn(params, prompt)
    jax.block_until_ready(output)
# View with: tensorboard --logdir /tmp/xla_trace
```

For HLO text (compiler IR):
```python
lowered = jax.jit(generate_fn).lower(params, prompt)
print(lowered.compile().as_text())  # XLA HLO
```

---

## 7. Common Pitfalls

1. **Dynamic shapes inside `jax.lax.while_loop`**: All arrays in the loop state must have static shapes. Pre-allocate and use position indices.

2. **Forgetting `jax.block_until_ready()`**: JAX dispatches asynchronously. Timing without blocking measures Python dispatch, not GPU compute.

3. **Sharding mismatch between weights and cache**: If attention weights are sharded on the head dimension, the KV-cache must match. Mismatches cause silent all-gathers that kill performance.

4. **Draft model on wrong device**: If the draft model lives on GPU:0 but the target's input embeddings are on GPU:1, you get unnecessary cross-device transfers. Pin the draft model explicitly.

5. **Not warming up JIT**: The first call to a JIT-compiled function includes compilation time. Always run one warmup pass before benchmarking.

6. **Using Python control flow in spec-dec**: `if n_accepted > 3:` inside a JIT-traced function becomes a static branch based on the trace-time value. Use `jax.lax.cond` instead.
