# XLA-Sharded

Built a JAX/Flax NNX inference system with static KV-cache, XLA decode loops, speculative draft/target verification, and tensor-parallel sharding utilities.

XLA-Sharded is an ML systems demo focused on correctness and architecture clarity for modern inference pipelines.

## What It Does

- Llama-style transformer implementation in Flax NNX (RoPE, RMSNorm, GQA, SwiGLU)
- Static KV-cache that works with JIT and loop-based generation
- Three decode paths: `naive`, `xla`, `speculative`
- Sharding helpers (`Mesh`, `NamedSharding`, `PartitionSpec`)
- Benchmark/report modules for throughput and acceptance metrics

## Design Structure (Plan + Code)

```text
xla-sharded/
├── pyproject.toml
├── README.md
├── demo.py
│
configs/
│   └── model_config.py           # Target + draft configs
│
models/
│   ├── layers.py                 # RMSNorm/RoPE/GQA/SwiGLU/block
│   ├── transformer.py            # Main transformer
│   ├── draft_model.py            # Draft model factory
│   └── kv_cache.py               # Static KV cache
│
engine/
│   ├── generate_naive.py         # Python loop + jitted step
│   ├── generate_xla.py           # jax.lax.while_loop decode
│   ├── spec_dec.py               # Draft/target speculative decode
│   └── sharder.py                # Mesh/sharding helpers
│
benchmark/
│   ├── throughput.py             # Benchmark harness
│   ├── report.py                 # Report formatter
│   └── profiler.py               # Optional trace helper
│
tests/
│   ├── test_model.py
│   ├── test_generation.py
│   ├── test_kv_cache.py
│   ├── test_sharding.py
│   └── test_spec_dec.py
└── tokenizer/
    └── tokenizer.py
```

## How It Works

1. Prompt is tokenized by `DummyTokenizer`.
2. Target model prefills KV-cache with prompt context.
3. Selected generation path runs:
- `naive`: Python loop calls a jitted single-token step.
- `xla`: decode loop is compiled into `jax.lax.while_loop`.
- `speculative`: draft proposes `k` tokens, target verifies/accepts/rejects, then rolls forward.
4. Output token IDs are decoded as bracketed IDs (`[123]`), since defaults use random weights and dummy tokenization.

## Timeline

Planned delivery timeline from `plan/XLA_SHARDED_PLAN_FINAL.md`:
- Days 1-3: model stack + tokenizer + core validation
- Days 4-5: single-device generation baseline
- Days 6-8: sharding + XLA generation path
- Days 9-12: speculative decoding orchestration
- Days 13-16: benchmarking, CLI polish, tests, docs

## Installation

```bash
pip install -e .
pip install -e ".[dev]"
```

## Run

```bash
python demo.py --mode naive --prompt "The meaning" --max-tokens 50
python demo.py --mode xla --prompt "The meaning" --max-tokens 50
python demo.py --mode speculative --prompt "The meaning" --max-tokens 50 --k 5
python demo.py --mode compare --prompt "The meaning" --max-tokens 50 --warmup 1 --runs 3
```

## Test

```bash
pytest -q
```

## Notes

- Default weights are random initialization, so decoded outputs are token IDs rather than meaningful text.
- `DummyTokenizer` is intentionally simple for pipeline validation.
- Sharding tests are skipped automatically when fewer than 2 JAX devices are available.
