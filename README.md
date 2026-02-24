# XLA-Sharded

Multi-phase JAX/Flax NNX project for speculative decoding across draft and target models.

## Implemented Phases

- Phase 1: core model stack (configs, layers, transformer, draft model, KV cache, tokenizer)
- Phase 2: naive single-device generation baseline (`engine/generate_naive.py`)
- Phase 3: sharding helpers + XLA while-loop generation (`engine/sharder.py`, `engine/generate_xla.py`)
- Phase 4: speculative decoding orchestrator (`engine/spec_dec.py`)
- Phase 5: benchmark/report/profiler modules + multi-mode CLI (`benchmark/*`, `demo.py`)

## Quickstart

```bash
pytest -q
python demo.py --mode naive --prompt "The meaning" --max-tokens 50
python demo.py --mode xla --prompt "The meaning" --max-tokens 50
python demo.py --mode speculative --prompt "The meaning" --max-tokens 50 --k 5
python demo.py --mode compare --prompt "The meaning" --max-tokens 50 --warmup 1 --runs 3
```

## Notes

- Current defaults use random model init, so output tokens are not natural language.
- Dummy tokenizer is intended for pipeline correctness testing.
- Sharding tests require at least 2 visible JAX devices.

