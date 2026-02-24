"""Timing utilities for generation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from time import perf_counter
from typing import Callable


@dataclass
class ThroughputStats:
    mode: str
    tokens_per_sec_mean: float
    tokens_per_sec_std: float
    ttft_ms_mean: float
    total_time_s_mean: float
    tokens_generated: int
    acceptance_rate: float | None = None
    rounds: int | None = None


def benchmark_runs(
    mode: str,
    run_fn: Callable[[], tuple[int, float, float, float | None, int | None]],
    *,
    warmup: int = 1,
    runs: int = 5,
) -> ThroughputStats:
    """
    Benchmark helper.

    run_fn returns:
      (tokens_generated, ttft_ms, total_time_s, acceptance_rate, rounds)
    """
    for _ in range(warmup):
        run_fn()

    tps_vals: list[float] = []
    ttft_vals: list[float] = []
    total_vals: list[float] = []
    acc_vals: list[float] = []
    rounds_vals: list[int] = []
    last_tokens = 0

    for _ in range(runs):
        start = perf_counter()
        tokens, ttft_ms, total_time_s, accept_rate, rounds_count = run_fn()
        _ = perf_counter() - start  # Keep a local end-to-end marker for consistency.
        last_tokens = tokens
        total_vals.append(total_time_s)
        ttft_vals.append(ttft_ms)
        tps_vals.append(tokens / max(total_time_s, 1e-9))
        if accept_rate is not None:
            acc_vals.append(accept_rate)
        if rounds_count is not None:
            rounds_vals.append(rounds_count)

    return ThroughputStats(
        mode=mode,
        tokens_per_sec_mean=mean(tps_vals),
        tokens_per_sec_std=pstdev(tps_vals) if len(tps_vals) > 1 else 0.0,
        ttft_ms_mean=mean(ttft_vals),
        total_time_s_mean=mean(total_vals),
        tokens_generated=last_tokens,
        acceptance_rate=(mean(acc_vals) if acc_vals else None),
        rounds=(int(mean(rounds_vals)) if rounds_vals else None),
    )

