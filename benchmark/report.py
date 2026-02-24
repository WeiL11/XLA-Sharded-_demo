"""Inline terminal report formatting for benchmark results."""

from __future__ import annotations

from benchmark.throughput import ThroughputStats


def render_report(rows: list[ThroughputStats]) -> str:
    headers = [
        "Mode",
        "Tokens/sec",
        "TTFT (ms)",
        "Total (s)",
        "Tokens",
        "Accept Rate",
        "Rounds",
    ]
    lines = []
    lines.append(" | ".join(headers))
    lines.append("-" * 86)
    for r in rows:
        tps = f"{r.tokens_per_sec_mean:.2f} (+/- {r.tokens_per_sec_std:.2f})"
        ttft = f"{r.ttft_ms_mean:.2f}"
        total = f"{r.total_time_s_mean:.2f}"
        acc = f"{r.acceptance_rate:.3f}" if r.acceptance_rate is not None else "-"
        rounds = str(r.rounds) if r.rounds is not None else "-"
        lines.append(
            f"{r.mode:<12} | {tps:<22} | {ttft:<10} | {total:<9} | "
            f"{r.tokens_generated:<6} | {acc:<11} | {rounds}"
        )
    return "\n".join(lines)

