"""CLI entry point for naive/xla/speculative generation and compare mode."""

from __future__ import annotations

from time import perf_counter

import typer

from benchmark.report import render_report
from benchmark.throughput import ThroughputStats, benchmark_runs
from engine.generate_naive import generate_naive
from engine.generate_xla import generate_xla
from engine.spec_dec import speculative_decode


app = typer.Typer(add_completion=False, help="XLA-Sharded demo CLI")


def _make_tokenizer(tokenizer_type: str):
    """Return a tokenizer instance for the given type string."""
    if tokenizer_type == "gemma":
        from tokenizer.tokenizer import GemmaTokenizer  # requires [gemma] extra
        return GemmaTokenizer()
    from tokenizer.tokenizer import DummyTokenizer
    return DummyTokenizer()


@app.command()
def main(
    prompt: str = typer.Option("The meaning", help="Input prompt"),
    max_tokens: int = typer.Option(200, min=1, help="Max new tokens to generate"),
    mode: str = typer.Option("naive", help="naive|xla|speculative|compare"),
    k: int = typer.Option(5, min=1, help="Speculation length for speculative mode"),
    seed: int = typer.Option(42, help="Model initialization seed"),
    warmup: int = typer.Option(1, min=0, help="Warmup runs for benchmark mode"),
    runs: int = typer.Option(3, min=1, help="Benchmark runs"),
    tokenizer_type: str = typer.Option(
        "dummy",
        "--tokenizer",
        help="dummy (V1, default) | gemma (V2, requires pip install -e '[gemma]' and vocab_size=256_000)",
    ),
    profile: bool = typer.Option(False, "--profile", help="Enable XLA profiling (requires [profile] extra)"),
    profile_dir: str = typer.Option("/tmp/xla_trace", "--profile-dir", help="Directory for XLA profile output"),
) -> None:
    """Run generation and print output or benchmark comparison."""
    valid = {"naive", "xla", "speculative", "compare"}
    if mode not in valid:
        raise typer.BadParameter(f"mode must be one of {sorted(valid)}")

    tok = _make_tokenizer(tokenizer_type)

    def _run_naive():
        t0 = perf_counter()
        r = generate_naive(prompt, max_new_tokens=max_tokens, seed=seed, tokenizer=tok)
        dt = perf_counter() - t0
        return len(r.generated_ids), 0.0, dt, None, None

    def _run_xla():
        t0 = perf_counter()
        r = generate_xla(prompt, max_new_tokens=max_tokens, seed=seed, tokenizer=tok)
        dt = perf_counter() - t0
        return len(r.generated_ids), 0.0, dt, None, None

    def _run_spec():
        t0 = perf_counter()
        r = speculative_decode(prompt, max_new_tokens=max_tokens, k=k, seed=seed, tokenizer=tok)
        dt = perf_counter() - t0
        return len(r.generated_ids), 0.0, dt, r.acceptance_rate, r.rounds

    # Optionally wrap execution in a profiling context.
    if profile:
        from benchmark.profiler import profile_and_dump
        ctx = profile_and_dump(logdir=profile_dir, hlo_dir=profile_dir)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    if mode == "compare":
        with ctx:
            rows: list[ThroughputStats] = []
            rows.append(benchmark_runs("naive",       _run_naive, warmup=warmup, runs=runs))
            rows.append(benchmark_runs("xla",         _run_xla,   warmup=warmup, runs=runs))
            rows.append(benchmark_runs("speculative", _run_spec,  warmup=warmup, runs=runs))
        typer.echo(render_report(rows))
        if profile:
            typer.echo(f"\nProfile written to: {profile_dir}")
        return

    with ctx:
        if mode == "naive":
            result = generate_naive(prompt, max_new_tokens=max_tokens, seed=seed, tokenizer=tok)
        elif mode == "xla":
            result = generate_xla(prompt, max_new_tokens=max_tokens, seed=seed, tokenizer=tok)
        else:
            result = speculative_decode(prompt, max_new_tokens=max_tokens, k=k, seed=seed, tokenizer=tok)

    typer.echo("Prompt:")
    typer.echo(prompt)
    typer.echo("\nGenerated token IDs:")
    typer.echo(result.generated_ids)
    typer.echo("\nDecoded generated text:")
    typer.echo(result.decoded_generated_text)
    typer.echo("\nDecoded full sequence (prompt + generated):")
    typer.echo(result.decoded_all_text)
    if mode == "speculative":
        typer.echo("\nSpeculative stats:")
        typer.echo(f"Rounds: {result.rounds}")
        typer.echo(f"Accepted/proposed: {result.accepted_tokens}/{result.proposed_tokens}")
        typer.echo(f"Acceptance rate: {result.acceptance_rate:.3f}")
    if profile:
        typer.echo(f"\nProfile written to: {profile_dir}")
        typer.echo("View with: tensorboard --logdir " + profile_dir + "/trace")


if __name__ == "__main__":
    app()
