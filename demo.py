"""Phase 2 CLI entry point: single-GPU naive generation baseline."""

from __future__ import annotations

import typer

from engine.generate_naive import generate_naive


app = typer.Typer(add_completion=False, help="XLA-Sharded demo CLI")


@app.command()
def main(
    prompt: str = typer.Option("The meaning", help="Input prompt"),
    max_tokens: int = typer.Option(200, min=1, help="Max new tokens to generate"),
    seed: int = typer.Option(42, help="Model initialization seed"),
) -> None:
    """Run naive generation and print decoded output."""
    result = generate_naive(prompt, max_new_tokens=max_tokens, seed=seed)
    typer.echo("Prompt:")
    typer.echo(prompt)
    typer.echo("\nGenerated token IDs:")
    typer.echo(result.generated_ids)
    typer.echo("\nDecoded generated text:")
    typer.echo(result.decoded_generated_text)
    typer.echo("\nDecoded full sequence (prompt + generated):")
    typer.echo(result.decoded_all_text)


if __name__ == "__main__":
    app()

