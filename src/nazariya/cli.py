from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nazariya import __version__
from nazariya.sample import sample_candidates

app = typer.Typer(
    help="Local visual search and clustering for image archives.",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the Nazariya version and exit.",
    ),
) -> None:
    if version:
        console.print(f"nazariya {__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def hello() -> None:
    """Smoke test command."""
    console.print("[bold green]nazariya[/bold green] is ready.")


@app.command()
def init(
    root: Path = typer.Argument(
        Path("data"),
        help="Project data directory to create.",
    )
) -> None:
    """Create a minimal data folder layout."""
    folders = [
        root / "inputs",
        root / "previews",
        root / "normalized",
        root / "features",
        root / "clusters" / "contact_sheets",
        root / "reports",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Created Nazariya data folders:[/bold green]")
    for folder in folders:
        console.print(f"  {folder}")


@app.command()
def sample(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input Lightroom candidate CSV.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output sampled CSV.",
    ),
    per_candidate: int = typer.Option(
        3,
        "--per-candidate",
        help="Number of images to sample from each candidate key.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for repeatable sampling.",
    ),
    group_column: str = typer.Option(
        "primary_candidate_key",
        "--group-column",
        help="CSV column used as the candidate grouping key.",
    ),
) -> None:
    """Create a repeatable stratified sample from a Lightroom candidate CSV."""
    summary = sample_candidates(
        input_path=input,
        output_path=output,
        per_candidate=per_candidate,
        seed=seed,
        group_column=group_column,
    )

    table = Table(title="Nazariya sample")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Input", str(summary.input_path))
    table.add_row("Output", str(summary.output_path))
    table.add_row("Total input rows", str(summary.total_rows))
    table.add_row("Candidate groups", str(summary.candidate_count))
    table.add_row("Per candidate", str(summary.per_candidate))
    table.add_row("Seed", str(summary.seed))
    table.add_row("Sampled rows", str(summary.sampled_rows))
    table.add_row(
        "Groups with fewer rows than requested",
        str(summary.candidates_with_fewer_than_requested),
    )

    console.print(table)


if __name__ == "__main__":
    app()
