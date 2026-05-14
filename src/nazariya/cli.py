from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from nazariya import __version__
from nazariya.preview import build_previews
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


@app.command("build-previews")
def build_previews_command(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input sampled candidate CSV.",
    ),
    output: Path = typer.Option(
        Path("data/previews/default"),
        "--output",
        "-o",
        help="Output preview root.",
    ),
    max_size: int = typer.Option(
        768,
        "--max-size",
        help="Long-edge size for generated previews.",
    ),
    low_pct: float = typer.Option(
        0.5,
        "--low-pct",
        help="Low luminance percentile for exposure normalization.",
    ),
    high_pct: float = typer.Option(
        99.5,
        "--high-pct",
        help="High luminance percentile for exposure normalization.",
    ),
    wb: str = typer.Option(
        "daylight",
        "--wb",
        help="White balance mode: daylight, camera, auto, gray-world, custom.",
    ),
    user_wb: str | None = typer.Option(
        None,
        "--user-wb",
        help="Custom WB multipliers for --wb custom, like 2.0,1.0,1.4,1.0.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing previews.",
    ),
) -> None:
    """Build normalized analysis previews directly from RAW files."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Building RAW previews...", total=None)
        result = build_previews(
            input_path=input,
            output_root=output,
            max_size=max_size,
            low_pct=low_pct,
            high_pct=high_pct,
            wb_mode=wb,
            user_wb_text=user_wb,
            overwrite=overwrite,
        )
        progress.update(task, description="Finished building RAW previews.")

    table = Table(title="Nazariya build-previews")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Input", str(result.input_path))
    table.add_row("Output root", str(result.output_root))
    table.add_row("Total rows", str(result.total_rows))
    table.add_row("Rendered", str(result.rendered))
    table.add_row("Skipped existing", str(result.skipped_existing))
    table.add_row("Failed", str(result.failed))
    table.add_row("WB mode", str(result.wb_mode))
    table.add_row("Preview map", str(result.preview_map_path))
    table.add_row("Failures", str(result.failures_path))

    console.print(table)

    if result.failed > 0:
        console.print(
            "[yellow]Some RAW files failed. Check failures.csv before continuing.[/yellow]"
        )


if __name__ == "__main__":
    app()
