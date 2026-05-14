from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from nazariya import __version__

app = typer.Typer(help="Local visual search and clustering for image archives.")
console = Console()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the Nazariya version and exit.",
    )
) -> None:
    if version:
        console.print(f"nazariya {__version__}")
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


if __name__ == "__main__":
    app()
