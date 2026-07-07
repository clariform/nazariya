from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    help="Analyze and learn global photo-editing starting points.",
    no_args_is_help=True,
)

console = Console()


@app.command("info")
def info() -> None:
    """Show the current status of Nazariya style-learning support."""
    console.print("[bold]Nazariya style learning[/bold]")
    console.print()
    console.print("Groundwork is ready.")
    console.print("Planned stages:")
    console.print("  1. Export Lightroom Develop settings")
    console.print("  2. Extract original RAW metadata")
    console.print("  3. Audit target consistency and trainability")
    console.print("  4. Build baseline previews and image features")
    console.print("  5. Train global edit predictors")
    console.print("  6. Generate candidate XMP sidecars")
