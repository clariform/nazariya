from __future__ import annotations

from pathlib import Path
from nazariya.style.cli import app as style_app
from nazariya.lureva.cli import app as lureva_app
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from nazariya import __version__
from nazariya.search import (
    build_previews,
    extract_background_features,
    extract_features,
    generate_neighbor_sheets,
    make_contact_sheets,
    make_overrides_template,
    sample_candidates,
    swap_sample_row,
)

app = typer.Typer(
    help="Local visual search and clustering for image archives.",
    invoke_without_command=True,
)
app.add_typer(
    style_app,
    name="style",
    help="Analyze and learn global photo-editing starting points.",
)
app.add_typer(
    lureva_app,
    name="lureva",
    help="Prepare and review the Lureva production-selection corpus.",
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
        root / "config",
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


@app.command("swap-sample")
def swap_sample_command(
    full: Path = typer.Option(
        ...,
        "--full",
        help="Full Lightroom candidate CSV.",
    ),
    sample: Path = typer.Option(
        ...,
        "--sample",
        help="Current sampled CSV.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output sampled CSV with one row swapped.",
    ),
    candidate: str = typer.Option(
        ...,
        "--candidate",
        help="Candidate key to modify, for example c014.",
    ),
    remove_file: str | None = typer.Option(
        None,
        "--remove-file",
        help="File name to remove from the sample, for example DSC01234.ARW.",
    ),
    remove_source_path: str | None = typer.Option(
        None,
        "--remove-source-path",
        help="Exact source_path to remove from the sample.",
    ),
    remove_image_id: str | None = typer.Option(
        None,
        "--remove-image-id",
        help="Image id to remove if the CSV contains image_id.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for choosing the replacement.",
    ),
    group_column: str = typer.Option(
        "primary_candidate_key",
        "--group-column",
        help="CSV column used as the candidate grouping key.",
    ),
) -> None:
    """Swap one sampled row for another image from the same candidate set."""
    result = swap_sample_row(
        full_input_path=full,
        sample_input_path=sample,
        output_path=output,
        candidate_key=candidate,
        remove_file=remove_file,
        remove_source_path=remove_source_path,
        remove_image_id=remove_image_id,
        seed=seed,
        group_column=group_column,
    )

    table = Table(title="Nazariya swap sample")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Full input", str(result.full_input_path))
    table.add_row("Sample input", str(result.sample_input_path))
    table.add_row("Output", str(result.output_path))
    table.add_row("Candidate", str(result.candidate_key))
    table.add_row("Removed rows", str(result.removed_count))
    table.add_row("Added rows", str(result.added_count))
    table.add_row("Final rows", str(result.final_rows))
    table.add_row("Added file", str(result.added_file_name))
    table.add_row("Added source", str(result.added_source_path))

    console.print(table)


@app.command("make-overrides-template")
def make_overrides_template_command(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input candidate CSV.",
    ),
    output: Path = typer.Option(
        Path("data/config/candidate_overrides.csv"),
        "--output",
        "-o",
        help="Output candidate override CSV template.",
    ),
    include_examples: bool = typer.Option(
        False,
        "--include-examples",
        help="Include example override rows at the top.",
    ),
) -> None:
    """Create a per-candidate override CSV template."""
    result = make_overrides_template(
        input_path=input,
        output_path=output,
        include_examples=include_examples,
    )

    table = Table(title="Nazariya override template")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Input", str(result.input_path))
    table.add_row("Output", str(result.output_path))
    table.add_row("Candidate groups", str(result.candidate_count))

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
        help="Low luminance percentile for percentile exposure normalization.",
    ),
    high_pct: float = typer.Option(
        99.5,
        "--high-pct",
        help="High luminance percentile for percentile exposure normalization.",
    ),
    exposure_mode: str = typer.Option(
        "center-midtone",
        "--exposure-mode",
        help="Exposure mode: percentile, midtone, center-midtone.",
    ),
    target_median: float = typer.Option(
        0.38,
        "--target-median",
        help="Target midtone median for midtone exposure modes.",
    ),
    overrides: Path | None = typer.Option(
        None,
        "--overrides",
        help="Optional candidate override CSV with per-set normalization settings.",
    ),
    candidate: str | None = typer.Option(
        None,
        "--candidate",
        help="Only build previews for one candidate set, for example c014.",
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
            exposure_mode=exposure_mode,
            target_median=target_median,
            overrides_path=overrides,
            candidate_filter=candidate,
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
    table.add_row("Exposure mode", str(result.exposure_mode))
    table.add_row("Overrides", str(result.overrides_path or "None"))
    table.add_row("Overrides loaded", str(result.overrides_loaded))
    table.add_row("Candidate filter", str(result.candidate_filter or "None"))
    table.add_row("Preview map", str(result.preview_map_path))
    table.add_row("Failures", str(result.failures_path))

    console.print(table)

    if result.failed > 0:
        console.print(
            "[yellow]Some RAW files failed. Check failures.csv before continuing.[/yellow]"
        )


@app.command("extract-features")
def extract_features_command(
    preview_map: Path = typer.Option(
        ...,
        "--preview-map",
        help="Preview map CSV generated by build-previews.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output compressed .npz feature file.",
    ),
    metadata: Path = typer.Option(
        ...,
        "--metadata",
        help="Output feature metadata CSV.",
    ),
    model_name: str = typer.Option(
        "ViT-B-32",
        "--model-name",
        help="OpenCLIP model name.",
    ),
    pretrained: str = typer.Option(
        "laion2b_s34b_b79k",
        "--pretrained",
        help="OpenCLIP pretrained checkpoint.",
    ),
    clip_weight: float = typer.Option(
        0.65,
        "--clip-weight",
        help="Weight for CLIP embedding block.",
    ),
    color_weight: float = typer.Option(
        0.35,
        "--color-weight",
        help="Weight for color/light feature block.",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Batch size for CLIP inference.",
    ),
) -> None:
    """Extract CLIP + color/light features from normalized previews."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Extracting image features...", total=None)
        result = extract_features(
            preview_map_path=preview_map,
            output_npz_path=output,
            output_metadata_path=metadata,
            model_name=model_name,
            pretrained=pretrained,
            clip_weight=clip_weight,
            color_weight=color_weight,
            batch_size=batch_size,
        )
        progress.update(task, description="Finished extracting image features.")

    table = Table(title="Nazariya extract-features")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Preview map", str(result.preview_map_path))
    table.add_row("Output NPZ", str(result.output_npz_path))
    table.add_row("Metadata CSV", str(result.output_metadata_path))
    table.add_row("Total rows", str(result.total_rows))
    table.add_row("Extracted", str(result.extracted))
    table.add_row("Failed", str(result.failed))
    table.add_row("Model", str(result.model_name))
    table.add_row("Pretrained", str(result.pretrained))
    table.add_row("Feature dim", str(result.feature_dim))
    table.add_row("CLIP dim", str(result.clip_dim))
    table.add_row("Color/light dim", str(result.color_dim))

    console.print(table)

    if result.failed > 0:
        console.print(
            "[yellow]Some previews failed feature extraction. Check *_failures.csv next to the metadata file.[/yellow]"
        )


@app.command("extract-bg-features")
def extract_bg_features_command(
    preview_map: Path = typer.Option(
        ...,
        "--preview-map",
        help="Preview map CSV generated by build-previews.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output .npz feature file.",
    ),
    metadata: Path = typer.Option(
        ...,
        "--metadata",
        help="Output feature metadata CSV.",
    ),
    model_name: str = typer.Option(
        "ViT-B-32",
        "--model-name",
        help="OpenCLIP image model name.",
    ),
    pretrained: str = typer.Option(
        "laion2b_s34b_b79k",
        "--pretrained",
        help="OpenCLIP pretrained checkpoint name.",
    ),
    segment_model: str = typer.Option(
        "briaai/RMBG-2.0",
        "--segment-model",
        help="Hugging Face segmentation/background-removal model.",
    ),
    clip_weight: float = typer.Option(
        0.50,
        "--clip-weight",
        help="Weight for CLIP image embedding.",
    ),
    background_weight: float = typer.Option(
        0.50,
        "--background-weight",
        help="Weight for segmented-background histogram embedding.",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Batch size for CLIP extraction.",
    ),
    debug_dir: Path | None = typer.Option(
        None,
        "--debug-dir",
        help="Optional directory for subject/background mask debug images.",
    ),
    debug_limit: int = typer.Option(
        0,
        "--debug-limit",
        help="Number of debug masks/background images to write.",
    ),
) -> None:
    """Extract CLIP + segmented-background histogram features."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            "Extracting CLIP + background segmentation features...",
            total=None,
        )
        result = extract_background_features(
            preview_map_path=preview_map,
            output_npz_path=output,
            output_metadata_path=metadata,
            clip_model_name=model_name,
            clip_pretrained=pretrained,
            segment_model=segment_model,
            clip_weight=clip_weight,
            background_weight=background_weight,
            batch_size=batch_size,
            debug_dir=debug_dir,
            debug_limit=debug_limit,
        )
        progress.update(task, description="Finished extracting CLIP + background features.")

    table = Table(title="Nazariya extract-bg-features")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Preview map", str(result.preview_map_path))
    table.add_row("Output NPZ", str(result.output_npz_path))
    table.add_row("Metadata CSV", str(result.output_metadata_path))
    table.add_row("Total rows", str(result.total_rows))
    table.add_row("Extracted", str(result.extracted))
    table.add_row("Failed", str(result.failed))
    table.add_row("CLIP model", str(result.clip_model_name))
    table.add_row("CLIP pretrained", str(result.clip_pretrained))
    table.add_row("Segment model", str(result.segment_model))
    table.add_row("Feature dim", str(result.feature_dim))
    table.add_row("CLIP dim", str(result.clip_dim))
    table.add_row("Background dim", str(result.background_dim))
    table.add_row("CLIP weight", str(result.clip_weight))
    table.add_row("Background weight", str(result.background_weight))
    table.add_row("Debug dir", str(result.debug_dir or "None"))

    console.print(table)

    if result.failed > 0:
        console.print("[yellow]Some previews failed. Check the failure CSV next to metadata.[/yellow]")


@app.command("neighbor-sheets")
def neighbor_sheets_command(
    features: Path = typer.Option(
        ...,
        "--features",
        help="Feature .npz file generated by extract-features or extract-bg-features.",
    ),
    preview_map: Path = typer.Option(
        ...,
        "--preview-map",
        help="Preview map CSV generated by build-previews.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output folder for neighbor contact sheets.",
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        help="Number of neighbors to show per query image.",
    ),
    exclude_same_candidate: bool = typer.Option(
        True,
        "--exclude-same-candidate/--include-same-candidate",
        help="Exclude images from the same candidate set.",
    ),
    thumb_size: int = typer.Option(
        260,
        "--thumb-size",
        help="Thumbnail size in pixels.",
    ),
    label_height: int = typer.Option(
        92,
        "--label-height",
        help="Label area height below each thumbnail.",
    ),
    feature_space: str = typer.Option(
        "combined",
        "--feature-space",
        help=(
            "Feature space: combined, clip, color, background, "
            "clip-then-background."
        ),
    ),
    metric: str = typer.Option(
        "cosine",
        "--metric",
        help="Metric: cosine, histogram-intersection, bhattacharyya.",
    ),
    clip_pool: int = typer.Option(
        80,
        "--clip-pool",
        help=(
            "For --feature-space clip-then-background, number of CLIP "
            "neighbors to rerank by background."
        ),
    ),
) -> None:
    """Generate per-candidate nearest-neighbor contact sheets."""
    result = generate_neighbor_sheets(
        features_path=features,
        preview_map_path=preview_map,
        output_dir=output,
        top_k=top_k,
        exclude_same_candidate=exclude_same_candidate,
        thumb_size=thumb_size,
        label_height=label_height,
        feature_space=feature_space,
        metric=metric,
        clip_pool=clip_pool,
    )

    table = Table(title="Nazariya neighbor sheets")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Features", str(result.features_path))
    table.add_row("Preview map", str(result.preview_map_path))
    table.add_row("Output", str(result.output_dir))
    table.add_row("Images", str(result.image_count))
    table.add_row("Candidates", str(result.candidate_count))
    table.add_row("Sheets written", str(result.sheets_written))
    table.add_row("Top K", str(result.top_k))
    table.add_row("Exclude same candidate", str(result.exclude_same_candidate))
    table.add_row("Feature space", str(result.feature_space))
    table.add_row("Metric", str(result.metric))
    table.add_row("CLIP pool", str(result.clip_pool))

    console.print(table)


@app.command("contact-sheets")
def contact_sheets_command(
    preview_map: Path = typer.Option(
        ...,
        "--preview-map",
        help="Preview map CSV generated by build-previews.",
    ),
    output: Path = typer.Option(
        Path("data/reports/contact_sheets"),
        "--output",
        "-o",
        help="Output folder for candidate contact sheets.",
    ),
    thumb_size: int = typer.Option(
        320,
        "--thumb-size",
        help="Thumbnail size in pixels.",
    ),
    columns: int = typer.Option(
        3,
        "--columns",
        help="Number of columns per contact sheet.",
    ),
) -> None:
    """Create one contact sheet per candidate set."""
    result = make_contact_sheets(
        preview_map_path=preview_map,
        output_dir=output,
        thumb_size=thumb_size,
        columns=columns,
    )

    table = Table(title="Nazariya contact sheets")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Preview map", str(result.preview_map_path))
    table.add_row("Output", str(result.output_dir))
    table.add_row("Sheets written", str(result.sheets_written))
    table.add_row("Images", str(result.image_count))

    console.print(table)


if __name__ == "__main__":
    app()
