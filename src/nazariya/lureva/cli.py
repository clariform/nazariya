from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nazariya.lureva.groups import build_group_pools, propose_groups
from nazariya.lureva.ingest import IngestError, ingest_milestone_one
from nazariya.lureva.review import build_group_review_contact_sheets
from nazariya.lureva.batches import (
    build_lightroom_review_structure,
    process_image_batch,
    sample_image_pools,
)
from nazariya.lureva.selection import (
    build_lightroom_review_manifest,
    finalize_groups,
    prepare_image_pools,
)
from nazariya.lureva.paths import LurevaPaths

app = typer.Typer(
    help="Prepare and review the Lureva 960-image production selection.",
    no_args_is_help=True,
)
console = Console()


@app.command("init")
def init_command(
    root: Path = typer.Option(Path("data/lureva"), "--root", help="Lureva selection workspace."),
) -> None:
    """Create the Lureva selection workspace."""
    folders = LurevaPaths(root).create()
    console.print("[bold green]Created Lureva workspace:[/bold green]")
    for folder in folders:
        console.print(f"  {folder}")


@app.command("ingest")
def ingest_command(
    catalog: Path = typer.Option(..., "--catalog", help="Lightroom candidate inventory CSV."),
    seed: Path = typer.Option(..., "--seed", help="Current 560-image Lureva corpus CSV."),
    root: Path = typer.Option(Path("data/lureva"), "--root", help="Lureva selection workspace."),
    exclude_part2: bool = typer.Option(
        True,
        "--exclude-part2/--include-part2",
        help="Exclude seed images carrying p001 through p008.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional stable run identifier."),
    expected_catalog_candidates: int | None = typer.Option(
        39393,
        "--expected-catalog-candidates",
        help="Expected number of unique c001-c325 RAW source files. Use 0 to disable.",
    ),
    overrides: Path | None = typer.Option(
        None,
        "--overrides",
        help="Optional seed replacement/override CSV.",
    ),
    expected_seed_rows: int | None = typer.Option(
        560,
        "--expected-seed-rows",
        help="Expected number of seed corpus images. Use 0 to disable.",
    ),
) -> None:
    """Normalize the catalog and seed corpus, match source identities, and write an audit."""
    try:
        result = ingest_milestone_one(
            catalog_csv=catalog,
            seed_csv=seed,
            root=root,
            exclude_part2=exclude_part2,
            run_id=run_id,
            expected_catalog_candidates=expected_catalog_candidates or None,
            expected_seed_rows=expected_seed_rows or None,
            overrides_csv=overrides,
        )
    except IngestError as error:
        console.print(f"[bold red]Ingest failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    table = Table(title="Lureva Milestone 1 ingest")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Run", result.run_id)
    table.add_row("Run directory", str(result.run_dir))
    table.add_row("SQLite state", str(result.database_path))
    table.add_row("Catalog rows read", f"{result.catalog_rows_read:,}")
    table.add_row("Candidate catalog rows", f"{result.catalog_candidate_rows:,}")
    table.add_row("RAW catalog records", f"{result.catalog_raw_rows:,}")
    table.add_row("Unique RAW source files", f"{result.catalog_unique_raws:,}")
    table.add_row("Duplicate catalog records", f"{result.catalog_duplicate_records:,}")
    table.add_row("Ignored non-RAW rows", f"{result.catalog_non_raw_rows:,}")
    table.add_row("Seed rows read", f"{result.seed_rows_read:,}")
    table.add_row("Seed rows after filter", f"{result.seed_rows_after_filter:,}")
    table.add_row("Matched", f"{result.matched_rows:,}")
    table.add_row("Ambiguous", f"{result.ambiguous_rows:,}")
    table.add_row("Unmatched", f"{result.unmatched_rows:,}")
    table.add_row("Represented groups", f"{result.represented_groups:,}")
    table.add_row("Overrides applied", f"{result.overrides_applied:,}")
    table.add_row("Part 2 excluded", str(result.exclude_part2))
    console.print(table)

    if result.ambiguous_rows or result.unmatched_rows:
        console.print(
            "[yellow]Identity issues remain. Review match_issues.csv before group selection.[/yellow]"
        )
    else:
        console.print("[bold green]All filtered seed images matched uniquely.[/bold green]")


@app.command("build-group-pools")
def build_group_pools_command(
    reference_run: str = typer.Option(..., "--reference-run", help="Effective Part 1 ingest run."),
    exclusion_run: str = typer.Option(..., "--exclusion-run", help="Effective full-corpus ingest run used to exclude all existing images."),
    root: Path = typer.Option(Path("data/lureva"), "--root", help="Lureva selection workspace."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional stable run identifier."),
    minimum_images: int = typer.Option(20, "--minimum-images", min=1),
    preferred_images: int = typer.Option(30, "--preferred-images", min=1),
) -> None:
    """Build unused RAW pools for Part 1-represented candidate groups."""
    try:
        result = build_group_pools(
            root=root,
            reference_run=reference_run,
            exclusion_run=exclusion_run,
            run_id=run_id,
            minimum_images=minimum_images,
            preferred_images=preferred_images,
        )
    except IngestError as error:
        console.print(f"[bold red]Group-pool build failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    table = Table(title="Lureva Milestone 2 group pools")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Run", result.run_id)
    table.add_row("Run directory", str(result.run_dir))
    table.add_row("Part 1 references", f"{result.reference_images:,}")
    table.add_row("Existing corpus excluded", f"{result.excluded_images:,}")
    table.add_row("Represented groups", f"{result.represented_groups:,}")
    table.add_row("Eligible groups", f"{result.eligible_groups:,}")
    table.add_row("Unique unused RAWs", f"{result.unique_pool_images:,}")
    table.add_row("Pool memberships", f"{result.pool_memberships:,}")
    console.print(table)


@app.command("propose-groups")
def propose_groups_command(
    pool_run: str = typer.Option(..., "--pool-run", help="Completed group-pool run."),
    count: int = typer.Option(48, "--count", min=1),
    root: Path = typer.Option(Path("data/lureva"), "--root", help="Lureva selection workspace."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional stable run identifier."),
) -> None:
    """Create a deterministic first proposal of candidate groups."""
    try:
        result = propose_groups(root=root, pool_run=pool_run, count=count, run_id=run_id)
    except IngestError as error:
        console.print(f"[bold red]Group proposal failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    table = Table(title="Lureva Milestone 2 group proposal")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Run", result.run_id)
    table.add_row("Run directory", str(result.run_dir))
    table.add_row("Eligible groups", f"{result.eligible_groups:,}")
    table.add_row("Requested groups", f"{result.requested_groups:,}")
    table.add_row("Proposed groups", f"{result.proposed_groups:,}")
    console.print(table)


@app.command("group-review-contact-sheets")
def group_review_contact_sheets_command(
    pool_run: str = typer.Option(..., "--pool-run", help="Completed group-pool run."),
    proposal_run: str = typer.Option(..., "--proposal-run", help="Completed group proposal run."),
    root: Path = typer.Option(Path("data/lureva"), "--root", help="Lureva selection workspace."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional stable run identifier."),
    selected_pool_samples: int = typer.Option(10, "--selected-pool-samples", min=1),
    unselected_pool_samples: int = typer.Option(6, "--unselected-pool-samples", min=1),
    reference_limit: int = typer.Option(6, "--reference-limit", min=1),
    max_preview_size: int = typer.Option(640, "--max-preview-size", min=128),
    overwrite_previews: bool = typer.Option(False, "--overwrite-previews"),
    resume: bool = typer.Option(False, "--resume", help="Reuse an existing run and rebuild sheets from cached previews."),
    sheets_only: bool = typer.Option(False, "--sheets-only", help="Skip RAW rendering and rebuild sheets from the existing preview map."),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show live progress while rendering and composing sheets."),
) -> None:
    """Render proposed and unselected eligible candidate groups for visual review."""
    try:
        result = build_group_review_contact_sheets(
            root=root,
            pool_run=pool_run,
            proposal_run=proposal_run,
            run_id=run_id,
            selected_pool_samples=selected_pool_samples,
            unselected_pool_samples=unselected_pool_samples,
            reference_limit=reference_limit,
            max_preview_size=max_preview_size,
            overwrite_previews=overwrite_previews,
            show_progress=progress,
            resume=resume,
            sheets_only=sheets_only,
        )
    except (IngestError, ValueError, RuntimeError) as error:
        console.print(f"[bold red]Group review failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    table = Table(title="Lureva group review contact sheets")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Run", result.run_id)
    table.add_row("Run directory", str(result.run_dir))
    table.add_row("Eligible groups", f"{result.eligible_groups:,}")
    table.add_row("Proposed groups", f"{result.proposed_groups:,}")
    table.add_row("Unselected groups", f"{result.unselected_groups:,}")
    table.add_row("Review images", f"{result.review_images:,}")
    table.add_row("Previews rendered", f"{result.previews_rendered:,}")
    table.add_row("Preview failures", f"{result.preview_failures:,}")
    table.add_row("Detail sheets", f"{result.detail_sheets:,}")
    table.add_row("Overview sheets", f"{result.overview_sheets:,}")
    console.print(table)


@app.command("finalize-groups")
def finalize_groups_command(
    review_run: str = typer.Option(..., "--review-run", help="Completed group review run."),
    root: Path = typer.Option(Path("data/lureva"), "--root", help="Lureva selection workspace."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional stable run identifier."),
    expected_groups: int = typer.Option(48, "--expected-groups", min=1),
    group_prefix: str = typer.Option("u", "--group-prefix", help="Prefix for final Lureva groups."),
) -> None:
    """Apply group review decisions and lock the final candidate groups."""
    try:
        result = finalize_groups(
            root=root,
            review_run=review_run,
            run_id=run_id,
            expected_groups=expected_groups,
            group_prefix=group_prefix,
        )
    except IngestError as error:
        console.print(f"[bold red]Finalize groups failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    table = Table(title="Lureva final groups")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Run", result.run_id)
    table.add_row("Run directory", str(result.run_dir))
    table.add_row("Selected groups", f"{result.selected_groups:,}")
    table.add_row("Removed proposed groups", f"{result.removed_groups:,}")
    table.add_row("Added unselected groups", f"{result.added_groups:,}")
    console.print(table)


@app.command("prepare-image-pools")
def prepare_image_pools_command(
    groups_run: str = typer.Option(..., "--groups-run", help="Finalized groups run."),
    pool_run: str = typer.Option(..., "--pool-run", help="Group-pool run."),
    root: Path = typer.Option(Path("data/lureva"), "--root", help="Lureva selection workspace."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional stable run identifier."),
) -> None:
    """Build deduplicated unused RAW pools for the finalized groups."""
    try:
        result = prepare_image_pools(
            root=root, groups_run=groups_run, pool_run=pool_run, run_id=run_id
        )
    except IngestError as error:
        console.print(f"[bold red]Prepare image pools failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    table = Table(title="Lureva image pools")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Run", result.run_id)
    table.add_row("Run directory", str(result.run_dir))
    table.add_row("Groups", f"{result.groups:,}")
    table.add_row("Pool memberships", f"{result.pool_memberships:,}")
    table.add_row("Unique RAWs", f"{result.unique_images:,}")
    table.add_row("Overlap assignments", f"{result.overlap_assignments:,}")
    console.print(table)


@app.command("build-lightroom-review-manifest")
def build_lightroom_review_manifest_command(
    proposal_csv: Path = typer.Option(..., "--proposal-csv", help="Image proposal CSV with primary and alternate roles."),
    groups_csv: Path = typer.Option(..., "--groups-csv", help="Finalized group manifest CSV."),
    root: Path = typer.Option(Path("data/lureva"), "--root", help="Lureva selection workspace."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional stable run identifier."),
    primary_count: int = typer.Option(20, "--primary-count", min=1),
    alternate_count: int = typer.Option(5, "--alternate-count", min=0),
) -> None:
    """Validate an image proposal and write the Lightroom review manifest."""
    try:
        result = build_lightroom_review_manifest(
            proposal_csv=proposal_csv,
            groups_csv=groups_csv,
            root=root,
            run_id=run_id,
            primary_count=primary_count,
            alternate_count=alternate_count,
        )
    except IngestError as error:
        console.print(f"[bold red]Lightroom manifest failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    table = Table(title="Lureva Lightroom review manifest")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Run", result.run_id)
    table.add_row("Manifest", str(result.manifest_path))
    table.add_row("Groups", f"{result.groups:,}")
    table.add_row("Primary images", f"{result.primary_images:,}")
    table.add_row("Alternates", f"{result.alternate_images:,}")
    table.add_row("Total review images", f"{result.total_images:,}")
    console.print(table)


@app.command("sample-image-pools")
def sample_image_pools_command(
    pool_run: str = typer.Option(..., "--pool-run"),
    root: Path = typer.Option(Path("data/lureva"), "--root"),
    max_per_group: int | None = typer.Option(40, "--max-per-group", min=20),
    strategy: str = typer.Option("temporal-spread", "--strategy"),
    seed: int = typer.Option(42, "--seed"),
    batch_size: int = typer.Option(8, "--batch-size", min=1),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Sample manageable per-group pools and split them into batches."""
    try:
        result = sample_image_pools(
            root=root, pool_run=pool_run, max_per_group=max_per_group,
            strategy=strategy, seed=seed, batch_size=batch_size, run_id=run_id,
        )
    except IngestError as error:
        console.print(f"[red]Sampling failed:[/red] {error}")
        raise typer.Exit(1) from error
    table = Table(title="Lureva sampled image pools")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in (("Run", result.run_id), ("Run directory", str(result.run_dir)),
                       ("Groups", result.groups), ("Sampled images", result.sampled_images),
                       ("Batches", result.batches), ("Warnings", result.warnings)):
        table.add_row(str(key), str(value))
    console.print(table)


@app.command("build-lightroom-review-structure")
def build_lightroom_review_structure_command(
    sample_run: str = typer.Option(..., "--sample-run"),
    root: Path = typer.Option(Path("data/lureva"), "--root"),
    version: str = typer.Option("0.1.0", "--version"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Write the one-time Lightroom group/batch structure manifest."""
    try:
        result = build_lightroom_review_structure(root=root, sample_run=sample_run, version=version, run_id=run_id)
    except IngestError as error:
        console.print(f"[red]Structure generation failed:[/red] {error}")
        raise typer.Exit(1) from error
    table = Table(title="Lureva Lightroom review structure")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in (("Run", result.run_id), ("Run directory", str(result.run_dir)),
                       ("Groups", result.groups), ("Batches", result.batches),
                       ("Manifest", str(result.manifest_path))):
        table.add_row(str(key), str(value))
    console.print(table)


@app.command("process-image-batch")
def process_image_batch_command(
    sample_run: str = typer.Option(..., "--sample-run"),
    batch: str = typer.Option(..., "--batch", help="Batch number or batch_XX identifier."),
    root: Path = typer.Option(Path("data/lureva"), "--root"),
    primary_count: int = typer.Option(20, "--primary-count", min=1),
    alternate_count: int = typer.Option(5, "--alternate-count", min=0),
    close_reference_count: int = typer.Option(5, "--close-reference-count", min=0),
    version: str = typer.Option("0.1.0", "--version"),
    collection_set: str | None = typer.Option(None, "--collection-set"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Create a fast Lightroom assignment manifest for one sampled batch."""
    try:
        result = process_image_batch(
            root=root,
            sample_run=sample_run,
            batch=batch,
            primary_count=primary_count,
            alternate_count=alternate_count,
            close_reference_count=close_reference_count,
            version=version,
            collection_set=collection_set,
            run_id=run_id,
        )
    except IngestError as error:
        console.print(f"[red]Process image batch failed:[/red] {error}")
        raise typer.Exit(1) from error
    table = Table(title="Lureva image batch assignment")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in (
        ("Run", result.run_id),
        ("Run directory", str(result.run_dir)),
        ("Batch", result.batch_id),
        ("Groups", result.groups),
        ("Primary images", result.primary_images),
        ("Alternates", result.alternate_images),
        ("Total review images", result.total_images),
        ("Assignment CSV", str(result.assignment_path)),
    ):
        table.add_row(str(key), str(value))
    console.print(table)
