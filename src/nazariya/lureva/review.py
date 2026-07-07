from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageDraw, ImageFont
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from nazariya.lureva.ingest import IngestError
from nazariya.lureva.paths import LurevaPaths


def _build_previews(**kwargs):
    # Imported lazily because nazariya.search exposes optional heavy ML dependencies.
    from nazariya.search.preview import build_previews

    return build_previews(**kwargs)


def _resolve_preview_source(path: Path) -> Path:
    # Keep path remapping consistent with preview generation while avoiding a
    # module-level import of optional RAW/ML dependencies.
    from nazariya.search.preview import resolve_source_path

    return resolve_source_path(path)


@dataclass(frozen=True)
class GroupReviewResult:
    run_id: str
    run_dir: Path
    eligible_groups: int
    proposed_groups: int
    unselected_groups: int
    review_images: int
    previews_rendered: int
    preview_failures: int
    detail_sheets: int
    overview_sheets: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise IngestError(f"Required CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise IngestError(f"CSV has no header: {path}")
        return [{str(k): str(v or "") for k, v in row.items()} for row in reader]


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _even_sample(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    if count == 1:
        return [rows[len(rows) // 2]]

    indexes = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    seen: set[int] = set()
    out: list[dict[str, str]] = []
    for index in indexes:
        if index not in seen:
            out.append(rows[index])
            seen.add(index)
    return out


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _fit_image(path: Path, width: int, height: int) -> Image.Image:
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (width, height), "lightgray")
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "missing", font=_font(14), fill="black")
        return image

    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def _preview_source_aliases(path_text: str) -> list[str]:
    """Return stable aliases for the same image tree across mounted roots."""
    text = path_text.strip()
    if not text:
        return []

    aliases: list[str] = [text.casefold()]

    try:
        resolved = str(_resolve_preview_source(Path(text)))
    except Exception:
        resolved = text
    if resolved.casefold() not in aliases:
        aliases.append(resolved.casefold())

    normalized = text.replace("\\", "/")
    resolved_normalized = resolved.replace("\\", "/")
    markers = (
        "/Pictures/Images/",
        "/proetus/images/",
        "/Images/",
    )
    for candidate in (normalized, resolved_normalized):
        for marker in markers:
            if marker in candidate:
                relative = candidate.split(marker, 1)[1].lstrip("/")
                alias = f"archive-relative:{relative}".casefold()
                if alias not in aliases:
                    aliases.append(alias)
                break

    return aliases


def _load_preview_map(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    preview_map: dict[str, dict[str, str]] = {}
    for row in rows:
        source_path = row.get("source_path", "").strip()
        for alias in _preview_source_aliases(source_path):
            preview_map.setdefault(alias, row)
    return preview_map


def _lookup_preview(
    preview_map: dict[str, dict[str, str]],
    source_path_text: str,
) -> dict[str, str]:
    for alias in _preview_source_aliases(source_path_text):
        preview = preview_map.get(alias)
        if preview is not None:
            return preview
    return {}


def _draw_detail_sheet(
    *,
    group: str,
    rows: list[dict[str, str]],
    summary: dict[str, str],
    output_path: Path,
    preview_map: dict[str, dict[str, str]],
    columns: int = 4,
    thumb_w: int = 300,
    thumb_h: int = 220,
) -> None:
    pad = 24
    gap = 16
    header_h = 92
    label_h = 54
    cell_h = thumb_h + label_h
    row_count = max(1, math.ceil(len(rows) / columns))
    width = pad * 2 + columns * thumb_w + (columns - 1) * gap
    height = pad * 2 + header_h + row_count * cell_h + (row_count - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    status = summary.get("proposal_status", "unselected").upper()
    title = f"{group}  |  {status}"
    draw.text((pad, pad), title, font=_font(26, bold=True), fill="black")
    subtitle = (
        f"references={summary.get('reference_count', '')}   "
        f"unused={summary.get('unused_raw_count', '')}   "
        f"rank={summary.get('proposal_rank', '') or '-'}"
    )
    draw.text((pad, pad + 38), subtitle, font=_font(17), fill="dimgray")

    for index, row in enumerate(rows):
        col = index % columns
        r = index // columns
        x = pad + col * (thumb_w + gap)
        y = pad + header_h + r * (cell_h + gap)
        preview = _lookup_preview(preview_map, row.get("source_path", ""))
        image = _fit_image(Path(preview.get("normalized_preview_path", "")), thumb_w, thumb_h)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline="black", width=1)
        role = row.get("review_role", "POOL")
        draw.text((x, y + thumb_h + 6), role, font=_font(15, bold=True), fill="black")
        name = row.get("file_name", "")
        draw.text((x, y + thumb_h + 26), name, font=_font(14), fill="dimgray")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=92, optimize=True)


def _draw_overview_pages(
    *,
    summaries: list[dict[str, str]],
    rows_by_group: dict[str, list[dict[str, str]]],
    preview_map: dict[str, dict[str, str]],
    output_dir: Path,
    groups_per_page: int = 12,
    page_callback: Callable[[int, int], None] | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_w, panel_h = 620, 310
    page_cols = 2
    page_rows = math.ceil(groups_per_page / page_cols)
    pad, gap = 24, 18
    header_h = 58
    page_w = pad * 2 + page_cols * panel_w + gap
    page_h = pad * 2 + header_h + page_rows * panel_h + (page_rows - 1) * gap
    pages = 0

    for start in range(0, len(summaries), groups_per_page):
        chunk = summaries[start : start + groups_per_page]
        canvas = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(canvas)
        page_num = start // groups_per_page + 1
        draw.text(
            (pad, pad),
            f"Lureva group review  |  page {page_num}",
            font=_font(24, bold=True),
            fill="black",
        )

        for index, summary in enumerate(chunk):
            col = index % page_cols
            row_index = index // page_cols
            x = pad + col * (panel_w + gap)
            y = pad + header_h + row_index * (panel_h + gap)
            draw.rectangle((x, y, x + panel_w, y + panel_h), outline="black", width=1)
            status = summary.get("proposal_status", "unselected").upper()
            title = f"{summary['candidate_group']}  |  {status}"
            draw.text((x + 12, y + 10), title, font=_font(18, bold=True), fill="black")
            info = (
                f"refs {summary.get('reference_count', '')}  "
                f"unused {summary.get('unused_raw_count', '')}  "
                f"rank {summary.get('proposal_rank', '') or '-'}"
            )
            draw.text((x + 12, y + 36), info, font=_font(14), fill="dimgray")

            sample_rows = rows_by_group.get(summary["candidate_group"], [])
            refs = [r for r in sample_rows if r.get("review_role") == "REF"][:1]
            pools = [r for r in sample_rows if r.get("review_role") == "POOL"][:3]
            tiles = refs + pools
            tile_w, tile_h = 140, 190
            for tile_index, item in enumerate(tiles):
                tx = x + 12 + tile_index * (tile_w + 8)
                ty = y + 68
                preview = _lookup_preview(preview_map, item.get("source_path", ""))
                image = _fit_image(Path(preview.get("normalized_preview_path", "")), tile_w, 150)
                canvas.paste(image, (tx, ty))
                draw.rectangle((tx, ty, tx + tile_w, ty + 150), outline="black", width=1)
                draw.text((tx, ty + 156), item.get("review_role", ""), font=_font(12, bold=True), fill="black")
                draw.text((tx, ty + 172), item.get("file_name", "")[:20], font=_font(11), fill="dimgray")

        output_path = output_dir / f"groups_{start + 1:03d}-{start + len(chunk):03d}.jpg"
        canvas.save(output_path, "JPEG", quality=92, optimize=True)
        pages += 1
        if page_callback is not None:
            page_callback(pages, math.ceil(len(summaries) / groups_per_page))

    return pages


def build_group_review_contact_sheets(
    *,
    root: Path = Path("data/lureva"),
    pool_run: str,
    proposal_run: str,
    run_id: str | None = None,
    selected_pool_samples: int = 10,
    unselected_pool_samples: int = 6,
    reference_limit: int = 6,
    max_preview_size: int = 640,
    overwrite_previews: bool = False,
    show_progress: bool = True,
    resume: bool = False,
    sheets_only: bool = False,
) -> GroupReviewResult:
    paths = LurevaPaths(root)
    paths.create()
    pool_dir = paths.runs / pool_run
    proposal_dir = paths.runs / proposal_run
    if not pool_dir.exists():
        raise IngestError(f"Pool run does not exist: {pool_dir}")
    if not proposal_dir.exists():
        raise IngestError(f"Proposal run does not exist: {proposal_dir}")

    run_id = run_id or datetime.now().strftime("group-review-%Y%m%d-%H%M%S")
    run_dir = paths.runs / run_id
    if run_dir.exists() and not resume:
        raise IngestError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries = _read_csv(pool_dir / "eligible_groups.csv")
    pools = _read_csv(pool_dir / "group_pools.csv")
    references = _read_csv(pool_dir / "reference_images.csv")
    proposed = _read_csv(proposal_dir / "proposed_groups.csv")
    proposal_by_group = {row["candidate_group"]: row for row in proposed}

    pools_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    refs_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pools:
        if row.get("is_selectable") == "1":
            pools_by_group[row["candidate_group"]].append(row)
    for row in references:
        refs_by_group[row["candidate_group"]].append(row)

    review_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for summary in summaries:
        group = summary["candidate_group"]
        proposal = proposal_by_group.get(group)
        selected = proposal is not None
        proposal_status = "proposed" if selected else "unselected"
        proposal_rank = proposal.get("proposal_rank", "") if proposal else ""
        sample_count = selected_pool_samples if selected else unselected_pool_samples

        ref_rows = sorted(refs_by_group[group], key=lambda r: (r.get("source_path", ""), r.get("corpus_file_name", "")))
        pool_rows = sorted(pools_by_group[group], key=lambda r: (r.get("capture_date", ""), r.get("source_path", "")))
        chosen_refs = _even_sample(ref_rows, reference_limit)
        chosen_pool = _even_sample(pool_rows, sample_count)

        for row in chosen_refs:
            review_rows.append({
                "candidate_group": group,
                "proposal_status": proposal_status,
                "proposal_rank": proposal_rank,
                "review_role": "REF",
                "source_path": row.get("source_path", ""),
                "file_name": row.get("source_file_name", ""),
                "corpus_file_name": row.get("corpus_file_name", ""),
                "primary_candidate_key": group,
                "candidate_keys": group,
                "file_stem": Path(row.get("source_file_name", "")).stem,
                "capture_time": "",
            })
        for row in chosen_pool:
            review_rows.append({
                "candidate_group": group,
                "proposal_status": proposal_status,
                "proposal_rank": proposal_rank,
                "review_role": "POOL",
                "source_path": row.get("source_path", ""),
                "file_name": row.get("source_file_name", ""),
                "corpus_file_name": "",
                "primary_candidate_key": group,
                "candidate_keys": group,
                "file_stem": Path(row.get("source_file_name", "")).stem,
                "capture_time": row.get("capture_date", ""),
            })

        summary_rows.append({
            **summary,
            "proposal_status": proposal_status,
            "proposal_rank": proposal_rank,
            "reference_samples": len(chosen_refs),
            "pool_samples": len(chosen_pool),
            "manual_status": "",
            "replacement_group": "",
            "manual_note": "",
        })

    summary_rows.sort(key=lambda r: (0 if r["proposal_status"] == "proposed" else 1, int(r["proposal_rank"] or 9999), r["candidate_group"]))
    review_rows.sort(key=lambda r: (0 if r["proposal_status"] == "proposed" else 1, int(r["proposal_rank"] or 9999), r["candidate_group"], 0 if r["review_role"] == "REF" else 1, r["source_path"]))

    review_fields = [
        "candidate_group", "proposal_status", "proposal_rank", "review_role",
        "source_path", "file_name", "corpus_file_name", "primary_candidate_key",
        "candidate_keys", "file_stem", "capture_time",
    ]
    summary_fields = list(summaries[0].keys()) + [
        "proposal_status", "proposal_rank", "reference_samples", "pool_samples",
        "manual_status", "replacement_group", "manual_note",
    ]
    _write_csv(run_dir / "review_images.csv", review_rows, review_fields)
    _write_csv(run_dir / "group_review.csv", summary_rows, summary_fields)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("left {task.fields[remaining]}"),
        TextColumn("group {task.fields[current_group]}"),
        TextColumn("failures {task.fields[failures]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=not show_progress,
    )

    with progress:
        preview_task = progress.add_task(
            "Rendering RAW previews",
            total=len(review_rows),
            remaining=len(review_rows),
            current_group="-",
            failures=0,
        )

        def preview_progress(
            completed: int,
            total: int,
            row: dict[str, str],
            status: str,
            failures: int,
        ) -> None:
            del status
            progress.update(
                preview_task,
                completed=completed,
                total=total,
                remaining=max(0, total - completed),
                current_group=row.get("primary_candidate_key", "-") or "-",
                failures=failures,
            )

        previews_root = run_dir / "previews"
        preview_map_path = previews_root / "preview_map.csv"
        failures_path = previews_root / "failures.csv"

        if sheets_only:
            if not preview_map_path.exists():
                raise IngestError(
                    f"--sheets-only requires an existing preview map: {preview_map_path}"
                )
            preview_failures = 0
            if failures_path.exists():
                preview_failures = len(_read_csv(failures_path))
            preview_rendered = 0
            progress.update(
                preview_task,
                completed=len(review_rows),
                remaining=0,
                failures=preview_failures,
                description="Using cached previews",
            )
        else:
            preview_result = _build_previews(
                input_path=run_dir / "review_images.csv",
                output_root=previews_root,
                max_size=max_preview_size,
                wb_mode="daylight",
                exposure_mode="center-midtone",
                overwrite=overwrite_previews,
                progress_callback=preview_progress,
            )
            preview_failures = preview_result.failed
            preview_rendered = preview_result.rendered
            preview_map_path = preview_result.preview_map_path
            progress.update(
                preview_task,
                completed=len(review_rows),
                remaining=0,
                failures=preview_failures,
            )

        preview_map = _load_preview_map(preview_map_path)
        rows_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in review_rows:
            rows_by_group[str(row["candidate_group"])].append({k: str(v) for k, v in row.items()})

        detail_dir = run_dir / "contact_sheets" / "groups"
        detail_task = progress.add_task(
            "Building group sheets",
            total=len(summary_rows),
            remaining=len(summary_rows),
            current_group="-",
            failures=0,
        )
        for completed, summary in enumerate(summary_rows, start=1):
            group = str(summary["candidate_group"])
            _draw_detail_sheet(
                group=group,
                rows=rows_by_group[group],
                summary={k: str(v) for k, v in summary.items()},
                output_path=detail_dir / f"{group}_{summary['proposal_status']}.jpg",
                preview_map=preview_map,
            )
            progress.update(
                detail_task,
                completed=completed,
                remaining=len(summary_rows) - completed,
                current_group=group,
            )

        overview_total = math.ceil(len(summary_rows) / 12)
        overview_task = progress.add_task(
            "Building overview pages",
            total=overview_total,
            remaining=overview_total,
            current_group="-",
            failures=0,
        )

        def overview_progress(completed: int, total: int) -> None:
            progress.update(
                overview_task,
                completed=completed,
                total=total,
                remaining=max(0, total - completed),
            )

        overview_count = _draw_overview_pages(
            summaries=[{k: str(v) for k, v in row.items()} for row in summary_rows],
            rows_by_group=rows_by_group,
            preview_map=preview_map,
            output_dir=run_dir / "contact_sheets" / "overview",
            page_callback=overview_progress,
        )

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pool_run": pool_run,
        "proposal_run": proposal_run,
        "selected_pool_samples": selected_pool_samples,
        "unselected_pool_samples": unselected_pool_samples,
        "reference_limit": reference_limit,
        "eligible_groups": len(summary_rows),
        "proposed_groups": len(proposed),
        "review_images": len(review_rows),
        "preview_failures": preview_failures,
        "detail_sheets": len(summary_rows),
        "overview_sheets": overview_count,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (run_dir / "audit.txt").write_text(
        "\n".join([
            f"Run: {run_id}",
            f"Eligible groups: {len(summary_rows)}",
            f"Proposed groups: {len(proposed)}",
            f"Unselected eligible groups: {len(summary_rows) - len(proposed)}",
            f"Review images: {len(review_rows)}",
            f"Preview failures: {preview_failures}",
            f"Detail sheets: {len(summary_rows)}",
            f"Overview sheets: {overview_count}",
        ]) + "\n",
        encoding="utf-8",
    )

    return GroupReviewResult(
        run_id=run_id,
        run_dir=run_dir,
        eligible_groups=len(summary_rows),
        proposed_groups=len(proposed),
        unselected_groups=len(summary_rows) - len(proposed),
        review_images=len(review_rows),
        previews_rendered=preview_rendered,
        preview_failures=preview_failures,
        detail_sheets=len(summary_rows),
        overview_sheets=overview_count,
    )
