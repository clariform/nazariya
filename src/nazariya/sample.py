from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleSummary:
    input_path: Path
    output_path: Path
    total_rows: int
    sampled_rows: int
    candidate_count: int
    per_candidate: int
    seed: int
    candidates_with_fewer_than_requested: int


@dataclass(frozen=True)
class SwapSampleSummary:
    full_input_path: Path
    sample_input_path: Path
    output_path: Path
    candidate_key: str
    removed_count: int
    added_count: int
    final_rows: int
    added_file_name: str
    added_source_path: str


def split_candidate_keys(value: str) -> list[str]:
    out: list[str] = []

    for item in str(value or "").split(";"):
        key = item.strip()
        if key:
            out.append(key)

    return out


def row_candidate_key(row: dict[str, str], group_column: str = "primary_candidate_key") -> str:
    value = str(row.get(group_column, "")).strip()
    if value:
        keys = split_candidate_keys(value)
        if keys:
            return keys[0]

    for fallback in ("primary_candidate_key", "candidate_keys"):
        value = str(row.get(fallback, "")).strip()
        if value:
            keys = split_candidate_keys(value)
            if keys:
                return keys[0]

    return ""


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        rows = list(reader)
        return list(reader.fieldnames), rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def sample_candidates(
    *,
    input_path: Path,
    output_path: Path,
    per_candidate: int,
    seed: int,
    group_column: str = "primary_candidate_key",
) -> SampleSummary:
    if per_candidate <= 0:
        raise ValueError("--per-candidate must be greater than zero")

    fieldnames, rows = read_csv_rows(input_path)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        key = row_candidate_key(row, group_column)

        if not key:
            continue

        grouped[key].append(row)

    rng = random.Random(seed)
    sampled: list[dict[str, str]] = []
    fewer_than_requested = 0

    for key in sorted(grouped.keys()):
        group_rows = grouped[key]

        if len(group_rows) <= per_candidate:
            selected = list(group_rows)

            if len(group_rows) < per_candidate:
                fewer_than_requested += 1
        else:
            selected = rng.sample(group_rows, per_candidate)

        selected.sort(
            key=lambda r: (
                str(r.get("capture_time", "")),
                str(r.get("source_path", "")),
            )
        )
        sampled.extend(selected)

    sampled.sort(
        key=lambda r: (
            row_candidate_key(r, group_column),
            str(r.get("capture_time", "")),
            str(r.get("source_path", "")),
        )
    )

    write_csv_rows(output_path, fieldnames, sampled)

    return SampleSummary(
        input_path=input_path,
        output_path=output_path,
        total_rows=len(rows),
        sampled_rows=len(sampled),
        candidate_count=len(grouped),
        per_candidate=per_candidate,
        seed=seed,
        candidates_with_fewer_than_requested=fewer_than_requested,
    )


def row_matches_remove_selector(
    row: dict[str, str],
    *,
    remove_file: str | None,
    remove_source_path: str | None,
    remove_image_id: str | None,
) -> bool:
    if remove_file:
        wanted = remove_file.strip()
        if str(row.get("file_name", "")).strip() == wanted:
            return True
        if str(row.get("file_stem", "")).strip() == Path(wanted).stem:
            return True

    if remove_source_path:
        if str(row.get("source_path", "")).strip() == remove_source_path.strip():
            return True

    if remove_image_id:
        # Some future CSVs may include image_id, current Lightroom candidate CSV does not.
        if str(row.get("image_id", "")).strip() == remove_image_id.strip():
            return True

    return False


def source_path_key(row: dict[str, str]) -> str:
    return str(row.get("source_path", "")).strip()


def swap_sample_row(
    *,
    full_input_path: Path,
    sample_input_path: Path,
    output_path: Path,
    candidate_key: str,
    remove_file: str | None = None,
    remove_source_path: str | None = None,
    remove_image_id: str | None = None,
    seed: int = 42,
    group_column: str = "primary_candidate_key",
) -> SwapSampleSummary:
    if not remove_file and not remove_source_path and not remove_image_id:
        raise ValueError("Provide one of --remove-file, --remove-source-path, or --remove-image-id")

    full_fieldnames, full_rows = read_csv_rows(full_input_path)
    sample_fieldnames, sample_rows = read_csv_rows(sample_input_path)

    # Preserve the sample CSV schema.
    fieldnames = sample_fieldnames or full_fieldnames

    candidate_key = candidate_key.strip()

    removed_rows: list[dict[str, str]] = []
    kept_rows: list[dict[str, str]] = []

    for row in sample_rows:
        is_target_candidate = row_candidate_key(row, group_column) == candidate_key
        should_remove = is_target_candidate and row_matches_remove_selector(
            row,
            remove_file=remove_file,
            remove_source_path=remove_source_path,
            remove_image_id=remove_image_id,
        )

        if should_remove:
            removed_rows.append(row)
        else:
            kept_rows.append(row)

    if not removed_rows:
        raise ValueError(
            f"No matching sampled row found for candidate={candidate_key!r}. "
            "Check --remove-file or --remove-source-path."
        )

    existing_paths = {source_path_key(row) for row in kept_rows if source_path_key(row)}

    replacement_pool = [
        row
        for row in full_rows
        if row_candidate_key(row, group_column) == candidate_key
        and source_path_key(row)
        and source_path_key(row) not in existing_paths
        and not row_matches_remove_selector(
            row,
            remove_file=remove_file,
            remove_source_path=remove_source_path,
            remove_image_id=remove_image_id,
        )
    ]

    if not replacement_pool:
        raise ValueError(f"No replacement candidates available for {candidate_key}")

    rng = random.Random(seed)
    added = rng.choice(replacement_pool)

    out_rows = kept_rows + [added]

    out_rows.sort(
        key=lambda r: (
            row_candidate_key(r, group_column),
            str(r.get("capture_time", "")),
            str(r.get("source_path", "")),
        )
    )

    write_csv_rows(output_path, fieldnames, out_rows)

    return SwapSampleSummary(
        full_input_path=full_input_path,
        sample_input_path=sample_input_path,
        output_path=output_path,
        candidate_key=candidate_key,
        removed_count=len(removed_rows),
        added_count=1,
        final_rows=len(out_rows),
        added_file_name=str(added.get("file_name", "")),
        added_source_path=str(added.get("source_path", "")),
    )
