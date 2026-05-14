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


def split_candidate_keys(value: str) -> list[str]:
    """
    Lightroom CSV uses fields like:
      c001
      c001 ; c002

    For v1, we keep every listed key, but primary_candidate_key should usually
    be the grouping key.
    """
    out: list[str] = []

    for item in str(value or "").split(";"):
        key = item.strip()
        if key:
            out.append(key)

    return out


def row_candidate_key(row: dict[str, str], group_column: str) -> str:
    """
    Prefer the requested group column.

    In your current Lightroom export, the best grouping column is:
      primary_candidate_key

    Fallbacks are included so older CSVs still work.
    """
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
    skipped_no_key = 0

    for row in rows:
        key = row_candidate_key(row, group_column)

        if not key:
            skipped_no_key += 1
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
