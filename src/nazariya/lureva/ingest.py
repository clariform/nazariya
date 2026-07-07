from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

from nazariya.lureva.identity import (
    candidate_keys,
    normalize_stem,
    parse_corpus_name,
    parse_date,
    part2_keys,
    source_key,
    split_keywords,
)
from nazariya.lureva.paths import LurevaPaths

CATALOG_ALIASES = {
    "photo_uuid": ("photo_uuid", "uuid", "lightroom_photo_uuid"),
    "source_path": ("source_path", "path", "original_path", "raw_path"),
    "file_name": ("file_name", "filename", "original_file_name", "source_file_name"),
    "file_stem": ("file_stem", "source_stem", "raw_stem"),
    "capture_time": ("capture_time", "date_time_original", "datetime_original", "capture_date"),
    "candidate_keys": ("candidate_keys", "candidate_group", "group_key"),
    "primary_candidate_key": ("primary_candidate_key", "candidate_group", "group_key"),
    "all_keywords": ("all_keywords", "keywords", "keyword_list"),
    "file_extension": ("file_extension", "extension"),
    "file_format": ("file_format", "format"),
    "rating": ("rating",),
    "label_color": ("label_color", "color_label"),
}

SEED_ALIASES = {
    "photo_uuid": ("photo_uuid", "uuid", "lightroom_photo_uuid"),
    "source_path": ("source_path", "original_path", "raw_path", "source_raw_path"),
    "file_name": ("file_name", "filename", "corpus_file_name", "dng_file_name"),
    "capture_time": ("capture_time", "capture_date", "date_time_original", "datetime_original"),
    "candidate_keys": ("candidate_keys", "candidate_group", "group_key"),
    "primary_candidate_key": ("primary_candidate_key", "candidate_group", "group_key"),
    "all_keywords": ("all_keywords", "keywords", "keyword_list"),
    "file_extension": ("file_extension", "extension"),
    "file_format": ("file_format", "format"),
    "rating": ("rating",),
    "label_color": ("label_color", "color_label"),
}


@dataclass(frozen=True)
class IngestResult:
    run_id: str
    run_dir: Path
    database_path: Path
    catalog_rows_read: int
    catalog_candidate_rows: int
    catalog_raw_rows: int
    catalog_unique_raws: int
    catalog_duplicate_records: int
    catalog_non_raw_rows: int
    seed_rows_read: int
    seed_rows_after_filter: int
    matched_rows: int
    ambiguous_rows: int
    unmatched_rows: int
    represented_groups: int
    overrides_applied: int
    exclude_part2: bool


class IngestError(RuntimeError):
    pass


def _first(row: Mapping[str, str], aliases: Iterable[str]) -> str:
    for key in aliases:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise IngestError(f"CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise IngestError(f"CSV has no header: {path}")
        rows = [{str(k): str(v or "") for k, v in row.items()} for row in reader]
        return list(reader.fieldnames), rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_extension(row: Mapping[str, str], file_name: str) -> str:
    value = _first(row, CATALOG_ALIASES["file_extension"]).strip().lower()
    if value and not value.startswith("."):
        value = f".{value}"
    return value or Path(file_name).suffix.lower()


RAW_EXTENSIONS = {
    ".3fr", ".arw", ".cr2", ".cr3", ".dng", ".erf", ".fff",
    ".iiq", ".kdc", ".mef", ".mos", ".mrw", ".nef", ".nrw",
    ".orf", ".pef", ".raf", ".raw", ".rw2", ".rwl", ".sr2",
    ".srf", ".srw", ".x3f",
}


def _rating_value(value: str) -> float:
    try:
        return float(value.strip())
    except (AttributeError, ValueError):
        return -1.0


def _normalize_catalog(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    candidate_rows: list[dict[str, object]] = []
    stats = {
        "ignored_without_candidate": 0,
        "non_raw_rows": 0,
        "raw_rows": 0,
        "duplicate_records": 0,
    }

    for row_number, row in enumerate(rows, start=2):
        path = _first(row, CATALOG_ALIASES["source_path"])
        file_name = _first(row, CATALOG_ALIASES["file_name"]) or Path(path).name
        all_keywords = _first(row, CATALOG_ALIASES["all_keywords"])
        raw_candidate_keys = _first(row, CATALOG_ALIASES["candidate_keys"])
        primary = _first(row, CATALOG_ALIASES["primary_candidate_key"]).lower()
        keys = candidate_keys(raw_candidate_keys, primary, all_keywords)
        if not keys:
            stats["ignored_without_candidate"] += 1
            continue

        extension = _raw_extension(row, file_name)
        file_format = _first(row, CATALOG_ALIASES["file_format"]).upper()
        if extension not in RAW_EXTENSIONS and file_format != "RAW":
            stats["non_raw_rows"] += 1
            continue
        stats["raw_rows"] += 1

        stem = _first(row, CATALOG_ALIASES["file_stem"]) or normalize_stem(file_name)
        capture_time = _first(row, CATALOG_ALIASES["capture_time"])
        capture_date = parse_date(capture_time) or parse_date(path)
        if primary not in keys:
            primary = keys[0]

        candidate_rows.append({
            "catalog_row_number": row_number,
            "photo_uuid": _first(row, CATALOG_ALIASES["photo_uuid"]),
            "source_path": path,
            "source_dir": str(Path(path).parent) if path else "",
            "source_file_name": file_name,
            "file_extension": extension,
            "file_format": file_format,
            "raw_stem": normalize_stem(stem),
            "capture_time": capture_time,
            "capture_date": capture_date,
            "source_key": source_key(stem, capture_date),
            "candidate_keys": " ; ".join(keys),
            "primary_candidate_key": primary,
            "all_keywords": all_keywords,
            "part2_keys": " ; ".join(part2_keys(all_keywords)),
            "rating": _first(row, CATALOG_ALIASES["rating"]),
            "label_color": _first(row, CATALOG_ALIASES["label_color"]),
            "original_row_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
        })

    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    missing_path: list[dict[str, object]] = []
    for row in candidate_rows:
        path = str(row.get("source_path", "")).strip()
        if path:
            by_source[path.casefold()].append(row)
        else:
            missing_path.append(row)

    normalized: list[dict[str, object]] = []
    groups = list(by_source.values()) + [[row] for row in missing_path]
    for group in groups:
        stats["duplicate_records"] += len(group) - 1
        representative = sorted(
            group,
            key=lambda row: (
                -_rating_value(str(row.get("rating", ""))),
                -len(candidate_keys(str(row.get("candidate_keys", "")))),
                str(row.get("photo_uuid", "")),
            ),
        )[0]
        merged_candidates = sorted({
            key
            for row in group
            for key in candidate_keys(str(row.get("candidate_keys", "")))
        })
        merged_part2 = sorted({
            key
            for row in group
            for key in part2_keys(str(row.get("part2_keys", "")), str(row.get("all_keywords", "")))
        })
        uuids = sorted({str(row.get("photo_uuid", "")).strip() for row in group if str(row.get("photo_uuid", "")).strip()})
        row_numbers = sorted(int(row["catalog_row_number"]) for row in group)
        ratings = sorted({str(row.get("rating", "")).strip() for row in group if str(row.get("rating", "")).strip()})
        labels = sorted({str(row.get("label_color", "")).strip() for row in group if str(row.get("label_color", "")).strip()})
        keyword_sets = sorted({str(row.get("all_keywords", "")).strip() for row in group if str(row.get("all_keywords", "")).strip()})
        primary = str(representative.get("primary_candidate_key", ""))
        if primary not in merged_candidates:
            primary = merged_candidates[0]

        normalized.append({
            **representative,
            "catalog_row_numbers": " ; ".join(str(value) for value in row_numbers),
            "catalog_photo_uuids": " ; ".join(uuids),
            "catalog_record_count": len(group),
            "is_duplicate_catalog_record": int(len(group) > 1),
            "candidate_keys": " ; ".join(merged_candidates),
            "primary_candidate_key": primary,
            "all_keywords": " || ".join(keyword_sets),
            "part2_keys": " ; ".join(merged_part2),
            "ratings": " ; ".join(ratings),
            "label_colors": " ; ".join(labels),
        })

    normalized.sort(key=lambda row: str(row.get("source_path", "")).casefold())
    return normalized, stats


def _normalize_seed(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=2):
        source_path = _first(row, SEED_ALIASES["source_path"])
        file_name = _first(row, SEED_ALIASES["file_name"]) or Path(source_path).name
        parsed_stem, parsed_date = parse_corpus_name(file_name)
        explicit_date = parse_date(_first(row, SEED_ALIASES["capture_time"]))
        capture_date = explicit_date or parsed_date
        all_keywords = _first(row, SEED_ALIASES["all_keywords"])
        raw_candidate_keys = _first(row, SEED_ALIASES["candidate_keys"])
        primary = _first(row, SEED_ALIASES["primary_candidate_key"]).lower()
        keys = candidate_keys(raw_candidate_keys, primary, all_keywords)
        p2 = part2_keys(all_keywords)
        normalized.append({
            "seed_row_number": row_number,
            "photo_uuid": _first(row, SEED_ALIASES["photo_uuid"]),
            "corpus_path": source_path,
            "corpus_file_name": file_name,
            "raw_stem": parsed_stem,
            "capture_date": capture_date,
            "source_key": source_key(parsed_stem, capture_date),
            "seed_candidate_keys": " ; ".join(keys),
            "seed_primary_candidate_key": primary if primary in keys else (keys[0] if keys else ""),
            "all_keywords": all_keywords,
            "part2_keys": " ; ".join(p2),
            "is_part2": int(bool(p2)),
            "original_row_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
        })
    return normalized



def _apply_seed_overrides(
    seed: list[dict[str, object]],
    overrides_csv: Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if overrides_csv is None:
        return seed, []

    _, raw_overrides = _read_csv(overrides_csv)
    applied: list[dict[str, object]] = []

    for override_number, override in enumerate(raw_overrides, start=2):
        action = override.get("action", "").strip().lower()
        if action != "replace":
            raise IngestError(
                f"Unsupported seed override action {action!r} on line {override_number}."
            )

        old_uuid = override.get("old_photo_uuid", "").strip()
        old_path = override.get("old_source_path", "").strip()
        old_name = override.get("old_corpus_file_name", "").strip()

        matches = [
            row for row in seed
            if (old_uuid and str(row.get("photo_uuid", "")) == old_uuid)
            or (old_path and str(row.get("corpus_path", "")) == old_path)
            or (old_name and str(row.get("corpus_file_name", "")) == old_name)
        ]
        if len(matches) != 1:
            raise IngestError(
                f"Seed override line {override_number} matched {len(matches)} rows; expected exactly 1."
            )

        row = matches[0]
        new_uuid = override.get("new_photo_uuid", "").strip()
        new_path = override.get("new_source_path", "").strip()
        new_name = override.get("new_corpus_file_name", "").strip() or Path(new_path).name
        new_stem, new_date = parse_corpus_name(new_name)
        new_date = new_date or parse_date(new_path)

        row.update({
            "photo_uuid": new_uuid,
            "corpus_path": new_path,
            "corpus_file_name": new_name,
            "raw_stem": new_stem,
            "capture_date": new_date,
            "source_key": source_key(new_stem, new_date),
            "seed_origin": "replacement",
            "replacement_of_photo_uuid": old_uuid,
            "replacement_of_source_path": old_path,
            "replacement_of_corpus_file_name": old_name,
            "override_reason": override.get("reason", "").strip(),
        })
        applied.append({
            "override_row_number": override_number,
            "action": action,
            **override,
        })

    return seed, applied


def _indexes(catalog: list[dict[str, object]]) -> dict[str, dict[str, list[int]]]:
    indexes: dict[str, dict[str, list[int]]] = {
        "photo_uuid": defaultdict(list),
        "source_path": defaultdict(list),
        "source_key": defaultdict(list),
        "raw_stem": defaultdict(list),
    }
    for index, row in enumerate(catalog):
        for key in indexes:
            if key == "photo_uuid":
                values = split_keywords(str(row.get("catalog_photo_uuids", "")).replace(" ; ", "|"))
                if not values:
                    values = [str(row.get("photo_uuid", "")).strip()]
            else:
                values = [str(row.get(key, "")).strip()]
            for value in values:
                if value:
                    normalized = value.casefold() if key == "source_path" else value.upper()
                    indexes[key][normalized].append(index)
    return indexes


def _match_seed(
    seed: list[dict[str, object]],
    catalog: list[dict[str, object]],
) -> list[dict[str, object]]:
    indexes = _indexes(catalog)
    output: list[dict[str, object]] = []
    for seed_row in seed:
        candidates: list[int] = []
        method = ""
        attempts = [
            ("photo_uuid", str(seed_row.get("photo_uuid", ""))),
            ("source_path", str(seed_row.get("corpus_path", ""))),
            ("source_key", str(seed_row.get("source_key", ""))),
            ("raw_stem", str(seed_row.get("raw_stem", ""))),
        ]
        for key, value in attempts:
            if not value:
                continue
            lookup = value.casefold() if key == "source_path" else value.upper()
            found = indexes[key].get(lookup, [])
            if found:
                candidates = list(found)
                method = key
                break

        status = "unmatched"
        if len(candidates) == 1:
            status = "matched"
        elif len(candidates) > 1:
            status = "ambiguous"

        matched_catalog = catalog[candidates[0]] if len(candidates) == 1 else {}
        output.append({
            **seed_row,
            "match_status": status,
            "match_method": method,
            "match_candidate_count": len(candidates),
            "matched_photo_uuid": matched_catalog.get("photo_uuid", ""),
            "matched_catalog_photo_uuids": matched_catalog.get("catalog_photo_uuids", ""),
            "matched_source_path": matched_catalog.get("source_path", ""),
            "matched_source_file_name": matched_catalog.get("source_file_name", ""),
            "matched_source_key": matched_catalog.get("source_key", ""),
            "matched_candidate_keys": matched_catalog.get("candidate_keys", ""),
            "matched_primary_candidate_key": matched_catalog.get("primary_candidate_key", ""),
            "matched_all_keywords": matched_catalog.get("all_keywords", ""),
        })
    return output


def _group_summary(matched: list[dict[str, object]], catalog: list[dict[str, object]]) -> list[dict[str, object]]:
    catalog_counts: Counter[str] = Counter()
    for row in catalog:
        for key in candidate_keys(str(row.get("candidate_keys", ""))):
            catalog_counts[key] += 1

    seed_counts: Counter[str] = Counter()
    part1_counts: Counter[str] = Counter()
    part2_counts: Counter[str] = Counter()
    for row in matched:
        if row.get("match_status") != "matched":
            continue
        keys = candidate_keys(str(row.get("matched_candidate_keys", "")))
        for key in keys:
            seed_counts[key] += 1
            if int(row.get("is_part2", 0)):
                part2_counts[key] += 1
            else:
                part1_counts[key] += 1

    return [
        {
            "candidate_group": key,
            "catalog_image_count": catalog_counts[key],
            "seed_image_count": seed_counts[key],
            "part1_seed_count": part1_counts[key],
            "part2_seed_count": part2_counts[key],
        }
        for key in sorted(seed_counts)
    ]


def _create_database(
    database_path: Path,
    run_id: str,
    catalog: list[dict[str, object]],
    seed: list[dict[str, object]],
    matched: list[dict[str, object]],
    groups: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.executescript("""
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS ingest_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS catalog_photos (
            run_id TEXT NOT NULL,
            catalog_row_number INTEGER NOT NULL,
            catalog_row_numbers TEXT,
            photo_uuid TEXT,
            catalog_photo_uuids TEXT,
            catalog_record_count INTEGER NOT NULL,
            is_duplicate_catalog_record INTEGER NOT NULL,
            source_path TEXT,
            source_file_name TEXT,
            raw_stem TEXT,
            capture_date TEXT,
            source_key TEXT,
            candidate_keys TEXT,
            primary_candidate_key TEXT,
            all_keywords TEXT,
            part2_keys TEXT,
            rating TEXT,
            ratings TEXT,
            label_color TEXT,
            label_colors TEXT,
            PRIMARY KEY (run_id, catalog_row_number)
        );
        CREATE TABLE IF NOT EXISTS seed_photos (
            run_id TEXT NOT NULL,
            seed_row_number INTEGER NOT NULL,
            corpus_path TEXT,
            corpus_file_name TEXT,
            raw_stem TEXT,
            capture_date TEXT,
            source_key TEXT,
            is_part2 INTEGER NOT NULL,
            match_status TEXT NOT NULL,
            match_method TEXT,
            match_candidate_count INTEGER NOT NULL,
            matched_photo_uuid TEXT,
            matched_source_path TEXT,
            matched_candidate_keys TEXT,
            matched_primary_candidate_key TEXT,
            PRIMARY KEY (run_id, seed_row_number)
        );
        CREATE TABLE IF NOT EXISTS represented_groups (
            run_id TEXT NOT NULL,
            candidate_group TEXT NOT NULL,
            catalog_image_count INTEGER NOT NULL,
            seed_image_count INTEGER NOT NULL,
            part1_seed_count INTEGER NOT NULL,
            part2_seed_count INTEGER NOT NULL,
            PRIMARY KEY (run_id, candidate_group)
        );
        """)
        connection.execute(
            "INSERT OR REPLACE INTO ingest_runs VALUES (?, ?, ?)",
            (run_id, str(metadata["created_at"]), json.dumps(metadata, sort_keys=True)),
        )
        connection.executemany(
            """INSERT OR REPLACE INTO catalog_photos (
                run_id, catalog_row_number, catalog_row_numbers, photo_uuid,
                catalog_photo_uuids, catalog_record_count,
                is_duplicate_catalog_record, source_path, source_file_name,
                raw_stem, capture_date, source_key, candidate_keys,
                primary_candidate_key, all_keywords, part2_keys, rating,
                ratings, label_color, label_colors
            ) VALUES (
                :run_id, :catalog_row_number, :catalog_row_numbers, :photo_uuid,
                :catalog_photo_uuids, :catalog_record_count,
                :is_duplicate_catalog_record, :source_path, :source_file_name,
                :raw_stem, :capture_date, :source_key, :candidate_keys,
                :primary_candidate_key, :all_keywords, :part2_keys, :rating,
                :ratings, :label_color, :label_colors
            )""",
            [{"run_id": run_id, **row} for row in catalog],
        )
        connection.executemany(
            """INSERT OR REPLACE INTO seed_photos VALUES
            (:run_id, :seed_row_number, :corpus_path, :corpus_file_name,
             :raw_stem, :capture_date, :source_key, :is_part2,
             :match_status, :match_method, :match_candidate_count,
             :matched_photo_uuid, :matched_source_path,
             :matched_candidate_keys, :matched_primary_candidate_key)""",
            [{"run_id": run_id, **row} for row in matched],
        )
        connection.executemany(
            """INSERT OR REPLACE INTO represented_groups VALUES
            (:run_id, :candidate_group, :catalog_image_count, :seed_image_count,
             :part1_seed_count, :part2_seed_count)""",
            [{"run_id": run_id, **row} for row in groups],
        )
        connection.commit()


def ingest_milestone_one(
    *,
    catalog_csv: Path,
    seed_csv: Path,
    root: Path = Path("data/lureva"),
    exclude_part2: bool = True,
    run_id: str | None = None,
    expected_catalog_candidates: int | None = 39393,
    expected_seed_rows: int | None = 560,
    overrides_csv: Path | None = None,
) -> IngestResult:
    paths = LurevaPaths(root)
    paths.create()
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = run_id or datetime.now().strftime("ingest-%Y%m%d-%H%M%S")
    run_dir = paths.runs / run_id
    if run_dir.exists():
        raise IngestError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    catalog_fields, catalog_raw = _read_csv(catalog_csv)
    seed_fields, seed_raw = _read_csv(seed_csv)
    catalog, catalog_stats = _normalize_catalog(catalog_raw)
    seed_all = _normalize_seed(seed_raw)
    for row in seed_all:
        row.setdefault("seed_origin", "historical")
        row.setdefault("replacement_of_photo_uuid", "")
        row.setdefault("replacement_of_source_path", "")
        row.setdefault("replacement_of_corpus_file_name", "")
        row.setdefault("override_reason", "")
    seed_all, applied_overrides = _apply_seed_overrides(seed_all, overrides_csv)
    matched_all = _match_seed(seed_all, catalog)
    for row in matched_all:
        effective_part2 = part2_keys(
            str(row.get("all_keywords", "")),
            str(row.get("matched_all_keywords", "")),
        )
        row["part2_keys"] = " ; ".join(effective_part2)
        row["is_part2"] = int(bool(effective_part2))
    matched = [
        row for row in matched_all
        if not (exclude_part2 and int(row["is_part2"]))
    ]
    seed = [dict(row) for row in matched]
    groups = _group_summary(matched, catalog)

    warnings: list[str] = []
    if expected_catalog_candidates is not None and len(catalog) != expected_catalog_candidates:
        warnings.append(
            f"Expected {expected_catalog_candidates} unique RAW source files, found {len(catalog)}."
        )
    if expected_seed_rows is not None and len(seed_all) != expected_seed_rows:
        warnings.append(f"Expected {expected_seed_rows} seed rows, found {len(seed_all)}.")

    matched_count = sum(row["match_status"] == "matched" for row in matched)
    ambiguous_count = sum(row["match_status"] == "ambiguous" for row in matched)
    unmatched_count = sum(row["match_status"] == "unmatched" for row in matched)
    method_counts = Counter(str(row["match_method"] or "none") for row in matched)

    metadata: dict[str, object] = {
        "run_id": run_id,
        "created_at": created_at,
        "catalog_csv": str(catalog_csv.resolve()),
        "seed_csv": str(seed_csv.resolve()),
        "catalog_sha256": _file_hash(catalog_csv),
        "seed_sha256": _file_hash(seed_csv),
        "overrides_csv": str(overrides_csv.resolve()) if overrides_csv else "",
        "overrides_sha256": _file_hash(overrides_csv) if overrides_csv else "",
        "overrides_applied": len(applied_overrides),
        "catalog_fields": catalog_fields,
        "seed_fields": seed_fields,
        "exclude_part2": exclude_part2,
        "catalog_rows_read": len(catalog_raw),
        "catalog_candidate_rows": catalog_stats["raw_rows"] + catalog_stats["non_raw_rows"],
        "catalog_raw_rows": catalog_stats["raw_rows"],
        "catalog_unique_raws": len(catalog),
        "catalog_duplicate_records": catalog_stats["duplicate_records"],
        "catalog_non_raw_rows": catalog_stats["non_raw_rows"],
        "catalog_rows_ignored_without_candidate": catalog_stats["ignored_without_candidate"],
        "seed_rows_read": len(seed_all),
        "seed_part2_rows": sum(int(row["is_part2"]) for row in matched_all),
        "seed_rows_after_filter": len(seed),
        "matched_rows": matched_count,
        "ambiguous_rows": ambiguous_count,
        "unmatched_rows": unmatched_count,
        "represented_groups": len(groups),
        "match_method_counts": dict(sorted(method_counts.items())),
        "warnings": warnings,
    }

    catalog_fields_out = [
        "catalog_row_number", "catalog_row_numbers", "photo_uuid",
        "catalog_photo_uuids", "catalog_record_count",
        "is_duplicate_catalog_record", "source_path", "source_dir",
        "source_file_name", "file_extension", "file_format", "raw_stem",
        "capture_time", "capture_date",
        "source_key", "candidate_keys", "primary_candidate_key",
        "all_keywords", "part2_keys", "rating", "ratings",
        "label_color", "label_colors",
    ]
    seed_fields_out = [
        "seed_row_number", "photo_uuid", "corpus_path", "corpus_file_name",
        "raw_stem", "capture_date", "source_key", "seed_candidate_keys",
        "seed_primary_candidate_key", "all_keywords", "part2_keys", "is_part2",
        "seed_origin", "replacement_of_photo_uuid",
        "replacement_of_source_path", "replacement_of_corpus_file_name",
        "override_reason",
    ]
    match_fields_out = seed_fields_out + [
        "match_status", "match_method", "match_candidate_count",
        "matched_photo_uuid", "matched_catalog_photo_uuids",
        "matched_source_path", "matched_source_file_name",
        "matched_source_key", "matched_candidate_keys",
        "matched_primary_candidate_key", "matched_all_keywords",
    ]
    group_fields_out = [
        "candidate_group", "catalog_image_count", "seed_image_count",
        "part1_seed_count", "part2_seed_count",
    ]

    _write_csv(run_dir / "catalog_candidates.csv", catalog, catalog_fields_out)
    _write_csv(run_dir / "seed_all.csv", seed_all, seed_fields_out)
    _write_csv(run_dir / "seed_matches_all.csv", matched_all, match_fields_out)
    _write_csv(run_dir / "seed_filtered.csv", seed, seed_fields_out)
    _write_csv(run_dir / "seed_matches.csv", matched, match_fields_out)
    _write_csv(run_dir / "represented_groups.csv", groups, group_fields_out)
    if applied_overrides:
        override_fields = list(applied_overrides[0].keys())
        _write_csv(run_dir / "applied_overrides.csv", applied_overrides, override_fields)
    _write_csv(
        run_dir / "match_issues.csv",
        [row for row in matched if row["match_status"] != "matched"],
        match_fields_out,
    )
    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_lines = [
        "Lureva Milestone 1 ingest audit",
        "=" * 34,
        f"Run: {run_id}",
        f"Created: {created_at}",
        "",
        f"Catalog rows read:                    {len(catalog_raw):,}",
        f"Candidate-keyword catalog rows:       {catalog_stats['raw_rows'] + catalog_stats['non_raw_rows']:,}",
        f"RAW catalog records:                  {catalog_stats['raw_rows']:,}",
        f"Unique RAW source files:              {len(catalog):,}",
        f"Duplicate RAW catalog records:        {catalog_stats['duplicate_records']:,}",
        f"Ignored non-RAW candidate rows:       {catalog_stats['non_raw_rows']:,}",
        f"Ignored rows without candidate:       {catalog_stats['ignored_without_candidate']:,}",
        f"Seed rows read:                       {len(seed_all):,}",
        f"Part 2 seed rows detected:            {sum(int(row['is_part2']) for row in matched_all):,}",
        f"Part 2 exclusion active:              {exclude_part2}",
        f"Seed rows after filter:               {len(seed):,}",
        f"Matched seed rows:                    {matched_count:,}",
        f"Ambiguous seed rows:                  {ambiguous_count:,}",
        f"Unmatched seed rows:                  {unmatched_count:,}",
        f"Represented candidate groups:         {len(groups):,}",
        "",
        "Match methods:",
    ]
    report_lines.extend(f"  {key}: {value:,}" for key, value in sorted(method_counts.items()))
    if warnings:
        report_lines.extend(["", "Warnings:", *[f"  - {warning}" for warning in warnings]])
    (run_dir / "audit.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    _create_database(paths.database, run_id, catalog, seed, matched, groups, metadata)

    for name in ("represented_groups.csv", "seed_matches.csv", "match_issues.csv", "audit.txt", "run.json"):
        shutil.copy2(run_dir / name, paths.manifests / f"{run_id}_{name}" if name.endswith(".csv") else paths.reports / f"{run_id}_{name}")

    return IngestResult(
        run_id=run_id,
        run_dir=run_dir,
        database_path=paths.database,
        catalog_rows_read=len(catalog_raw),
        catalog_candidate_rows=catalog_stats["raw_rows"] + catalog_stats["non_raw_rows"],
        catalog_raw_rows=catalog_stats["raw_rows"],
        catalog_unique_raws=len(catalog),
        catalog_duplicate_records=catalog_stats["duplicate_records"],
        catalog_non_raw_rows=catalog_stats["non_raw_rows"],
        seed_rows_read=len(seed_all),
        seed_rows_after_filter=len(seed),
        matched_rows=matched_count,
        ambiguous_rows=ambiguous_count,
        unmatched_rows=unmatched_count,
        represented_groups=len(groups),
        overrides_applied=len(applied_overrides),
        exclude_part2=exclude_part2,
    )
