from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from nazariya.lureva.identity import candidate_keys
from nazariya.lureva.ingest import IngestError
from nazariya.lureva.paths import LurevaPaths


@dataclass(frozen=True)
class GroupPoolResult:
    run_id: str
    run_dir: Path
    represented_groups: int
    eligible_groups: int
    reference_images: int
    excluded_images: int
    pool_memberships: int
    unique_pool_images: int


@dataclass(frozen=True)
class GroupProposalResult:
    run_id: str
    run_dir: Path
    requested_groups: int
    proposed_groups: int
    eligible_groups: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise IngestError(f"Required CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise IngestError(f"CSV has no header: {path}")
        return [{str(key): str(value or "") for key, value in row.items()} for row in reader]


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_ingest_run(root: Path, run_id: str) -> tuple[Path, list[dict[str, str]], list[dict[str, str]]]:
    run_dir = root / "runs" / run_id
    if not run_dir.exists():
        raise IngestError(f"Ingest run does not exist: {run_dir}")
    catalog = _read_csv(run_dir / "catalog_candidates.csv")
    matches = _read_csv(run_dir / "seed_matches.csv")
    unresolved = [row for row in matches if row.get("match_status") != "matched"]
    if unresolved:
        raise IngestError(
            f"Ingest run {run_id!r} contains {len(unresolved)} unresolved seed rows."
        )
    return run_dir, catalog, matches


def _identity(row: dict[str, str], *, matched: bool = False) -> str:
    if matched:
        return (
            row.get("matched_source_path", "").strip().casefold()
            or row.get("matched_photo_uuid", "").strip().upper()
        )
    return (
        row.get("source_path", "").strip().casefold()
        or row.get("photo_uuid", "").strip().upper()
    )


def build_group_pools(
    *,
    root: Path = Path("data/lureva"),
    reference_run: str,
    exclusion_run: str,
    run_id: str | None = None,
    minimum_images: int = 20,
    preferred_images: int = 30,
) -> GroupPoolResult:
    if minimum_images < 1:
        raise IngestError("minimum_images must be at least 1.")
    if preferred_images < minimum_images:
        raise IngestError("preferred_images must be greater than or equal to minimum_images.")

    paths = LurevaPaths(root)
    paths.create()
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = run_id or datetime.now().strftime("group-pools-%Y%m%d-%H%M%S")
    run_dir = paths.runs / run_id
    if run_dir.exists():
        raise IngestError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    reference_dir, catalog, references = _load_ingest_run(root, reference_run)
    exclusion_dir, exclusion_catalog, exclusions = _load_ingest_run(root, exclusion_run)

    # Both runs should originate from the same normalized Lightroom candidate inventory.
    catalog_ids = {_identity(row) for row in catalog}
    exclusion_catalog_ids = {_identity(row) for row in exclusion_catalog}
    if catalog_ids != exclusion_catalog_ids:
        raise IngestError(
            "Reference and exclusion runs do not contain the same normalized catalog universe."
        )

    represented: set[str] = set()
    reference_rows: list[dict[str, object]] = []
    reference_counts: Counter[str] = Counter()
    for row in references:
        groups = candidate_keys(row.get("matched_candidate_keys", ""))
        represented.update(groups)
        for group in groups:
            reference_counts[group] += 1
            reference_rows.append({
                "candidate_group": group,
                "photo_uuid": row.get("matched_photo_uuid", ""),
                "source_path": row.get("matched_source_path", ""),
                "source_file_name": row.get("matched_source_file_name", ""),
                "corpus_file_name": row.get("corpus_file_name", ""),
                "seed_origin": row.get("seed_origin", ""),
                "replacement_of_corpus_file_name": row.get(
                    "replacement_of_corpus_file_name", ""
                ),
            })

    exclusion_ids = {_identity(row, matched=True) for row in exclusions}
    exclusion_ids.discard("")

    pool_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    group_total: Counter[str] = Counter()
    group_excluded: Counter[str] = Counter()
    group_unused: Counter[str] = Counter()
    group_multigroup_unused: Counter[str] = Counter()
    unique_unused_ids: set[str] = set()

    for row in catalog:
        all_groups = candidate_keys(row.get("candidate_keys", ""))
        groups = [group for group in all_groups if group in represented]
        if not groups:
            continue
        identity = _identity(row)
        is_excluded = identity in exclusion_ids
        represented_membership_count = len(groups)

        for group in groups:
            group_total[group] += 1
            if is_excluded:
                group_excluded[group] += 1
            else:
                group_unused[group] += 1
                unique_unused_ids.add(identity)
                if represented_membership_count > 1:
                    group_multigroup_unused[group] += 1
            pool_rows.append({
                "candidate_group": group,
                "photo_uuid": row.get("photo_uuid", ""),
                "catalog_photo_uuids": row.get("catalog_photo_uuids", ""),
                "source_path": row.get("source_path", ""),
                "source_file_name": row.get("source_file_name", ""),
                "capture_date": row.get("capture_date", ""),
                "all_candidate_keys": row.get("candidate_keys", ""),
                "represented_membership_count": represented_membership_count,
                "is_existing_corpus_image": int(is_excluded),
                "is_selectable": int(not is_excluded),
            })

        if not is_excluded and represented_membership_count > 1:
            overlap_rows.append({
                "photo_uuid": row.get("photo_uuid", ""),
                "source_path": row.get("source_path", ""),
                "source_file_name": row.get("source_file_name", ""),
                "represented_candidate_groups": " ; ".join(groups),
                "represented_membership_count": represented_membership_count,
            })

    summaries: list[dict[str, object]] = []
    for group in sorted(represented):
        unused = group_unused[group]
        overlap = group_multigroup_unused[group]
        if unused >= preferred_images:
            readiness = "strong"
        elif unused >= minimum_images:
            readiness = "borderline"
        else:
            readiness = "ineligible"
        summaries.append({
            "candidate_group": group,
            "reference_count": reference_counts[group],
            "catalog_raw_count": group_total[group],
            "excluded_existing_count": group_excluded[group],
            "unused_raw_count": unused,
            "multi_group_unused_count": overlap,
            "single_group_unused_count": unused - overlap,
            "eligible_for_20": int(unused >= minimum_images),
            "readiness": readiness,
        })

    eligible = [row for row in summaries if int(row["eligible_for_20"])]
    pool_rows.sort(key=lambda row: (str(row["candidate_group"]), str(row["source_path"])))
    overlap_rows.sort(key=lambda row: str(row["source_path"]))
    reference_rows.sort(key=lambda row: (str(row["candidate_group"]), str(row["source_path"])))

    pool_fields = [
        "candidate_group", "photo_uuid", "catalog_photo_uuids", "source_path",
        "source_file_name", "capture_date", "all_candidate_keys",
        "represented_membership_count", "is_existing_corpus_image", "is_selectable",
    ]
    summary_fields = [
        "candidate_group", "reference_count", "catalog_raw_count",
        "excluded_existing_count", "unused_raw_count", "multi_group_unused_count",
        "single_group_unused_count", "eligible_for_20", "readiness",
    ]
    reference_fields = [
        "candidate_group", "photo_uuid", "source_path", "source_file_name",
        "corpus_file_name", "seed_origin", "replacement_of_corpus_file_name",
    ]
    overlap_fields = [
        "photo_uuid", "source_path", "source_file_name",
        "represented_candidate_groups", "represented_membership_count",
    ]

    _write_csv(run_dir / "group_pools.csv", pool_rows, pool_fields)
    _write_csv(run_dir / "group_summary.csv", summaries, summary_fields)
    _write_csv(run_dir / "eligible_groups.csv", eligible, summary_fields)
    _write_csv(
        run_dir / "ineligible_groups.csv",
        [row for row in summaries if not int(row["eligible_for_20"])],
        summary_fields,
    )
    _write_csv(run_dir / "reference_images.csv", reference_rows, reference_fields)
    _write_csv(run_dir / "cross_group_memberships.csv", overlap_rows, overlap_fields)

    metadata = {
        "run_id": run_id,
        "created_at": created_at,
        "reference_run": reference_run,
        "reference_run_dir": str(reference_dir),
        "exclusion_run": exclusion_run,
        "exclusion_run_dir": str(exclusion_dir),
        "minimum_images": minimum_images,
        "preferred_images": preferred_images,
        "represented_groups": len(represented),
        "eligible_groups": len(eligible),
        "reference_images": len({_identity(row, matched=True) for row in references}),
        "excluded_images": len(exclusion_ids),
        "pool_memberships": len(pool_rows),
        "unique_pool_images": len(unique_unused_ids),
    }
    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    readiness_counts = Counter(str(row["readiness"]) for row in summaries)
    report = [
        "Lureva Milestone 2 group-pool audit",
        "=" * 38,
        f"Run: {run_id}",
        f"Created: {created_at}",
        f"Reference ingest run: {reference_run}",
        f"Exclusion ingest run: {exclusion_run}",
        "",
        f"Part 1 reference images:             {metadata['reference_images']:,}",
        f"Existing corpus images excluded:     {metadata['excluded_images']:,}",
        f"Represented candidate groups:        {len(represented):,}",
        f"Eligible groups (>= {minimum_images} unused):      {len(eligible):,}",
        f"Unique unused RAWs across pools:     {len(unique_unused_ids):,}",
        f"Candidate-group pool memberships:   {len(pool_rows):,}",
        f"Cross-group unused RAWs:             {len(overlap_rows):,}",
        "",
        "Readiness:",
        f"  strong (>= {preferred_images}): {readiness_counts['strong']:,}",
        f"  borderline ({minimum_images}-{preferred_images - 1}): {readiness_counts['borderline']:,}",
        f"  ineligible (< {minimum_images}): {readiness_counts['ineligible']:,}",
    ]
    (run_dir / "audit.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    for name in ("group_summary.csv", "eligible_groups.csv", "audit.txt", "run.json"):
        destination = (
            paths.manifests / f"{run_id}_{name}"
            if name.endswith(".csv")
            else paths.reports / f"{run_id}_{name}"
        )
        shutil.copy2(run_dir / name, destination)

    return GroupPoolResult(
        run_id=run_id,
        run_dir=run_dir,
        represented_groups=len(represented),
        eligible_groups=len(eligible),
        reference_images=int(metadata["reference_images"]),
        excluded_images=len(exclusion_ids),
        pool_memberships=len(pool_rows),
        unique_pool_images=len(unique_unused_ids),
    )


def propose_groups(
    *,
    root: Path = Path("data/lureva"),
    pool_run: str,
    count: int = 48,
    run_id: str | None = None,
) -> GroupProposalResult:
    if count < 1:
        raise IngestError("count must be at least 1.")

    paths = LurevaPaths(root)
    paths.create()
    pool_dir = paths.runs / pool_run
    summaries = _read_csv(pool_dir / "group_summary.csv")
    eligible = [row for row in summaries if row.get("eligible_for_20") == "1"]
    if len(eligible) < count:
        raise IngestError(
            f"Requested {count} groups, but only {len(eligible)} are eligible in {pool_run!r}."
        )

    def rank_key(row: dict[str, str]) -> tuple[int, int, int, int, str]:
        unused = int(row.get("unused_raw_count", "0"))
        references = int(row.get("reference_count", "0"))
        overlap = int(row.get("multi_group_unused_count", "0"))
        single = int(row.get("single_group_unused_count", "0"))
        return (-unused, -single, -references, overlap, row.get("candidate_group", ""))

    ranked = sorted(eligible, key=rank_key)
    selected = ranked[:count]
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = run_id or datetime.now().strftime("group-proposal-%Y%m%d-%H%M%S")
    run_dir = paths.runs / run_id
    if run_dir.exists():
        raise IngestError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    proposal_rows: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        proposal_rows.append({
            "proposal_rank": rank,
            **row,
            "status": "proposed",
            "replacement_group": "",
            "manual_note": "",
        })

    fields = [
        "proposal_rank", "candidate_group", "reference_count", "catalog_raw_count",
        "excluded_existing_count", "unused_raw_count", "multi_group_unused_count",
        "single_group_unused_count", "eligible_for_20", "readiness", "status",
        "replacement_group", "manual_note",
    ]
    _write_csv(run_dir / "proposed_groups.csv", proposal_rows, fields)
    _write_csv(run_dir / "ranked_eligible_groups.csv", [
        {"proposal_rank": rank, **row}
        for rank, row in enumerate(ranked, start=1)
    ], fields[:-3])

    metadata = {
        "run_id": run_id,
        "created_at": created_at,
        "pool_run": pool_run,
        "requested_groups": count,
        "proposed_groups": len(selected),
        "eligible_groups": len(eligible),
        "ranking": [
            "unused_raw_count descending",
            "single_group_unused_count descending",
            "reference_count descending",
            "multi_group_unused_count ascending",
            "candidate_group ascending",
        ],
    }
    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "Lureva Milestone 2 initial group proposal",
        "=" * 42,
        f"Run: {run_id}",
        f"Created: {created_at}",
        f"Pool run: {pool_run}",
        f"Eligible groups: {len(eligible):,}",
        f"Proposed groups: {len(selected):,}",
        "",
        "This is a deterministic first proposal based on pool depth and low overlap.",
        "Visual review and swaps happen before the 48 groups are finalized.",
    ]
    (run_dir / "audit.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    shutil.copy2(run_dir / "proposed_groups.csv", paths.manifests / f"{run_id}_proposed_groups.csv")
    shutil.copy2(run_dir / "audit.txt", paths.reports / f"{run_id}_audit.txt")
    shutil.copy2(run_dir / "run.json", paths.reports / f"{run_id}_run.json")

    return GroupProposalResult(
        run_id=run_id,
        run_dir=run_dir,
        requested_groups=count,
        proposed_groups=len(selected),
        eligible_groups=len(eligible),
    )
