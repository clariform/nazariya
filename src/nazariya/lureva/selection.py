from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from nazariya.lureva.ingest import IngestError
from nazariya.lureva.paths import LurevaPaths


@dataclass(frozen=True)
class FinalizeGroupsResult:
    run_id: str
    run_dir: Path
    selected_groups: int
    removed_groups: int
    added_groups: int


@dataclass(frozen=True)
class ImagePoolsResult:
    run_id: str
    run_dir: Path
    groups: int
    pool_memberships: int
    unique_images: int
    overlap_assignments: int


@dataclass(frozen=True)
class LightroomManifestResult:
    run_id: str
    run_dir: Path
    groups: int
    primary_images: int
    alternate_images: int
    total_images: int
    manifest_path: Path


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


def _truthy_status(value: str) -> str:
    return value.strip().casefold().replace("-", "_")


def finalize_groups(
    *,
    root: Path = Path("data/lureva"),
    review_run: str,
    run_id: str | None = None,
    expected_groups: int = 48,
    group_prefix: str = "u",
) -> FinalizeGroupsResult:
    group_prefix = group_prefix.strip().lower()
    if not group_prefix or not group_prefix.isalpha():
        raise IngestError("Group prefix must contain letters only.")

    paths = LurevaPaths(root)
    paths.create()
    review_dir = paths.runs / review_run
    review_path = review_dir / "group_review.csv"
    rows = _read_csv(review_path)

    selected: list[dict[str, object]] = []
    removed = 0
    added = 0
    seen: set[str] = set()

    for row in rows:
        group = row.get("candidate_group", "").strip()
        if not group or group in seen:
            continue
        proposal_status = _truthy_status(row.get("proposal_status", ""))
        manual_status = _truthy_status(row.get("manual_status", ""))

        include = proposal_status == "proposed"
        if manual_status in {"remove", "reject", "excluded", "unselect"}:
            include = False
            if proposal_status == "proposed":
                removed += 1
        elif manual_status in {"add", "approve", "selected", "select", "keep"}:
            if proposal_status != "proposed":
                added += 1
            include = True

        if include:
            seen.add(group)
            selected.append({**row})

    if len(selected) != expected_groups:
        raise IngestError(
            f"Expected {expected_groups} final groups, found {len(selected)}. "
            "Update group_review.csv so removes and adds are balanced."
        )

    def sort_key(row: dict[str, object]) -> tuple[int, int, str]:
        proposed = str(row.get("proposal_status", "")).casefold() == "proposed"
        try:
            rank = int(str(row.get("proposal_rank", "")) or 999999)
        except ValueError:
            rank = 999999
        return (0 if proposed else 1, rank, str(row.get("candidate_group", "")))

    selected.sort(key=sort_key)
    for index, row in enumerate(selected, start=1):
        row["final_rank"] = index
        row["final_group"] = f"{group_prefix}{index:03d}"
        row["final_status"] = "selected"

    run_id = run_id or datetime.now().strftime("final-groups-%Y%m%d-%H%M%S")
    run_dir = paths.runs / run_id
    if run_dir.exists():
        raise IngestError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    fields = list(rows[0].keys()) if rows else []
    for field in ("final_rank", "final_group", "final_status"):
        if field not in fields:
            fields.append(field)
    _write_csv(run_dir / "final_48_groups.csv", selected, fields)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "review_run": review_run,
                "review_file": str(review_path),
                "selected_groups": len(selected),
                "removed_groups": removed,
                "added_groups": added,
                "group_prefix": group_prefix,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return FinalizeGroupsResult(run_id, run_dir, len(selected), removed, added)


def prepare_image_pools(
    *,
    root: Path = Path("data/lureva"),
    groups_run: str,
    pool_run: str,
    run_id: str | None = None,
) -> ImagePoolsResult:
    paths = LurevaPaths(root)
    paths.create()
    groups_dir = paths.runs / groups_run
    pools_dir = paths.runs / pool_run
    groups = _read_csv(groups_dir / "final_48_groups.csv")
    pool_rows = _read_csv(pools_dir / "group_pools.csv")
    references = _read_csv(pools_dir / "reference_images.csv")

    group_info = {row["candidate_group"]: row for row in groups}
    memberships: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pool_rows:
        if row.get("is_selectable") != "1":
            continue
        group = row.get("candidate_group", "")
        if group not in group_info:
            continue
        identity = row.get("source_path", "").strip().casefold() or row.get("photo_uuid", "")
        memberships[identity].append(row)

    selected_rows: list[dict[str, object]] = []
    overlaps: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for identity, options in memberships.items():
        options.sort(key=lambda row: int(group_info[row["candidate_group"]]["final_rank"]))
        chosen = options[0]
        group = chosen["candidate_group"]
        info = group_info[group]
        counts[group] += 1
        selected_rows.append({
            **chosen,
            "final_group": info["final_group"],
            "final_group_rank": info["final_rank"],
            "assigned_candidate_group": group,
            "assignment_method": "only_membership" if len(options) == 1 else "lowest_final_rank",
            "selected_group_memberships": " ; ".join(row["candidate_group"] for row in options),
        })
        if len(options) > 1:
            overlaps.append({
                "identity": identity,
                "photo_uuid": chosen.get("photo_uuid", ""),
                "source_path": chosen.get("source_path", ""),
                "candidate_groups": " ; ".join(row["candidate_group"] for row in options),
                "assigned_candidate_group": group,
                "assignment_method": "lowest_final_rank",
            })

    summaries: list[dict[str, object]] = []
    for row in sorted(groups, key=lambda r: int(r["final_rank"])):
        group = row["candidate_group"]
        count = counts[group]
        if count < 20:
            raise IngestError(f"Final group {group} has only {count} selectable unused RAWs.")
        summaries.append({
            "final_rank": row["final_rank"],
            "final_group": row["final_group"],
            "candidate_group": group,
            "reference_count": row.get("reference_count", ""),
            "unused_pool_count": count,
        })

    selected_rows.sort(key=lambda row: (int(str(row["final_group_rank"])), str(row["source_path"])))
    selected_refs = [
        {**row, "final_group": group_info[row["candidate_group"]]["final_group"]}
        for row in references
        if row.get("candidate_group") in group_info
    ]

    run_id = run_id or datetime.now().strftime("image-pools-%Y%m%d-%H%M%S")
    run_dir = paths.runs / run_id
    if run_dir.exists():
        raise IngestError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    pool_fields = list(selected_rows[0].keys()) if selected_rows else []
    _write_csv(run_dir / "image_pools.csv", selected_rows, pool_fields)
    _write_csv(
        run_dir / "group_summary.csv",
        summaries,
        ["final_rank", "final_group", "candidate_group", "reference_count", "unused_pool_count"],
    )
    ref_fields = list(selected_refs[0].keys()) if selected_refs else [
        "candidate_group", "photo_uuid", "source_path", "source_file_name", "corpus_file_name", "final_group"
    ]
    _write_csv(run_dir / "reference_images.csv", selected_refs, ref_fields)
    _write_csv(
        run_dir / "overlap_assignments.csv",
        overlaps,
        ["identity", "photo_uuid", "source_path", "candidate_groups", "assigned_candidate_group", "assignment_method"],
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "groups_run": groups_run,
                "pool_run": pool_run,
                "groups": len(groups),
                "pool_memberships": sum(len(v) for v in memberships.values()),
                "unique_images": len(selected_rows),
                "overlap_assignments": len(overlaps),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ImagePoolsResult(
        run_id,
        run_dir,
        len(groups),
        sum(len(v) for v in memberships.values()),
        len(selected_rows),
        len(overlaps),
    )


def build_lightroom_review_manifest(
    *,
    proposal_csv: Path,
    groups_csv: Path,
    root: Path = Path("data/lureva"),
    run_id: str | None = None,
    primary_count: int = 20,
    alternate_count: int = 5,
) -> LightroomManifestResult:
    proposals = _read_csv(proposal_csv)
    groups = _read_csv(groups_csv)
    group_info = {row["candidate_group"]: row for row in groups}
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in proposals:
        group = row.get("candidate_group", "") or row.get("assigned_candidate_group", "")
        if group not in group_info:
            raise IngestError(f"Proposal contains unknown candidate group: {group!r}")
        by_group[group].append(row)

    output_rows: list[dict[str, object]] = []
    total_primary = 0
    total_alternates = 0
    for group, info in sorted(group_info.items(), key=lambda item: int(item[1]["final_rank"])):
        rows = by_group.get(group, [])
        primaries = [r for r in rows if _truthy_status(r.get("selection_role", "")) in {"primary", "selected", "anchor_proxy", "farthest_point"}]
        alternates = [r for r in rows if _truthy_status(r.get("selection_role", "")) in {"alternate", "reserve"}]
        if len(primaries) != primary_count or len(alternates) < alternate_count:
            raise IngestError(
                f"{group} requires exactly {primary_count} primaries and at least {alternate_count} alternates; "
                f"found {len(primaries)} and {len(alternates)}."
            )
        alternates = sorted(alternates, key=lambda r: int(r.get("selection_rank", "999999") or 999999))[:alternate_count]
        primaries = sorted(primaries, key=lambda r: int(r.get("selection_rank", "999999") or 999999))
        for role, selected_rows in (("primary", primaries), ("alternate", alternates)):
            for local_rank, row in enumerate(selected_rows, start=1):
                source_file_name = row.get("source_file_name", "") or row.get("file_name", "") or Path(row.get("source_path", "")).name
                output_rows.append({
                    "final_group": info["final_group"],
                    "final_group_rank": info["final_rank"],
                    "candidate_group": group,
                    "photo_uuid": row.get("photo_uuid", ""),
                    "source_path": row.get("source_path", ""),
                    "file_name": source_file_name,
                    "proposal_role": role,
                    "proposal_rank": local_rank,
                    "initial_pick_status": 1 if role == "primary" else 0,
                    "nearest_reference_photo_uuid": row.get("nearest_reference_photo_uuid", ""),
                    "nearest_reference_file_name": row.get("nearest_reference_file_name", ""),
                    "reference_distance": row.get("reference_distance", ""),
                    "selection_method": row.get("selection_method", ""),
                })
        total_primary += len(primaries)
        total_alternates += len(alternates)

    paths = LurevaPaths(root)
    paths.create()
    run_id = run_id or datetime.now().strftime("lightroom-review-%Y%m%d-%H%M%S")
    run_dir = paths.runs / run_id
    if run_dir.exists():
        raise IngestError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "lightroom_review_manifest.csv"
    fields = [
        "final_group", "final_group_rank", "candidate_group", "photo_uuid",
        "source_path", "file_name", "proposal_role", "proposal_rank",
        "initial_pick_status", "nearest_reference_photo_uuid",
        "nearest_reference_file_name", "reference_distance", "selection_method",
    ]
    _write_csv(manifest_path, output_rows, fields)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "proposal_csv": str(proposal_csv),
                "groups_csv": str(groups_csv),
                "groups": len(group_info),
                "primary_count_per_group": primary_count,
                "alternate_count_per_group": alternate_count,
                "primary_images": total_primary,
                "alternate_images": total_alternates,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return LightroomManifestResult(
        run_id, run_dir, len(group_info), total_primary, total_alternates,
        len(output_rows), manifest_path,
    )
