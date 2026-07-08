from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from nazariya.lureva.ingest import IngestError
from nazariya.lureva.paths import LurevaPaths


@dataclass(frozen=True)
class SamplePoolsResult:
    run_id: str
    run_dir: Path
    groups: int
    sampled_images: int
    batches: int
    warnings: int


@dataclass(frozen=True)
class LightroomStructureResult:
    run_id: str
    run_dir: Path
    groups: int
    batches: int
    manifest_path: Path


@dataclass(frozen=True)
class ProcessImageBatchResult:
    run_id: str
    run_dir: Path
    batch_id: str
    groups: int
    primary_images: int
    alternate_images: int
    total_images: int
    assignment_path: Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise IngestError(f"Required CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise IngestError(f"CSV has no header: {path}")
        return [{str(k): str(v or "") for k, v in row.items()} for row in reader]


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)




def _archive_relative_path(path: str) -> str:
    normalized = str(path or "").replace("\\", "/")
    markers = ("/Pictures/Images/", "/proetus/images/")
    lower = normalized.lower()
    for marker in markers:
        index = lower.find(marker.lower())
        if index >= 0:
            return normalized[index + len(marker):].lstrip("/")
    parts = [part for part in normalized.split("/") if part]
    for index, part in enumerate(parts):
        if len(part) == 4 and part.isdigit() and 1900 <= int(part) <= 2100:
            return "/".join(parts[index:])
    return ""


def _env_source_path(relative_path: str, env_name: str = "PROETUS_IMAGES_ROOT") -> str:
    if not relative_path:
        return ""
    return f"${{{env_name}}}/{relative_path}"


def _resolved_env_source_path(relative_path: str, env_name: str = "PROETUS_IMAGES_ROOT") -> str:
    root = os.environ.get(env_name, "").strip()
    if not root or not relative_path:
        return ""
    return str(Path(root) / relative_path)

def _capture_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("capture_time", ""),
        row.get("source_path", ""),
        row.get("file_name", ""),
    )


def _spread_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    if count <= 1:
        return [total // 2]
    indices = {round(i * (total - 1) / (count - 1)) for i in range(count)}
    if len(indices) < count:
        for index in range(total):
            indices.add(index)
            if len(indices) == count:
                break
    return sorted(indices)[:count]


def sample_image_pools(
    *,
    root: Path = Path("data/lureva"),
    pool_run: str,
    max_per_group: int | None = 40,
    strategy: str = "temporal-spread",
    seed: int = 42,
    batch_size: int = 8,
    minimum_images: int = 20,
    warning_images: int = 25,
    run_id: str | None = None,
) -> SamplePoolsResult:
    if max_per_group is not None and max_per_group < minimum_images:
        raise IngestError("max_per_group cannot be lower than minimum_images.")
    if batch_size < 1:
        raise IngestError("batch_size must be at least 1.")
    if strategy not in {"temporal-spread", "random"}:
        raise IngestError("strategy must be temporal-spread or random.")

    paths = LurevaPaths(root)
    source_dir = paths.runs / pool_run
    rows = _read_csv(source_dir / "image_pools.csv")
    summaries = _read_csv(source_dir / "group_summary.csv")
    summary_by_group = {row["final_group"]: row for row in summaries}

    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[row["final_group"]].append(row)

    ordered_groups = sorted(
        by_group,
        key=lambda group: int(summary_by_group[group].get("final_rank", "999999")),
    )
    sampled: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    rng = random.Random(seed)

    for group_index, group in enumerate(ordered_groups):
        pool = sorted(by_group[group], key=_capture_key)
        available = len(pool)
        if available < minimum_images:
            raise IngestError(f"{group} has only {available} available images.")
        target = available if max_per_group is None else min(max_per_group, available)
        if strategy == "temporal-spread":
            chosen = [pool[index] for index in _spread_indices(available, target)]
        else:
            indices = sorted(rng.sample(range(available), target))
            chosen = [pool[index] for index in indices]

        batch_number = group_index // batch_size + 1
        batch_id = f"batch_{batch_number:02d}"
        for sample_rank, row in enumerate(chosen, start=1):
            sampled.append({
                **row,
                "batch_id": batch_id,
                "sample_rank": sample_rank,
                "sample_strategy": strategy,
                "source_pool_count": available,
            })
        group_rows.append({
            "final_rank": summary_by_group[group].get("final_rank", ""),
            "final_group": group,
            "candidate_group": summary_by_group[group].get("candidate_group", ""),
            "batch_id": batch_id,
            "available_images": available,
            "sampled_images": len(chosen),
            "sample_strategy": strategy,
            "warning": "low_headroom" if available < warning_images else "",
        })
        if available < warning_images:
            warnings.append({
                "final_group": group,
                "candidate_group": summary_by_group[group].get("candidate_group", ""),
                "available_images": available,
                "warning": f"Only {available} images available; limited swap headroom.",
            })

    run_id = run_id or datetime.now().strftime("sampled-pools-%Y%m%d-%H%M%S")
    run_dir = paths.runs / run_id
    if run_dir.exists():
        raise IngestError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    sampled_fields = list(sampled[0].keys()) if sampled else []
    _write_csv(run_dir / "sampled_image_pools.csv", sampled, sampled_fields)
    _write_csv(
        run_dir / "group_summary.csv",
        group_rows,
        ["final_rank", "final_group", "candidate_group", "batch_id", "available_images", "sampled_images", "sample_strategy", "warning"],
    )
    batch_rows: list[dict[str, object]] = []
    batch_count = math.ceil(len(ordered_groups) / batch_size)
    for batch_number in range(1, batch_count + 1):
        batch_id = f"batch_{batch_number:02d}"
        batch_group_rows = [row for row in group_rows if row["batch_id"] == batch_id]
        batch_samples = [row for row in sampled if row["batch_id"] == batch_id]
        batch_rows.append({
            "batch_id": batch_id,
            "group_count": len(batch_group_rows),
            "image_count": len(batch_samples),
            "first_group": batch_group_rows[0]["final_group"] if batch_group_rows else "",
            "last_group": batch_group_rows[-1]["final_group"] if batch_group_rows else "",
        })
        _write_csv(run_dir / f"{batch_id}.csv", batch_samples, sampled_fields)

    _write_csv(run_dir / "batches.csv", batch_rows, ["batch_id", "group_count", "image_count", "first_group", "last_group"])
    _write_csv(run_dir / "warnings.csv", warnings, ["final_group", "candidate_group", "available_images", "warning"])
    (run_dir / "run.json").write_text(
        json.dumps({
            "run_id": run_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "pool_run": pool_run,
            "groups": len(ordered_groups),
            "sampled_images": len(sampled),
            "batches": batch_count,
            "batch_size": batch_size,
            "max_per_group": max_per_group,
            "strategy": strategy,
            "seed": seed,
            "warnings": len(warnings),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return SamplePoolsResult(run_id, run_dir, len(ordered_groups), len(sampled), batch_count, len(warnings))


def build_lightroom_review_structure(
    *,
    root: Path = Path("data/lureva"),
    sample_run: str,
    version: str = "0.1.0",
    run_id: str | None = None,
) -> LightroomStructureResult:
    paths = LurevaPaths(root)
    source_dir = paths.runs / sample_run
    groups = _read_csv(source_dir / "group_summary.csv")
    batches = _read_csv(source_dir / "batches.csv")
    run_id = run_id or f"lightroom-structure-v{version}"
    run_dir = paths.runs / run_id
    if run_dir.exists():
        raise IngestError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    for row in groups:
        rows.append({
            "version": version,
            "collection_set": f"Lureva 960 Review v{version}",
            "batch_id": row["batch_id"],
            "final_group": row["final_group"],
            "candidate_group": row["candidate_group"],
            "keyword_root": f"projects/lureva/selection_v{version}",
            "group_keyword": f"projects/lureva/selection_v{version}/groups/{row['final_group']}",
            "batch_keyword": f"projects/lureva/selection_v{version}/batches/{row['batch_id']}",
            "primary_keyword": f"projects/lureva/selection_v{version}/roles/primary",
            "alternate_keyword": f"projects/lureva/selection_v{version}/roles/alternate",
        })
    manifest = run_dir / "lightroom_structure.csv"
    _write_csv(manifest, rows, list(rows[0].keys()) if rows else [])
    (run_dir / "run.json").write_text(
        json.dumps({
            "run_id": run_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "sample_run": sample_run,
            "version": version,
            "groups": len(groups),
            "batches": len(batches),
            "manifest": str(manifest),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return LightroomStructureResult(run_id, run_dir, len(groups), len(batches), manifest)



def process_image_batch(
    *,
    root: Path = Path("data/lureva"),
    sample_run: str,
    batch: int | str,
    primary_count: int = 20,
    alternate_count: int = 5,
    close_reference_count: int = 5,
    version: str = "0.1.0",
    collection_set: str | None = None,
    run_id: str | None = None,
) -> ProcessImageBatchResult:
    """Write a fast Lightroom assignment manifest for one sampled batch.

    This is intentionally lightweight: it uses the deterministic sampled order as
    the first-pass proposal so Lightroom review can begin immediately. The
    first ``primary_count`` images per group are marked as primaries/Picks; the
    next ``alternate_count`` images are marked as alternates/Unflagged.

    ``close_reference_count`` is accepted now so the CLI contract matches the
    later embedding-aware implementation, but it is not used by this fast path.
    """
    if primary_count < 1:
        raise IngestError("primary_count must be at least 1.")
    if alternate_count < 0:
        raise IngestError("alternate_count cannot be negative.")
    if close_reference_count < 0:
        raise IngestError("close_reference_count cannot be negative.")

    if isinstance(batch, int):
        if batch < 1:
            raise IngestError("batch must be at least 1.")
        batch_id = f"batch_{batch:02d}"
    else:
        batch_text = str(batch).strip()
        if batch_text.isdigit():
            batch_id = f"batch_{int(batch_text):02d}"
        elif batch_text.startswith("batch_"):
            batch_id = batch_text
        else:
            raise IngestError("batch must be an integer or batch_XX identifier.")

    paths = LurevaPaths(root)
    source_dir = paths.runs / sample_run
    batch_csv = source_dir / f"{batch_id}.csv"
    sample_rows = _read_csv(batch_csv)
    if not sample_rows:
        raise IngestError(f"Batch has no sampled images: {batch_csv}")

    groups = _read_csv(source_dir / "group_summary.csv")
    group_info = {row["final_group"]: row for row in groups}
    batch_groups = [row for row in groups if row.get("batch_id") == batch_id]
    if not batch_groups:
        raise IngestError(f"No groups found for {batch_id} in {source_dir / 'group_summary.csv'}")

    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sample_rows:
        by_group[row["final_group"]].append(row)
    for rows in by_group.values():
        rows.sort(key=lambda row: int(row.get("sample_rank", "999999") or 999999))

    collection_set_name = collection_set or f"Lureva 960 Review v{version}"
    keyword_root = f"projects/lureva/selection_v{version}"
    assignment_rows: list[dict[str, object]] = []
    proposal_rows: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    for group in sorted(batch_groups, key=lambda row: int(row.get("final_rank", "999999") or 999999)):
        final_group = group["final_group"]
        candidate_group = group.get("candidate_group", "")
        rows = by_group.get(final_group, [])
        if len(rows) < primary_count:
            raise IngestError(
                f"{final_group} has only {len(rows)} sampled images; "
                f"need at least {primary_count} primary images."
            )
        target_count = primary_count + alternate_count
        if len(rows) < target_count:
            warnings.append({
                "batch_id": batch_id,
                "final_group": final_group,
                "candidate_group": candidate_group,
                "sampled_images": len(rows),
                "warning": f"Only {len(rows)} sampled images; fewer than {alternate_count} alternates available.",
            })
        chosen = rows[:target_count]
        for index, row in enumerate(chosen, start=1):
            is_primary = index <= primary_count
            role = "primary" if is_primary else "alternate"
            pick_status = "1" if is_primary else "0"
            proposal_rank = index if is_primary else index - primary_count
            archive_relative_path = _archive_relative_path(row.get("source_path", ""))
            source_root_env = "PROETUS_IMAGES_ROOT"
            common = {
                **row,
                "collection_set": collection_set_name,
                "batch_id": batch_id,
                "archive_relative_path": archive_relative_path,
                "source_root_env": source_root_env,
                "source_path_env": _env_source_path(archive_relative_path, source_root_env),
                "resolved_source_path": _resolved_env_source_path(archive_relative_path, source_root_env),
                "final_group": final_group,
                "final_group_rank": group.get("final_rank", ""),
                "candidate_group": candidate_group,
                "proposal_role": role,
                "proposal_rank": proposal_rank,
                "selection_method": "sample_order_fast_path",
                "initial_pick_status": pick_status,
                "group_keyword": f"{keyword_root}/groups/{final_group}",
                "batch_keyword": f"{keyword_root}/batches/{batch_id}",
                "role_keyword": f"{keyword_root}/roles/{role}",
                "nearest_reference_photo_uuid": "",
                "nearest_reference_file_name": "",
                "reference_distance": "",
            }
            assignment_rows.append(common)
            proposal_rows.append(common)

    run_id = run_id or datetime.now().strftime(f"image-{batch_id}-%Y%m%d-%H%M%S")
    run_dir = paths.runs / run_id
    if run_dir.exists():
        raise IngestError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    fields = [
        "collection_set", "batch_id", "final_group", "final_group_rank", "candidate_group",
        "source_path", "archive_relative_path", "source_root_env", "source_path_env",
        "resolved_source_path", "file_name", "photo_uuid", "proposal_role", "proposal_rank",
        "selection_method", "initial_pick_status", "group_keyword", "batch_keyword",
        "role_keyword", "nearest_reference_photo_uuid", "nearest_reference_file_name",
        "reference_distance", "sample_rank", "source_pool_count",
    ]
    # Preserve any important source columns that are not already in the fixed schema.
    for row in assignment_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    assignment_path = run_dir / f"{batch_id}_assignments.csv"
    _write_csv(assignment_path, assignment_rows, fields)
    _write_csv(run_dir / "image_proposal.csv", proposal_rows, fields)
    _write_csv(run_dir / "warnings.csv", warnings, ["batch_id", "final_group", "candidate_group", "sampled_images", "warning"])
    (run_dir / "run.json").write_text(
        json.dumps({
            "run_id": run_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "sample_run": sample_run,
            "batch_id": batch_id,
            "groups": len(batch_groups),
            "primary_images": sum(1 for row in assignment_rows if row["proposal_role"] == "primary"),
            "alternate_images": sum(1 for row in assignment_rows if row["proposal_role"] == "alternate"),
            "total_images": len(assignment_rows),
            "primary_count": primary_count,
            "alternate_count": alternate_count,
            "close_reference_count": close_reference_count,
            "version": version,
            "collection_set": collection_set_name,
            "assignment_manifest": str(assignment_path),
            "selection_method": "sample_order_fast_path",
            "warnings": len(warnings),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return ProcessImageBatchResult(
        run_id=run_id,
        run_dir=run_dir,
        batch_id=batch_id,
        groups=len(batch_groups),
        primary_images=sum(1 for row in assignment_rows if row["proposal_role"] == "primary"),
        alternate_images=sum(1 for row in assignment_rows if row["proposal_role"] == "alternate"),
        total_images=len(assignment_rows),
        assignment_path=assignment_path,
    )
