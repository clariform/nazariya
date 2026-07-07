from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from nazariya.lureva.review import (
    _load_preview_map,
    _lookup_preview,
    build_group_review_contact_sheets,
)
from types import SimpleNamespace


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class GroupReviewTests(unittest.TestCase):

    def test_preview_lookup_survives_archive_root_move(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            preview = root / "preview.jpg"
            Image.new("RGB", (20, 20), "white").save(preview)
            map_path = root / "preview_map.csv"
            write_csv(
                map_path,
                ["source_path", "normalized_preview_path"],
                [{
                    "source_path": "/Volumes/whisk/work/ml/datasets/proetus/images/2017/2017-10/2017-10-12/DSC03407.ARW",
                    "normalized_preview_path": str(preview),
                }],
            )

            preview_map = _load_preview_map(map_path)
            row = _lookup_preview(
                preview_map,
                "/Volumes/dataLib/Pictures/Images/2017/2017-10/2017-10-12/DSC03407.ARW",
            )

            self.assertEqual(row["normalized_preview_path"], str(preview))

    def test_builds_selected_and_unselected_review_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "lureva"
            pool = root / "runs" / "pool"
            proposal = root / "runs" / "proposal"
            pool.mkdir(parents=True)
            proposal.mkdir(parents=True)

            summaries = [
                {"candidate_group": "c001", "reference_count": "1", "catalog_raw_count": "30", "excluded_existing_count": "1", "unused_raw_count": "29", "multi_group_unused_count": "0", "single_group_unused_count": "29", "eligible_for_20": "1", "readiness": "borderline"},
                {"candidate_group": "c002", "reference_count": "1", "catalog_raw_count": "40", "excluded_existing_count": "1", "unused_raw_count": "39", "multi_group_unused_count": "0", "single_group_unused_count": "39", "eligible_for_20": "1", "readiness": "strong"},
            ]
            write_csv(pool / "eligible_groups.csv", list(summaries[0]), summaries)
            pool_rows = []
            refs = []
            for group in ("c001", "c002"):
                refs.append({"candidate_group": group, "photo_uuid": f"ref-{group}", "source_path": f"/{group}/ref.ARW", "source_file_name": "ref.ARW", "corpus_file_name": "ref.DNG", "seed_origin": "historical", "replacement_of_corpus_file_name": ""})
                for index in range(8):
                    pool_rows.append({"candidate_group": group, "photo_uuid": f"{group}-{index}", "catalog_photo_uuids": "", "source_path": f"/{group}/{index}.ARW", "source_file_name": f"{index}.ARW", "capture_date": f"2020-01-{index+1:02d}", "all_candidate_keys": group, "represented_membership_count": "1", "is_existing_corpus_image": "0", "is_selectable": "1"})
            write_csv(pool / "group_pools.csv", list(pool_rows[0]), pool_rows)
            write_csv(pool / "reference_images.csv", list(refs[0]), refs)
            proposal_rows = [{**summaries[1], "proposal_rank": "1", "status": "proposed", "replacement_group": "", "manual_note": ""}]
            write_csv(proposal / "proposed_groups.csv", list(proposal_rows[0]), proposal_rows)

            def fake_previews(**kwargs):
                input_path = kwargs["input_path"]
                output_root = kwargs["output_root"]
                output_root.mkdir(parents=True, exist_ok=True)
                with input_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                map_rows = []
                for index, row in enumerate(rows):
                    image_path = output_root / f"{index}.jpg"
                    Image.new("RGB", (100, 80), "white").save(image_path)
                    map_rows.append({"source_path": row["source_path"], "normalized_preview_path": str(image_path)})
                map_path = output_root / "preview_map.csv"
                write_csv(map_path, ["source_path", "normalized_preview_path"], map_rows)
                failures = output_root / "failures.csv"
                write_csv(failures, ["row", "source_path", "candidate_key", "error"], [])
                return SimpleNamespace(input_path=input_path, output_root=output_root, total_rows=len(rows), rendered=len(rows), skipped_existing=0, failed=0, preview_map_path=map_path, failures_path=failures)

            with patch("nazariya.lureva.review._build_previews", side_effect=fake_previews):
                result = build_group_review_contact_sheets(root=root, pool_run="pool", proposal_run="proposal", run_id="review", selected_pool_samples=4, unselected_pool_samples=3, reference_limit=1)

            self.assertEqual(result.eligible_groups, 2)
            self.assertEqual(result.proposed_groups, 1)
            self.assertEqual(result.unselected_groups, 1)
            self.assertTrue((result.run_dir / "contact_sheets/groups/c002_proposed.jpg").exists())
            self.assertTrue((result.run_dir / "contact_sheets/groups/c001_unselected.jpg").exists())
            self.assertGreater(result.overview_sheets, 0)


if __name__ == "__main__":
    unittest.main()
