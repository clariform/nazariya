from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from nazariya.lureva.selection import (
    build_lightroom_review_manifest,
    finalize_groups,
    prepare_image_pools,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SelectionTests(unittest.TestCase):
    def test_finalize_prepare_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "lureva"
            review = root / "runs" / "review"
            pool = root / "runs" / "pool"
            review_rows = [
                {"candidate_group": "c001", "proposal_status": "proposed", "proposal_rank": "1", "manual_status": "remove", "reference_count": "1", "replacement_group": "c003", "manual_note": ""},
                {"candidate_group": "c002", "proposal_status": "proposed", "proposal_rank": "2", "manual_status": "", "reference_count": "1", "replacement_group": "", "manual_note": ""},
                {"candidate_group": "c003", "proposal_status": "unselected", "proposal_rank": "", "manual_status": "add", "reference_count": "1", "replacement_group": "", "manual_note": ""},
            ]
            write_csv(review / "group_review.csv", review_rows)
            final = finalize_groups(root=root, review_run="review", run_id="final", expected_groups=2)
            self.assertEqual(final.selected_groups, 2)
            with (final.run_dir / "final_48_groups.csv").open(newline="", encoding="utf-8") as handle:
                finalized_rows = list(csv.DictReader(handle))
            self.assertEqual([row["final_group"] for row in finalized_rows], ["u001", "u002"])

            pool_rows = []
            for group in ("c002", "c003"):
                for index in range(25):
                    pool_rows.append({
                        "candidate_group": group,
                        "photo_uuid": f"{group}-{index}",
                        "catalog_photo_uuids": "",
                        "source_path": f"/archive/{group}/{index}.ARW",
                        "source_file_name": f"{index}.ARW",
                        "capture_date": "2020-01-01",
                        "all_candidate_keys": group,
                        "represented_membership_count": "1",
                        "is_existing_corpus_image": "0",
                        "is_selectable": "1",
                    })
            write_csv(pool / "group_pools.csv", pool_rows)
            refs = [
                {"candidate_group": group, "photo_uuid": f"ref-{group}", "source_path": f"/archive/{group}/ref.ARW", "source_file_name": "ref.ARW", "corpus_file_name": "ref.DNG"}
                for group in ("c002", "c003")
            ]
            write_csv(pool / "reference_images.csv", refs)
            image_pools = prepare_image_pools(root=root, groups_run="final", pool_run="pool", run_id="images")
            self.assertEqual(image_pools.groups, 2)
            self.assertEqual(image_pools.unique_images, 50)

            with (final.run_dir / "final_48_groups.csv").open(newline="", encoding="utf-8") as handle:
                groups = list(csv.DictReader(handle))
            proposals = []
            for group in groups:
                for index in range(25):
                    proposals.append({
                        "candidate_group": group["candidate_group"],
                        "photo_uuid": f"{group['candidate_group']}-{index}",
                        "source_path": f"/archive/{group['candidate_group']}/{index}.ARW",
                        "source_file_name": f"{index}.ARW",
                        "selection_role": "primary" if index < 20 else "alternate",
                        "selection_rank": str(index + 1),
                    })
            proposal_path = root / "proposal.csv"
            write_csv(proposal_path, proposals)
            manifest = build_lightroom_review_manifest(
                proposal_csv=proposal_path,
                groups_csv=final.run_dir / "final_48_groups.csv",
                root=root,
                run_id="lr",
            )
            self.assertEqual(manifest.primary_images, 40)
            self.assertEqual(manifest.alternate_images, 10)
            self.assertTrue(manifest.manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
