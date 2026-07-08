from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from nazariya.lureva.batches import build_lightroom_review_structure, process_image_batch, sample_image_pools


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class BatchTests(unittest.TestCase):
    def test_samples_and_batches_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            run = root / "runs" / "image-pools"
            pool_rows = []
            summary_rows = []
            for group_index, group in enumerate(("u001", "u002", "u003"), start=1):
                summary_rows.append({
                    "final_rank": group_index,
                    "final_group": group,
                    "candidate_group": f"c{group_index:03d}",
                    "reference_count": 1,
                    "unused_pool_count": 50,
                })
                for image_index in range(50):
                    pool_rows.append({
                        "final_group": group,
                        "final_group_rank": group_index,
                        "assigned_candidate_group": f"c{group_index:03d}",
                        "source_path": f"/images/{group}/{image_index:03d}.ARW",
                        "file_name": f"{image_index:03d}.ARW",
                        "capture_time": f"2020-01-{image_index % 28 + 1:02d}",
                    })
            write_csv(run / "image_pools.csv", pool_rows)
            write_csv(run / "group_summary.csv", summary_rows)

            result = sample_image_pools(root=root, pool_run="image-pools", max_per_group=40, batch_size=2, run_id="sampled")
            self.assertEqual(result.groups, 3)
            self.assertEqual(result.sampled_images, 120)
            self.assertEqual(result.batches, 2)

            structure = build_lightroom_review_structure(root=root, sample_run="sampled", run_id="structure")
            self.assertEqual(structure.groups, 3)
            self.assertEqual(structure.batches, 2)
            self.assertTrue(structure.manifest_path.exists())


    def test_process_image_batch_writes_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            run = root / "runs" / "image-pools"
            pool_rows = []
            summary_rows = []
            for group_index, group in enumerate(("u001", "u002"), start=1):
                summary_rows.append({
                    "final_rank": group_index,
                    "final_group": group,
                    "candidate_group": f"c{group_index:03d}",
                    "reference_count": 1,
                    "unused_pool_count": 30,
                })
                for image_index in range(30):
                    pool_rows.append({
                        "photo_uuid": f"uuid-{group}-{image_index:03d}",
                        "final_group": group,
                        "final_group_rank": group_index,
                        "assigned_candidate_group": f"c{group_index:03d}",
                        "source_path": f"/images/{group}/{image_index:03d}.ARW",
                        "file_name": f"{image_index:03d}.ARW",
                        "capture_time": f"2020-01-{image_index % 28 + 1:02d}",
                    })
            write_csv(run / "image_pools.csv", pool_rows)
            write_csv(run / "group_summary.csv", summary_rows)
            sample_image_pools(root=root, pool_run="image-pools", max_per_group=30, batch_size=2, run_id="sampled")

            result = process_image_batch(
                root=root,
                sample_run="sampled",
                batch=1,
                primary_count=20,
                alternate_count=5,
                run_id="batch-01",
            )
            self.assertEqual(result.batch_id, "batch_01")
            self.assertEqual(result.groups, 2)
            self.assertEqual(result.primary_images, 40)
            self.assertEqual(result.alternate_images, 10)
            self.assertTrue(result.assignment_path.exists())

            with result.assignment_path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 50)
            self.assertEqual(rows[0]["proposal_role"], "primary")
            self.assertEqual(rows[0]["initial_pick_status"], "1")
            self.assertEqual(rows[20]["proposal_role"], "alternate")
            self.assertEqual(rows[20]["initial_pick_status"], "0")
            self.assertEqual(rows[0]["final_group"], "u001")
            self.assertEqual(rows[-1]["final_group"], "u002")
            self.assertIn("archive_relative_path", rows[0])
            self.assertIn("source_path_env", rows[0])
            self.assertEqual(rows[0]["source_root_env"], "PROETUS_IMAGES_ROOT")


if __name__ == "__main__":
    unittest.main()
