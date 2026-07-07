from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from nazariya.lureva.ingest import ingest_milestone_one


class IngestTests(unittest.TestCase):
    def test_matches_by_stem_date_and_filters_part2_from_catalog_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.csv"
            seed = root / "seed.csv"

            with catalog.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "source_path", "file_name", "file_stem", "photo_uuid",
                        "candidate_keys", "primary_candidate_key", "all_keywords",
                        "capture_time",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "source_path": "/archive/2013/DSC02683.ARW",
                    "file_name": "DSC02683.ARW",
                    "file_stem": "DSC02683",
                    "photo_uuid": "uuid-1",
                    "candidate_keys": "c014",
                    "primary_candidate_key": "c014",
                    "all_keywords": "c014 ; part1",
                    "capture_time": "2013-12-12T10:00:00",
                })
                writer.writerow({
                    "source_path": "/archive/2015/DSC00001.ARW",
                    "file_name": "DSC00001.ARW",
                    "file_stem": "DSC00001",
                    "photo_uuid": "uuid-2",
                    "candidate_keys": "c020",
                    "primary_candidate_key": "c020",
                    "all_keywords": "c020 ; p001",
                    "capture_time": "2015-01-02T10:00:00",
                })

            with seed.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file_name"])
                writer.writeheader()
                writer.writerow({"file_name": "DSC02683_2013_12_12_SUHAIL.DNG"})
                writer.writerow({"file_name": "DSC00001_2015_01_02_SUHAIL.DNG"})

            result = ingest_milestone_one(
                catalog_csv=catalog,
                seed_csv=seed,
                root=root / "data",
                run_id="test",
                expected_catalog_candidates=None,
                expected_seed_rows=None,
            )

            self.assertEqual(result.seed_rows_read, 2)
            self.assertEqual(result.seed_rows_after_filter, 1)
            self.assertEqual(result.matched_rows, 1)
            self.assertEqual(result.represented_groups, 1)

    def test_filters_non_raw_and_collapses_duplicate_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "catalog.csv"
            seed = root / "seed.csv"

            fields = [
                "source_path", "file_name", "file_extension", "file_format",
                "photo_uuid", "candidate_keys", "primary_candidate_key",
                "all_keywords", "capture_time", "rating", "label_color",
            ]
            with catalog.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "source_path": "/archive/2023/2023-08-03/_SUH0329.ARW",
                    "file_name": "_SUH0329.ARW",
                    "file_extension": "ARW",
                    "file_format": "RAW",
                    "photo_uuid": "uuid-gray",
                    "candidate_keys": "c283",
                    "primary_candidate_key": "c283",
                    "all_keywords": "c283 ; g018",
                    "capture_time": "712780464.118",
                    "rating": "",
                    "label_color": "gray",
                })
                writer.writerow({
                    "source_path": "/archive/2023/2023-08-03/_SUH0329.ARW",
                    "file_name": "_SUH0329.ARW",
                    "file_extension": "ARW",
                    "file_format": "RAW",
                    "photo_uuid": "uuid-purple",
                    "candidate_keys": "c283",
                    "primary_candidate_key": "c283",
                    "all_keywords": "c283 ; g018",
                    "capture_time": "712780464.118",
                    "rating": "5",
                    "label_color": "purple",
                })
                writer.writerow({
                    "source_path": "/archive/2023/2023-08-03/render.jpg",
                    "file_name": "render.jpg",
                    "file_extension": "JPG",
                    "file_format": "JPEG",
                    "photo_uuid": "uuid-jpeg",
                    "candidate_keys": "c283",
                    "primary_candidate_key": "c283",
                    "all_keywords": "c283",
                })

            with seed.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file_name"] )
                writer.writeheader()
                writer.writerow({"file_name": "_SUH0329_2023_08_03_SUHAIL.DNG"})

            result = ingest_milestone_one(
                catalog_csv=catalog,
                seed_csv=seed,
                root=root / "data",
                run_id="test",
                expected_catalog_candidates=None,
                expected_seed_rows=None,
            )

            self.assertEqual(result.catalog_candidate_rows, 3)
            self.assertEqual(result.catalog_raw_rows, 2)
            self.assertEqual(result.catalog_unique_raws, 1)
            self.assertEqual(result.catalog_duplicate_records, 1)
            self.assertEqual(result.catalog_non_raw_rows, 1)
            self.assertEqual(result.matched_rows, 1)

            with (result.run_dir / "catalog_candidates.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["catalog_record_count"], "2")
            self.assertIn("uuid-gray", rows[0]["catalog_photo_uuids"])
            self.assertIn("uuid-purple", rows[0]["catalog_photo_uuids"])
            self.assertEqual(rows[0]["rating"], "5")
            self.assertEqual(rows[0]["ratings"], "5")
            self.assertEqual(rows[0]["label_colors"], "gray ; purple")


if __name__ == "__main__":
    unittest.main()
