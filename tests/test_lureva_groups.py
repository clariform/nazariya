from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from nazariya.lureva.groups import build_group_pools, propose_groups


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class GroupPoolTests(unittest.TestCase):
    def test_builds_unused_pools_and_proposes_top_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "lureva"
            ref_dir = root / "runs" / "part1"
            full_dir = root / "runs" / "all560"
            catalog = []
            for group, count in (("c001", 25), ("c002", 35), ("c003", 22)):
                for index in range(count):
                    catalog.append({
                        "photo_uuid": f"{group}-{index}",
                        "catalog_photo_uuids": f"{group}-{index}",
                        "source_path": f"/archive/{group}/{index}.ARW",
                        "source_file_name": f"{index}.ARW",
                        "capture_date": "2020-01-01",
                        "candidate_keys": group,
                    })
            catalog_fields = list(catalog[0])
            write_csv(ref_dir / "catalog_candidates.csv", catalog_fields, catalog)
            write_csv(full_dir / "catalog_candidates.csv", catalog_fields, catalog)

            match_fields = [
                "match_status", "matched_source_path", "matched_photo_uuid",
                "matched_source_file_name", "matched_candidate_keys", "corpus_file_name",
                "seed_origin", "replacement_of_corpus_file_name",
            ]
            references = []
            exclusions = []
            for group in ("c001", "c002", "c003"):
                row = {
                    "match_status": "matched",
                    "matched_source_path": f"/archive/{group}/0.ARW",
                    "matched_photo_uuid": f"{group}-0",
                    "matched_source_file_name": "0.ARW",
                    "matched_candidate_keys": group,
                    "corpus_file_name": f"{group}.DNG",
                    "seed_origin": "historical",
                    "replacement_of_corpus_file_name": "",
                }
                references.append(row)
                exclusions.append(row)
            # Exclude two more existing corpus assets from c002.
            for index in (1, 2):
                exclusions.append({
                    **references[1],
                    "matched_source_path": f"/archive/c002/{index}.ARW",
                    "matched_photo_uuid": f"c002-{index}",
                })
            write_csv(ref_dir / "seed_matches.csv", match_fields, references)
            write_csv(full_dir / "seed_matches.csv", match_fields, exclusions)

            result = build_group_pools(
                root=root,
                reference_run="part1",
                exclusion_run="all560",
                minimum_images=20,
                preferred_images=30,
                run_id="pools",
            )
            self.assertEqual(result.represented_groups, 3)
            self.assertEqual(result.eligible_groups, 3)

            proposal = propose_groups(root=root, pool_run="pools", count=2, run_id="proposal")
            self.assertEqual(proposal.proposed_groups, 2)
            with (proposal.run_dir / "proposed_groups.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["candidate_group"] for row in rows], ["c002", "c001"])


if __name__ == "__main__":
    unittest.main()
