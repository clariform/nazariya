from __future__ import annotations

import unittest

from nazariya.lureva.identity import candidate_keys, parse_corpus_name, part2_keys, source_key


class IdentityTests(unittest.TestCase):
    def test_corpus_name_maps_back_to_raw_identity(self) -> None:
        stem, date = parse_corpus_name("DSC02683_2013_12_12_SUHAIL.DNG")
        self.assertEqual(stem, "DSC02683")
        self.assertEqual(date, "2013-12-12")
        self.assertEqual(source_key(stem, date), "DSC02683|2013-12-12")

    def test_candidate_keys_are_bounded_to_catalog_range(self) -> None:
        self.assertEqual(candidate_keys("c001 ; project|c325 ; c326"), ["c001", "c325"])

    def test_part2_keywords_are_exact(self) -> None:
        self.assertEqual(part2_keys("p001 ; project|p008 ; p009"), ["p001", "p008"])


if __name__ == "__main__":
    unittest.main()
