from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

CANDIDATE_RE = re.compile(r"(?<![A-Za-z0-9])c(\d{3})(?![A-Za-z0-9])", re.IGNORECASE)
PART_RE = re.compile(r"(?<![A-Za-z0-9])p(00[1-8])(?![A-Za-z0-9])", re.IGNORECASE)
CORPUS_DNG_RE = re.compile(
    r"^(?P<stem>.+)_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_SUHAIL\.DNG$",
    re.IGNORECASE,
)


def split_keywords(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in re.split(r"\s*;\s*|\s*\|\s*", value) if item.strip()]


def candidate_keys(*values: str | None) -> list[str]:
    keys: set[str] = set()
    for value in values:
        for match in CANDIDATE_RE.finditer(value or ""):
            number = int(match.group(1))
            if 1 <= number <= 325:
                keys.add(f"c{number:03d}")
    return sorted(keys)


def part2_keys(*values: str | None) -> list[str]:
    keys: set[str] = set()
    for value in values:
        for match in PART_RE.finditer(value or ""):
            keys.add(f"p{int(match.group(1)):03d}")
    return sorted(keys)


def normalize_stem(file_name_or_stem: str | None) -> str:
    value = Path(file_name_or_stem or "").name
    return Path(value).stem.strip().upper()


def parse_date(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""

    match = re.search(r"(?P<year>\d{4})[-:/](?P<month>\d{2})[-:/](?P<day>\d{2})", raw)
    if match:
        return f"{match.group('year')}-{match.group('month')}-{match.group('day')}"

    for fmt in ("%Y%m%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw[:10], fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return ""


def parse_corpus_name(file_name: str) -> tuple[str, str]:
    name = Path(file_name).name
    match = CORPUS_DNG_RE.match(name)
    if not match:
        return normalize_stem(name), ""
    date = f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
    return match.group("stem").strip().upper(), date


def source_key(stem: str, capture_date: str) -> str:
    stem = normalize_stem(stem)
    capture_date = parse_date(capture_date)
    if not stem or not capture_date:
        return ""
    return f"{stem}|{capture_date}"
