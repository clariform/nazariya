from __future__ import annotations

"""Canonical schemas for style-learning inputs, targets, and predictions."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StylePhoto:
    photo_uuid: str
    source_path: Path
    file_name: str
    group_key: str
    camera_model: str
