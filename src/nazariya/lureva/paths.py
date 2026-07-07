from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LurevaPaths:
    root: Path

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def groups(self) -> Path:
        return self.root / "groups"

    @property
    def selections(self) -> Path:
        return self.root / "selections"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    @property
    def contact_sheets(self) -> Path:
        return self.root / "contact_sheets"

    @property
    def lightroom(self) -> Path:
        return self.root / "lightroom"

    @property
    def database(self) -> Path:
        return self.state / "lureva_selection.sqlite3"

    def create(self) -> list[Path]:
        folders = [
            self.root,
            self.inputs,
            self.state,
            self.runs,
            self.manifests,
            self.reports,
            self.groups,
            self.selections,
            self.embeddings,
            self.contact_sheets,
            self.lightroom,
        ]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
        return folders
