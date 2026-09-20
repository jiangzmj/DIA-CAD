"""Run manifest: stage progress, artifact paths, checks, handoff."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Four cameras on the unit-box corners looking at the AABB center
# (think of the box as [0,1]³ with center 0.5,0.5,0.5):
#   iso (1,1,1) · z_corner (0,0,1) · x_corner (1,0,0) · y_corner (0,1,0)
VIEW_NAMES = ("iso", "z_corner", "x_corner", "y_corner")


@dataclass
class Manifest:
    run_id: str
    root: str
    stage: str = "init"
    status: str = "running"  # running | done | failed
    input_step: str = ""
    description_txt: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)
    handoff: dict[str, str] = field(default_factory=lambda: {"summary": ""})
    error: str | None = None

    def set_summary(self, text: str) -> None:
        self.handoff["summary"] = text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        return cls(
            run_id=str(data.get("run_id", "")),
            root=str(data.get("root", "")),
            stage=str(data.get("stage", "init")),
            status=str(data.get("status", "running")),
            input_step=str(data.get("input_step", "")),
            description_txt=str(data.get("description_txt", "")),
            artifacts=dict(data.get("artifacts") or {}),
            checks=dict(data.get("checks") or {}),
            sessions=dict(data.get("sessions") or {}),
            handoff=dict(data.get("handoff") or {"summary": ""}),
            error=data.get("error"),
        )

    def save(self, path: Path | None = None) -> Path:
        out = path or (Path(self.root) / "manifest.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, path: Path) -> Manifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
