"""Session metadata and turn record shapes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SessionStatus = Literal["active", "done", "failed"]


@dataclass
class SessionMeta:
    session_id: str
    label: str = ""
    agent_role: str = "main"
    parent_id: str | None = None
    root_id: str | None = None
    depth: int = 0
    status: SessionStatus = "active"
    created_at: str = ""
    last_active: str = ""
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMeta:
        return cls(
            session_id=data["session_id"],
            label=data.get("label") or "",
            agent_role=data.get("agent_role") or "main",
            parent_id=data.get("parent_id"),
            root_id=data.get("root_id"),
            depth=int(data.get("depth") or 0),
            status=data.get("status") or "active",  # type: ignore[arg-type]
            created_at=data.get("created_at") or "",
            last_active=data.get("last_active") or "",
            message_count=int(data.get("message_count") or 0),
        )


@dataclass
class TurnRecord:
    type: str
    content: Any = None
    ts: float = 0.0
    tool_use_id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "ts": self.ts}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_use_id is not None:
            d["tool_use_id"] = self.tool_use_id
        if self.name is not None:
            d["name"] = self.name
        if self.input is not None:
            d["input"] = self.input
        if self.extra:
            d.update(self.extra)
        return d
