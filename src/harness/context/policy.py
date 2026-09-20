"""Adjustable context policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

HandoffMode = Literal["none", "summary", "window", "full"]


@dataclass
class ContextPolicy:
    max_messages: int = 80
    max_tool_chars: int = 20000
    compact_keep_recent: float = 0.2
    compact_summarize_oldest: float = 0.5
    handoff_default: HandoffMode = "summary"
    handoff_window: int = 12
    system_layers: list[str] = field(
        default_factory=lambda: ["identity", "soul", "tools", "runtime"]
    )

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Soft-trim history before an API call (does not mutate caller list)."""
        trimmed = self._truncate_tool_contents(messages)
        if len(trimmed) > self.max_messages:
            # Keep system-like leading notes + recent tail
            head = trimmed[:2] if len(trimmed) > 2 else []
            tail = trimmed[-self.max_messages :]
            # Avoid duplicating if overlap
            if head and tail and head[0] is tail[0]:
                trimmed = tail
            else:
                trimmed = head + [m for m in tail if m not in head]
                if len(trimmed) > self.max_messages:
                    trimmed = trimmed[-self.max_messages :]
        return trimmed

    def _truncate_tool_contents(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            msg = deepcopy(msg)
            if (
                msg.get("role") == "tool"
                and isinstance(msg.get("content"), str)
                and len(msg["content"]) > self.max_tool_chars
            ):
                original = len(msg["content"])
                msg["content"] = (
                    msg["content"][: self.max_tool_chars]
                    + f"\n\n[... truncated {original} chars, "
                    f"showing first {self.max_tool_chars} ...]"
                )
            out.append(msg)
        return out

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise KeyError(f"Unknown context policy field: {key}")
            setattr(self, key, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_messages": self.max_messages,
            "max_tool_chars": self.max_tool_chars,
            "compact_keep_recent": self.compact_keep_recent,
            "compact_summarize_oldest": self.compact_summarize_oldest,
            "handoff_default": self.handoff_default,
            "handoff_window": self.handoff_window,
            "system_layers": list(self.system_layers),
        }
