"""Bounded, text-only conversation memory for the edit harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else str(content or "")


@dataclass
class HarnessConversationMemory:
    """Keep recent exchanges and a compact deterministic summary of older ones.

    Images deliberately never enter this object. Every live request attaches the
    current image set once, while history preserves model decisions and controller
    feedback without duplicating base64 payloads.
    """

    max_messages: int = 8
    max_chars: int = 24000
    summary_chars: int = 6000
    messages: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""

    def record(self, *, user_summary: str, assistant: str) -> None:
        self.messages.extend(
            [
                {"role": "user", "content": str(user_summary).strip()},
                {"role": "assistant", "content": str(assistant).strip()},
            ]
        )
        self._compact()

    def api_messages(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if self.summary:
            out.append(
                {
                    "role": "user",
                    "content": "## Compressed earlier S2 history\n" + self.summary,
                }
            )
            out.append(
                {
                    "role": "assistant",
                    "content": "Understood. I will use that earlier history with the recent turns.",
                }
            )
        out.extend(dict(m) for m in self.messages)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": {
                "max_messages": self.max_messages,
                "max_chars": self.max_chars,
                "summary_chars": self.summary_chars,
                "images_in_history": False,
            },
            "summary": self.summary,
            "messages": list(self.messages),
        }

    def _compact(self) -> None:
        max_messages = max(2, int(self.max_messages))
        max_chars = max(1000, int(self.max_chars))
        while len(self.messages) > max_messages or self._message_chars() > max_chars:
            take = 2 if len(self.messages) >= 2 else 1
            removed = self.messages[:take]
            del self.messages[:take]
            snippets = []
            for msg in removed:
                text = _content_text(msg).replace("\x00", "").strip()
                if len(text) > 1200:
                    text = text[:1200] + " …[truncated]"
                snippets.append(f"{msg.get('role', 'unknown')}: {text}")
            addition = "\n".join(snippets)
            self.summary = (self.summary + "\n" + addition).strip()
            if len(self.summary) > max(500, int(self.summary_chars)):
                self.summary = self.summary[-int(self.summary_chars) :]

    def _message_chars(self) -> int:
        return sum(len(_content_text(m)) for m in self.messages)
