"""LLM provider protocol and shared turn types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw: Any = None
    reasoning: str | None = None
    usage: dict[str, Any] | None = None


class LLMProvider(Protocol):
    name: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AssistantTurn: ...


class MediaProvider(Protocol):
    """Optional image generation (real or StubMediaProvider)."""

    name: str

    def generate_image(self, prompt: str, out_path: Any, **kwargs: Any) -> dict[str, Any]: ...
