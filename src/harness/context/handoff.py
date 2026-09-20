"""Parent → child context handoff builders."""

from __future__ import annotations

from typing import Any

from harness.context.policy import ContextPolicy, HandoffMode
from harness.providers.base import LLMProvider


def _flatten(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(f"[{role}] {content.strip()}")
    return "\n".join(parts)


def build_handoff_messages(
    parent_messages: list[dict[str, Any]],
    task: str,
    policy: ContextPolicy,
    mode: HandoffMode | None = None,
    window_n: int | None = None,
    provider: LLMProvider | None = None,
) -> list[dict[str, Any]]:
    """Build the initial message list for a child agent (excluding system prompt)."""
    effective: HandoffMode = mode or policy.handoff_default  # type: ignore[assignment]
    n = window_n if window_n is not None else policy.handoff_window

    if effective == "none":
        return [{"role": "user", "content": task}]

    if effective == "full":
        body = _flatten(parent_messages)
        return [
            {
                "role": "user",
                "content": (
                    "Parent conversation (full handoff):\n"
                    f"{body}\n\n---\nYour task:\n{task}"
                ),
            }
        ]

    if effective == "window":
        window = parent_messages[-n:] if n > 0 else []
        body = _flatten(window)
        return [
            {
                "role": "user",
                "content": (
                    f"Parent conversation (last {len(window)} messages):\n"
                    f"{body}\n\n---\nYour task:\n{task}"
                ),
            }
        ]

    # summary
    body = _flatten(parent_messages[-max(n * 2, 20) :])
    summary = body
    if provider is not None and body.strip():
        try:
            turn = provider.chat(
                [
                    {
                        "role": "system",
                        "content": "Summarize parent context for a sub-agent. Be concise.",
                    },
                    {"role": "user", "content": body},
                ],
                tools=None,
            )
            if turn.content:
                summary = turn.content
        except Exception:
            summary = body[:4000]

    return [
        {
            "role": "user",
            "content": (
                "Parent context summary:\n"
                f"{summary}\n\n---\nYour task:\n{task}"
            ),
        }
    ]
