"""Context overflow guard: truncate then compact."""

from __future__ import annotations

from typing import Any, Callable

from harness.context.policy import ContextPolicy
from harness.providers.base import AssistantTurn, LLMProvider


def _serialize_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        if isinstance(content, str) and content:
            parts.append(f"[{role}]: {content}")
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args_preview = (fn.get("arguments") or "")[:500]
            parts.append(f"[{role} called {fn.get('name', '?')}]: {args_preview}")
    return "\n".join(parts)


class ContextGuard:
    """Three-stage protection around provider.chat."""

    def __init__(self, policy: ContextPolicy, provider: LLMProvider):
        self.policy = policy
        self.provider = provider

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    def estimate_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            for tc in msg.get("tool_calls") or []:
                args = tc.get("function", {}).get("arguments", "")
                total += self.estimate_tokens(args)
        return total

    def truncate_tool_results(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.policy._truncate_tool_contents(messages)

    def compact_history(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total = len(messages)
        if total <= 4:
            return messages

        keep_count = max(4, int(total * self.policy.compact_keep_recent))
        compress_count = max(2, int(total * self.policy.compact_summarize_oldest))
        compress_count = min(compress_count, total - keep_count)
        if compress_count < 2:
            return messages

        old_messages = messages[:compress_count]
        recent_messages = messages[compress_count:]
        old_text = _serialize_messages_for_summary(old_messages)
        summary_prompt = (
            "Summarize the following conversation concisely, "
            "preserving key facts and decisions. "
            "Output only the summary, no preamble.\n\n"
            f"{old_text}"
        )
        try:
            turn = self.provider.chat(
                [
                    {
                        "role": "system",
                        "content": "You are a conversation summarizer. Be concise and factual.",
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                tools=None,
            )
            summary_text = turn.content or ""
        except Exception:
            return recent_messages

        return [
            {
                "role": "user",
                "content": "[Previous conversation summary]\n" + summary_text,
            },
            {
                "role": "assistant",
                "content": "Understood, I have the context from our previous conversation.",
            },
        ] + recent_messages

    def guard_api_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        is_overflow: Callable[[Exception], bool] | None = None,
    ) -> tuple[AssistantTurn, list[dict[str, Any]]]:
        """Return (turn, possibly-compacted messages used)."""

        def _overflow(exc: Exception) -> bool:
            if is_overflow:
                return is_overflow(exc)
            text = str(exc).lower()
            return any(
                k in text
                for k in (
                    "context_length",
                    "maximum context",
                    "too many tokens",
                    "token limit",
                    "context window",
                    "overflow",
                )
            )

        working = self.policy.apply(messages)
        try:
            return self.provider.chat(working, tools=tools), working
        except Exception as exc0:
            if not _overflow(exc0):
                raise

        working = self.truncate_tool_results(working)
        try:
            return self.provider.chat(working, tools=tools), working
        except Exception as exc1:
            if not _overflow(exc1):
                raise

        working = self.compact_history(working)
        return self.provider.chat(working, tools=tools), working
