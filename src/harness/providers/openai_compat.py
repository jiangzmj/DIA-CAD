"""OpenAI-compatible chat completions provider (any base_url + api key)."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from harness.providers.base import AssistantTurn, ToolCall


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for openai_compat provider")
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AssistantTurn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )

        finish = choice.finish_reason or ("tool_calls" if tool_calls else "stop")
        return AssistantTurn(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=finish,
            raw=resp,
        )
