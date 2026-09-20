"""Select primary provider with failover to fallback."""

from __future__ import annotations

from typing import Any

from harness.config import Settings
from harness.providers.base import AssistantTurn, LLMProvider
from harness.providers.chatgpt_oauth import ChatGPTOAuthProvider
from harness.providers.openai_compat import OpenAICompatProvider


def _is_failover_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        k in text
        for k in (
            "401",
            "403",
            "429",
            "rate",
            "auth",
            "unauthorized",
            "forbidden",
            "quota",
            "billing",
            "invalid or disabled credential",
        )
    )


def build_provider(name: str, settings: Settings) -> LLMProvider:
    if name == "chatgpt_oauth":
        path = settings.provider.chatgpt_auth_path
        if path is None:
            raise ValueError("CHATGPT_AUTH_PATH is not set")
        return ChatGPTOAuthProvider(
            path,
            model=settings.provider.chatgpt_model,
            proxy=settings.provider.proxy or settings.proxy,
            auto_detect_proxy=settings.provider.auto_detect_proxy,
            reasoning_effort=settings.provider.chatgpt_reasoning_effort,
        )
    if name == "openai_compat":
        return OpenAICompatProvider(
            api_key=settings.provider.openai_api_key,
            model=settings.provider.model,
            base_url=settings.provider.openai_base_url,
        )
    raise ValueError(f"Unknown provider: {name}")


class ProviderRouter:
    name = "router"

    def __init__(self, settings: Settings, primary: str | None = None):
        self.settings = settings
        self.primary_name = primary or settings.provider.primary
        self.fallback_name = settings.provider.fallback
        self._primary = build_provider(self.primary_name, settings)
        self._fallback: LLMProvider | None = None
        if self.fallback_name and self.fallback_name != self.primary_name:
            try:
                self._fallback = build_provider(self.fallback_name, settings)
            except Exception:
                self._fallback = None
        self.active_name = self.primary_name

    @property
    def active(self) -> LLMProvider:
        return self._primary if self.active_name == self.primary_name else (
            self._fallback or self._primary
        )

    def switch(self, name: str) -> None:
        provider = build_provider(name, self.settings)
        self.primary_name = name
        self._primary = provider
        self.active_name = name

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AssistantTurn:
        try:
            self.active_name = self.primary_name
            return self._primary.chat(messages, tools=tools)
        except Exception as exc:
            if self._fallback is None or not _is_failover_error(exc):
                raise
            self.active_name = self.fallback_name
            return self._fallback.chat(messages, tools=tools)
