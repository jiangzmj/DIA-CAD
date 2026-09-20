"""Pipeline media provider via ChatGPT OAuth (gpt-5.6-sol).

Chat and image generation both use the same Codex OAuth backend —
no separate OpenAI Images / img2 API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.providers.base import AssistantTurn
from harness.providers.chatgpt_oauth import ChatGPTOAuthProvider


class GptMediaProvider:
    """Multimodal chat + target-render images, all through ChatGPT OAuth."""

    name = "gpt_media"

    def __init__(self, *, chat: ChatGPTOAuthProvider):
        self.backend = "chatgpt_oauth"
        self._chat = chat
        self.model = chat.model
        self.meter = None

    def set_meter(self, meter) -> None:
        self.meter = meter
        self._chat.meter = meter

    @classmethod
    def from_settings(cls, settings: Any) -> GptMediaProvider:
        prov = settings.provider
        path = prov.chatgpt_auth_path
        if path is None or not Path(path).is_file():
            raise ValueError(
                "chatgpt_oauth requires ~/.codex/auth.json "
                "(or CHATGPT_AUTH_PATH). Run `codex login` first."
            )
        chat = ChatGPTOAuthProvider(
            Path(path),
            model=prov.chatgpt_model or "gpt-5.6-sol",
            proxy=prov.proxy,
            auto_detect_proxy=prov.auto_detect_proxy,
            reasoning_effort=getattr(prov, "chatgpt_reasoning_effort", None) or "high",
        )
        return cls(chat=chat)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AssistantTurn:
        return self._chat.chat(messages, tools=tools)

    def generate_image(
        self,
        prompt: str,
        out_path: Path | str,
        *,
        reference_images: list[Path | str] | None = None,
        size: str = "1024x1024",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Same OAuth model as chat; reference PNGs attached as vision input."""
        return self._chat.generate_image(
            prompt,
            out_path,
            reference_images=reference_images,
            size=size,
            **kwargs,
        )
