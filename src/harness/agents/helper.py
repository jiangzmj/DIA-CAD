"""Helper sub-agent: prompt chores (QC, labeling helpers, etc.).

Pipeline stages pass a task prompt (+ optional images); the helper returns text.
No tools, no spawn — multimodal chat with a short session transcript.

Default is one-shot. Pass ``session_id`` + ``history`` to continue the same
conversation across turns (e.g. S5 result QC retries).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from harness.agents.roles import build_system_prompt
from harness.pipeline.multimodal import user_content


def _text_only_content(content: Any) -> str:
    """Flatten multimodal content to text for compact multi-turn history."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = part.get("text")
                if text:
                    parts.append(str(text))
            elif part.get("type") == "image_url":
                parts.append("[image]")
        return "\n".join(parts).strip() or "[multimodal turn]"
    return str(content)


@dataclass
class HelperResult:
    content: str
    session_id: str | None = None
    ok: bool = True
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # Text-only prior+this-turn messages (no system); reuse as ``history=``.
    history: list[dict[str, Any]] = field(default_factory=list)


class HelperAgent:
    """Reusable helper for small pipeline chores.

    Usage::

        result = helper.run(prompt_text, images=[...], label="view_qc")
        text = result.content

        # Multi-turn (same context):
        r1 = helper.run(p1, images=imgs, label="qc")
        r2 = helper.run(p2, images=imgs, session_id=r1.session_id, history=r1.history)
    """

    role = "helper"

    def __init__(
        self,
        media: Any,
        *,
        workspace: Path | str | None = None,
        session_store: Any | None = None,
        agent_id: str = "cad_edit",
        system_layers: list[str] | None = None,
    ):
        self.media = media
        self.workspace = Path(workspace) if workspace else None
        self._session_store = session_store
        self.agent_id = agent_id
        self.system_layers = system_layers or ["runtime"]

    def _store(self) -> Any:
        if self._session_store is not None:
            return self._session_store
        if self.workspace is None:
            return None
        from harness.session.store import SessionStore

        self._session_store = SessionStore(self.workspace, agent_id=self.agent_id)
        return self._session_store

    def _system_prompt(self) -> str | None:
        if self.workspace is None:
            return None
        return build_system_prompt(self.workspace, self.role, self.system_layers)

    def run(
        self,
        prompt: str,
        *,
        images: list[Path | str] | None = None,
        label: str = "helper_task",
        parent_session_id: str | None = None,
        session_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        detail: str = "low",
        finalize: bool = True,
        append: Callable[[dict[str, Any]], None] | None = None,
    ) -> HelperResult:
        """Run one helper turn: ``prompt`` (+ images) → assistant text.

        When ``session_id`` / ``history`` are provided, continues that conversation
        instead of starting a fresh one-shot context. Prior history is sent as
        text-only (images stripped) so token use stays bounded; the current turn
        still carries full images.
        """
        imgs = [Path(p) for p in (images or []) if Path(p).is_file()]
        content = user_content(prompt, imgs, detail=detail)
        prior = list(history or [])

        store = self._store()
        sid: str | None = session_id
        if store is not None:
            if sid is None:
                sid = store.create_session(
                    label=label[:80],
                    agent_role=self.role,
                    parent_id=parent_session_id,
                )
            else:
                try:
                    store.update_meta(sid, status="running")
                except Exception:  # noqa: BLE001
                    pass
            store.append_transcript(
                sid,
                {
                    "type": "user",
                    "content": prompt,
                    "images": [str(p) for p in imgs],
                },
            )

        if append is not None:
            append(
                {
                    "type": "user",
                    "content": prompt,
                    "images": [str(p) for p in imgs],
                    "helper_session": sid,
                }
            )

        messages: list[dict[str, Any]] = []
        system = self._system_prompt()
        if system:
            messages.append({"role": "system", "content": system})
        for msg in prior:
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            messages.append(
                {
                    "role": role,
                    "content": _text_only_content(msg.get("content")),
                }
            )
        messages.append({"role": "user", "content": content})

        try:
            turn = self.media.chat(messages)
            reply = turn.content or ""
            reasoning = getattr(turn, "reasoning", None) or None
            ok = True
            err = None
        except Exception as exc:  # noqa: BLE001
            reply = ""
            reasoning = None
            ok = False
            err = str(exc)

        asst_text = reply if ok else f"[helper error] {err}"
        history_out = prior + [
            {"role": "user", "content": _text_only_content(content)},
            {"role": "assistant", "content": asst_text},
        ]

        if store is not None and sid is not None:
            store.append_transcript(
                sid,
                {
                    "type": "assistant",
                    "content": asst_text,
                    "reasoning": reasoning,
                },
            )
            try:
                if finalize:
                    store.update_meta(sid, status="done" if ok else "failed")
                else:
                    store.update_meta(sid, status="running" if ok else "failed")
            except Exception:  # noqa: BLE001
                pass

        if append is not None:
            append(
                {
                    "type": "assistant",
                    "content": asst_text,
                    "reasoning": reasoning,
                    "helper_session": sid,
                }
            )

        return HelperResult(
            content=reply,
            session_id=sid,
            ok=ok,
            error=err,
            history=history_out,
            meta={
                "label": label,
                "image_count": len(imgs),
                "reasoning": reasoning,
                "history_turns": len(history_out) // 2,
            },
        )

    def run_prompt_file(
        self,
        prompt_path: Path | str,
        *,
        extra: str = "",
        images: list[Path | str] | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> HelperResult:
        """Load a markdown/text prompt file, append ``extra``, then ``run``."""
        path = Path(prompt_path)
        body = path.read_text(encoding="utf-8") if path.is_file() else str(prompt_path)
        prompt = f"{body}\n\n{extra}".strip() if extra else body
        return self.run(
            prompt,
            images=images,
            label=label or path.stem,
            **kwargs,
        )
