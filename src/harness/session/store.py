"""JSONL session store with parent/child chain metadata."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.session.models import SessionMeta


class SessionStore:
    """Persist turns as JSONL; index sessions with parent/child links."""

    def __init__(self, workspace: Path, agent_id: str = "default"):
        self.agent_id = agent_id
        self.base_dir = workspace / ".sessions" / "agents" / agent_id / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir.parent / "sessions.json"
        self._index: dict[str, dict[str, Any]] = self._load_index()
        self.current_session_id: str | None = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _session_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.jsonl"

    def get_meta(self, session_id: str) -> SessionMeta | None:
        raw = self._index.get(session_id)
        if not raw:
            return None
        data = dict(raw)
        data["session_id"] = session_id
        return SessionMeta.from_dict(data)

    def update_meta(self, session_id: str, **kwargs: Any) -> None:
        if session_id not in self._index:
            return
        self._index[session_id].update(kwargs)
        self._index[session_id]["last_active"] = self._now()
        self._save_index()

    def create_session(
        self,
        label: str = "",
        agent_role: str = "main",
        parent_id: str | None = None,
    ) -> str:
        session_id = uuid.uuid4().hex[:12]
        now = self._now()
        parent = self.get_meta(parent_id) if parent_id else None
        depth = (parent.depth + 1) if parent else 0
        root_id = parent.root_id if parent and parent.root_id else (
            parent.session_id if parent else session_id
        )
        if not parent:
            root_id = session_id

        meta = SessionMeta(
            session_id=session_id,
            label=label or agent_role,
            agent_role=agent_role,
            parent_id=parent_id,
            root_id=root_id,
            depth=depth,
            status="active",
            created_at=now,
            last_active=now,
            message_count=0,
        )
        self._index[session_id] = meta.to_dict()
        # store without duplicating session_id key inside? keep it for round-trip
        self._save_index()
        self._session_path(session_id).touch()
        self.current_session_id = session_id
        return session_id

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        path = self._session_path(session_id)
        if not path.exists():
            return []
        self.current_session_id = session_id
        return self._rebuild_history(path)

    def save_turn(self, role: str, content: Any) -> None:
        if not self.current_session_id:
            return
        self.append_transcript(
            self.current_session_id,
            {"type": role, "content": content, "ts": time.time()},
        )

    def save_system_note(self, content: str, kind: str = "note") -> None:
        if not self.current_session_id:
            return
        self.append_transcript(
            self.current_session_id,
            {
                "type": "system_note",
                "content": content,
                "kind": kind,
                "ts": time.time(),
            },
        )

    def save_tool_exchange(
        self,
        tool_use_id: str,
        name: str,
        tool_input: dict[str, Any],
        result: str,
    ) -> None:
        if not self.current_session_id:
            return
        ts = time.time()
        sid = self.current_session_id
        self.append_transcript(
            sid,
            {
                "type": "tool_use",
                "tool_use_id": tool_use_id,
                "name": name,
                "input": tool_input,
                "ts": ts,
            },
        )
        self.append_transcript(
            sid,
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
                "ts": ts,
            },
        )

    def append_transcript(self, session_id: str, record: dict[str, Any]) -> None:
        path = self._session_path(session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if session_id in self._index:
            self._index[session_id]["last_active"] = self._now()
            self._index[session_id]["message_count"] = (
                int(self._index[session_id].get("message_count") or 0) + 1
            )
            self._save_index()

    def _rebuild_history(self, path: Path) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return messages

        for line in text.split("\n"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")

            if rtype == "user":
                messages.append({"role": "user", "content": record.get("content")})

            elif rtype == "assistant":
                content = record.get("content")
                if isinstance(content, list):
                    text_parts = [
                        b["text"]
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    content = " ".join(text_parts)
                messages.append({"role": "assistant", "content": content or None})

            elif rtype == "tool_use":
                tool_call = {
                    "id": record["tool_use_id"],
                    "type": "function",
                    "function": {
                        "name": record["name"],
                        "arguments": json.dumps(record.get("input") or {}),
                    },
                }
                if messages and messages[-1]["role"] == "assistant":
                    messages[-1].setdefault("tool_calls", []).append(tool_call)
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call],
                        }
                    )

            elif rtype == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": record["tool_use_id"],
                        "content": record.get("content") or "",
                    }
                )

            elif rtype == "system_note":
                # Not sent as a system role by default; keep as user note for API continuity
                kind = record.get("kind") or "note"
                messages.append(
                    {
                        "role": "user",
                        "content": f"[system_note:{kind}]\n{record.get('content') or ''}",
                    }
                )

        return messages

    def list_sessions(self) -> list[tuple[str, SessionMeta]]:
        items: list[tuple[str, SessionMeta]] = []
        for sid, raw in self._index.items():
            data = dict(raw)
            data["session_id"] = sid
            items.append((sid, SessionMeta.from_dict(data)))
        items.sort(key=lambda x: x[1].last_active, reverse=True)
        return items

    def get_children(self, session_id: str) -> list[SessionMeta]:
        out: list[SessionMeta] = []
        for sid, meta in self.list_sessions():
            if meta.parent_id == session_id:
                out.append(meta)
        out.sort(key=lambda m: m.created_at)
        return out

    def get_ancestry(self, session_id: str) -> list[SessionMeta]:
        chain: list[SessionMeta] = []
        seen: set[str] = set()
        cur = session_id
        while cur and cur not in seen:
            seen.add(cur)
            meta = self.get_meta(cur)
            if not meta:
                break
            chain.append(meta)
            cur = meta.parent_id or ""
        return chain

    def get_tree(self, root_id: str) -> list[SessionMeta]:
        meta = self.get_meta(root_id)
        effective_root = (meta.root_id if meta else None) or root_id
        nodes = [
            m
            for _, m in self.list_sessions()
            if (m.root_id or m.session_id) == effective_root
        ]
        nodes.sort(key=lambda m: (m.depth, m.created_at))
        return nodes
