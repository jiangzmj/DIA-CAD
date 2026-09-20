"""Spawn child agents with adjustable context handoff."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from harness.agents.roles import build_system_prompt
from harness.context.handoff import build_handoff_messages
from harness.context.policy import ContextPolicy, HandoffMode
from harness.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from harness.loop import AgentRunner
    from harness.session.store import SessionStore
    from harness.providers.base import LLMProvider


class SpawnService:
    def __init__(
        self,
        store: SessionStore,
        provider: LLMProvider,
        policy: ContextPolicy,
        workspace,
        max_depth: int,
        max_child_turns: int,
        make_runner,
    ):
        self.store = store
        self.provider = provider
        self.policy = policy
        self.workspace = workspace
        self.max_depth = max_depth
        self.max_child_turns = max_child_turns
        self.make_runner = make_runner

    def run(
        self,
        parent_session_id: str,
        role: str,
        task: str,
        context_mode: str | None = None,
        window_n: int | None = None,
        max_turns: int | None = None,
        parent_messages: list[dict[str, Any]] | None = None,
        child_tools: ToolRegistry | None = None,
    ) -> dict[str, Any]:
        parent = self.store.get_meta(parent_session_id)
        if not parent:
            return {"status": "failed", "error": f"parent session not found: {parent_session_id}"}
        if parent.depth >= self.max_depth:
            return {
                "status": "failed",
                "error": f"max spawn depth {self.max_depth} reached (parent depth={parent.depth})",
            }

        mode: HandoffMode | None = None
        if context_mode:
            if context_mode not in ("none", "summary", "window", "full"):
                return {"status": "failed", "error": f"invalid context_mode: {context_mode}"}
            mode = context_mode  # type: ignore[assignment]

        child_id = self.store.create_session(
            label=f"{role}:{task[:40]}",
            agent_role=role,
            parent_id=parent_session_id,
        )

        parent_msgs = parent_messages
        if parent_msgs is None:
            # temporarily load parent
            prev = self.store.current_session_id
            parent_msgs = self.store.load_session(parent_session_id)
            self.store.current_session_id = prev

        handoff = build_handoff_messages(
            parent_msgs,
            task=task,
            policy=self.policy,
            mode=mode,
            window_n=window_n,
            provider=self.provider,
        )

        # Seed child transcript
        self.store.current_session_id = child_id
        for msg in handoff:
            self.store.save_turn(msg["role"], msg.get("content"))
        self.store.save_system_note(
            f"spawned by {parent_session_id} role={role} mode={mode or self.policy.handoff_default}",
            kind="handoff",
        )

        layers = list(self.policy.system_layers)
        if role != "main":
            # minimal layers for sub-agents
            layers = [l for l in layers if l in ("tools", "runtime", "identity")]

        system_prompt = build_system_prompt(self.workspace, role, layers)
        runner: AgentRunner = self.make_runner(
            tools=child_tools,
            allow_spawn=False,
            system_prompt=system_prompt,
            session_id=child_id,
        )

        turns = max_turns or self.max_child_turns
        # Drive child: first handoff user message already saved; run until stop
        # by feeding empty continuation via run_loaded_session
        try:
            summary = runner.run_existing_session(max_user_turns=turns)
            self.store.update_meta(child_id, status="done")
            status = "done"
        except Exception as exc:  # noqa: BLE001
            self.store.update_meta(child_id, status="failed")
            summary = f"Child agent failed: {exc}"
            status = "failed"

        # restore parent as current
        self.store.current_session_id = parent_session_id
        return {
            "status": status,
            "session_id": child_id,
            "role": role,
            "summary": summary,
        }

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "description": (
                    "Spawn a child agent with its own session. "
                    "Use for specialized subtasks. Returns child session_id and summary."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["researcher", "coder", "worker", "helper"],
                        },
                        "task": {"type": "string"},
                        "context_mode": {
                            "type": "string",
                            "enum": ["none", "summary", "window", "full"],
                        },
                        "window_n": {"type": "integer"},
                        "max_turns": {"type": "integer"},
                    },
                    "required": ["role", "task"],
                },
            },
        }

    def as_handler(self, get_parent_session_id, get_parent_messages, child_tools: ToolRegistry):
        def spawn_agent(
            role: str,
            task: str,
            context_mode: str | None = None,
            window_n: int | None = None,
            max_turns: int | None = None,
        ) -> str:
            result = self.run(
                parent_session_id=get_parent_session_id(),
                role=role,
                task=task,
                context_mode=context_mode,
                window_n=window_n,
                max_turns=max_turns,
                parent_messages=get_parent_messages(),
                child_tools=child_tools,
            )
            return json.dumps(result, ensure_ascii=False)

        return spawn_agent
