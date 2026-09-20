"""Core agent loop: messages + tools + finish_reason."""

from __future__ import annotations

import json
from typing import Any, Callable

from harness.agents.roles import build_system_prompt
from harness.agents.spawn import SpawnService
from harness.config import Settings
from harness.context.guard import ContextGuard
from harness.context.policy import ContextPolicy
from harness.providers.base import LLMProvider
from harness.session.store import SessionStore
from harness.tools.registry import ToolRegistry


class AgentRunner:
    def __init__(
        self,
        settings: Settings,
        store: SessionStore,
        provider: LLMProvider,
        tools: ToolRegistry,
        policy: ContextPolicy,
        system_prompt: str | None = None,
        session_id: str | None = None,
        allow_spawn: bool = True,
        on_event: Callable[[str], None] | None = None,
    ):
        self.settings = settings
        self.store = store
        self.provider = provider
        self.tools = tools
        self.policy = policy
        self.workspace = settings.workspace
        self.system_prompt = system_prompt or build_system_prompt(
            self.workspace, "main", policy.system_layers
        )
        self.session_id = session_id or store.current_session_id
        self.allow_spawn = allow_spawn
        self.on_event = on_event or (lambda _m: None)
        self.guard = ContextGuard(policy, provider)
        self._messages: list[dict[str, Any]] = []

        if self.session_id:
            self._messages = store.load_session(self.session_id)
            store.current_session_id = self.session_id

    def _emit(self, msg: str) -> None:
        self.on_event(msg)

    def _api_messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system_prompt}] + self._messages

    def _persist_assistant(self, content: str | None, tool_calls: list) -> None:
        # Save assistant text first
        if content:
            self.store.save_turn("assistant", content)
        elif tool_calls:
            self.store.save_turn("assistant", "")

    def run_turn(self, user_text: str) -> str:
        if not self.store.current_session_id:
            raise RuntimeError("No active session")

        self._messages.append({"role": "user", "content": user_text})
        self.store.save_turn("user", user_text)

        final_text = ""
        for _ in range(self.settings.agent.max_tool_iterations):
            turn, used = self.guard.guard_api_call(
                self._api_messages(),
                tools=self.tools.schemas() or None,
            )
            # If compaction changed history beyond system, sync soft view:
            # keep self._messages as source of truth; guard only retries API payload.

            if turn.tool_calls:
                # Append assistant with tool_calls to in-memory history
                tc_payload = []
                for tc in turn.tool_calls:
                    tc_payload.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                    )
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": tc_payload,
                }
                self._messages.append(assistant_msg)
                self._persist_assistant(turn.content, turn.tool_calls)

                for tc in turn.tool_calls:
                    self._emit(f"→ tool {tc.name}({json.dumps(tc.arguments)[:120]})")
                    result = self.tools.call(tc.name, tc.arguments)
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
                    self.store.save_tool_exchange(tc.id, tc.name, tc.arguments, result)
                continue

            final_text = turn.content or ""
            self._messages.append({"role": "assistant", "content": final_text})
            self.store.save_turn("assistant", final_text)
            break
        else:
            final_text = final_text or "[stopped: max tool iterations]"
            self._messages.append({"role": "assistant", "content": final_text})
            self.store.save_turn("assistant", final_text)

        return final_text

    def run_existing_session(self, max_user_turns: int = 20) -> str:
        """Continue a pre-seeded session (used by spawn). Returns last assistant text."""
        # Ensure messages loaded
        if self.session_id:
            self._messages = self.store.load_session(self.session_id)
            self.store.current_session_id = self.session_id

        last = ""
        # One "kick" if last message is user (handoff), otherwise ask model to proceed
        if not self._messages or self._messages[-1].get("role") != "user":
            return self.run_turn("Continue and complete your assigned task.")

        # Drive tool loop without adding another user message — reuse last user
        # by calling provider on current history.
        final_text = ""
        for _ in range(min(max_user_turns, self.settings.agent.max_tool_iterations)):
            turn, _used = self.guard.guard_api_call(
                self._api_messages(),
                tools=self.tools.schemas() or None,
            )
            if turn.tool_calls:
                tc_payload = []
                for tc in turn.tool_calls:
                    tc_payload.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                    )
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": turn.content,
                        "tool_calls": tc_payload,
                    }
                )
                self._persist_assistant(turn.content, turn.tool_calls)
                for tc in turn.tool_calls:
                    result = self.tools.call(tc.name, tc.arguments)
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
                    self.store.save_tool_exchange(tc.id, tc.name, tc.arguments, result)
                continue

            final_text = turn.content or ""
            self._messages.append({"role": "assistant", "content": final_text})
            self.store.save_turn("assistant", final_text)
            last = final_text
            break
        return last or final_text


def attach_spawn_tool(
    runner: AgentRunner,
    spawn: SpawnService,
    child_tools: ToolRegistry,
) -> None:
    if not runner.allow_spawn:
        return
    if "spawn_agent" in runner.tools.names():
        return

    def get_parent_session_id() -> str:
        sid = runner.store.current_session_id
        if not sid:
            raise RuntimeError("No parent session")
        return sid

    def get_parent_messages() -> list[dict[str, Any]]:
        return list(runner._messages)

    runner.tools.register(
        spawn.tool_schema(),
        spawn.as_handler(get_parent_session_id, get_parent_messages, child_tools),
    )


def create_app_bundle(settings: Settings, provider: LLMProvider):
    """Wire store, policy, tools, spawn, and a factory for runners."""
    from harness.tools.builtin import build_builtin_tools

    store = SessionStore(settings.workspace)
    policy = ContextPolicy(
        max_messages=settings.context.max_messages,
        max_tool_chars=settings.context.max_tool_chars,
        compact_keep_recent=settings.context.compact_keep_recent,
        compact_summarize_oldest=settings.context.compact_summarize_oldest,
        handoff_default=settings.context.handoff_default,  # type: ignore[arg-type]
        handoff_window=settings.context.handoff_window,
        system_layers=list(settings.context.system_layers),
    )
    base_tools = build_builtin_tools(settings, settings.workspace)
    child_tools = base_tools.without()  # full builtins for children, spawn added separately only on main

    def make_runner(
        tools: ToolRegistry | None = None,
        allow_spawn: bool = False,
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> AgentRunner:
        t = tools or child_tools
        return AgentRunner(
            settings=settings,
            store=store,
            provider=provider,
            tools=t,
            policy=policy,
            system_prompt=system_prompt,
            session_id=session_id,
            allow_spawn=allow_spawn,
            on_event=lambda m: print(m),
        )

    spawn = SpawnService(
        store=store,
        provider=provider,
        policy=policy,
        workspace=settings.workspace,
        max_depth=settings.agent.max_depth,
        max_child_turns=settings.agent.max_child_turns,
        make_runner=make_runner,
    )

    main_tools = build_builtin_tools(settings, settings.workspace)

    def make_main_runner(session_id: str | None = None) -> AgentRunner:
        runner = AgentRunner(
            settings=settings,
            store=store,
            provider=provider,
            tools=main_tools,
            policy=policy,
            session_id=session_id or store.current_session_id,
            allow_spawn=True,
            on_event=lambda m: print(m),
        )
        attach_spawn_tool(runner, spawn, child_tools)
        return runner

    return {
        "store": store,
        "policy": policy,
        "spawn": spawn,
        "make_main_runner": make_main_runner,
        "make_runner": make_runner,
        "base_tools": base_tools,
    }
