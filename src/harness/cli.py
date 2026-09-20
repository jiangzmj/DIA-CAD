"""CLI REPL for Autodesk3 Harness."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from harness.config import load_settings
from harness.loop import create_app_bundle
from harness.providers.router import ProviderRouter


BANNER = """
Autodesk3 Harness
  /new [label]              new root session
  /list                     list sessions
  /switch <id>              switch session
  /tree                     session tree
  /context [k=v ...]        show/set context policy
  /provider [name]          show/switch provider (openai_compat|chatgpt_oauth)
  /spawn <role> <task...>   spawn child agent
  /quit                     exit
""".strip()


def _print_sessions(store) -> None:
    for sid, meta in store.list_sessions():
        mark = "*" if sid == store.current_session_id else " "
        print(
            f"{mark} {sid}  d={meta.depth}  {meta.status:6}  "
            f"role={meta.agent_role:10}  msgs={meta.message_count:3}  {meta.label}"
        )


def _print_tree(store) -> None:
    cur = store.current_session_id
    if not cur:
        print("No active session")
        return
    meta = store.get_meta(cur)
    root = (meta.root_id if meta else None) or cur
    for node in store.get_tree(root):
        indent = "  " * node.depth
        mark = ">" if node.session_id == cur else " "
        print(
            f"{mark}{indent}{node.session_id} [{node.agent_role}] "
            f"{node.status} — {node.label}"
        )


def _handle_context(policy, args: list[str]) -> None:
    if not args:
        for k, v in policy.as_dict().items():
            print(f"  {k}={v}")
        return
    updates = {}
    for token in args:
        if "=" not in token:
            print(f"Expected key=value, got: {token}")
            return
        k, v = token.split("=", 1)
        if k in ("max_messages", "max_tool_chars", "handoff_window"):
            updates[k] = int(v)
        elif k in ("compact_keep_recent", "compact_summarize_oldest"):
            updates[k] = float(v)
        elif k == "system_layers":
            updates[k] = [x for x in v.split(",") if x]
        else:
            updates[k] = v
    policy.update(**updates)
    print("Updated context policy:")
    for k, v in policy.as_dict().items():
        print(f"  {k}={v}")


def run_repl(provider_override: str | None = None) -> int:
    settings = load_settings()
    if provider_override:
        settings.provider.primary = provider_override

    try:
        router = ProviderRouter(settings)
    except Exception as exc:
        print(f"Failed to init provider '{settings.provider.primary}': {exc}")
        if settings.provider.primary != "openai_compat":
            print("Falling back to openai_compat...")
            settings.provider.primary = "openai_compat"
            router = ProviderRouter(settings)
        else:
            return 1

    bundle = create_app_bundle(settings, router)
    store = bundle["store"]
    policy = bundle["policy"]
    spawn = bundle["spawn"]
    make_main = bundle["make_main_runner"]

    if not store.current_session_id:
        sessions = store.list_sessions()
        if sessions:
            store.load_session(sessions[0][0])
        else:
            store.create_session(label="main", agent_role="main")

    runner = make_main(store.current_session_id)
    print(BANNER)
    print(f"provider={router.active_name}  session={store.current_session_id}")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            try:
                parts = shlex.split(line)
            except ValueError as exc:
                print(f"parse error: {exc}")
                continue
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in ("/quit", "/exit", "/q"):
                break
            if cmd == "/new":
                label = " ".join(args) if args else "main"
                sid = store.create_session(label=label, agent_role="main")
                runner = make_main(sid)
                print(f"created session {sid}")
                continue
            if cmd == "/list":
                _print_sessions(store)
                continue
            if cmd == "/switch":
                if not args:
                    print("usage: /switch <session_id>")
                    continue
                sid = args[0]
                if not store.get_meta(sid):
                    print(f"unknown session {sid}")
                    continue
                store.load_session(sid)
                runner = make_main(sid)
                print(f"switched to {sid}")
                continue
            if cmd == "/tree":
                _print_tree(store)
                continue
            if cmd == "/context":
                _handle_context(policy, args)
                continue
            if cmd == "/provider":
                if not args:
                    print(f"active={router.active_name} primary={router.primary_name}")
                    continue
                name = args[0]
                try:
                    router.switch(name)
                    # rebuild bundle pieces that hold provider
                    bundle = create_app_bundle(settings, router)
                    store = bundle["store"]
                    policy = bundle["policy"]
                    spawn = bundle["spawn"]
                    make_main = bundle["make_main_runner"]
                    runner = make_main(store.current_session_id)
                    print(f"provider={router.active_name}")
                except Exception as exc:
                    print(f"switch failed: {exc}")
                continue
            if cmd == "/spawn":
                if len(args) < 2:
                    print("usage: /spawn <role> <task...>")
                    continue
                role, task = args[0], " ".join(args[1:])
                parent = store.current_session_id
                result = spawn.run(
                    parent_session_id=parent,
                    role=role,
                    task=task,
                    parent_messages=list(runner._messages),
                    child_tools=bundle["base_tools"],
                )
                print(result)
                # reload parent runner messages
                runner = make_main(parent)
                continue
            if cmd == "/help":
                print(BANNER)
                continue
            print(f"unknown command: {cmd} (try /help)")
            continue

        reply = runner.run_turn(line)
        print(f"agent> {reply}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autodesk3 Harness CLI")
    parser.add_argument(
        "command",
        nargs="?",
        default="repl",
        choices=["repl", "login"],
        help="repl (default) or login help",
    )
    parser.add_argument(
        "--provider",
        choices=["openai_compat", "chatgpt_oauth"],
        default=None,
        help="Override DEFAULT_PROVIDER",
    )
    args = parser.parse_args(argv)

    if args.command == "login":
        auth = Path.home() / ".codex" / "auth.json"
        print(
            "ChatGPT OAuth uses the Codex CLI credential cache.\n"
            "  1. Install Codex CLI and run: codex login\n"
            f"  2. Confirm auth file exists: {auth}\n"
            "  3. Set DEFAULT_PROVIDER=chatgpt_oauth in .env\n"
            "  4. Run: PYTHONPATH=src python -m harness.cli\n"
            "\nThis harness does not open a browser itself; it reuses Codex tokens."
        )
        return 0

    return run_repl(provider_override=args.provider)


if __name__ == "__main__":
    sys.exit(main())
