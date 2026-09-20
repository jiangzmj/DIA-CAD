"""Built-in tools: bash, files, http_request."""

from __future__ import annotations

import ipaddress
import json
import socket
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from harness.config import Settings
from harness.net import resolve_proxy
from harness.tools.registry import ToolRegistry


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated {len(text)} chars ...]"


def _resolve_under_workspace(workspace: Path, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = workspace / p
    resolved = p.resolve()
    workspace_resolved = workspace.resolve()
    if workspace_resolved not in resolved.parents and resolved != workspace_resolved:
        raise PermissionError(f"Path escapes workspace: {path}")
    return resolved


def _host_is_private(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
        ):
            return True
    return False


def build_builtin_tools(settings: Settings, workspace: Path) -> ToolRegistry:
    registry = ToolRegistry()
    max_bash = settings.tools.max_bash_output_chars
    http_cfg = settings.tools.http_request

    def bash(command: str) -> str:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=120,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode != 0:
                out = f"[exit {proc.returncode}]\n{out}"
            return _clip(out or "(no output)", max_bash)
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 120s"

    def read_file(path: str) -> str:
        target = _resolve_under_workspace(workspace, path)
        return _clip(target.read_text(encoding="utf-8"), max_bash)

    def write_file(path: str, content: str) -> str:
        target = _resolve_under_workspace(workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {target}"

    def edit_file(path: str, old_string: str, new_string: str) -> str:
        target = _resolve_under_workspace(workspace, path)
        text = target.read_text(encoding="utf-8")
        if old_string not in text:
            return "Error: old_string not found in file"
        count = text.count(old_string)
        if count != 1:
            return f"Error: old_string matched {count} times; must be unique"
        target.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        return f"Edited {target}"

    def list_directory(path: str = ".") -> str:
        target = _resolve_under_workspace(workspace, path)
        if not target.is_dir():
            return f"Error: not a directory: {path}"
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for e in entries:
            suffix = "/" if e.is_dir() else ""
            lines.append(e.name + suffix)
        return "\n".join(lines) or "(empty)"

    def http_request(
        method: str,
        url: str,
        headers: dict[str, Any] | None = None,
        body: Any = None,
    ) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "Error: only http/https URLs allowed"
        if not parsed.hostname:
            return "Error: missing hostname"
        if not http_cfg.allow_private_hosts and _host_is_private(parsed.hostname):
            return "Error: private/loopback hosts are blocked (set tools.http_request.allow_private_hosts)"

        req_headers = {str(k): str(v) for k, v in (headers or {}).items()}
        content: bytes | None = None
        if body is not None:
            if isinstance(body, (dict, list)):
                content = json.dumps(body).encode("utf-8")
                req_headers.setdefault("Content-Type", "application/json")
            elif isinstance(body, str):
                content = body.encode("utf-8")
            else:
                content = str(body).encode("utf-8")

        try:
            proxy = resolve_proxy(
                settings.proxy or settings.provider.proxy,
                auto_detect=settings.provider.auto_detect_proxy,
            )
            client_kwargs: dict[str, Any] = {
                "timeout": http_cfg.timeout_seconds,
                "follow_redirects": True,
                "trust_env": False,
            }
            if proxy:
                client_kwargs["proxy"] = proxy
            with httpx.Client(**client_kwargs) as client:
                resp = client.request(method.upper(), url, headers=req_headers, content=content)
            text = resp.text
            return _clip(
                f"HTTP {resp.status_code}\n{text}",
                http_cfg.max_response_chars,
            )
        except Exception as exc:  # noqa: BLE001
            return f"Error: http_request failed: {exc}"

    registry.register(
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command in the workspace directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        bash,
    )
    registry.register(
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file under the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        read_file,
    )
    registry.register(
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write UTF-8 text to a file under the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        write_file,
    )
    registry.register(
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Exact string replacement in a workspace file (old_string must be unique).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
        },
        edit_file,
    )
    registry.register(
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files in a workspace directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        },
        list_directory,
    )
    registry.register(
        {
            "type": "function",
            "function": {
                "name": "http_request",
                "description": "Call an arbitrary HTTP API (method/url/headers/body).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string"},
                        "url": {"type": "string"},
                        "headers": {"type": "object"},
                        "body": {},
                    },
                    "required": ["method", "url"],
                },
            },
        },
        http_request,
    )
    return registry
