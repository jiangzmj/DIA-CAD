"""Tool schema registry + handler dispatch."""

from __future__ import annotations

import json
from typing import Any, Callable


Handler = Callable[..., str]


class ToolRegistry:
    def __init__(self) -> None:
        self._schemas: list[dict[str, Any]] = []
        self._handlers: dict[str, Handler] = {}

    def register(self, schema: dict[str, Any], handler: Handler) -> None:
        name = (schema.get("function") or {}).get("name")
        if not name:
            raise ValueError("Tool schema missing function.name")
        self._schemas.append(schema)
        self._handlers[name] = handler

    def schemas(self) -> list[dict[str, Any]]:
        return list(self._schemas)

    def names(self) -> list[str]:
        return list(self._handlers.keys())

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            return handler(**arguments)
        except TypeError as exc:
            return f"Error calling {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 — surface to model
            return f"Error in {name}: {exc}"

    def without(self, *names: str) -> ToolRegistry:
        blocked = set(names)
        clone = ToolRegistry()
        for schema in self._schemas:
            fname = (schema.get("function") or {}).get("name")
            if fname in blocked:
                continue
            clone.register(schema, self._handlers[fname])
        return clone
