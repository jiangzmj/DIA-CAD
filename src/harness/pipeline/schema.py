"""Hard schema validation for edit.json (step 6)."""

from __future__ import annotations

from typing import Any


REQUIRED_TOP = ("version", "edit_type", "operations")
ALLOWED_EDIT_TYPES = {"modify", "add", "remove", "replace", "other"}
OP_REQUIRED = ("op", "target")


def validate_edit_json(data: Any) -> dict[str, Any]:
    """Return ``{ok, errors, hard}`` for hard schema gate."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return {"ok": False, "errors": ["root must be object"], "hard": "fail"}

    for key in REQUIRED_TOP:
        if key not in data:
            errors.append(f"missing field: {key}")

    version = data.get("version")
    if version is not None and not isinstance(version, int):
        errors.append("version must be int")

    edit_type = data.get("edit_type")
    if edit_type is not None and edit_type not in ALLOWED_EDIT_TYPES:
        errors.append(f"edit_type must be one of {sorted(ALLOWED_EDIT_TYPES)}")

    ops = data.get("operations")
    if ops is not None:
        if not isinstance(ops, list):
            errors.append("operations must be list")
        else:
            for i, op in enumerate(ops):
                if not isinstance(op, dict):
                    errors.append(f"operations[{i}] must be object")
                    continue
                for k in OP_REQUIRED:
                    if k not in op:
                        errors.append(f"operations[{i}] missing {k}")
                if "params" in op and not isinstance(op["params"], dict):
                    errors.append(f"operations[{i}].params must be object")

    ok = not errors
    return {"ok": ok, "errors": errors, "hard": "pass" if ok else "fail"}


def skeleton_edit_json(description: str = "") -> dict[str, Any]:
    return {
        "version": 1,
        "edit_type": "modify",
        "operations": [
            {
                "op": "modify_feature",
                "target": "body",
                "params": {"note": description[:200] if description else "stub"},
            }
        ],
        "notes": description[:500] if description else "",
    }
