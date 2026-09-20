"""Controller-owned capability registry.

The MLLM may *propose* UNSUPPORTED_EDIT; only the controller confirms it
after checking this table and the failure history.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_OPERATIONS = frozenset(
    {
        "add_hole",
        "add_boss",
        "cut_slot",
        "extend_cylinder",
        "chamfer",
        "fillet",
        "modify_feature",
    }
)

EDIT_CLASSES = (
    "local_modify",
    "additive",
    "subtractive",
    "replace",
    "global_transform",
)

# Gate usage only. Does not expand SUPPORTED_OPERATIONS / UNSUPPORTED confirmation.
_CANONICAL_EDIT_CLASS = {
    "add_hole": "subtractive",
    "add_boss": "additive",
    "cut_slot": "subtractive",
    "extend_cylinder": "additive",
    "chamfer": "local_modify",
    "fillet": "local_modify",
    "modify_feature": "replace",
}

_GLOBAL_TOKENS = (
    "mirror",
    "pattern",
    "scale",
    "duplicate",
    "array",
    "boolean_union",
)
_LOCAL_TOKENS = (
    "fillet",
    "chamfer",
    "bevel",
    "blend",
    "round_inner",
    "round_outer",
    "edge_radii",
    "edge_treatment",
    "apply_draft",
)
_REPLACE_TOKENS = ("replace", "change_", "reshape", "modify_")
_CYLINDER_OPS = frozenset({"extend_cylinder", "add_boss", "add_hole"})

OPERATION_ALIASES = {
    "modify_cylindrical_boss": "extend_cylinder",
    "extend_cylindrical_boss": "extend_cylinder",
    "increase_height": "extend_cylinder",
    "extend_boss": "extend_cylinder",
    "pad": "add_boss",
    "boss": "add_boss",
    "add_pad": "add_boss",
    "add_cylinder": "add_boss",
    "hole": "add_hole",
    "cut_hole": "add_hole",
    "drill": "add_hole",
    "bore": "add_hole",
    "slot": "cut_slot",
    "cut_pocket": "cut_slot",
    "pocket": "cut_slot",
    "round": "fillet",
    "blend": "fillet",
    "bevel": "chamfer",
    "modify": "modify_feature",
}

_INSTRUCTION_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("chamfer", "bevel"), "chamfer"),
    (("fillet", "round the", "blend"), "fillet"),
    (("hole", "drill", "bore"), "add_hole"),
    (("slot", "pocket"), "cut_slot"),
    (("boss", "pad", "taller", "height", "cylinder", "extrude"), "extend_cylinder"),
)


def canonicalize_operation(operation: str | None) -> str | None:
    raw = (operation or "").strip().lower().replace(" ", "_")
    if not raw:
        return None
    if raw in SUPPORTED_OPERATIONS:
        return raw
    return OPERATION_ALIASES.get(raw)


def _classify_one(operation: str | None) -> str:
    raw = (operation or "").strip().lower().replace(" ", "_")
    if not raw:
        return "replace"
    mapped = canonicalize_operation(raw)
    if mapped in _CANONICAL_EDIT_CLASS:
        return _CANONICAL_EDIT_CLASS[mapped]
    if any(tok in raw for tok in _GLOBAL_TOKENS):
        return "global_transform"
    if any(tok in raw for tok in _LOCAL_TOKENS):
        return "local_modify"
    if any(tok in raw for tok in _REPLACE_TOKENS) or raw in {"replace", "modify"}:
        return "replace"
    if raw.startswith(("cut_", "remove", "shorten")) or raw in {"trim", "remove"}:
        return "subtractive"
    if "reduce" in raw or "shorten" in raw:
        return "subtractive"
    if raw.startswith(("add_", "insert", "create", "mount")):
        return "additive"
    if raw in {"extend", "emboss_text", "position_blade"} or "increase" in raw or "extend" in raw:
        return "additive"
    return "replace"


def edit_class(
    operation: str | None,
    extra_ops: list[str] | None = None,
) -> str:
    """Classify an edit for gate policy. Not a capability registry."""
    ops = [operation] if operation else []
    ops.extend(extra_ops or [])
    classes = [_classify_one(op) for op in ops if str(op or "").strip()]
    if not classes:
        return "replace"
    if "global_transform" in classes:
        return "global_transform"
    if "additive" in classes and "subtractive" in classes:
        return "replace"
    if "replace" in classes:
        return "replace"
    if "additive" in classes:
        return "additive"
    if "subtractive" in classes:
        return "subtractive"
    return "local_modify"


def edit_class_from_legacy(edit_json: dict[str, Any] | None) -> str:
    ops = []
    for item in (edit_json or {}).get("operations") or []:
        if isinstance(item, dict) and item.get("op"):
            ops.append(str(item["op"]))
    first = ops[0] if ops else str((edit_json or {}).get("edit_type") or "")
    return edit_class(first, extra_ops=ops[1:])


def is_cylinder_intent(operation: str | None, parameters: dict[str, Any] | None = None) -> bool:
    mapped = canonicalize_operation(operation)
    if mapped in _CYLINDER_OPS:
        return True
    params = parameters or {}
    numeric = any(
        k in params
        for k in (
            "target_height_mm",
            "height_mm",
            "target_radius_mm",
            "radius_mm",
            "hole_diameter_mm",
            "diameter_mm",
        )
    )
    return bool(numeric and edit_class(operation) in {"additive", "subtractive"})


def infer_operation_from_instruction(instruction: str) -> str | None:
    text = (instruction or "").lower()
    if not text.strip():
        return None
    for keywords, op in _INSTRUCTION_HINTS:
        if any(k in text for k in keywords):
            return op
    return None


def confirm_unsupported_edit(
    *,
    claimed_by_mllm: bool,
    operation: str | None,
    instruction: str = "",
    failure_errors: list[str] | None = None,
    min_repeated_failures: int = 3,
) -> dict[str, Any]:
    """Controller gate. MLLM claim is evidence, not a terminal decision."""
    mapped = canonicalize_operation(operation)
    if mapped is None:
        mapped = infer_operation_from_instruction(instruction)
    errors = [e.strip()[:240] for e in (failure_errors or []) if str(e).strip()]
    repeated = len(errors) >= min_repeated_failures and len(set(errors[-min_repeated_failures:])) == 1

    if mapped in SUPPORTED_OPERATIONS:
        if repeated:
            return {
                "confirmed": True,
                "canonical_operation": mapped,
                "reason": (
                    f"supported operation {mapped!r} failed {min_repeated_failures} "
                    "times in a row; controller confirms UNSUPPORTED_EDIT"
                ),
            }
        return {
            "confirmed": False,
            "canonical_operation": mapped,
            "reason": (
                f"operation {mapped!r} is in SUPPORTED_OPERATIONS; "
                "reject MLLM UNSUPPORTED_EDIT and require a new proposal"
            ),
        }

    if claimed_by_mllm or mapped is None:
        return {
            "confirmed": True,
            "canonical_operation": mapped,
            "reason": (
                "operation is not in the controller registry"
                if mapped is None
                else f"operation {mapped!r} is not supported"
            ),
        }
    return {
        "confirmed": True,
        "canonical_operation": mapped,
        "reason": f"operation {mapped!r} is not supported",
    }
