"""Parse and validate MLLM EditSpec + CadQuery responses.

``tool_trace`` is a strategy plan only (registry nouns). CadQuery is the landing."""

from __future__ import annotations

import json
import re
from typing import Any

from harness.pipeline.edit_harness.execute import lint_cadquery_code
from harness.pipeline.edit_harness.operations import SUPPORTED_OPERATIONS, canonicalize_operation
from harness.pipeline.edit_harness.spec import EditIntent, EditSpec
from harness.pipeline.schema import ALLOWED_EDIT_TYPES

ALLOWED_ACTIONS = {
    "APPLY_EDIT",
    "REVISE_INTENT",
    "REQUEST_VIEWS",
    "REQUEST_REVIEW",
    "UNSUPPORTED_EDIT",
}
ALLOWED_INTENT_ACTIONS = {"PLAN_INTENT", "REQUEST_VIEWS", "REQUEST_REVIEW", "UNSUPPORTED_EDIT"}
ALLOWED_FALLBACKS = {"REQUEST_REVIEW", "UNSUPPORTED_EDIT", "REFINE_EDIT"}
MAX_STRATEGY = 12


def extract_python_code(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_mllm_edit_response(text: str, *, require_tool_trace: bool = False) -> dict[str, Any]:
    """Validate structured MLLM output. Illegal responses are not executed.

    ``require_tool_trace`` is ignored: landing is always CadQuery.
    """
    del require_tool_trace
    data = extract_json_object(text)
    if data is None:
        # Bare python fence with no JSON is still illegal for the harness.
        return {
            "ok": False,
            "errors": ["JSON cannot be parsed"],
            "failure_type": "INVALID_RESPONSE",
            "action": None,
            "edit_spec": None,
            "cadquery_code": None,
            "tool_trace": None,
            "from_commit": None,
            "raw": text,
        }

    errors: list[str] = []
    action = str(data.get("action") or "APPLY_EDIT").strip().upper()
    if action not in ALLOWED_ACTIONS:
        errors.append(f"action not allowed: {action}")

    fallback = data.get("fallback_action")
    if fallback is not None and str(fallback).strip().upper() not in ALLOWED_FALLBACKS:
        errors.append(f"fallback_action not allowed: {fallback}")

    edit_raw = data.get("edit_spec") or data.get("edit")
    edit_spec = None
    if action == "APPLY_EDIT" and isinstance(edit_raw, dict):
        # Optional: the loop overwrites spec from the frozen intent.
        edit_spec = EditSpec.from_dict(edit_raw)
        if edit_spec.edit_type and edit_spec.edit_type not in ALLOWED_EDIT_TYPES:
            errors.append(f"edit_type must be one of {sorted(ALLOWED_EDIT_TYPES)}")
        edit_spec.canonical_operation = canonicalize_operation(edit_spec.operation)

    code = data.get("cadquery_code") or data.get("code") or ""
    if isinstance(code, str) and "```" in code:
        code = extract_python_code(code)
    elif isinstance(code, str):
        code = code.strip() + ("\n" if code.strip() else "")
    else:
        code = ""

    tool_trace, trace_errors = _parse_strategy_trace(data.get("tool_trace"))
    from_commit = data.get("from_commit")
    if from_commit is not None:
        from_commit = str(from_commit).strip() or "base"
    else:
        from_commit = "base"

    if action == "APPLY_EDIT":
        has_code = bool(code.strip())
        if trace_errors and has_code:
            # Primitive JSON (box/cut/…) is not executed; CadQuery is the landing.
            tool_trace = None
            trace_errors = []
        errors.extend(trace_errors)
        if not has_code:
            errors.append(
                "cadquery code is empty; write cadquery_code. "
                "tool_trace is only a strategy noun list (cut_slot/fillet/…), not geometry ops"
            )
        else:
            lint = lint_cadquery_code(code)
            if not lint["ok"]:
                errors.append(str(lint["error"]))

    intent_patch = data.get("intent_patch")
    revision_reason = data.get("revision_reason")
    revision_evidence = data.get("revision_evidence")
    allowed_patch_fields = {
        "what",
        "where",
        "how",
        "why",
        "constraints",
        "target_description",
        "change_type",
        "target_relationship",
        "spatial_constraints",
        "geometric_constraints",
        "preserve_constraints",
        "ambiguities",
        "candidate_refs",
    }
    if action == "REVISE_INTENT":
        if not isinstance(intent_patch, dict) or not intent_patch:
            errors.append("REVISE_INTENT requires a non-empty intent_patch object")
        else:
            unknown = sorted(set(intent_patch) - allowed_patch_fields)
            if unknown:
                errors.append(f"intent_patch fields not allowed: {unknown}")
        if not isinstance(revision_reason, str) or not revision_reason.strip():
            errors.append("REVISE_INTENT requires revision_reason")
        if not isinstance(revision_evidence, list) or not revision_evidence:
            errors.append("REVISE_INTENT requires non-empty revision_evidence")
        elif any(not isinstance(item, str) or not item.strip() for item in revision_evidence):
            errors.append("revision_evidence entries must be non-empty strings")
        if code.strip():
            errors.append(
                "REVISE_INTENT must not include cadquery_code; revise first, then regenerate code"
            )

    requested = data.get("requested_views")
    if requested is not None and not isinstance(requested, list):
        errors.append("requested_views must be a list")
    if isinstance(requested, list) and any(not isinstance(v, str) for v in requested):
        errors.append("requested_views entries must be strings")
    if action == "REQUEST_VIEWS" and not requested:
        errors.append("REQUEST_VIEWS requires at least one requested_views entry")

    ok = not errors
    return {
        "ok": ok,
        "errors": errors,
        "failure_type": None if ok else "INVALID_RESPONSE",
        "action": action if action in ALLOWED_ACTIONS else None,
        "edit_spec": edit_spec,
        "cadquery_code": code if code.strip() else None,
        "tool_trace": tool_trace,
        "from_commit": from_commit,
        "intent_patch": dict(intent_patch) if isinstance(intent_patch, dict) else None,
        "revision_reason": revision_reason.strip() if isinstance(revision_reason, str) else None,
        "revision_evidence": list(revision_evidence) if isinstance(revision_evidence, list) else [],
        "requested_views": list(requested) if isinstance(requested, list) else [],
        "fallback_action": (
            str(fallback).strip().upper() if fallback is not None else "REQUEST_REVIEW"
        ),
        "raw_obj": data,
        "raw": text,
    }


def _parse_strategy_trace(raw: Any) -> tuple[list[dict[str, Any]] | None, list[str]]:
    if raw is None:
        return None, []
    if not isinstance(raw, list):
        return None, ["tool_trace must be a list of strategy steps"]
    if not raw:
        return None, []
    if len(raw) > MAX_STRATEGY:
        return None, [f"tool_trace longer than {MAX_STRATEGY}"]
    steps: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            item = {"op": item}
        if not isinstance(item, dict):
            return None, [f"tool_trace[{i}] is not an object"]
        op = str(item.get("op") or "").strip().lower()
        mapped = canonicalize_operation(op) or op
        if mapped not in SUPPORTED_OPERATIONS:
            return None, [
                f"tool_trace[{i}] {op or '(missing)'} is not a strategy noun; "
                "use cut_slot/fillet/chamfer/add_hole/add_boss/extend_cylinder/modify_feature "
                "and put the implementation in cadquery_code"
            ]
        rec = dict(item)
        rec["op"] = mapped
        steps.append(rec)
    return steps, []


def parse_mllm_intent_response(text: str) -> dict[str, Any]:
    """Validate the planning beat. CadQuery is not accepted here."""
    data = extract_json_object(text)
    empty = {
        "ok": False,
        "errors": ["JSON cannot be parsed"],
        "failure_type": "INVALID_RESPONSE",
        "action": None,
        "intent": None,
        "edit_spec": None,
        "raw": text,
    }
    if data is None:
        return empty

    errors: list[str] = []
    action = str(data.get("action") or "PLAN_INTENT").strip().upper()
    if action == "APPLY_EDIT":
        return {
            "ok": False,
            "errors": ["plan intent first; do not send APPLY_EDIT / cadquery_code on this turn"],
            "failure_type": "INVALID_RESPONSE",
            "action": None,
            "intent": None,
            "edit_spec": None,
            "raw_obj": data,
            "raw": text,
        }
    if action not in ALLOWED_INTENT_ACTIONS:
        errors.append(f"action not allowed on intent turn: {action}")

    requested = data.get("requested_views")
    if requested is not None and not isinstance(requested, list):
        errors.append("requested_views must be a list")
    if isinstance(requested, list) and any(not isinstance(v, str) for v in requested):
        errors.append("requested_views entries must be strings")
    if action == "REQUEST_VIEWS" and not requested:
        errors.append("REQUEST_VIEWS requires at least one requested_views entry")

    intent = None
    edit_spec = None
    if action == "UNSUPPORTED_EDIT":
        if isinstance(data.get("how"), dict) or isinstance(data.get("what"), dict) or isinstance(
            data.get("intent"), dict
        ):
            intent = EditIntent.from_dict(data)
        edit_raw = data.get("edit_spec") or data.get("edit")
        if isinstance(edit_raw, dict):
            edit_spec = EditSpec.from_dict(edit_raw)
    if action == "PLAN_INTENT":
        intent = EditIntent.from_dict(data)
        if not intent.what:
            errors.append("what is missing")
        if not (intent.what.get("feature") or intent.what.get("parameters") or intent.what.get("operation")):
            errors.append("what needs a feature and/or parameters")
        loc = str(intent.where.get("semantic_location") or intent.where.get("feature_id") or "")
        if not loc:
            errors.append("where needs semantic_location or feature_id")
        if not intent.how:
            errors.append("how is missing")
        op = str(intent.how.get("operation") or intent.what.get("operation") or "").strip()
        mapped = canonicalize_operation(op) or op
        if mapped not in SUPPORTED_OPERATIONS:
            errors.append(
                "how.operation must be one of "
                "cut_slot/fillet/chamfer/add_hole/add_boss/extend_cylinder/modify_feature"
            )
        else:
            intent.how["operation"] = mapped
        if not intent.why:
            errors.append("why is missing")
        if data.get("cadquery_code") or data.get("code"):
            errors.append("cadquery_code is not allowed on the intent turn")
        if not errors:
            edit_spec = intent.to_edit_spec()
            edit_spec.canonical_operation = mapped

    ok = not errors
    return {
        "ok": ok,
        "errors": errors,
        "failure_type": None if ok else "INVALID_RESPONSE",
        "action": action if action in ALLOWED_INTENT_ACTIONS else None,
        "intent": intent,
        "edit_spec": edit_spec,
        "requested_views": list(requested) if isinstance(requested, list) else [],
        "raw_obj": data,
        "raw": text,
    }


def edit_spec_to_legacy_json(
    edit_spec: EditSpec,
    instruction: str = "",
    *,
    intent: EditIntent | None = None,
) -> dict[str, Any]:
    """Keep 04_structure/edit.json compatible with the previous schema."""
    target = edit_spec.target.semantic_location or edit_spec.target.feature_id or "body"
    return {
        "version": 1,
        "edit_type": edit_spec.edit_type or "modify",
        "operations": [
            {
                "op": edit_spec.operation,
                "target": target,
                "params": dict(edit_spec.parameters),
            }
        ],
        "notes": instruction[:500] if instruction else "",
        "design_intent": {
            "shape": edit_spec.operation,
            "position": target,
            "purpose": "; ".join(edit_spec.expected_changes[:4]),
        },
        "edit_spec": edit_spec.to_dict(),
        "intent": intent.to_dict() if intent is not None else None,
    }
