"""Classify candidate effects and build concrete retry feedback."""

from __future__ import annotations

from typing import Any


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_edit_effect(
    comparison: dict[str, Any] | None,
    visual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comparison = comparison or {}
    delta = comparison.get("delta") or {}
    base = comparison.get("base") or {}
    volume_delta = _number(delta.get("volume_mm3"))
    diag_delta = _number(delta.get("diag_mm"))
    face_delta = _number(delta.get("face_count"))
    solid_delta = _number(delta.get("solid_count"))
    center_delta = _number(delta.get("center_distance_mm"))
    bbox_delta = _number(delta.get("bbox_max_abs_mm"))
    base_volume = abs(_number(base.get("volume_mm3")) or 0.0)
    base_diag = abs(_number(base.get("diag_mm")) or 0.0)
    volume_tol = max(base_volume * 1e-7, 1e-5)
    diag_tol = max(base_diag * 1e-7, 1e-6)
    geometry_unchanged = (
        volume_delta is not None
        and diag_delta is not None
        and abs(volume_delta) <= volume_tol
        and abs(diag_delta) <= diag_tol
        and abs(face_delta or 0.0) < 0.5
        and abs(solid_delta or 0.0) < 0.5
        and (center_delta is None or center_delta <= diag_tol)
        and (bbox_delta is None or bbox_delta <= diag_tol)
    )
    visual_noop = bool(
        (((visual or {}).get("layers") or {}).get("unchanged_region") or {}).get("noop")
    )
    # Full-frame renders can miss a real small/local edit. Visual similarity
    # alone must not overrule measurable B-rep deltas.
    no_effect = geometry_unchanged and (
        visual_noop
        or visual is None
        or bool((visual or {}).get("skipped_target_match"))
    )
    intent_ok = bool(comparison.get("intent_ok"))
    preservation_ok = bool(comparison.get("preservation_ok"))
    visual_ok = None if visual is None else bool(visual.get("passed"))
    if no_effect:
        effect_class = "no_effect"
        reason = "candidate is geometrically/visually indistinguishable from the input"
    elif not preservation_ok:
        effect_class = "wrong_effect"
        reason = "candidate changed geometry outside preservation constraints"
    elif not intent_ok:
        effect_class = "wrong_effect"
        reason = "candidate changed geometry but deterministic intent checks failed"
    elif visual is not None and not visual_ok:
        score = float(visual.get("mean_target_iou") or 0.0)
        effect_class = "partial_effect" if score > 0.0 else "wrong_effect"
        reason = "candidate changed geometry but did not match the requested edit region"
    elif visual_ok:
        effect_class = "targeted_effect"
        reason = "geometry, intent, preservation and visual gates passed"
    else:
        effect_class = "changed_unverified"
        reason = "candidate changed geometry; visual verification has not run"
    return {
        "effect_class": effect_class,
        "reason": reason,
        "no_effect": no_effect,
        "geometry_unchanged": geometry_unchanged,
        "visual_noop": visual_noop,
        "intent_ok": intent_ok,
        "preservation_ok": preservation_ok,
        "visual_ok": visual_ok,
        "deltas": {
            "volume_mm3": volume_delta,
            "diag_mm": diag_delta,
            "face_count": face_delta,
            "solid_count": solid_delta,
            "center_distance_mm": center_delta,
            "bbox_max_abs_mm": bbox_delta,
        },
    }


def retry_feedback(
    effect: dict[str, Any],
    comparison: dict[str, Any] | None,
    visual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comparison = comparison or {}
    visual = visual or {}
    intent_checks = ((comparison.get("intent_constraints") or {}).get("checks") or {})
    preservation_checks = ((comparison.get("preservation_constraints") or {}).get("checks") or {})
    failed_intent = {
        key: value
        for key, value in intent_checks.items()
        if isinstance(value, dict) and not value.get("passed", False)
    }
    failed_preservation = {
        key: value
        for key, value in preservation_checks.items()
        if isinstance(value, dict) and not value.get("passed", False)
    }
    layers = visual.get("layers") or {}
    failed_visual = {
        key: value
        for key, value in layers.items()
        if isinstance(value, dict) and value.get("passed") is False
    }
    cls = str(effect.get("effect_class") or "unknown")
    if cls == "no_effect":
        directive = (
            "The previous program produced no measurable edit. Do not return the same code or "
            "the same selector. Re-check candidate_refs/target relationship, then make a real "
            "geometry change that satisfies the frozen intent."
        )
    elif cls == "wrong_effect":
        directive = (
            "The previous program changed the wrong geometry or violated preservation. Do not "
            "build on that candidate; restart from base unless a verified commit is explicitly useful. "
            "Re-check the target candidate and operation before rewriting the code."
        )
    elif cls == "partial_effect":
        directive = (
            "The previous program made a relevant but incomplete edit. Preserve the verified parts, "
            "then correct the failed region/measurements; do not repeat the unchanged script."
        )
    else:
        directive = "Use the failed checks below to revise the candidate; do not repeat identical code."
    return {
        "effect_class": cls,
        "reason": effect.get("reason"),
        "deltas": effect.get("deltas"),
        "failed_intent_checks": failed_intent,
        "failed_preservation_checks": failed_preservation,
        "failed_visual_layers": failed_visual,
        "directive": directive,
    }
