"""Harness states, actions, budget rules, and oscillation detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.pipeline.edit_harness.execute import sha256_text
from harness.pipeline.edit_harness.spec import HarnessBudget

STATES = (
    "READY",
    "INTENT_READY",
    "EDIT_PROPOSED",
    "INVALID_RESPONSE",
    "CODE_ERROR",
    "BUILD_ERROR",
    "GEOMETRY_INVALID",
    "PRESERVATION_VIOLATION",
    "VIEW_UNCERTAIN",
    "INTENT_MISMATCH",
    "VISUAL_MISMATCH",
    "OSCILLATION",
    "SUCCESS",
    "UNSUPPORTED_EDIT",
    "NEEDS_REVIEW",
    "FAILED",
)

# Why we rolled back, and which snapshot to restore.
# Snapshots live in 08_harness/geom_repo/ (STEP commit DAG, not git).
ROLLBACK_POLICY = {
    "GEOMETRY_INVALID": "last_valid",
    "PRESERVATION_VIOLATION": "best_valid_or_base",
    "CODE_ERROR": "last_valid",
    "BUILD_ERROR": "last_valid",
    "OSCILLATION": "best_valid_or_base",
    "BUDGET": "best_valid_or_base",
}

TERMINAL = {"SUCCESS", "UNSUPPORTED_EDIT", "NEEDS_REVIEW", "FAILED", "OSCILLATION"}

ACTIONS = (
    "PROPOSE_INTENT",
    "PROPOSE_CODE",
    "EXECUTE_CANDIDATE",
    "REPAIR_RESPONSE",
    "REPAIR_CODE",
    "SEARCH_VIEW",
    "REFINE_EDIT",
    "ROLLBACK",
    "FINISH",
    "REQUEST_REVIEW",
)


def decide_next(
    state: str,
    *,
    budget: HarnessBudget,
    oscillating: bool,
    camera_known: bool,
    geometry_fault: str | None = None,
) -> str:
    """Deterministic controller. The MLLM never chooses FINISH.

    ``geometry_fault`` is ``invalid_solid`` (code) or ``volume_anomaly`` (edit).
    After ROLLBACK the loop resumes with REPAIR_CODE or REFINE_EDIT — never
    by rewriting GEOMETRY_INVALID into INTENT_MISMATCH.
    """
    if oscillating and state not in TERMINAL:
        return "REQUEST_REVIEW"
    rem = budget.remaining()
    if state in TERMINAL:
        return "FINISH" if state == "SUCCESS" else "REQUEST_REVIEW"
    if rem["mllm_calls"] <= 0 and state not in {"EDIT_PROPOSED"}:
        return "REQUEST_REVIEW"
    if rem["edit_rounds"] <= 0 and state in {
        "READY",
        "INTENT_READY",
        "VISUAL_MISMATCH",
        "GEOMETRY_INVALID",
        "PRESERVATION_VIOLATION",
    }:
        return "REQUEST_REVIEW"

    if state == "READY":
        return "PROPOSE_INTENT"
    if state == "INTENT_READY":
        return "PROPOSE_CODE"
    if state == "INVALID_RESPONSE":
        return "REPAIR_RESPONSE" if rem["mllm_calls"] > 0 else "REQUEST_REVIEW"
    if state == "EDIT_PROPOSED":
        return "EXECUTE_CANDIDATE"
    if state in {"CODE_ERROR", "BUILD_ERROR"}:
        if rem["code_repairs"] > 0 and rem["mllm_calls"] > 0:
            return "REPAIR_CODE"
        return "ROLLBACK"
    if state == "GEOMETRY_INVALID":
        return "ROLLBACK"
    if state == "PRESERVATION_VIOLATION":
        return "ROLLBACK"
    if state == "VIEW_UNCERTAIN":
        return "SEARCH_VIEW" if (not camera_known and rem["view_searches"] > 0) else "REQUEST_REVIEW"
    if state == "INTENT_MISMATCH":
        if rem["code_repairs"] > 0 and rem["mllm_calls"] > 0:
            return "REPAIR_CODE"
        if rem["edit_rounds"] > 0 and rem["mllm_calls"] > 0:
            return "PROPOSE_CODE"
        return "REQUEST_REVIEW"
    if state == "VISUAL_MISMATCH":
        if rem["mllm_calls"] > 0:
            return "PROPOSE_INTENT"
        return "REQUEST_REVIEW"
    return "REQUEST_REVIEW"


def resume_after_geometry_invalid(geometry_fault: str | None) -> str:
    """Pick the post-rollback action from the diagnosis, not from a fake state."""
    if geometry_fault == "invalid_solid":
        return "REPAIR_CODE"
    return "REFINE_EDIT"


@dataclass
class OscillationGuard:
    code_hashes: list[str] = field(default_factory=list)
    spec_hashes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    visual_scores: list[float] = field(default_factory=list)

    def observe(
        self,
        *,
        code: str | None = None,
        spec: dict[str, Any] | None = None,
        error: str | None = None,
        visual_score: float | None = None,
    ) -> None:
        if code:
            self.code_hashes.append(sha256_text(code))
        if spec is not None:
            self.spec_hashes.append(sha256_text(repr(sorted(spec.items())) if spec else ""))
        if error:
            self.errors.append(error.strip()[:240])
        if visual_score is not None:
            self.visual_scores.append(float(visual_score))

    def reason(self) -> str | None:
        if len(self.code_hashes) >= 2 and self.code_hashes[-1] == self.code_hashes[-2]:
            return "same code generated twice in a row"
        if len(self.errors) >= 3 and len(set(self.errors[-3:])) == 1:
            return "same error repeated three times"
        if len(self.spec_hashes) >= 4:
            a, b, c, d = self.spec_hashes[-4:]
            if a == c and b == d and a != b:
                return "two EditSpecs alternating"
        if len(self.visual_scores) >= 3:
            recent = self.visual_scores[-3:]
            if recent[-1] <= recent[0] + 0.01 and recent[-1] <= recent[-2] + 0.005:
                return "visual score not improving"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_hashes": list(self.code_hashes),
            "spec_hashes": list(self.spec_hashes),
            "errors": list(self.errors),
            "visual_scores": list(self.visual_scores),
            "reason": self.reason(),
        }
