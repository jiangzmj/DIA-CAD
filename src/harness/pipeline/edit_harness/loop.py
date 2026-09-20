"""State-driven S2+ edit harness. The MLLM never decides FINISH."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.pipeline.artifacts import (
    REL_EDIT_JSON,
    REL_EDIT_PY,
    REL_INTENT_JSON,
    REL_OUTPUT_STEP,
    REL_OUTPUT_STL,
    REL_RESULT,
    ArtifactStore,
    rel_result_view,
)
from harness.pipeline.edit_harness.compare import (
    compare_before_after,
    compare_images,
    visual_gates,
)
from harness.pipeline.edit_harness.cq_tools import GeomRepo
from harness.pipeline.edit_harness.diagnostics import classify_edit_effect, retry_feedback
from harness.pipeline.edit_harness.execute import (
    copy_valid_step,
    execute_cadquery,
    sha256_file,
    sha256_text,
)
from harness.pipeline.edit_harness.inspect import inspect_step
from harness.pipeline.edit_harness.memory import HarnessConversationMemory
from harness.pipeline.edit_harness.propose import propose_edit, propose_intent
from harness.pipeline.edit_harness.operations import (
    SUPPORTED_OPERATIONS,
    canonicalize_operation,
    confirm_unsupported_edit,
    edit_class,
)
from harness.pipeline.edit_harness.schema import (
    edit_spec_to_legacy_json,
    parse_mllm_edit_response,
    parse_mllm_intent_response,
)
from harness.pipeline.edit_harness.spec import (
    EditIntent,
    EditSpec,
    ExecutionResult,
    HarnessBudget,
    ReferenceView,
    TaskSpec,
    merge_intent_patch,
)
from harness.pipeline.edit_harness.state import (
    OscillationGuard,
    ROLLBACK_POLICY,
    TERMINAL,
    decide_next,
    resume_after_geometry_invalid,
)
from harness.pipeline.edit_harness.validate import compare_geometry, validate_step
from harness.pipeline.edit_harness.views import (
    known_view_spec,
    render_step,
    task_spec_from_manifest,
    unknown_camera_views,
)
from harness.pipeline.manifest import Manifest
from harness.pipeline.skills.cad_views import VIEW_CAMERAS


DIR_HARNESS = "08_harness"


def _progress(msg: str) -> None:
    print(f"[harness] {msg}", flush=True)


def _dump(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def _read_instruction(path: Any) -> str:
    if not path:
        return ""
    p = Path(str(path))
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


@dataclass
class CandidateRecord:
    step_path: str | None = None
    code: str = ""
    edit_spec: dict[str, Any] | None = None
    score: float = -1.0
    iteration: int = -1
    renders: dict[str, str] = field(default_factory=dict)
    geometry: dict[str, Any] | None = None
    visual: dict[str, Any] | None = None
    geometry_ok: bool = False
    preservation_ok: bool = False
    intent_ok: bool = False
    commit_id: str | None = None
    note: str = ""
    effect_class: str = "unknown"
    retry_feedback: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "step_path": self.step_path,
            "iteration": self.iteration,
            "score": self.score,
            "geometry_ok": self.geometry_ok,
            "preservation_ok": self.preservation_ok,
            "intent_ok": self.intent_ok,
            "commit_id": self.commit_id,
            "effect_class": self.effect_class,
        }


def _candidate_score(comparison: dict[str, Any] | None, visual: dict[str, Any] | None) -> float:
    intent = 1.0 if (comparison or {}).get("intent_ok") else 0.0
    pres = 1.0 if (comparison or {}).get("preservation_ok") else 0.0
    vis = float((visual or {}).get("mean_target_iou") or 0.0)
    score = 0.35 * intent + 0.30 * pres + 0.35 * vis
    effect_class = str(((comparison or {}).get("edit_effect") or {}).get("effect_class") or "")
    if effect_class == "no_effect":
        score -= 0.50
    elif effect_class == "wrong_effect":
        score -= 0.25
    return round(score, 4)


@dataclass
class HarnessConfig:
    max_edit_rounds: int = 5
    max_code_repairs: int = 3
    max_mllm_calls: int = 12
    max_view_searches: int = 3
    max_views_per_search: int = 3
    memory_max_messages: int = 8
    memory_max_chars: int = 24000
    memory_summary_chars: int = 6000
    exec_timeout_seconds: float = 120.0
    silhouette_iou_min: float = 0.55
    silhouette_iou_floor: float = 0.40
    cross_renderer_iou_min: float = 0.28
    cross_renderer_iou_floor: float = 0.15
    preservation_bbox_rel_tol: float = 0.50
    preservation_bbox_rel_tol_replace: float = 1.20
    volume_explode_ratio: float = 8.0
    volume_explode_ratio_global: float = 12.0
    wrecked_iou_min: float = 0.25
    view_alignment_min: float = 0.40
    view_size: int = 1024
    region_change_iou_min: float = 0.12
    region_edge_iou_min: float = 0.12
    region_mask_max_frac: float = 0.25
    region_mask_min_frac: float = 0.001
    noop_silhouette_iou_min: float = 0.997
    noop_pixel_mae_max: float = 0.008
    thresholds_are_placeholders: bool = True
    calibrated_on: str = ""

    @classmethod
    def from_settings(cls, settings: Any, ctx: dict[str, Any]) -> HarnessConfig:
        pipe = getattr(settings, "pipeline", None)
        h = getattr(pipe, "harness", None)
        def _g(name: str, default: Any) -> Any:
            if name in ctx:
                return ctx[name]
            if h is not None and hasattr(h, name):
                return getattr(h, name)
            return default

        return cls(
            max_edit_rounds=int(_g("max_edit_rounds", 5)),
            max_code_repairs=int(_g("max_code_repairs", 3)),
            max_mllm_calls=int(_g("max_mllm_calls", 12)),
            max_view_searches=int(_g("max_view_searches", 3)),
            max_views_per_search=int(_g("max_views_per_search", 3)),
            memory_max_messages=int(_g("memory_max_messages", 8)),
            memory_max_chars=int(_g("memory_max_chars", 24000)),
            memory_summary_chars=int(_g("memory_summary_chars", 6000)),
            exec_timeout_seconds=float(_g("exec_timeout_seconds", 120.0)),
            silhouette_iou_min=float(_g("silhouette_iou_min", 0.55)),
            silhouette_iou_floor=float(_g("silhouette_iou_floor", 0.40)),
            cross_renderer_iou_min=float(_g("cross_renderer_iou_min", 0.28)),
            cross_renderer_iou_floor=float(_g("cross_renderer_iou_floor", 0.15)),
            preservation_bbox_rel_tol=float(_g("preservation_bbox_rel_tol", 0.50)),
            preservation_bbox_rel_tol_replace=float(_g("preservation_bbox_rel_tol_replace", 1.20)),
            volume_explode_ratio=float(_g("volume_explode_ratio", 8.0)),
            volume_explode_ratio_global=float(_g("volume_explode_ratio_global", 12.0)),
            wrecked_iou_min=float(_g("wrecked_iou_min", 0.25)),
            view_alignment_min=float(_g("view_alignment_min", 0.40)),
            view_size=int(ctx.get("view_size") or getattr(pipe, "view_size", 1024) or 1024),
            region_change_iou_min=float(_g("region_change_iou_min", 0.12)),
            region_edge_iou_min=float(_g("region_edge_iou_min", 0.12)),
            region_mask_max_frac=float(_g("region_mask_max_frac", 0.25)),
            region_mask_min_frac=float(_g("region_mask_min_frac", 0.001)),
            noop_silhouette_iou_min=float(_g("noop_silhouette_iou_min", 0.997)),
            noop_pixel_mae_max=float(_g("noop_pixel_mae_max", 0.008)),
            thresholds_are_placeholders=bool(_g("thresholds_are_placeholders", True)),
            calibrated_on=str(_g("calibrated_on", "") or ""),
        )


class EditHarness:
    def __init__(
        self,
        store: ArtifactStore,
        manifest: Manifest,
        ctx: dict[str, Any],
    ):
        self.store = store
        self.manifest = manifest
        self.ctx = ctx
        self.settings = ctx.get("settings")
        self.cfg = HarnessConfig.from_settings(self.settings, ctx)
        self.budget = HarnessBudget(
            max_edit_rounds=self.cfg.max_edit_rounds,
            max_code_repairs=self.cfg.max_code_repairs,
            max_mllm_calls=self.cfg.max_mllm_calls,
            max_view_searches=self.cfg.max_view_searches,
            exec_timeout_seconds=self.cfg.exec_timeout_seconds,
        )
        self.guard = OscillationGuard()
        self.transitions: list[dict[str, Any]] = []
        base_step = str(manifest.input_step)
        self.base_step = base_step
        self.last_valid = CandidateRecord(
            step_path=base_step,
            iteration=-1,
            geometry_ok=True,
            preservation_ok=True,
            commit_id="base",
            note="base",
        )
        self.best_valid = CandidateRecord()
        self.current_code = ""
        self.current_spec: EditSpec | None = None
        self.frozen_intent: EditIntent | None = None
        self.pending_kind = "intent"
        self.current_trace: list[dict[str, Any]] | None = None
        self.current_from_commit = "base"
        self.last_trace_result: dict[str, Any] | None = None
        self.geom_repo: GeomRepo | None = None
        self.diagnosis = ""
        self.geometry_fault: str | None = None
        self.rollback_reason: str | None = None
        self.iter_idx = 0
        self.feature_note = ""
        self.root = self.store.path(DIR_HARNESS)
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory = HarnessConversationMemory(
            max_messages=self.cfg.memory_max_messages,
            max_chars=self.cfg.memory_max_chars,
            summary_chars=self.cfg.memory_summary_chars,
        )
        self.supplemental_views: list[ReferenceView] = []
        self.pending_requested_views: list[str] = []
        self._persist_memory()

    def run(self) -> Manifest:
        task = task_spec_from_manifest(self.manifest, size=self.cfg.view_size)
        _dump(self.root / "task.json", task.to_dict())
        unknown = unknown_camera_views(task)
        if unknown:
            # Six pipeline images are known; if a path is missing we still refuse to invent.
            self._transition("READY", "VIEW_UNCERTAIN", "REQUEST_REVIEW", "missing camera on some views")
            return self._finish("NEEDS_REVIEW", "VIEW_UNCERTAIN: some reference images have no camera")

        geometry = inspect_step(task.base_step_path)
        _dump(self.root / "base_geometry.json", geometry)
        self.task = task
        self.base_geometry = geometry
        self.feature_note = self._feature_probe_note()
        self.geom_repo = GeomRepo(self.root / "geom_repo", task.base_step_path)

        state = "READY"
        while state not in TERMINAL:
            oscillating = self.guard.reason() is not None
            if oscillating and state not in TERMINAL:
                self._transition(state, "OSCILLATION", "REQUEST_REVIEW", self.guard.reason() or "")
                state = "OSCILLATION"
                break
            action = decide_next(
                state,
                budget=self.budget,
                oscillating=False,
                camera_known=True,
                geometry_fault=self.geometry_fault,
            )
            _progress(f"state={state} action={action} budget={self.budget.remaining()}")
            if action == "PROPOSE_INTENT":
                state = self._propose_intent(task, geometry, extra="")
            elif action == "PROPOSE_CODE":
                state = self._propose_code(task, geometry, extra="")
            elif action == "REPAIR_RESPONSE":
                if self.pending_kind == "code":
                    extra = (
                        "Previous reply was not valid JSON / schema. Return ONLY APPLY_EDIT + "
                        "cadquery_code, or REVISE_INTENT with a minimal intent_patch, "
                        "revision_reason, and revision_evidence."
                    )
                    state = self._propose_code(task, geometry, extra=extra)
                else:
                    extra = (
                        "Previous reply was not valid JSON / schema. Return ONLY the intent JSON "
                        "(PLAN_INTENT + target_description/change_type/target_relationship + "
                        "what/where/how/why/constraints). No cadquery_code."
                    )
                    if "UNSUPPORTED_EDIT" in (self.diagnosis or ""):
                        extra = (
                            self.diagnosis
                            + " Propose action=PLAN_INTENT with a registry how.operation, or keep UNSUPPORTED_EDIT."
                        )
                    state = self._propose_intent(task, geometry, extra=extra)
            elif action == "REFINE_EDIT":
                state = self._propose_intent(task, geometry, extra="")
            elif action == "EXECUTE_CANDIDATE":
                state = self._execute_and_evaluate(task, geometry)
            elif action == "REPAIR_CODE":
                state = self._propose_code(
                    task,
                    geometry,
                    extra=(
                        "Previous candidate failed. Decide from the evidence whether CadQuery is "
                        "wrong or whether one or more frozen-intent fields are wrong. Return "
                        "APPLY_EDIT to rewrite code while retaining intent, or REVISE_INTENT with "
                        "the smallest evidence-backed intent_patch. On APPLY_EDIT, set from_commit "
                        "to reuse a prior STEP or 'base' to restart.\n"
                        f"Runtime:\n{self.diagnosis}"
                    ),
                    code_repair=True,
                )
            elif action == "ROLLBACK":
                state = self._handle_rollback(state)
            elif action == "SEARCH_VIEW":
                self._transition(state, "VIEW_UNCERTAIN", "REQUEST_REVIEW", "cameras are known; search disabled")
                state = "NEEDS_REVIEW"
            elif action in {"FINISH", "REQUEST_REVIEW"}:
                state = "SUCCESS" if action == "FINISH" else "NEEDS_REVIEW"
                self._transition(state, state, action, "controller halt")
            else:
                state = "NEEDS_REVIEW"
                self._transition("READY", state, "REQUEST_REVIEW", f"unhandled action {action}")

        if state == "OSCILLATION":
            self._apply_rollback("OSCILLATION")
            return self._finish("NEEDS_REVIEW", f"OSCILLATION: {self.guard.reason()}")
        if state == "SUCCESS":
            return self._finish("SUCCESS", "all terminal gates passed")
        if state == "UNSUPPORTED_EDIT":
            return self._finish("UNSUPPORTED_EDIT", self.diagnosis or "unsupported edit")
        if state == "FAILED":
            return self._finish("FAILED", self.diagnosis or "failed")
        return self._finish("NEEDS_REVIEW", self.diagnosis or state)

    def _iter_dir(self) -> Path:
        path = self.root / "iterations" / f"{self.iter_idx:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _feature_probe_note(self) -> str:
        from harness.pipeline.feature_probe import SCHEMA_VERSION

        cached = self.manifest.checks.get("feature_probe")
        if self.ctx.get("skip_feature_probe"):
            return str((cached or {}).get("coder_note") or "")
        cached_schema = None
        if isinstance(cached, dict):
            cached_schema = cached.get("schema_version")
            if cached_schema is None and isinstance(cached.get("inspect"), dict):
                cached_schema = cached["inspect"].get("schema_version")
        if (
            isinstance(cached, dict)
            and cached_schema == SCHEMA_VERSION
            and cached.get("coder_note")
        ):
            return str(cached.get("coder_note"))
        try:
            from harness.agents.feature_probe import FeatureProbeAgent
            from harness.pipeline.feature_probe import compact_probe_for_prompt

            probe_cfg = getattr(getattr(self.settings, "pipeline", None), "feature_probe", None)
            root = getattr(self.settings, "root", Path.cwd())
            cache_name = str(getattr(probe_cfg, "cache_dir", "feature_probe_cache"))
            cache_dir = Path(cache_name)
            if not cache_dir.is_absolute():
                cache_dir = Path(root) / cache_dir
            report = FeatureProbeAgent().run(
                self.manifest.input_step,
                cache_dir=cache_dir,
                palmetto_engine=getattr(probe_cfg, "palmetto_engine_path", "") or None,
                palmetto_enabled=bool(getattr(probe_cfg, "palmetto_enabled", True)),
                palmetto_timeout_s=float(
                    getattr(probe_cfg, "palmetto_timeout_seconds", 120.0)
                ),
            )
            note = str(report.get("coder_note") or "")
            unified = report.get("inspect") if isinstance(report.get("inspect"), dict) else {}
            prompt_probe = compact_probe_for_prompt(unified)
            compact = {
                "ok": report.get("ok"),
                "schema_version": unified.get("schema_version"),
                "model": prompt_probe.get("model"),
                "candidate_count": prompt_probe.get("candidate_count"),
                "sources": prompt_probe.get("sources"),
                "plan": report.get("plan"),
                "coder_note": note,
            }
            self.store.write_text(
                "05_code/feature_probe.json",
                json.dumps(unified or compact, indent=2, ensure_ascii=False),
            )
            self.manifest.checks["feature_probe"] = compact
            return note
        except Exception as exc:  # noqa: BLE001
            return f"FeatureProbe skipped: {exc}"

    def _propose_intent(
        self,
        task: TaskSpec,
        geometry: dict[str, Any],
        *,
        extra: str,
    ) -> str:
        self.pending_kind = "intent"
        media = self.ctx.get("media_provider")
        if media is None:
            self.diagnosis = "no media_provider"
            return "FAILED"
        if self.budget.remaining()["mllm_calls"] <= 0:
            return "NEEDS_REVIEW"
        work = self._iter_dir()
        self.budget.used_mllm_calls += 1
        diagnosis = self.diagnosis
        history_messages = self.memory.api_messages()
        raw, request_content = propose_intent(
            media,
            settings=self.settings,
            task=task,
            geometry=geometry,
            diagnosis=diagnosis,
            feature_note=self.feature_note,
            extra=extra,
            history_messages=history_messages,
            supplemental_views=self.supplemental_views,
        )
        self._record_memory("intent", raw, diagnosis=diagnosis, extra=extra)
        _dump(
            work / "mllm_intent_request.json",
            {
                "extra": extra,
                "diagnosis": diagnosis,
                "history_messages": len(history_messages),
                "supplemental_views": [v.view_id for v in self.supplemental_views],
                "text_preview": extra[:500],
            },
        )
        (work / "mllm_intent_response.txt").write_text(raw or "", encoding="utf-8")
        parsed = parse_mllm_intent_response(raw or "")
        _dump(
            work / "mllm_intent_response.json",
            {
                "ok": parsed["ok"],
                "errors": parsed.get("errors"),
                "action": parsed.get("action"),
                "intent": parsed["intent"].to_dict() if parsed.get("intent") else None,
            },
        )
        del request_content

        action = parsed.get("action")
        if action == "REQUEST_VIEWS" and parsed.get("ok"):
            return self._handle_view_request(
                parsed.get("requested_views") or [], resume_state="READY", phase="intent"
            )
        if action == "UNSUPPORTED_EDIT":
            decision = confirm_unsupported_edit(
                claimed_by_mllm=True,
                operation=(
                    (parsed["intent"].how.get("operation") if parsed.get("intent") else None)
                    or (parsed["edit_spec"].operation if parsed.get("edit_spec") else None)
                ),
                instruction=task.instruction,
                failure_errors=self.guard.errors,
            )
            _dump(work / "unsupported_decision.json", decision)
            if decision["confirmed"]:
                self.diagnosis = f"controller confirmed UNSUPPORTED_EDIT: {decision['reason']}"
                self._transition("READY", "UNSUPPORTED_EDIT", "FINISH", self.diagnosis)
                return "UNSUPPORTED_EDIT"
            self.diagnosis = (
                "MLLM claimed UNSUPPORTED_EDIT but controller rejected it: "
                + str(decision["reason"])
            )
            self.guard.observe(error="unsupported_claim_rejected")
            self._transition("READY", "INVALID_RESPONSE", "REPAIR_RESPONSE", self.diagnosis)
            self.iter_idx += 1
            return "INVALID_RESPONSE"
        if action == "REQUEST_REVIEW":
            self.diagnosis = "model requested review"
            self._transition("READY", "NEEDS_REVIEW", "REQUEST_REVIEW", self.diagnosis)
            return "NEEDS_REVIEW"
        if not parsed["ok"] or parsed.get("intent") is None:
            self.diagnosis = "; ".join(parsed.get("errors") or ["invalid intent"])
            self.guard.observe(error=self.diagnosis)
            self._transition("READY", "INVALID_RESPONSE", "REPAIR_RESPONSE", self.diagnosis)
            self.iter_idx += 1
            return "INVALID_RESPONSE"

        intent: EditIntent = parsed["intent"]
        spec = parsed.get("edit_spec") or intent.to_edit_spec()
        spec.canonical_operation = canonicalize_operation(spec.operation) or spec.canonical_operation
        self.frozen_intent = intent
        self.current_spec = spec
        self.guard.observe(spec=intent.to_dict())
        _dump(work / "intent.json", intent.to_dict())
        _dump(work / "edit_spec.json", spec.to_dict())
        self.store.write_text(
            REL_INTENT_JSON, json.dumps(intent.to_dict(), indent=2, ensure_ascii=False) + "\n"
        )
        self.manifest.artifacts["intent_json"] = str(self.store.path(REL_INTENT_JSON).resolve())
        self._transition("READY", "INTENT_READY", "PROPOSE_CODE", spec.operation)
        return "INTENT_READY"

    def _propose_code(
        self,
        task: TaskSpec,
        geometry: dict[str, Any],
        *,
        extra: str,
        code_repair: bool = False,
    ) -> str:
        self.pending_kind = "code"
        media = self.ctx.get("media_provider")
        if media is None:
            self.diagnosis = "no media_provider"
            return "FAILED"
        if self.frozen_intent is None:
            self.diagnosis = "no frozen intent; plan first"
            self._transition("INTENT_READY", "READY", "PROPOSE_INTENT", self.diagnosis)
            return "READY"
        if self.budget.remaining()["mllm_calls"] <= 0:
            return "NEEDS_REVIEW"
        work = self._iter_dir()
        self.budget.used_mllm_calls += 1
        diagnosis = self.diagnosis
        history_messages = self.memory.api_messages()
        raw, request_content = propose_edit(
            media,
            settings=self.settings,
            task=task,
            geometry=geometry,
            diagnosis=diagnosis,
            feature_note=self.feature_note,
            extra=extra,
            frozen_intent=self.frozen_intent,
            history_messages=history_messages,
            supplemental_views=self.supplemental_views,
        )
        self._record_memory("code_repair" if code_repair else "code", raw, diagnosis=diagnosis, extra=extra)
        _dump(
            work / "mllm_request.json",
            {
                "extra": extra,
                "diagnosis": diagnosis,
                "history_messages": len(history_messages),
                "image_roles": [
                    {"view_id": v.view_id, "role": v.role, "camera_known": bool(v.camera and v.camera.camera_known)}
                    for v in (*task.base_views, *task.target_views, *self.supplemental_views)
                ],
                "text_preview": extra[:500],
            },
        )
        (work / "mllm_response.txt").write_text(raw or "", encoding="utf-8")
        parsed = parse_mllm_edit_response(raw or "")
        _dump(
            work / "mllm_response.json",
            {
                "ok": parsed["ok"],
                "errors": parsed.get("errors"),
                "action": parsed.get("action"),
                "cadquery_code": parsed.get("cadquery_code"),
                "from_commit": parsed.get("from_commit"),
                "intent_patch": parsed.get("intent_patch"),
                "revision_reason": parsed.get("revision_reason"),
                "revision_evidence": parsed.get("revision_evidence"),
            },
        )
        del request_content

        action = parsed.get("action")
        if action == "REQUEST_VIEWS" and parsed.get("ok"):
            return self._handle_view_request(
                parsed.get("requested_views") or [], resume_state="INTENT_READY", phase="code"
            )
        if action == "UNSUPPORTED_EDIT":
            self.diagnosis = "UNSUPPORTED_EDIT is only allowed on the intent turn"
            self.guard.observe(error=self.diagnosis)
            self._transition("INTENT_READY", "INVALID_RESPONSE", "REPAIR_RESPONSE", self.diagnosis)
            self.iter_idx += 1
            return "INVALID_RESPONSE"
        if action == "REQUEST_REVIEW" and not parsed.get("cadquery_code"):
            self.diagnosis = "model requested review"
            self._transition("INTENT_READY", "NEEDS_REVIEW", "REQUEST_REVIEW", self.diagnosis)
            return "NEEDS_REVIEW"
        if not parsed["ok"]:
            self.diagnosis = "; ".join(parsed.get("errors") or ["invalid response"])
            self.guard.observe(error=self.diagnosis)
            self._transition("INTENT_READY", "INVALID_RESPONSE", "REPAIR_RESPONSE", self.diagnosis)
            self.iter_idx += 1
            return "INVALID_RESPONSE"

        if action == "REVISE_INTENT":
            old_intent = self.frozen_intent
            patched = merge_intent_patch(old_intent, parsed.get("intent_patch") or {})
            validated = parse_mllm_intent_response(
                json.dumps(
                    {"action": "PLAN_INTENT", **patched.to_dict()}, ensure_ascii=False
                )
            )
            if not validated.get("ok") or validated.get("intent") is None:
                self.diagnosis = "invalid revised intent: " + "; ".join(
                    validated.get("errors") or ["unknown validation error"]
                )
                self.guard.observe(error=self.diagnosis)
                self._transition(
                    "INTENT_READY", "INVALID_RESPONSE", "REPAIR_RESPONSE", self.diagnosis
                )
                self.iter_idx += 1
                return "INVALID_RESPONSE"
            revised: EditIntent = validated["intent"]
            if revised.to_dict() == old_intent.to_dict():
                self.diagnosis = "REVISE_INTENT patch made no effective change"
                self.guard.observe(error=self.diagnosis)
                self._transition(
                    "INTENT_READY", "INVALID_RESPONSE", "REPAIR_RESPONSE", self.diagnosis
                )
                self.iter_idx += 1
                return "INVALID_RESPONSE"
            self.frozen_intent = revised
            spec = validated.get("edit_spec") or revised.to_edit_spec()
            spec.canonical_operation = canonicalize_operation(spec.operation) or spec.canonical_operation
            self.current_spec = spec
            self.guard.observe(spec=revised.to_dict())
            _dump(
                work / "intent_revision.json",
                {
                    "intent_patch": parsed.get("intent_patch"),
                    "revision_reason": parsed.get("revision_reason"),
                    "revision_evidence": parsed.get("revision_evidence"),
                    "before": old_intent.to_dict(),
                    "after": revised.to_dict(),
                },
            )
            _dump(work / "intent.json", revised.to_dict())
            _dump(work / "edit_spec.json", spec.to_dict())
            self.store.write_text(
                REL_INTENT_JSON,
                json.dumps(revised.to_dict(), indent=2, ensure_ascii=False) + "\n",
            )
            self.diagnosis = (
                "MLLM revised selected intent fields: "
                + str(parsed.get("revision_reason") or "evidence-backed revision")
            )
            self._transition(
                "INTENT_READY", "INTENT_READY", "PROPOSE_CODE", self.diagnosis[:200]
            )
            self.iter_idx += 1
            return "INTENT_READY"

        if code_repair:
            self.budget.used_code_repairs += 1
        else:
            self.budget.used_edit_rounds += 1

        spec = self.frozen_intent.to_edit_spec()
        spec.canonical_operation = canonicalize_operation(spec.operation) or spec.canonical_operation
        self.current_spec = spec
        self.current_trace = parsed.get("tool_trace")
        self.current_from_commit = str(parsed.get("from_commit") or "base")
        self.pending_requested_views = self._normalize_requested_views(
            parsed.get("requested_views") or []
        )
        code = str(parsed.get("cadquery_code") or "")
        self.current_code = code
        (work / "generated_cadquery.py").write_text(code, encoding="utf-8")
        self.guard.observe(code=code)
        _dump(work / "intent.json", self.frozen_intent.to_dict())
        _dump(work / "edit_spec.json", spec.to_dict())
        self._transition("INTENT_READY", "EDIT_PROPOSED", "EXECUTE_CANDIDATE", spec.operation)
        return "EDIT_PROPOSED"

    def _execute_and_evaluate(self, task: TaskSpec, base_geometry: dict[str, Any]) -> str:
        work = self._iter_dir()
        exec_result = self._run_candidate(task, work)
        _dump(work / "execution.json", exec_result.to_dict())
        (work / "stdout.txt").write_text(exec_result.stdout or "", encoding="utf-8")
        (work / "stderr.txt").write_text(exec_result.stderr or "", encoding="utf-8")
        _dump(
            work / "hashes.json",
            {
                "code": sha256_text(self.current_code),
                "base_step": sha256_file(task.base_step_path),
                "candidate_step": sha256_file(exec_result.output_step_path),
                "last_valid_step": sha256_file(self.last_valid.step_path),
                "best_valid_step": sha256_file(self.best_valid.step_path),
                "geom_repo_head": self.geom_repo.head if self.geom_repo else None,
            },
        )
        if not exec_result.success:
            self._set_diagnosis("CODE_ERROR" if exec_result.failure_type == "CODE_ERROR" else "BUILD_ERROR", exec_result.stderr or exec_result.failure_type or "exec failed")
            self.guard.observe(error=self.diagnosis)
            nxt = "CODE_ERROR" if exec_result.failure_type == "CODE_ERROR" else "BUILD_ERROR"
            self._transition("EDIT_PROPOSED", nxt, "REPAIR_CODE", self.diagnosis[:200])
            self.iter_idx += 1
            return nxt

        geom = validate_step(
            exec_result.output_step_path or "",
            base_report=base_geometry,
            volume_explode_ratio=self.cfg.volume_explode_ratio,
            volume_explode_ratio_global=self.cfg.volume_explode_ratio_global,
            edit_class_name=self._edit_class(),
        )
        _dump(work / "geometry_metrics.json", geom)
        if not geom["ok"]:
            self._set_diagnosis("GEOMETRY_INVALID", str(geom.get("error") or "geometry invalid"))
            self.geometry_fault = str(geom.get("fault_kind") or "invalid_solid")
            self.rollback_reason = "GEOMETRY_INVALID"
            self.guard.observe(error=self.diagnosis)
            self._transition("EDIT_PROPOSED", "GEOMETRY_INVALID", "ROLLBACK", self.diagnosis)
            self.iter_idx += 1
            return "GEOMETRY_INVALID"

        frozen_parameters = None
        if self.frozen_intent is not None:
            frozen_parameters = dict(self.frozen_intent.what.get("parameters") or {})
        comparison = compare_geometry(
            task.base_step_path,
            exec_result.output_step_path or "",
            self.current_spec,
            bbox_rel_tol=self.cfg.preservation_bbox_rel_tol,
            bbox_rel_tol_replace=self.cfg.preservation_bbox_rel_tol_replace,
            edit_class_name=self._edit_class(),
            frozen_parameters=frozen_parameters,
        )
        effect = classify_edit_effect(comparison)
        feedback = retry_feedback(effect, comparison)
        comparison["edit_effect"] = effect
        comparison["retry_feedback"] = feedback
        _dump(work / "geometry_comparison.json", comparison)
        _dump(work / "edit_effect.json", {"effect": effect, "retry_feedback": feedback})
        self._update_last_valid(exec_result, work, comparison, None)
        if not comparison["preservation_ok"]:
            self._set_diagnosis(
                "PRESERVATION_VIOLATION",
                "preservation constraints failed\nstructured_retry_feedback: "
                + json.dumps(feedback, ensure_ascii=False),
            )
            self.rollback_reason = "PRESERVATION_VIOLATION"
            self.guard.observe(error=self.diagnosis)
            self._transition("EDIT_PROPOSED", "PRESERVATION_VIOLATION", "ROLLBACK", self.diagnosis)
            self.iter_idx += 1
            return "PRESERVATION_VIOLATION"
        if not comparison["intent_ok"]:
            self._set_diagnosis(
                "INTENT_MISMATCH",
                "intent constraints failed\nstructured_retry_feedback: "
                + json.dumps(feedback, ensure_ascii=False),
            )
            self.guard.observe(error=self.diagnosis)
            self._update_best_valid(exec_result, work, comparison, None)
            self._transition("EDIT_PROPOSED", "INTENT_MISMATCH", "REPAIR_CODE", self.diagnosis)
            self.iter_idx += 1
            return "INTENT_MISMATCH"

        visual = self._render_and_compare(
            task, exec_result.output_step_path or "", work, edit_class_name=self._edit_class()
        )
        self._capture_candidate_feedback(
            task,
            exec_result.output_step_path or "",
            work,
            visual,
            self.pending_requested_views,
        )
        _dump(work / "visual_metrics.json", visual)
        intent_checks = (comparison.get("intent_constraints") or {}).get("checks") or {}
        deterministic_effect = any(
            isinstance(check, dict)
            and check.get("passed") is True
            and check.get("deterministic") is True
            for check in intent_checks.values()
        )
        visual_layers = visual.get("layers") or {}
        unchanged_layer = visual_layers.get("unchanged_region") or {}
        edit_region_layer = visual_layers.get("edit_region_vs_target") or {}
        change_iou = float(edit_region_layer.get("mean_change_iou") or 0.0)
        edge_iou = float(edit_region_layer.get("mean_region_edge_iou") or 0.0)
        change_threshold = float(edit_region_layer.get("threshold") or 1.0)
        edge_threshold = float(edit_region_layer.get("region_edge_iou_min") or 1.0)
        visual_region_supports_edit = (
            edit_region_layer.get("mode") == "unreliable_fallback"
            or (
                change_iou >= 0.8 * change_threshold
                and edge_iou >= edge_threshold
            )
        )
        if (
            deterministic_effect
            and comparison.get("intent_ok")
            and comparison.get("preservation_ok")
            and unchanged_layer.get("noop")
            and visual_region_supports_edit
        ):
            unchanged_layer["full_frame_noop_suppressed"] = True
            unchanged_layer["noop"] = False
            edit_region_layer["geometry_override"] = (
                "deterministic local B-rep intent check passed; full-frame no-op contradicted "
                "by local region evidence"
            )
            visual["passed"] = True
            visual["failure_type"] = None
            visual["reason"] = (
                "geometry-verified local edit; unreliable full-frame no-op signal ignored"
            )
        effect = classify_edit_effect(comparison, visual)
        feedback = retry_feedback(effect, comparison, visual)
        comparison["edit_effect"] = effect
        comparison["retry_feedback"] = feedback
        visual["edit_effect"] = effect
        visual["retry_feedback"] = feedback
        if effect.get("no_effect") and visual.get("passed"):
            visual["passed"] = False
            visual["failure_type"] = "VISUAL_MISMATCH"
            visual["reason"] = "NO_EFFECT: candidate did not measurably change the input"
        _dump(work / "geometry_comparison.json", comparison)
        _dump(work / "visual_metrics.json", visual)
        _dump(work / "edit_effect.json", {"effect": effect, "retry_feedback": feedback})
        self.guard.observe(visual_score=float(visual.get("mean_target_iou") or 0.0))
        self._update_best_valid(exec_result, work, comparison, visual)

        if effect.get("no_effect"):
            self._set_diagnosis(
                "NO_EFFECT",
                "candidate did not measurably change the input\nstructured_retry_feedback: "
                + json.dumps(feedback, ensure_ascii=False),
            )
            self.guard.observe(error="NO_EFFECT")
            self._transition(
                "EDIT_PROPOSED", "INTENT_MISMATCH", "REPAIR_CODE", self.diagnosis[:200]
            )
            self.iter_idx += 1
            return "INTENT_MISMATCH"

        if visual.get("failure_type") == "VIEW_UNCERTAIN":
            self._set_diagnosis("VIEW_UNCERTAIN", "view alignment below placeholder threshold; not treating as geometry error")
            self._transition("EDIT_PROPOSED", "VIEW_UNCERTAIN", "SEARCH_VIEW", self.diagnosis)
            self.iter_idx += 1
            return "VIEW_UNCERTAIN"
        if visual.get("failure_type") == "PRESERVATION_VIOLATION":
            self._set_diagnosis(
                "PRESERVATION_VIOLATION",
                "visual wreckage: unmodified region destroyed\nstructured_retry_feedback: "
                + json.dumps(feedback, ensure_ascii=False),
            )
            self.rollback_reason = "PRESERVATION_VIOLATION"
            self._transition("EDIT_PROPOSED", "PRESERVATION_VIOLATION", "ROLLBACK", self.diagnosis[:200])
            self.iter_idx += 1
            return "PRESERVATION_VIOLATION"
        if not visual.get("passed"):
            self._set_diagnosis(
                "VISUAL_MISMATCH",
                "visual mismatch: "
                + json.dumps(visual.get("layers") or {}, ensure_ascii=False)[:800]
                + "\nstructured_retry_feedback: "
                + json.dumps(feedback, ensure_ascii=False),
            )
            self._transition("EDIT_PROPOSED", "VISUAL_MISMATCH", "PROPOSE_INTENT", self.diagnosis[:200])
            self.iter_idx += 1
            return "VISUAL_MISMATCH"

        # All gates passed — harness, not the model, declares SUCCESS.
        self._promote_current(exec_result, work, comparison, visual)
        self._transition("EDIT_PROPOSED", "SUCCESS", "FINISH", "geometry+intent+preservation+visual passed")
        self.iter_idx += 1
        return "SUCCESS"

    def _run_candidate(self, task: TaskSpec, work: Path) -> ExecutionResult:
        if self.geom_repo is None:
            self.geom_repo = GeomRepo(self.root / "geom_repo", task.base_step_path)
        parent = self.current_from_commit or "base"
        try:
            input_step = self.geom_repo.get_step(parent)
        except KeyError:
            parent = "base"
            input_step = self.geom_repo.get_step("base")
        exec_result = execute_cadquery(
            self.current_code,
            input_step,
            work,
            timeout=self.cfg.exec_timeout_seconds,
        )
        if exec_result.success and exec_result.output_step_path:
            cid = self.geom_repo.commit(
                parent=parent,
                op="cadquery",
                args={"strategy": self.current_trace or [], "from_commit": parent},
                step_path=exec_result.output_step_path,
            )
            self.last_trace_result = {
                "ok": True,
                "commit": cid,
                "steps": [{"op": "cadquery", "ok": True, "commit": cid}],
            }
        else:
            self.last_trace_result = {
                "ok": False,
                "commit": parent,
                "error": exec_result.stderr or exec_result.failure_type,
                "steps": [],
            }
        return exec_result

    def _record_memory(
        self,
        phase: str,
        assistant: str,
        *,
        diagnosis: str,
        extra: str,
    ) -> None:
        frozen = self.frozen_intent.to_dict() if self.frozen_intent is not None else None
        summary = {
            "phase": phase,
            "iteration": self.iter_idx,
            "diagnosis": diagnosis,
            "controller_instruction": extra,
            "frozen_intent": frozen,
            "geom_head": self.geom_repo.head if self.geom_repo else None,
            "attached_supplemental_views": [v.view_id for v in self.supplemental_views],
        }
        self.memory.record(
            user_summary=json.dumps(summary, ensure_ascii=False, default=str),
            assistant=assistant,
        )
        self._persist_memory()

    def _persist_memory(self) -> None:
        _dump(self.root / "conversation_memory.json", self.memory.to_dict())

    def _normalize_requested_views(self, requested: list[Any]) -> list[str]:
        aliases = {"rear": "back", "forward": "front", "upper": "top", "lower": "bottom"}
        out: list[str] = []
        for raw in requested:
            name = str(raw).strip().lower()
            for prefix in ("target_", "base_", "candidate_", "result_"):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            name = aliases.get(name, name)
            if name in VIEW_CAMERAS and name not in out:
                out.append(name)
        return out[: max(1, self.cfg.max_views_per_search)]

    def _handle_view_request(
        self,
        requested: list[Any],
        *,
        resume_state: str,
        phase: str,
    ) -> str:
        if self.budget.remaining()["view_searches"] <= 0:
            self.diagnosis = "view-search budget exhausted"
            self._transition(resume_state, "NEEDS_REVIEW", "REQUEST_REVIEW", self.diagnosis)
            return "NEEDS_REVIEW"
        names = self._normalize_requested_views(requested)
        if not names:
            self.diagnosis = (
                "requested_views contained no known camera; choose from "
                + ", ".join(VIEW_CAMERAS)
            )
            self.guard.observe(error=self.diagnosis)
            self._transition(resume_state, "INVALID_RESPONSE", "REPAIR_RESPONSE", self.diagnosis)
            self.iter_idx += 1
            return "INVALID_RESPONSE"

        self.budget.used_view_searches += 1
        work = self._iter_dir() / "requested_views"
        source = self.last_valid.step_path or self.base_step
        rendered = self._render_feedback_views(
            names,
            base_step=self.base_step,
            candidate_step=source if Path(str(source)).resolve() != Path(self.base_step).resolve() else None,
            output_dir=work,
            candidate_label=self.last_valid.commit_id or "current",
        )
        self.supplemental_views = self._merge_supplemental(rendered)
        report = {
            "phase": phase,
            "requested": list(requested),
            "accepted": names,
            "rendered": [v.to_dict() for v in rendered],
            "available_cameras": list(VIEW_CAMERAS),
        }
        _dump(work / "request.json", report)
        self.diagnosis = (
            f"fulfilled requested views {names}; inspect the attached requested_base/candidate_result images"
        )
        self._transition(resume_state, resume_state, "SEARCH_VIEW", self.diagnosis)
        self.iter_idx += 1
        return resume_state

    def _capture_candidate_feedback(
        self,
        task: TaskSpec,
        candidate_step: str,
        work: Path,
        visual: dict[str, Any],
        requested: list[str],
    ) -> None:
        feedback: list[ReferenceView] = []
        render_paths = dict(visual.get("render_paths") or {})
        for target in task.target_views:
            cam = target.camera
            if cam is None or not cam.camera_known:
                continue
            if not target.image_path or not Path(str(target.image_path)).is_file():
                continue
            path = render_paths.get(cam.view_name)
            if path and Path(path).is_file():
                feedback.append(
                    ReferenceView(
                        view_id=f"candidate_result_{cam.view_name}",
                        role="candidate_result",
                        image_path=str(path),
                        camera=cam,
                        image_source="vtk",
                        expected_view_known=True,
                        view_alignment_confidence=1.0,
                        estimated_observed_view_spec=cam.to_dict(),
                    )
                )
        extra = [name for name in requested if name not in render_paths]
        if extra:
            feedback.extend(
                self._render_feedback_views(
                    extra,
                    base_step=self.base_step,
                    candidate_step=candidate_step,
                    output_dir=work / "feedback_views",
                    candidate_label=self.geom_repo.head if self.geom_repo else "candidate",
                )
            )
        self.supplemental_views = self._merge_supplemental(feedback)
        _dump(
            work / "mllm_feedback_views.json",
            {"views": [v.to_dict() for v in self.supplemental_views]},
        )

    def _render_feedback_views(
        self,
        names: list[str],
        *,
        base_step: str,
        candidate_step: str | None,
        output_dir: Path,
        candidate_label: str,
    ) -> list[ReferenceView]:
        output_dir.mkdir(parents=True, exist_ok=True)
        up_axis = "+Z"
        task = getattr(self, "task", None)
        if task and task.base_views and task.base_views[0].camera:
            up_axis = task.base_views[0].camera.up_axis
        views: list[ReferenceView] = []
        for name in names:
            spec = known_view_spec(name, up_axis=up_axis, size=self.cfg.view_size)
            sources = [("requested_base", base_step)]
            if candidate_step:
                sources.append(("candidate_result", candidate_step))
            for role, step in sources:
                out = output_dir / f"{role}_{name}.png"
                result = render_step(step, spec, out)
                if not result.get("ok"):
                    continue
                label = candidate_label if role == "candidate_result" else "base"
                views.append(
                    ReferenceView(
                        view_id=f"{role}_{label}_{name}",
                        role=role,
                        image_path=str(Path(result["path"]).resolve()),
                        camera=spec,
                        image_source="vtk",
                        expected_view_known=True,
                        view_alignment_confidence=1.0,
                        estimated_observed_view_spec=spec.to_dict(),
                    )
                )
        return views

    def _merge_supplemental(self, new_views: list[ReferenceView]) -> list[ReferenceView]:
        merged: dict[tuple[str, str], ReferenceView] = {
            (v.role, v.camera.view_name if v.camera else v.view_id): v
            for v in self.supplemental_views
        }
        for view in new_views:
            merged[(view.role, view.camera.view_name if view.camera else view.view_id)] = view
        limit = max(6, self.cfg.max_view_searches * self.cfg.max_views_per_search * 2 + 4)
        return list(merged.values())[-limit:]

    def _set_diagnosis(self, layer: str, msg: str) -> None:
        parts = [f"failed_layer: {layer}", msg]
        if self.frozen_intent is not None:
            parts.append(
                "## Frozen intent (keep what/where unless this is a visual replan)\n"
                + json.dumps(self.frozen_intent.to_dict(), ensure_ascii=False)[:2000]
            )
        if self.current_trace:
            parts.append(
                "## Strategy (not executed; rewrite cadquery_code)\n"
                + json.dumps(
                    {"from_commit": self.current_from_commit, "strategy": self.current_trace},
                    ensure_ascii=False,
                )[:2000]
            )
        if self.current_code:
            parts.append("## Previous cadquery_code\n" + self.current_code[:4000])
        if self.geom_repo:
            parts.append(
                f"last_commit: {self.geom_repo.head}. "
                "Set from_commit to that id to reuse the STEP as INPUT_STEP, or 'base' to restart."
            )
        self.diagnosis = "\n".join(parts)

    def _edit_class(self) -> str:
        spec = self.current_spec
        if spec is None:
            return "replace"
        return edit_class(spec.canonical_operation or spec.operation)

    def _render_and_compare(
        self,
        task: TaskSpec,
        step_path: str,
        work: Path,
        *,
        edit_class_name: str | None = None,
    ) -> dict[str, Any]:
        render_dir = work / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        per_view: list[dict[str, Any]] = []
        vs_before: list[dict[str, Any]] = []
        render_paths: dict[str, str] = {}
        for target in task.target_views:
            cam = target.camera
            if cam is None or not cam.camera_known:
                continue
            if not target.image_path or not Path(str(target.image_path)).is_file():
                continue
            out = render_dir / f"{cam.view_name}.png"
            rendered = render_step(step_path, cam, out)
            if not rendered.get("ok"):
                return {
                    "passed": False,
                    "failure_type": "BUILD_ERROR",
                    "reason": rendered.get("detail") or "render failed",
                    "mean_target_iou": 0.0,
                    "render": rendered,
                }
            render_paths[cam.view_name] = str(Path(rendered["path"]).resolve())
            before = next((v for v in task.base_views if v.camera and v.camera.view_name == cam.view_name), None)
            before_path = (
                before.image_path if before and Path(before.image_path).is_file() else None
            )
            per_view.append(
                compare_images(
                    rendered["path"],
                    target.image_path,
                    before_image=before_path,
                    mask_dir=render_dir,
                    camera_confidence=None,
                    view_id=target.view_id,
                    requested_view_spec=cam.to_dict(),
                    view_alignment_confidence=target.view_alignment_confidence,
                    image_source=target.image_source,
                    expected_view_known=target.expected_view_known,
                    region_mask_max_frac=self.cfg.region_mask_max_frac,
                    region_mask_min_frac=self.cfg.region_mask_min_frac,
                )
            )
            if before_path:
                vs_before.append(
                    compare_before_after(
                        before_path,
                        rendered["path"],
                        view_id=cam.view_name,
                    )
                )
        gates = visual_gates(
            per_view,
            vs_before,
            same_renderer_iou_min=self.cfg.silhouette_iou_min,
            cross_renderer_iou_min=self.cfg.cross_renderer_iou_min,
            wrecked_iou_min=self.cfg.wrecked_iou_min,
            view_alignment_min=self.cfg.view_alignment_min,
            same_renderer_iou_floor=self.cfg.silhouette_iou_floor,
            cross_renderer_iou_floor=self.cfg.cross_renderer_iou_floor,
            edit_class_name=edit_class_name,
            region_change_iou_min=self.cfg.region_change_iou_min,
            region_edge_iou_min=self.cfg.region_edge_iou_min,
            noop_silhouette_iou_min=self.cfg.noop_silhouette_iou_min,
            noop_pixel_mae_max=self.cfg.noop_pixel_mae_max,
        )
        gates["per_view"] = per_view
        gates["vs_before"] = vs_before
        gates["render_paths"] = render_paths
        return gates

    def _record(
        self,
        exec_result: ExecutionResult,
        work: Path,
        comparison: dict[str, Any],
        visual: dict[str, Any] | None,
    ) -> CandidateRecord:
        effect = (comparison.get("edit_effect") or {}) if comparison else {}
        return CandidateRecord(
            step_path=exec_result.output_step_path,
            code=self.current_code,
            edit_spec=self.current_spec.to_dict() if self.current_spec else None,
            score=_candidate_score(comparison, visual),
            iteration=self.iter_idx,
            renders=dict((visual or {}).get("render_paths") or {}),
            geometry=comparison,
            visual=visual,
            geometry_ok=True,
            preservation_ok=bool(comparison.get("preservation_ok")),
            intent_ok=bool(comparison.get("intent_ok")),
            commit_id=self.geom_repo.head if self.geom_repo else None,
            effect_class=str(effect.get("effect_class") or "unknown"),
            retry_feedback=(comparison.get("retry_feedback") if comparison else None),
        )

    def _update_last_valid(
        self,
        exec_result: ExecutionResult,
        work: Path,
        comparison: dict[str, Any],
        visual: dict[str, Any] | None,
    ) -> None:
        rec = self._record(exec_result, work, comparison, visual)
        rec.note = "last_valid"
        self.last_valid = rec

    def _update_best_valid(
        self,
        exec_result: ExecutionResult,
        work: Path,
        comparison: dict[str, Any],
        visual: dict[str, Any] | None,
    ) -> None:
        if not comparison.get("preservation_ok") or bool(
            (comparison.get("edit_effect") or {}).get("no_effect")
        ):
            return
        rec = self._record(exec_result, work, comparison, visual)
        rec.note = "best_valid"
        if rec.score >= self.best_valid.score:
            self.best_valid = rec

    def _promote_current(
        self,
        exec_result: ExecutionResult,
        work: Path,
        comparison: dict[str, Any],
        visual: dict[str, Any],
    ) -> None:
        rec = self._record(exec_result, work, comparison, visual)
        rec.note = "accepted"
        self.last_valid = rec
        self.best_valid = rec

    def _apply_rollback(self, reason: str) -> CandidateRecord:
        policy = ROLLBACK_POLICY.get(reason, "best_valid_or_base")
        if policy == "best_valid_or_base":
            if self.best_valid.step_path:
                chosen = self.best_valid
                label = "best_valid"
            else:
                chosen = CandidateRecord(step_path=self.base_step, note="base", geometry_ok=True)
                label = "base"
        else:
            chosen = self.last_valid
            label = "last_valid"
        if chosen.code:
            self.current_code = chosen.code
        if chosen.commit_id:
            self.current_from_commit = chosen.commit_id
        _progress(f"rollback[{reason}] policy={policy} restore={label} iter={chosen.iteration}")
        return chosen

    def _handle_rollback(self, from_state: str) -> str:
        reason = self.rollback_reason or from_state
        chosen = self._apply_rollback(reason)
        rem = self.budget.remaining()
        if rem["mllm_calls"] <= 0:
            self._transition("ROLLBACK", "NEEDS_REVIEW", "REQUEST_REVIEW", f"no MLLM budget after {reason}")
            return "NEEDS_REVIEW"

        if from_state == "GEOMETRY_INVALID":
            resume = resume_after_geometry_invalid(self.geometry_fault)
            if resume == "REPAIR_CODE" and rem["code_repairs"] > 0:
                self._transition(
                    "ROLLBACK",
                    "CODE_ERROR",
                    "REPAIR_CODE",
                    f"geometry invalid ({self.geometry_fault}); restore {chosen.note or 'last_valid'} then repair code",
                )
                return "CODE_ERROR"
            if rem["edit_rounds"] > 0:
                self._transition(
                    "ROLLBACK",
                    "GEOMETRY_INVALID",
                    "REFINE_EDIT",
                    f"geometry invalid ({self.geometry_fault}); restore then refine edit — not intent mismatch",
                )
                extra = (
                    "Previous candidate was geometrically invalid. "
                    f"Diagnosis: {self.diagnosis}. "
                    "Replan what/where if the feature choice was wrong; do not treat this as a numeric intent mismatch. "
                    "Return ONLY PLAN_INTENT JSON, no cadquery_code."
                )
                return self._propose_resume(extra=extra, code_repair=False)
            self._transition("ROLLBACK", "NEEDS_REVIEW", "REQUEST_REVIEW", "no budget after geometry rollback")
            return "NEEDS_REVIEW"

        if from_state == "PRESERVATION_VIOLATION":
            if rem["edit_rounds"] > 0:
                extra = (
                    "Preservation failed: unmodified structure was changed. "
                    "Replan where (and what if needed). Keep registry how.operation. "
                    "Return ONLY PLAN_INTENT JSON, no cadquery_code."
                )
                self._transition(
                    "ROLLBACK",
                    "PRESERVATION_VIOLATION",
                    "PROPOSE_INTENT",
                    f"restore {chosen.note or 'best_valid_or_base'} then replan where",
                )
                return self._propose_resume(extra=extra, code_repair=False)
            self._transition("ROLLBACK", "NEEDS_REVIEW", "REQUEST_REVIEW", "no budget after preservation rollback")
            return "NEEDS_REVIEW"

        if rem["code_repairs"] > 0:
            self._transition("ROLLBACK", "CODE_ERROR", "REPAIR_CODE", f"restore after {reason}")
            return "CODE_ERROR"
        self._transition("ROLLBACK", "NEEDS_REVIEW", "REQUEST_REVIEW", f"no budget after {reason}")
        return "NEEDS_REVIEW"

    def _propose_resume(self, *, extra: str, code_repair: bool) -> str:
        task = getattr(self, "task", None) or task_spec_from_manifest(self.manifest, size=self.cfg.view_size)
        geometry = getattr(self, "base_geometry", None) or inspect_step(task.base_step_path)
        if code_repair:
            return self._propose_code(task, geometry, extra=extra, code_repair=True)
        return self._propose_intent(task, geometry, extra=extra)

    def _transition(self, src: str, dest: str, action: str, detail: str) -> None:
        rec = {
            "from": src,
            "to": dest,
            "action": action,
            "detail": detail[:500],
            "iteration": self.iter_idx,
            "budget": self.budget.remaining(),
        }
        self.transitions.append(rec)
        _progress(f"{src} -> {dest} via {action}: {detail[:160]}")

    def _finish(self, status: str, summary: str) -> Manifest:
        self._persist_memory()
        self.manifest.artifacts["harness_memory"] = str(
            (self.root / "conversation_memory.json").resolve()
        )
        final_dir = self.root / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        spec = self.current_spec
        export = self.best_valid if self.best_valid.step_path else self.last_valid
        if export.edit_spec and status in {"SUCCESS", "NEEDS_REVIEW"}:
            spec = EditSpec.from_dict(export.edit_spec)
        instruction = _read_instruction(self.manifest.description_txt)
        if self.frozen_intent is not None:
            self.store.write_text(
                REL_INTENT_JSON,
                json.dumps(self.frozen_intent.to_dict(), indent=2, ensure_ascii=False) + "\n",
            )
            self.manifest.artifacts["intent_json"] = str(self.store.path(REL_INTENT_JSON).resolve())
        if spec is not None:
            legacy = edit_spec_to_legacy_json(spec, instruction, intent=self.frozen_intent)
            self.store.write_text(
                REL_EDIT_JSON, json.dumps(legacy, indent=2, ensure_ascii=False) + "\n"
            )
            self.manifest.artifacts["edit_json"] = str(self.store.path(REL_EDIT_JSON).resolve())
        code = export.code or self.current_code
        if code:
            self.store.write_text(REL_EDIT_PY, code)
            self.manifest.artifacts["edit_py"] = str(self.store.path(REL_EDIT_PY).resolve())
            (final_dir / "solution.py").write_text(code, encoding="utf-8")

        step_src = None
        if status == "SUCCESS" and self.best_valid.step_path:
            step_src = self.best_valid.step_path
        elif status == "NEEDS_REVIEW" and self.best_valid.step_path and self.best_valid.score >= 0:
            step_src = self.best_valid.step_path

        if step_src and Path(step_src).is_file() and Path(step_src).resolve() != Path(self.manifest.input_step).resolve():
            dest = copy_valid_step(step_src, self.store.path(REL_OUTPUT_STEP))
            self.manifest.artifacts["output_step"] = str(dest.resolve())
            shutil.copy2(dest, final_dir / "output.step")
            stl_src = Path(step_src).with_suffix(".stl")
            if not stl_src.is_file():
                stl_src = Path(step_src).parent / "candidate.stl"
            if stl_src.is_file():
                shutil.copy2(stl_src, self.store.path(REL_OUTPUT_STL))
                self.manifest.artifacts["output_stl"] = str(self.store.path(REL_OUTPUT_STL).resolve())
        elif status == "SUCCESS":
            # still copy last_valid if it is an edited candidate
            pass

        result_renders: dict[str, str] = {}
        for name, src in (export.renders or {}).items():
            if src and Path(src).is_file():
                dest = self.store.path(rel_result_view(name))
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                result_renders[name] = str(dest.resolve())
                final_png = final_dir / "renders" / f"{name}.png"
                final_png.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, final_png)
        if result_renders:
            first = next(iter(result_renders.values()))
            shutil.copy2(first, self.store.path(REL_RESULT))
            self.manifest.artifacts["result_render"] = str(self.store.path(REL_RESULT).resolve())
            self.manifest.artifacts["result_renders"] = result_renders

        report = {
            "status": status,
            "summary": summary,
            "budget": self.budget.to_dict(),
            "view_search": {
                "available_cameras": list(VIEW_CAMERAS),
                "max_views_per_search": self.cfg.max_views_per_search,
                "supplemental_views": [v.view_id for v in self.supplemental_views],
            },
            "conversation_memory": self.memory.to_dict()["policy"],
            "oscillation": self.guard.to_dict(),
            "last_valid_candidate": self.last_valid.snapshot(),
            "best_valid_candidate": self.best_valid.snapshot(),
            "best_candidate_score": self.best_valid.score,
            "edit_effect": (
                (export.geometry or {}).get("edit_effect")
                or ((export.visual or {}).get("edit_effect") if export.visual else None)
            ),
            "retry_feedback": export.retry_feedback,
            "rollback_policy": dict(ROLLBACK_POLICY),
            "supported_operations": sorted(SUPPORTED_OPERATIONS),
            "thresholds": {
                "status": "engineering_placeholder" if self.cfg.thresholds_are_placeholders else "calibrated",
                "calibrated": not bool(self.cfg.thresholds_are_placeholders),
                "calibrated_on": self.cfg.calibrated_on or None,
                "silhouette_iou_min": self.cfg.silhouette_iou_min,
                "silhouette_iou_floor": self.cfg.silhouette_iou_floor,
                "cross_renderer_iou_min": self.cfg.cross_renderer_iou_min,
                "cross_renderer_iou_floor": self.cfg.cross_renderer_iou_floor,
                "preservation_bbox_rel_tol": self.cfg.preservation_bbox_rel_tol,
                "preservation_bbox_rel_tol_replace": self.cfg.preservation_bbox_rel_tol_replace,
                "volume_explode_ratio": self.cfg.volume_explode_ratio,
                "volume_explode_ratio_global": self.cfg.volume_explode_ratio_global,
                "wrecked_iou_min": self.cfg.wrecked_iou_min,
                "view_alignment_min": self.cfg.view_alignment_min,
                "region_change_iou_min": self.cfg.region_change_iou_min,
                "region_edge_iou_min": self.cfg.region_edge_iou_min,
                "region_mask_max_frac": self.cfg.region_mask_max_frac,
                "noop_silhouette_iou_min": self.cfg.noop_silhouette_iou_min,
                "note": (
                    "Placeholder gates to keep the MVP runnable. Not validated on a "
                    "real MLLM / GPT-target / complex-STEP task set."
                    if self.cfg.thresholds_are_placeholders
                    else (
                        "Calibrated on old-pipeline artifacts "
                        f"({self.cfg.calibrated_on or 'artifacts'}). "
                        "Pass-weighted ~P10 visual / ~P95 geometry."
                    )
                ),
            },
            "best": {
                "iteration": self.best_valid.iteration,
                "score": self.best_valid.score,
                "step": self.best_valid.step_path,
            },
            "geometry": self.best_valid.geometry or self.last_valid.geometry,
            "visual": self.best_valid.visual,
            "cadquery": str(self.store.path(REL_EDIT_PY).resolve()) if code else None,
            "output_step": self.manifest.artifacts.get("output_step"),
            "renders": result_renders,
            "ml_cannot_finish": True,
            "geom_repo_head": self.geom_repo.head if self.geom_repo else None,
            "cameras": (
                "Standard and requested CAD views: VTK requested=observed. "
                "GPT targets: expected_view_known, view_alignment_confidence < 1.0."
            ),
        }
        _dump(final_dir / "report.json", report)
        _dump(self.root / "transition_history.json", self.transitions)
        self.manifest.artifacts["harness_root"] = str(self.root.resolve())
        self.manifest.artifacts["harness_report"] = str((final_dir / "report.json").resolve())
        self.manifest.checks["edit_harness"] = {
            "status": status,
            "summary": summary,
            "iterations": self.iter_idx,
            "budget": self.budget.to_dict(),
            "visual": {
                "mean_target_iou": (self.best_valid.visual or {}).get("mean_target_iou"),
                "layers": (self.best_valid.visual or {}).get("layers"),
            },
            "geometry": self.best_valid.geometry,
            "last_valid": self.last_valid.snapshot(),
            "best_valid": self.best_valid.snapshot(),
        }
        self.manifest.checks["edit_json"] = {
            "hard": "pass" if spec is not None else "fail",
            "source": "edit_harness",
        }
        self.manifest.checks["edit_py"] = {
            "hard": "pass" if code else "fail",
            "source": "edit_harness",
        }
        self.manifest.checks["exec"] = {
            "hard": "pass" if self.manifest.artifacts.get("output_step") else "fail",
            "source": "edit_harness",
        }
        self.manifest.checks["result_qc"] = {
            "hard": "pass" if status == "SUCCESS" else "fail",
            "source": "edit_harness",
            "status": status,
            "layers": (self.best_valid.visual or {}).get("layers"),
        }
        self.manifest.set_summary(f"edit harness {status}: {summary[:200]}")
        if status == "SUCCESS":
            self.manifest.status = "done"
            self.manifest.stage = "done"
            self.manifest.error = None
        elif status == "NEEDS_REVIEW":
            self.manifest.status = "needs_review"
            self.manifest.stage = "step4_edit_harness"
            self.manifest.error = summary
        else:
            self.manifest.status = "failed"
            self.manifest.stage = "step4_edit_harness"
            self.manifest.error = summary
        self.store.persist(self.manifest)
        return self.manifest


def run_edit_harness(store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]) -> Manifest:
    return EditHarness(store, manifest, ctx).run()
