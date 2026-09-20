"""Task / view / edit / execution contracts for the S2+ edit harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


def _as_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _as_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [_as_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _as_dict(v) for k, v in obj.items()}
    return obj


@dataclass
class ViewSpec:
    """Numeric camera used by the VTK renderer.

    ``camera_known`` means a *requested* pose exists (from ``VIEW_CAMERAS``).
    That is not the same as observing that pose in a GPT img2img target.
    """

    view_name: str
    projection: str = "orthographic"
    look_from: tuple[float, float, float] = (1.0, 1.0, 1.0)
    view_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    azimuth_deg: float | None = None
    elevation_deg: float | None = None
    roll_deg: float = 0.0
    scale: float | None = 1.18
    target: tuple[float, float, float] | None = None
    image_width: int = 1024
    image_height: int = 1024
    crop_mode: str = "none"
    up_axis: str = "+Z"
    # ``look_from``/``view_up`` are expressed after the render-only upright
    # transform. CadQuery continues to operate in the untouched STEP frame.
    # These two vectors make that inverse mapping explicit to the model.
    step_look_from: tuple[float, float, float] | None = None
    step_view_up: tuple[float, float, float] | None = None
    coordinate_frame: str = "upright_render"
    modeling_coordinate_frame: str = "original_step"
    camera_known: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ViewSpec | None:
        if not data:
            return None
        look = data.get("look_from") or (1.0, 1.0, 1.0)
        up = data.get("view_up") or (0.0, 0.0, 1.0)
        tgt = data.get("target")
        return cls(
            view_name=str(data.get("view_name") or "iso"),
            projection=str(data.get("projection") or "orthographic"),
            look_from=_triple(look, (1.0, 1.0, 1.0)),
            view_up=_triple(up, (0.0, 0.0, 1.0)),
            azimuth_deg=_opt_float(data.get("azimuth_deg")),
            elevation_deg=_opt_float(data.get("elevation_deg")),
            roll_deg=float(data.get("roll_deg") or 0.0),
            scale=_opt_float(data.get("scale"), 1.18),
            target=_triple(tgt, None) if tgt is not None else None,
            image_width=int(data.get("image_width") or 1024),
            image_height=int(data.get("image_height") or 1024),
            crop_mode=str(data.get("crop_mode") or "none"),
            up_axis=str(data.get("up_axis") or "+Z"),
            step_look_from=(
                _triple(data.get("step_look_from"), None)
                if data.get("step_look_from") is not None
                else None
            ),
            step_view_up=(
                _triple(data.get("step_view_up"), None)
                if data.get("step_view_up") is not None
                else None
            ),
            coordinate_frame=str(data.get("coordinate_frame") or "upright_render"),
            modeling_coordinate_frame=str(
                data.get("modeling_coordinate_frame") or "original_step"
            ),
            camera_known=bool(data.get("camera_known", True)),
        )


@dataclass
class ReferenceView:
    view_id: str
    role: str  # base_view | target_reference
    image_path: str
    camera: ViewSpec | None = None
    # requested_view_spec is ``camera``. GPT targets may not obey it.
    image_source: str = "unknown"  # vtk | gpt_img2img | stub | unknown
    expected_view_known: bool = False
    view_alignment_confidence: float | None = None
    estimated_observed_view_spec: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "role": self.role,
            "image_path": self.image_path,
            "camera": self.camera.to_dict() if self.camera else None,
            "requested_view_spec": self.camera.to_dict() if self.camera else None,
            "image_source": self.image_source,
            "expected_view_known": self.expected_view_known,
            "view_alignment_confidence": self.view_alignment_confidence,
            "estimated_observed_view_spec": self.estimated_observed_view_spec,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceView:
        cam = data.get("camera") or data.get("requested_view_spec")
        conf = data.get("view_alignment_confidence")
        return cls(
            view_id=str(data.get("view_id") or ""),
            role=str(data.get("role") or "base_view"),
            image_path=str(data.get("image_path") or ""),
            camera=ViewSpec.from_dict(cam) if isinstance(cam, dict) else None,
            image_source=str(data.get("image_source") or "unknown"),
            expected_view_known=bool(data.get("expected_view_known", False)),
            view_alignment_confidence=float(conf) if isinstance(conf, (int, float)) else None,
            estimated_observed_view_spec=(
                dict(data["estimated_observed_view_spec"])
                if isinstance(data.get("estimated_observed_view_spec"), dict)
                else None
            ),
        )


@dataclass
class TaskSpec:
    task_id: str
    base_step_path: str
    instruction: str
    base_views: list[ReferenceView] = field(default_factory=list)
    target_views: list[ReferenceView] = field(default_factory=list)
    output_directory: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "base_step_path": self.base_step_path,
            "instruction": self.instruction,
            "base_views": [v.to_dict() for v in self.base_views],
            "target_views": [v.to_dict() for v in self.target_views],
            "output_directory": self.output_directory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        return cls(
            task_id=str(data.get("task_id") or ""),
            base_step_path=str(data.get("base_step_path") or ""),
            instruction=str(data.get("instruction") or ""),
            base_views=[
                ReferenceView.from_dict(v)
                for v in (data.get("base_views") or [])
                if isinstance(v, dict)
            ],
            target_views=[
                ReferenceView.from_dict(v)
                for v in (data.get("target_views") or [])
                if isinstance(v, dict)
            ],
            output_directory=str(data.get("output_directory") or ""),
        )


@dataclass
class EditTarget:
    feature_id: str = ""
    semantic_location: str = ""
    geometric_conditions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EditTarget:
        data = data or {}
        cond = data.get("geometric_conditions")
        return cls(
            feature_id=str(data.get("feature_id") or ""),
            semantic_location=str(data.get("semantic_location") or data.get("target") or ""),
            geometric_conditions=dict(cond) if isinstance(cond, dict) else {},
        )


@dataclass
class EditIntent:
    """Frozen plan that steers CadQuery. Not executed."""

    what: dict[str, Any] = field(default_factory=dict)
    where: dict[str, Any] = field(default_factory=dict)
    how: dict[str, Any] = field(default_factory=dict)
    why: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    target_description: str = ""
    change_type: str = "unknown"
    target_relationship: str = "unknown"
    spatial_constraints: list[Any] = field(default_factory=list)
    geometric_constraints: list[Any] = field(default_factory=list)
    preserve_constraints: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    candidate_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "what": dict(self.what),
            "where": dict(self.where),
            "how": dict(self.how),
            "why": self.why,
            "constraints": dict(self.constraints),
            "target_description": self.target_description,
            "change_type": self.change_type,
            "target_relationship": self.target_relationship,
            "spatial_constraints": list(self.spatial_constraints),
            "geometric_constraints": list(self.geometric_constraints),
            "preserve_constraints": list(self.preserve_constraints),
            "ambiguities": list(self.ambiguities),
            "candidate_refs": list(self.candidate_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EditIntent:
        data = data or {}
        nested = data.get("intent") if isinstance(data.get("intent"), dict) else data
        why = nested.get("why") or ""
        what = dict(nested.get("what") or {})
        where = dict(nested.get("where") or {})
        how = dict(nested.get("how") or {})
        constraints = _as_constraints(nested.get("constraints"))
        op = str(how.get("operation") or what.get("operation") or "").lower()
        if op.startswith(("add_", "extend_")):
            inferred_change = "add"
        elif op.startswith(("cut_", "remove_")) or op == "add_hole":
            inferred_change = "remove"
        elif op in {"mirror_body", "move", "translate", "rotate"}:
            inferred_change = "move"
        else:
            inferred_change = "modify"
        target_description = str(
            nested.get("target_description")
            or where.get("semantic_location")
            or where.get("target")
            or what.get("feature")
            or ""
        )
        candidate_raw = nested.get("candidate_refs") or where.get("candidate_refs") or where.get("candidate_id") or []
        if isinstance(candidate_raw, str):
            candidate_raw = [candidate_raw]
        preserve = nested.get("preserve_constraints") or constraints.get("keep") or []
        ambiguities = nested.get("ambiguities") or constraints.get("unknowns") or []
        return cls(
            what=what,
            where=where,
            how=how,
            why=str(why).strip() if not isinstance(why, dict) else json_why(why),
            constraints=constraints,
            target_description=target_description,
            change_type=str(nested.get("change_type") or inferred_change),
            target_relationship=str(
                nested.get("target_relationship") or where.get("relationship") or "unknown"
            ),
            spatial_constraints=_any_list(
                nested.get("spatial_constraints") or where.get("spatial_constraints")
            ),
            geometric_constraints=_any_list(
                nested.get("geometric_constraints") or what.get("geometric_constraints")
            ),
            preserve_constraints=_str_list(preserve),
            ambiguities=_str_list(ambiguities),
            candidate_refs=_str_list(candidate_raw),
        )

    def to_edit_spec(self) -> EditSpec:
        op = str(self.how.get("operation") or self.what.get("operation") or "modify_feature")
        params = dict(self.what.get("parameters") or {})
        if self.how.get("pattern"):
            params["pattern"] = self.how["pattern"]
        cond = {
            k: v
            for k, v in self.where.items()
            if k not in {"semantic_location", "feature_id", "target"}
        }
        # Carry high-level intent into deterministic validation.  These are
        # constraints, not CadQuery decisions.
        cond["target_relationship"] = self.target_relationship
        cond["change_type"] = self.change_type
        if self.spatial_constraints:
            cond["spatial_constraints"] = list(self.spatial_constraints)
        if self.geometric_constraints:
            cond["geometric_constraints"] = list(self.geometric_constraints)
        keep = _str_list(
            self.preserve_constraints
            or self.constraints.get("keep")
            or self.constraints.get("invariants")
        )
        forbidden = _str_list(
            self.constraints.get("forbidden") or self.constraints.get("forbidden_changes")
        )
        unknowns = _str_list(self.ambiguities or self.constraints.get("unknowns"))
        expected = [self.why] if self.why else []
        if unknowns:
            expected.append("unknowns: " + "; ".join(unknowns[:4]))
        return EditSpec(
            operation=op,
            target=EditTarget(
                feature_id=str(self.where.get("feature_id") or ""),
                semantic_location=str(
                    self.where.get("semantic_location") or self.where.get("target") or ""
                ),
                geometric_conditions=cond,
            ),
            parameters=params,
            invariants=keep,
            expected_changes=expected,
            forbidden_changes=forbidden,
            edit_type="modify",
        )


def json_why(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("purpose") or raw.get("text") or "").strip()
    return str(raw or "").strip()


_INTENT_PATCH_FIELDS = {
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


def merge_intent_patch(intent: EditIntent, patch: dict[str, Any]) -> EditIntent:
    """Deep-merge an evidence-backed partial intent revision.

    Nested dictionaries can be changed field by field. Lists and scalar values
    are replaced only when the patch explicitly supplies them.
    """

    def merge(current: Any, update: Any) -> Any:
        if isinstance(current, dict) and isinstance(update, dict):
            result = dict(current)
            for key, value in update.items():
                normalized = str(key)
                result[normalized] = merge(result.get(normalized), value)
            return result
        return update

    base = intent.to_dict()
    for key, value in patch.items():
        if key in _INTENT_PATCH_FIELDS:
            base[key] = merge(base.get(key), value)
    return EditIntent.from_dict(base)


def _as_constraints(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"keep": [], "forbidden": [], "unknowns": []}
    return {
        "keep": _str_list(raw.get("keep") or raw.get("invariants")),
        "forbidden": _str_list(raw.get("forbidden") or raw.get("forbidden_changes")),
        "unknowns": _str_list(raw.get("unknowns")),
    }


def _any_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, tuple):
        return list(raw)
    return [raw]


@dataclass
class EditSpec:
    operation: str = "modify_feature"
    target: EditTarget = field(default_factory=EditTarget)
    parameters: dict[str, Any] = field(default_factory=dict)
    invariants: list[str] = field(default_factory=list)
    expected_changes: list[str] = field(default_factory=list)
    forbidden_changes: list[str] = field(default_factory=list)
    confidence: float | None = None
    edit_type: str = "modify"
    canonical_operation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "canonical_operation": self.canonical_operation,
            "target": self.target.to_dict(),
            "parameters": dict(self.parameters),
            "invariants": list(self.invariants),
            "expected_changes": list(self.expected_changes),
            "forbidden_changes": list(self.forbidden_changes),
            "confidence": self.confidence,
            "edit_type": self.edit_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EditSpec:
        data = data or {}
        target_raw = data.get("target")
        if isinstance(target_raw, str):
            target = EditTarget(semantic_location=target_raw)
        else:
            target = EditTarget.from_dict(target_raw if isinstance(target_raw, dict) else {})
        conf = data.get("confidence")
        return cls(
            operation=str(data.get("operation") or data.get("op") or "modify_feature"),
            target=target,
            parameters=dict(data.get("parameters") or data.get("params") or {}),
            invariants=_str_list(data.get("invariants")),
            expected_changes=_str_list(data.get("expected_changes")),
            forbidden_changes=_str_list(data.get("forbidden_changes")),
            confidence=float(conf) if isinstance(conf, (int, float)) else None,
            edit_type=str(data.get("edit_type") or "modify"),
            canonical_operation=(
                str(data["canonical_operation"])
                if data.get("canonical_operation")
                else None
            ),
        )


@dataclass
class ExecutionResult:
    success: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    output_step_path: str | None = None
    output_stl_path: str | None = None
    duration_seconds: float = 0.0
    failure_type: str | None = None
    code_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)


@dataclass
class HarnessBudget:
    max_edit_rounds: int = 5
    max_code_repairs: int = 3
    max_mllm_calls: int = 12
    max_view_searches: int = 0
    exec_timeout_seconds: float = 120.0
    used_edit_rounds: int = 0
    used_code_repairs: int = 0
    used_mllm_calls: int = 0
    used_view_searches: int = 0

    def remaining(self) -> dict[str, int]:
        return {
            "edit_rounds": max(0, self.max_edit_rounds - self.used_edit_rounds),
            "code_repairs": max(0, self.max_code_repairs - self.used_code_repairs),
            "mllm_calls": max(0, self.max_mllm_calls - self.used_mllm_calls),
            "view_searches": max(0, self.max_view_searches - self.used_view_searches),
        }

    def exhausted(self) -> bool:
        rem = self.remaining()
        return rem["edit_rounds"] <= 0 or rem["mllm_calls"] <= 0

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)


def _triple(raw: Any, default: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    return default


def _opt_float(raw: Any, default: float | None = None) -> float | None:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    return [str(raw)]
