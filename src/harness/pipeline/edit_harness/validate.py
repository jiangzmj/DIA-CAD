"""Geometry validity, intent constraints, and preservation checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from harness.pipeline.edit_harness.inspect import inspect_step
from harness.pipeline.edit_harness.operations import (
    canonicalize_operation,
    edit_class,
    is_cylinder_intent,
)
from harness.pipeline.edit_harness.spec import EditSpec


def _explode_limit(
    *,
    cls: str,
    volume_explode_ratio: float,
    volume_explode_ratio_global: float,
) -> float:
    if cls in {"global_transform", "additive"}:
        return max(float(volume_explode_ratio), float(volume_explode_ratio_global))
    return float(volume_explode_ratio)


def validate_step(
    step_path: Path | str,
    *,
    base_report: dict[str, Any] | None = None,
    volume_explode_ratio: float = 8.0,
    volume_explode_ratio_global: float = 24.0,
    edit_class_name: str | None = None,
) -> dict[str, Any]:
    path = Path(step_path)
    if not path.is_file() or path.stat().st_size <= 0:
        return {
            "ok": False,
            "failure_type": "GEOMETRY_INVALID",
            "fault_kind": "invalid_solid",
            "error": "output STEP missing or empty",
            "report": None,
        }
    report = inspect_step(path, include_features=False)
    if not report.get("ok"):
        return {
            "ok": False,
            "failure_type": "GEOMETRY_INVALID",
            "fault_kind": "invalid_solid",
            "error": report.get("error") or "STEP re-read failed",
            "report": report,
        }
    if int(report.get("solid_count") or 0) < 1:
        return {
            "ok": False,
            "failure_type": "GEOMETRY_INVALID",
            "fault_kind": "invalid_solid",
            "error": "no valid solid",
            "report": report,
        }
    volume = float(report.get("volume_mm3") or 0.0)
    if volume <= 0:
        return {
            "ok": False,
            "failure_type": "GEOMETRY_INVALID",
            "fault_kind": "invalid_solid",
            "error": "volume is not positive",
            "report": report,
        }
    bbox = report.get("bounding_box") or {}
    coords = []
    for axis in ("x", "y", "z"):
        pair = bbox.get(axis) or [0, 0]
        coords.extend(pair)
    if any(not math.isfinite(float(c)) for c in coords):
        return {
            "ok": False,
            "failure_type": "GEOMETRY_INVALID",
            "fault_kind": "invalid_solid",
            "error": "NaN or infinite coordinates",
            "report": report,
        }
    if base_report and base_report.get("ok"):
        cls = edit_class_name or "replace"
        limit = _explode_limit(
            cls=cls,
            volume_explode_ratio=volume_explode_ratio,
            volume_explode_ratio_global=volume_explode_ratio_global,
        )
        base_vol = float(base_report.get("volume_mm3") or 0.0) or 1e-6
        if volume / base_vol > limit or base_vol / max(volume, 1e-9) > limit:
            return {
                "ok": False,
                "failure_type": "GEOMETRY_INVALID",
                "fault_kind": "volume_anomaly",
                "error": (
                    f"volume exploded or collapsed ({volume:.3f} vs base {base_vol:.3f})"
                ),
                "report": report,
            }
        base_diag = float(base_report.get("diag_mm") or 0.0) or 1e-6
        diag = float(report.get("diag_mm") or 0.0)
        if diag / base_diag > limit or base_diag / max(diag, 1e-9) > limit:
            return {
                "ok": False,
                "failure_type": "GEOMETRY_INVALID",
                "fault_kind": "volume_anomaly",
                "error": f"bounding box exploded ({diag:.3f} vs base {base_diag:.3f})",
                "report": report,
            }
    return {
        "ok": True,
        "failure_type": None,
        "fault_kind": None,
        "error": None,
        "report": report,
    }


def compare_geometry(
    base_step: Path | str,
    candidate_step: Path | str,
    edit_spec: EditSpec | None = None,
    *,
    bbox_rel_tol: float = 0.45,
    bbox_rel_tol_replace: float = 0.90,
    edit_class_name: str | None = None,
    frozen_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = (
        dict(frozen_parameters)
        if frozen_parameters is not None
        else (dict(edit_spec.parameters) if edit_spec else {})
    )
    cls = edit_class_name or (
        edit_class(edit_spec.canonical_operation or edit_spec.operation) if edit_spec else "replace"
    )
    need_features = is_cylinder_intent(
        edit_spec.operation if edit_spec else None,
        params,
    )
    base = inspect_step(base_step, include_features=need_features)
    cand = inspect_step(candidate_step, include_features=need_features)
    intent = evaluate_intent(
        base,
        cand,
        edit_spec,
        edit_class_name=cls,
        frozen_parameters=frozen_parameters,
    )
    occupancy = probe_opening_occupancy(base_step, candidate_step, edit_spec)
    if occupancy.get("applicable"):
        checks = dict(intent.get("checks") or {})
        checks["opening_occupancy"] = occupancy
        intent["checks"] = checks
        intent["passed"] = bool(intent.get("passed")) and bool(occupancy.get("passed"))
    preservation = evaluate_preservation(
        base,
        cand,
        edit_spec,
        bbox_rel_tol=bbox_rel_tol,
        bbox_rel_tol_replace=bbox_rel_tol_replace,
        edit_class_name=cls,
    )
    return {
        "base": _compact_metrics(base),
        "candidate": _compact_metrics(cand),
        "delta": {
            "volume_mm3": _delta(cand, base, "volume_mm3"),
            "diag_mm": _delta(cand, base, "diag_mm"),
            "solid_count": _delta(cand, base, "solid_count"),
            "face_count": _delta(cand, base, "face_count"),
            "center_distance_mm": _center_distance(base.get("center"), cand.get("center")),
            "bbox_max_abs_mm": _bbox_abs_delta(
                base.get("bounding_box"), cand.get("bounding_box")
            ),
        },
        "intent_constraints": intent,
        "preservation_constraints": preservation,
        "intent_ok": bool(intent.get("passed")),
        "preservation_ok": bool(preservation.get("passed")),
        "edit_class": cls,
    }


def evaluate_intent(
    base: dict[str, Any],
    cand: dict[str, Any],
    edit_spec: EditSpec | None,
    *,
    edit_class_name: str | None = None,
    frozen_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Numeric checks when the spec has numbers; otherwise class-direction.

    Expected numbers come from the frozen intent ``what.parameters`` when
    provided — not from a per-round ``edit_spec`` the model might rewrite.
    """
    checks: dict[str, Any] = {}
    params = (
        dict(frozen_parameters)
        if frozen_parameters is not None
        else (dict(edit_spec.parameters) if edit_spec else {})
    )
    op = (edit_spec.operation if edit_spec else "") or ""
    cls = edit_class_name or edit_class(edit_spec.canonical_operation or op if edit_spec else op)
    numeric_keys = {
        "target_height_mm": "height",
        "height_mm": "height",
        "target_radius_mm": "radius",
        "radius_mm": "radius",
        "hole_diameter_mm": "diameter",
        "diameter_mm": "diameter",
    }
    cylinderish = is_cylinder_intent(op, params)
    matched_feature = _match_feature(cand.get("features") or [], edit_spec) if cylinderish else None
    base_feature = _match_feature(base.get("features") or [], edit_spec) if cylinderish else None

    duplicate_check = _detached_duplicate_check(base, cand, edit_spec)
    if duplicate_check is not None:
        checks["detached_duplicate_instances"] = duplicate_check

    if cylinderish:
        for key, kind in numeric_keys.items():
            if key not in params:
                continue
            expected = _as_float(params.get(key))
            if expected is None:
                continue
            actual = _feature_measure(matched_feature, kind)
            if kind == "height":
                combined = _combined_cylinder_height(
                    cand.get("features") or [],
                    edit_spec,
                )
                if combined is not None:
                    actual = combined
            passed = actual is not None and abs(actual - expected) <= max(0.15 * abs(expected), 0.5)
            checks[key] = {
                "expected": expected,
                "actual": actual,
                "passed": passed,
                "deterministic": actual is not None,
            }

    if not checks:
        dv = _delta(cand, base, "volume_mm3")
        d_diag = _delta(cand, base, "diag_mm")
        d_faces = _delta(cand, base, "face_count")
        if cls == "additive":
            checks["relative_change"] = {
                "expected": "volume_should_increase",
                "actual_volume_delta": dv,
                "passed": dv is not None and dv > -1e-3,
                "deterministic": False,
                "certainty": "medium",
            }
        elif cls == "subtractive":
            checks["relative_change"] = {
                "expected": "volume_should_decrease",
                "actual_volume_delta": dv,
                "passed": dv is not None and dv < 1e-3,
                "deterministic": False,
                "certainty": "medium",
            }
        elif cls == "replace":
            changed = any(
                v is not None and abs(v) > 1e-4 for v in (dv, d_diag, d_faces)
            )
            checks["relative_change"] = {
                "expected": "geometry_should_change",
                "actual_volume_delta": dv,
                "passed": changed,
                "deterministic": False,
                "certainty": "low",
            }
        elif cls == "global_transform":
            checks["relative_change"] = {
                "expected": "global_transform_deferred_to_visual",
                "actual_volume_delta": dv,
                "passed": True,
                "deterministic": False,
                "certainty": "low",
            }
        else:
            checks["deferred_to_visual"] = {
                "expected": "local_modify_no_numeric",
                "passed": True,
                "deterministic": False,
                "certainty": "low",
            }

    if cylinderish and base_feature and matched_feature:
        if "axis" in base_feature and "axis" in matched_feature:
            aligned = _dot(base_feature["axis"], matched_feature["axis"])
            checks["axis_unchanged"] = {
                "expected": 1.0,
                "actual": aligned,
                "passed": abs(aligned) >= 0.98,
            }

    passed = all(bool(v.get("passed")) for v in checks.values()) if checks else True
    return {"passed": passed, "checks": checks, "matched_feature": matched_feature, "edit_class": cls}


def _detached_duplicate_check(
    base: dict[str, Any], cand: dict[str, Any], edit_spec: EditSpec | None
) -> dict[str, Any] | None:
    """Verify source-preserving detached copies at frozen destination centers."""
    if edit_spec is None:
        return None
    cond = dict(edit_spec.target.geometric_conditions or {})
    relationship = str(cond.get("target_relationship") or "").strip().lower()
    change_type = str(cond.get("change_type") or "").strip().lower()
    source_center = _vec3(cond.get("origin_mm"))
    raw_destinations = cond.get("destination_origins_mm")
    destinations = [
        point for point in (_vec3(item) for item in (raw_destinations or [])) if point is not None
    ] if isinstance(raw_destinations, list) else []
    if relationship != "detached_solid" or change_type != "duplicate" or not source_center or not destinations:
        return None

    base_solids = list(base.get("solid_metrics") or [])
    cand_solids = list(cand.get("solid_metrics") or [])
    if not base_solids or not cand_solids:
        return {
            "passed": False,
            "deterministic": False,
            "reason": "per-solid metrics unavailable",
            "expected_destinations_mm": [list(point) for point in destinations],
        }

    def center_of(item: dict[str, Any]) -> tuple[float, float, float] | None:
        return _vec3(item.get("center_mm"))

    def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    source = min(
        (item for item in base_solids if center_of(item) is not None),
        key=lambda item: distance(center_of(item), source_center),  # type: ignore[arg-type]
        default=None,
    )
    if source is None:
        return {
            "passed": False,
            "deterministic": True,
            "reason": "source solid not found",
            "expected_destinations_mm": [list(point) for point in destinations],
        }
    source_actual = center_of(source)
    source_span = [float(v) for v in (source.get("span_mm") or [0.0, 0.0, 0.0])]
    source_volume = float(source.get("volume_mm3") or 0.0)
    center_tol = max(0.5, 0.01 * max(source_span or [0.0]))
    source_ok = source_actual is not None and distance(source_actual, source_center) <= center_tol

    def same_shape(item: dict[str, Any]) -> bool:
        span = [float(v) for v in (item.get("span_mm") or [])]
        if len(span) != 3 or len(source_span) != 3:
            return False
        spans_ok = all(
            abs(span[i] - source_span[i]) <= max(0.2, 0.02 * abs(source_span[i]))
            for i in range(3)
        )
        volume = float(item.get("volume_mm3") or 0.0)
        volume_ok = abs(volume - source_volume) <= max(0.5, 0.01 * abs(source_volume))
        return spans_ok and volume_ok

    matches: list[dict[str, Any]] = []
    all_destinations_ok = True
    for destination in destinations:
        eligible = [item for item in cand_solids if center_of(item) is not None and same_shape(item)]
        nearest = min(
            eligible,
            key=lambda item: distance(center_of(item), destination),  # type: ignore[arg-type]
            default=None,
        )
        actual = center_of(nearest) if nearest is not None else None
        error = distance(actual, destination) if actual is not None else None
        passed = error is not None and error <= center_tol
        all_destinations_ok = all_destinations_ok and passed
        matches.append(
            {
                "expected_center_mm": [round(v, 5) for v in destination],
                "actual_center_mm": [round(v, 5) for v in actual] if actual else None,
                "center_error_mm": round(error, 5) if error is not None else None,
                "passed": passed,
            }
        )

    source_preserved = any(
        center_of(item) is not None
        and same_shape(item)
        and distance(center_of(item), source_center) <= center_tol  # type: ignore[arg-type]
        for item in cand_solids
    )
    return {
        "passed": bool(source_ok and source_preserved and all_destinations_ok),
        "deterministic": True,
        "source_center_mm": list(source_center),
        "source_center_error_mm": (
            round(distance(source_actual, source_center), 5) if source_actual is not None else None
        ),
        "source_preserved": source_preserved,
        "expected_destinations_mm": [list(point) for point in destinations],
        "destination_matches": matches,
        "shape_reference": {
            "span_mm": source_span,
            "volume_mm3": source_volume,
        },
        "center_tolerance_mm": round(center_tol, 5),
    }


def probe_opening_occupancy(
    base_step: Path | str,
    candidate_step: Path | str,
    edit_spec: EditSpec | None,
) -> dict[str, Any]:
    """Sample the claimed opening: base should contain material, candidate should not."""
    skipped = {
        "applicable": False,
        "passed": True,
        "skipped": True,
        "reason": "not an opening spec",
        "n_inside_base": None,
        "n_inside_candidate": None,
        "n_points": 0,
    }
    if edit_spec is None:
        return skipped
    op = canonicalize_operation(edit_spec.canonical_operation or edit_spec.operation) or (
        edit_spec.operation or ""
    ).strip().lower()
    if op not in {"cut_slot", "add_hole"}:
        return skipped
    points = _opening_probe_points(edit_spec)
    if not points:
        return {
            **skipped,
            "reason": "opening plane/size incomplete",
        }
    try:
        import cadquery as cq
    except ImportError:
        return {**skipped, "reason": "cadquery not installed"}
    try:
        base_solids = cq.importers.importStep(str(Path(base_step).resolve())).solids().vals()
        cand_solids = cq.importers.importStep(str(Path(candidate_step).resolve())).solids().vals()
    except Exception as exc:  # noqa: BLE001
        return {
            "applicable": True,
            "passed": False,
            "skipped": False,
            "reason": f"import failed: {exc}",
            "n_inside_base": None,
            "n_inside_candidate": None,
            "n_points": len(points),
        }
    n_base = sum(1 for p in points if _point_inside_solids(base_solids, p))
    n_cand = sum(1 for p in points if _point_inside_solids(cand_solids, p))
    n = len(points)
    majority = max(1, (n + 1) // 2)
    base_has_material = n_base >= majority
    cand_is_open = n_cand <= n - majority
    passed = base_has_material and cand_is_open
    reason = None
    if not base_has_material:
        reason = "claimed opening plane has no solid on the base"
    elif not cand_is_open:
        reason = f"opening still occupied ({n_cand}/{n} probe points inside)"
    return {
        "applicable": True,
        "passed": passed,
        "deterministic": True,
        "skipped": False,
        "reason": reason,
        "n_inside_base": n_base,
        "n_inside_candidate": n_cand,
        "n_points": n,
        "offset_mm": 0.8,
        "expected": "base_inside_candidate_empty",
    }


def _opening_probe_points(edit_spec: EditSpec) -> list[tuple[float, float, float]] | None:
    params = dict(edit_spec.parameters or {})
    cond = dict(edit_spec.target.geometric_conditions or {})
    width = _coalesce_float(params.get("width_mm"), params.get("hole_diameter_mm"), params.get("diameter_mm"))
    height = _coalesce_float(params.get("height_mm"), params.get("hole_diameter_mm"), params.get("diameter_mm"))
    if width is None and height is None:
        diameter = _coalesce_float(
            params.get("hole_diameter_mm"), params.get("diameter_mm"), params.get("radius_mm")
        )
        if diameter is None:
            return None
        if params.get("radius_mm") is not None and params.get("diameter_mm") is None and params.get(
            "hole_diameter_mm"
        ) is None:
            diameter = 2.0 * diameter
        width = height = diameter
    if width is None:
        width = height
    if height is None:
        height = width
    if width is None or height is None or width <= 0 or height <= 0:
        return None

    normal = _vec3(cond.get("normal"))
    cut_dir = _vec3(params.get("cut_direction")) or _vec3(cond.get("axis"))
    if cut_dir is None and normal is not None:
        cut_dir = (-normal[0], -normal[1], -normal[2])
    if normal is None and cut_dir is not None:
        normal = (-cut_dir[0], -cut_dir[1], -cut_dir[2])
    if normal is None or cut_dir is None:
        return None
    normal = _unit3(normal)
    cut_dir = _unit3(cut_dir)

    axis = max(range(3), key=lambda i: abs(normal[i]))
    plane_coord = _coalesce_float(
        cond.get("plane_coordinate_mm"),
        cond.get("plane_y_mm"),
        cond.get("plane_x_mm"),
        cond.get("plane_z_mm"),
    )
    center = [
        _coalesce_float(params.get("center_x_mm"), cond.get("center_x_mm")),
        _coalesce_float(params.get("center_y_mm"), cond.get("center_y_mm")),
        _coalesce_float(params.get("center_z_mm"), cond.get("center_z_mm")),
    ]
    if plane_coord is not None:
        center[axis] = plane_coord
    origin = (
        cond.get("origin_mm")
        or cond.get("origin")
        or params.get("origin")
        or cond.get("center")
    )
    if isinstance(origin, (list, tuple)) and len(origin) == 3:
        for i, v in enumerate(origin):
            if center[i] is None:
                center[i] = _as_float(v)
    bbox = cond.get("bbox") if isinstance(cond.get("bbox"), dict) else None
    if bbox:
        for i, key in enumerate(("x", "y", "z")):
            pair = bbox.get(key) or [None, None]
            if center[i] is None and isinstance(pair, (list, tuple)) and len(pair) == 2:
                a, b = _as_float(pair[0]), _as_float(pair[1])
                if a is not None and b is not None:
                    center[i] = 0.5 * (a + b)
    if any(c is None for c in center):
        return None
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])  # type: ignore[arg-type]

    u, v = _plane_axes(normal)
    inset = 0.6
    offsets = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    into = 0.8
    points: list[tuple[float, float, float]] = []
    for su, sv in offsets:
        px = cx + su * inset * (width / 2.0) * u[0] + sv * inset * (height / 2.0) * v[0]
        py = cy + su * inset * (width / 2.0) * u[1] + sv * inset * (height / 2.0) * v[1]
        pz = cz + su * inset * (width / 2.0) * u[2] + sv * inset * (height / 2.0) * v[2]
        points.append(
            (
                px + into * cut_dir[0],
                py + into * cut_dir[1],
                pz + into * cut_dir[2],
            )
        )
    return points


def _plane_axes(normal: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    n = _unit3(normal)
    helper = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (1.0, 0.0, 0.0)
    u = _unit3(_cross(helper, n))
    v = _unit3(_cross(n, u))
    return u, v


def _vec3(raw: Any) -> tuple[float, float, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        try:
            return (float(raw[0]), float(raw[1]), float(raw[2]))
        except (TypeError, ValueError):
            return None
    return None


def _unit3(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(c * c for c in vec)) or 1.0
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _point_inside_solids(solids: list[Any], point: tuple[float, float, float]) -> bool:
    try:
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        from OCP.TopAbs import TopAbs_IN, TopAbs_ON
        from OCP.gp import gp_Pnt
    except ImportError:
        return False
    pnt = gp_Pnt(float(point[0]), float(point[1]), float(point[2]))
    for solid in solids:
        wrapped = getattr(solid, "wrapped", None)
        if wrapped is None:
            continue
        try:
            classifier = BRepClass3d_SolidClassifier(wrapped, pnt, 1.0e-4)
            state = classifier.State()
            if state in (TopAbs_IN, TopAbs_ON):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def evaluate_preservation(
    base: dict[str, Any],
    cand: dict[str, Any],
    edit_spec: EditSpec | None,
    *,
    bbox_rel_tol: float = 0.45,
    bbox_rel_tol_replace: float = 0.90,
    edit_class_name: str | None = None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    params = dict(edit_spec.parameters) if edit_spec else {}
    cond = (edit_spec.target.geometric_conditions if edit_spec else {}) or {}
    op = (edit_spec.operation if edit_spec else "") or ""
    cls = edit_class_name or edit_class(edit_spec.canonical_operation or op if edit_spec else op)
    relax_axes = _axes_expected_to_change(params, cond, edit_spec)
    if cls in {"additive", "subtractive"}:
        relax_axes = set(relax_axes) | _inferred_growth_axes(base, cand)

    base_bbox = base.get("bounding_box") or {}
    cand_bbox = cand.get("bounding_box") or {}
    if cls == "global_transform":
        checks["bbox_exempt"] = {
            "passed": True,
            "reason": "global_transform skips bbox stability",
        }
    else:
        axis_tol = float(bbox_rel_tol_replace if cls in {"replace", "additive", "subtractive"} else bbox_rel_tol)
        for axis in ("x", "y", "z"):
            b = base_bbox.get(axis) or [0, 0]
            c = cand_bbox.get(axis) or [0, 0]
            span = max(abs(b[1] - b[0]), 1e-6)
            rel = max(abs(c[0] - b[0]), abs(c[1] - b[1])) / span
            tol = 12.0 if axis in relax_axes and cls in {"additive", "subtractive"} else (
                2.5 if axis in relax_axes else axis_tol
            )
            checks[f"bbox_{axis}_stable"] = {
                "relative_delta": round(rel, 4),
                "tolerance": tol,
                "relaxed": axis in relax_axes,
                "passed": rel <= tol,
            }

    solid_delta = abs(int(cand.get("solid_count") or 0) - int(base.get("solid_count") or 0))
    solid_limit = {"local_modify": 3, "additive": 8, "subtractive": 6, "replace": 16, "global_transform": 24}.get(
        cls, 6
    )
    if cls in {"additive", "subtractive"}:
        checks["no_unexpected_solids"] = {
            "delta": solid_delta,
            "passed": True,
            "limit": None,
            "note": "boolean fragmenting is recorded, not a preservation failure",
        }
    elif cls == "global_transform":
        checks["no_unexpected_solids"] = {
            "delta": solid_delta,
            "passed": True,
            "limit": solid_limit,
            "note": "global_transform does not fail on solid count",
        }
    else:
        checks["no_unexpected_solids"] = {
            "delta": solid_delta,
            "passed": solid_delta <= solid_limit,
            "limit": solid_limit,
        }

    base_solids = int(base.get("solid_count") or 0)
    cand_solids = int(cand.get("solid_count") or 0)
    # Adding/cutting may fragment a body, but it must not silently fuse most
    # pre-existing assembly components into one replacement solid. Small
    # 1–3-solid models are exempt because a legitimate local union commonly
    # changes 2 solids to 1.
    if cls in {"local_modify", "additive", "subtractive"} and base_solids >= 4:
        minimum_kept = max(2, math.ceil(base_solids * 0.5))
        checks["preserve_existing_solid_components"] = {
            "base_solid_count": base_solids,
            "candidate_solid_count": cand_solids,
            "minimum_candidate_count": minimum_kept,
            "passed": cand_solids >= minimum_kept,
            "note": "local booleans may fragment, but may not collapse most existing components",
        }

    relationship = str(cond.get("target_relationship") or "unknown").strip().lower()
    change_type = str(cond.get("change_type") or "unknown").strip().lower()
    if relationship == "fused_feature" and cls in {"local_modify", "additive"}:
        checks["fused_feature_connectivity"] = {
            "target_relationship": relationship,
            "base_solid_count": base_solids,
            "candidate_solid_count": cand_solids,
            "passed": cand_solids <= base_solids,
            "note": "a fused additive feature must not introduce detached solids",
        }
    elif relationship == "detached_solid" and change_type in {"add", "duplicate"}:
        checks["detached_feature_created"] = {
            "target_relationship": relationship,
            "change_type": change_type,
            "base_solid_count": base_solids,
            "candidate_solid_count": cand_solids,
            "passed": cand_solids > base_solids,
            "note": "an additive detached target must add at least one solid without collapsing the assembly",
        }

    keep_radius = _as_float(params.get("keep_radius_mm") or params.get("original_radius_mm"))
    if keep_radius is not None:
        feat = _match_feature(cand.get("features") or [], edit_spec)
        actual = _feature_measure(feat, "radius")
        checks["radius_unchanged"] = {
            "expected": keep_radius,
            "actual": actual,
            "passed": actual is not None and abs(actual - keep_radius) <= max(0.05 * keep_radius, 0.2),
        }

    passed = all(bool(v.get("passed")) for v in checks.values())
    return {"passed": passed, "checks": checks, "edit_class": cls}


def _compact_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": report.get("ok"),
        "volume_mm3": report.get("volume_mm3"),
        "diag_mm": report.get("diag_mm"),
        "solid_count": report.get("solid_count"),
        "face_count": report.get("face_count"),
        "bounding_box": report.get("bounding_box"),
        "center": report.get("center"),
        "feature_count": len(report.get("features") or []),
    }


def _delta(cand: dict[str, Any], base: dict[str, Any], key: str) -> float | None:
    try:
        return float(cand.get(key) or 0) - float(base.get(key) or 0)
    except (TypeError, ValueError):
        return None


def _center_distance(a: Any, b: Any) -> float | None:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return None
    if len(a) < 3 or len(b) < 3:
        return None
    try:
        return math.sqrt(sum((float(b[i]) - float(a[i])) ** 2 for i in range(3)))
    except (TypeError, ValueError):
        return None


def _bbox_abs_delta(a: Any, b: Any) -> float | None:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    values: list[float] = []
    try:
        for axis in ("x", "y", "z"):
            av = a.get(axis) or []
            bv = b.get(axis) or []
            values.extend(abs(float(bv[i]) - float(av[i])) for i in (0, 1))
    except (IndexError, TypeError, ValueError):
        return None
    return max(values) if values else None


def _axes_expected_to_change(
    params: dict[str, Any],
    cond: dict[str, Any],
    edit_spec: EditSpec | None,
) -> set[str]:
    """Don't treat the intended growth axis as a preservation failure."""
    keys = set(params)
    text = " ".join(
        [
            str(edit_spec.operation if edit_spec else ""),
            " ".join(edit_spec.expected_changes if edit_spec else []),
        ]
    ).lower()
    height_like = bool(keys & {"target_height_mm", "height_mm"}) or any(
        w in text for w in ("taller", "height", "boss", "extrude")
    )
    if not height_like:
        return set()
    axis = cond.get("axis") or [0.0, 0.0, 1.0]
    try:
        ax, ay, az = abs(float(axis[0])), abs(float(axis[1])), abs(float(axis[2]))
    except (TypeError, ValueError, IndexError):
        return {"z"}
    if az >= ax and az >= ay:
        return {"z"}
    if ay >= ax:
        return {"y"}
    return {"x"}


def _bbox_rel_for_axis(base: dict[str, Any], cand: dict[str, Any], axis: str) -> float:
    b = (base.get("bounding_box") or {}).get(axis) or [0, 0]
    c = (cand.get("bounding_box") or {}).get(axis) or [0, 0]
    span = max(abs(b[1] - b[0]), 1e-6)
    return max(abs(c[0] - b[0]), abs(c[1] - b[1])) / span


def _max_bbox_rel(base: dict[str, Any], cand: dict[str, Any]) -> float:
    return max(_bbox_rel_for_axis(base, cand, axis) for axis in ("x", "y", "z"))


def _inferred_growth_axes(base: dict[str, Any], cand: dict[str, Any]) -> set[str]:
    """Treat the axis with the largest bbox move as intended growth."""
    rels = {axis: _bbox_rel_for_axis(base, cand, axis) for axis in ("x", "y", "z")}
    mx = max(rels.values())
    if mx < 0.12:
        return set()
    return {axis for axis, rel in rels.items() if rel >= 0.45 * mx or rel >= 0.20}


def _coalesce_float(*values: Any) -> float | None:
    for raw in values:
        parsed = _as_float(raw)
        if parsed is not None:
            return parsed
    return None


def _as_float(raw: Any) -> float | None:
    try:
        if raw is None or raw == "unknown":
            return None
        if isinstance(raw, dict) and "value" in raw:
            return _as_float(raw.get("value"))
        return float(raw)
    except (TypeError, ValueError):
        return None


def _combined_cylinder_height(
    features: list[dict[str, Any]],
    edit_spec: EditSpec | None,
) -> float | None:
    """Union z-span of coaxial same-radius cylinders (split faces after a pad)."""
    cond = (edit_spec.target.geometric_conditions if edit_spec else {}) or {}
    want_radius = _as_float(cond.get("radius_mm") or (edit_spec.parameters.get("radius_mm") if edit_spec else None))
    want_axis = cond.get("axis") or [0.0, 0.0, 1.0]
    zs: list[float] = []
    for feat in features:
        if feat.get("surface_type") != "cylinder":
            continue
        if want_radius is not None and feat.get("radius_mm") is not None:
            if abs(float(feat["radius_mm"]) - want_radius) > max(0.15 * want_radius, 0.4):
                continue
        if feat.get("axis") and abs(_dot(want_axis, feat["axis"])) < 0.95:
            continue
        z = feat.get("z_range") or []
        if len(z) >= 2:
            zs.extend([float(z[0]), float(z[1])])
    if len(zs) < 2:
        return None
    return max(zs) - min(zs)


def _feature_measure(feat: dict[str, Any] | None, kind: str) -> float | None:
    if not feat:
        return None
    if kind == "radius":
        return _as_float(feat.get("radius_mm"))
    if kind == "diameter":
        r = _as_float(feat.get("radius_mm"))
        return None if r is None else 2.0 * r
    if kind == "height":
        z = feat.get("z_range") or []
        if len(z) >= 2:
            return abs(float(z[1]) - float(z[0]))
    return None


def _match_feature(features: list[dict[str, Any]], edit_spec: EditSpec | None) -> dict[str, Any] | None:
    if not features:
        return None
    if not edit_spec:
        return next((f for f in features if f.get("surface_type") == "cylinder"), features[0])
    want_id = edit_spec.target.feature_id
    if want_id:
        for feat in features:
            if feat.get("feature_id") == want_id:
                return feat
    cond = edit_spec.target.geometric_conditions or {}
    want_radius = _as_float(cond.get("radius_mm") or edit_spec.parameters.get("radius_mm"))
    want_axis = cond.get("axis")
    best = None
    best_score = -1.0
    for feat in features:
        score = 0.0
        if feat.get("surface_type") == "cylinder":
            score += 1.0
        if want_radius is not None and feat.get("radius_mm") is not None:
            if abs(float(feat["radius_mm"]) - want_radius) <= max(0.15 * want_radius, 0.4):
                score += 2.0
        if want_axis and feat.get("axis"):
            if abs(_dot(want_axis, feat["axis"])) >= 0.95:
                score += 1.5
        loc = (edit_spec.target.semantic_location or "").lower()
        z = feat.get("z_range") or [0, 0]
        if "top" in loc and float(z[1]) >= float(z[0]):
            score += 0.2 * float(z[1])
        if score > best_score:
            best_score = score
            best = feat
    return best


def _dot(a: Any, b: Any) -> float:
    try:
        ax, ay, az = float(a[0]), float(a[1]), float(a[2])
        bx, by, bz = float(b[0]), float(b[1]), float(b[2])
        return ax * bx + ay * by + az * bz
    except (TypeError, ValueError, IndexError):
        return 0.0
