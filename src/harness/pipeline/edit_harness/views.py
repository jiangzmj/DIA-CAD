"""Known-camera ViewSpec + STEP rendering.

The pipeline images use known poses:
  4 CAD views (iso / z_corner / x_corner / y_corner)
  0–2 target references (same cameras as the pair-QC-accepted base_views)

VTK uses orthographic (parallel) projection. Never invent a camera.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from harness.pipeline.base_view import resolve_base_views
from harness.pipeline.edit_harness.compare import estimate_view_alignment
from harness.pipeline.edit_harness.spec import ReferenceView, TaskSpec, ViewSpec
from harness.pipeline.manifest import VIEW_NAMES
from harness.pipeline.orient import axis_vector, normalize_up_axis
from harness.pipeline.skills.cad_views import (
    VIEW_CAMERAS,
    render_oriented_view_png,
    rotation_aligning_up,
)
from harness.pipeline.view_qc_adjust import DEFAULT_FACTOR, clamp_factor


def look_from_to_az_el(
    look_from: tuple[float, float, float],
) -> tuple[float, float]:
    x, y, z = look_from
    azimuth = math.degrees(math.atan2(y, x))
    elevation = math.degrees(math.atan2(z, math.hypot(x, y)))
    return (round(azimuth, 4), round(elevation, 4))


def upright_vector_to_step(
    vector: tuple[float, float, float], up_axis: str
) -> tuple[float, float, float]:
    """Map a render-upright direction back to original STEP coordinates."""
    import numpy as np

    axis = normalize_up_axis(up_axis) or "+Z"
    rotation = rotation_aligning_up(axis_vector(axis))
    # render_point = rotation @ step_point, hence step_vector = R.T @ render_vector.
    mapped = rotation.T @ np.asarray(vector, dtype=np.float64)
    return tuple(0.0 if abs(float(x)) < 1e-10 else round(float(x), 10) for x in mapped)  # type: ignore[return-value]


def known_view_spec(
    view_name: str,
    *,
    up_axis: str = "+Z",
    scale: float = DEFAULT_FACTOR,
    size: int = 1024,
) -> ViewSpec:
    if view_name not in VIEW_CAMERAS:
        raise ValueError(f"unknown view_name={view_name!r}; camera would be invented")
    look_from, view_up = VIEW_CAMERAS[view_name]
    az, el = look_from_to_az_el(look_from)
    step_look_from = upright_vector_to_step(look_from, up_axis)
    step_view_up = upright_vector_to_step(view_up, up_axis)
    return ViewSpec(
        view_name=view_name,
        projection="orthographic",
        look_from=tuple(float(x) for x in look_from),  # type: ignore[arg-type]
        view_up=tuple(float(x) for x in view_up),  # type: ignore[arg-type]
        azimuth_deg=az,
        elevation_deg=el,
        roll_deg=0.0,
        scale=clamp_factor(float(scale)),
        target=None,
        image_width=int(size),
        image_height=int(size),
        crop_mode="none",
        up_axis=str(up_axis or "+Z"),
        step_look_from=step_look_from,
        step_view_up=step_view_up,
        camera_known=True,
    )


def task_spec_from_manifest(
    manifest: Any,
    *,
    size: int = 1024,
) -> TaskSpec:
    arts = dict(getattr(manifest, "artifacts", None) or {})
    views = dict(arts.get("views") or {})
    targets = dict(arts.get("target_renders") or {})
    factors = dict(arts.get("view_factors") or {})
    orient = dict(arts.get("orient") or {})
    up_axis = str(orient.get("up_axis") or "+Z")
    chosen = resolve_base_views(arts, count=2)

    instruction = ""
    desc = getattr(manifest, "description_txt", None)
    if desc and Path(str(desc)).is_file():
        instruction = Path(str(desc)).read_text(encoding="utf-8")

    base_views: list[ReferenceView] = []
    for name in VIEW_NAMES:
        path = views.get(name)
        if not path or not Path(str(path)).is_file():
            continue
        scale = float(factors.get(name, DEFAULT_FACTOR))
        spec = known_view_spec(name, up_axis=up_axis, scale=scale, size=size)
        base_views.append(
            ReferenceView(
                view_id=name,
                role="base_view",
                image_path=str(path),
                camera=spec,
                image_source="vtk",
                expected_view_known=True,
                view_alignment_confidence=1.0,
                estimated_observed_view_spec=spec.to_dict(),
            )
        )

    declared_source = _declared_target_source(manifest)
    target_views: list[ReferenceView] = []
    seen: set[str] = set()
    # Prefer the chosen cameras, then any extra keys still listed in artifacts.
    # Missing files are skipped so 0/1/2 accepted targets all load cleanly.
    for name in (*chosen, *targets.keys()):
        key = str(name)
        if key in seen:
            continue
        seen.add(key)
        path = targets.get(name)
        if path is None:
            path = targets.get(key)
        if not path or not Path(str(path)).is_file():
            continue
        try:
            scale = float(factors.get(key, DEFAULT_FACTOR))
            camera = known_view_spec(key, up_axis=up_axis, scale=scale, size=size)
        except ValueError:
            continue
        cad = next((v for v in base_views if v.camera and v.camera.view_name == key), None)
        target_views.append(
            _target_reference_view(
                view_name=key,
                image_path=str(path),
                camera=camera,
                cad_view=cad,
                declared_source=declared_source,
            )
        )

    return TaskSpec(
        task_id=str(getattr(manifest, "run_id", "") or "task"),
        base_step_path=str(getattr(manifest, "input_step", "") or arts.get("input_step") or ""),
        instruction=instruction,
        base_views=base_views,
        target_views=target_views,
        output_directory=str(getattr(manifest, "root", "") or ""),
    )


def unknown_camera_views(task: TaskSpec) -> list[ReferenceView]:
    """CAD base views that cannot be used. Missing-camera targets are skipped, not fatal."""
    unknown: list[ReferenceView] = []
    for view in task.base_views:
        if view.image_path and not Path(view.image_path).is_file():
            continue
        if view.camera is None or not view.camera.camera_known:
            unknown.append(view)
    return unknown


def render_step(
    step_path: Path | str,
    view_spec: ViewSpec,
    output_image_path: Path | str,
) -> dict[str, Any]:
    """Render STEP at a numeric ViewSpec. Requires a known named camera."""
    if not view_spec.camera_known or view_spec.view_name not in VIEW_CAMERAS:
        return {
            "ok": False,
            "path": str(Path(output_image_path).resolve()),
            "backend": "view_uncertain",
            "detail": "camera unknown; refusing to invent a pose",
            "view_spec": view_spec.to_dict(),
        }
    result = render_oriented_view_png(
        step_path,
        output_image_path,
        up_axis=view_spec.up_axis,
        view_name=view_spec.view_name,
        camera_distance_factor=float(view_spec.scale or DEFAULT_FACTOR),
        size=int(view_spec.image_width or 1024),
    )
    result["view_spec"] = view_spec.to_dict()
    result["requested_view_spec"] = view_spec.to_dict()
    # VTK render of our own STEP at a named camera: requested == observed.
    result["camera_confidence"] = 1.0 if result.get("ok") else 0.0
    result["view_alignment_confidence"] = result["camera_confidence"]
    result["image_source"] = "vtk"
    return result


def _declared_target_source(manifest: Any) -> str:
    checks = dict(getattr(manifest, "checks", None) or {})
    mode = str((checks.get("target_render") or {}).get("mode") or "").lower()
    if mode in {"gpt_images", "gpt_img2img", "gpt-image-1"}:
        return "gpt_img2img"
    if mode == "stub":
        return "stub"
    arts = dict(getattr(manifest, "artifacts", None) or {})
    src = str(arts.get("target_image_source") or "").lower()
    if src in {"vtk", "gpt_img2img", "stub"}:
        return src
    return ""


def _target_reference_view(
    *,
    view_name: str,
    image_path: str,
    camera: ViewSpec,
    cad_view: ReferenceView | None,
    declared_source: str,
) -> ReferenceView:
    source = declared_source or "unknown"
    alignment = None
    observed = None
    cad_path = cad_view.image_path if cad_view else ""
    if cad_path and Path(cad_path).is_file() and Path(image_path).is_file():
        est = estimate_view_alignment(cad_path, image_path)
        observed = est.get("estimated_observed_view_spec")
        alignment = est.get("view_alignment_confidence")
        if not source or source == "unknown":
            from harness.pipeline.edit_harness.compare import compare_images

            probe = compare_images(
                image_path,
                cad_path,
                view_id=view_name,
                image_source="unknown",
                expected_view_known=True,
            )
            source = (
                "vtk"
                if probe.get("reference_looks_like_vtk") and probe.get("candidate_looks_like_vtk")
                else "gpt_img2img"
            )

    if source == "vtk":
        alignment = 1.0
        observed = camera.to_dict()
    elif alignment is not None:
        alignment = min(float(alignment), 0.99)

    return ReferenceView(
        view_id=f"target_{view_name}",
        role="target_reference",
        image_path=str(image_path),
        camera=camera,
        image_source=source or "unknown",
        expected_view_known=True,
        view_alignment_confidence=None if alignment is None else float(alignment),
        estimated_observed_view_spec=observed,
    )
