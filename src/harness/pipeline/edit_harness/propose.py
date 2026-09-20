"""Partitioned MLLM prompts for the S2+ edit harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.pipeline.edit_harness.spec import EditIntent, ReferenceView, TaskSpec
from harness.pipeline.multimodal import image_part
from harness.pipeline.skills.cad_views import VIEW_CAMERAS


def load_cad_prompt(settings: Any, name: str) -> str:
    prompts_dir = getattr(getattr(settings, "pipeline", None), "prompts_dir", None)
    if prompts_dir:
        path = Path(prompts_dir) / name
    else:
        workspace = getattr(settings, "workspace", None)
        path = Path(workspace) / "prompts" / "cad_edit" / name if workspace else Path(name)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return f"(missing prompt {name})"


def view_caption(view: ReferenceView) -> str:
    cam = view.camera
    if cam is None or not cam.camera_known:
        return (
            f"[{view.role}:{view.view_id}] CAMERA UNKNOWN — do not assume a default pose; "
            "do not treat view differences as geometry."
        )
    lf = cam.look_from
    vu = cam.view_up
    step_lf = cam.step_look_from or lf
    step_vu = cam.step_view_up or vu
    requested = (
        f"requested_view_spec projection={cam.projection} view_name={cam.view_name} "
        f"upright_look_from=({lf[0]:.3f},{lf[1]:.3f},{lf[2]:.3f}) "
        f"upright_view_up=({vu[0]:.3f},{vu[1]:.3f},{vu[2]:.3f}) "
        f"step_look_from=({step_lf[0]:.3f},{step_lf[1]:.3f},{step_lf[2]:.3f}) "
        f"step_view_up=({step_vu[0]:.3f},{step_vu[1]:.3f},{step_vu[2]:.3f}) "
        f"azimuth_deg={cam.azimuth_deg} elevation_deg={cam.elevation_deg} "
        f"scale={cam.scale} up_axis={cam.up_axis} "
        f"size={cam.image_width}x{cam.image_height}"
    )
    if view.role == "base_view" or view.image_source == "vtk":
        return (
            f"[{view.role}:{view.view_id}] image_source={view.image_source or 'vtk'} "
            f"expected_view_known=true view_alignment_confidence="
            f"{1.0 if view.view_alignment_confidence is None else view.view_alignment_confidence} "
            f"{requested} (VTK: requested == observed)"
        )
    align = view.view_alignment_confidence
    align_txt = "unmeasured" if align is None else f"{align:.3f}"
    return (
        f"[{view.role}:{view.view_id}] image_source={view.image_source or 'gpt_img2img'} "
        f"expected_view_known={str(view.expected_view_known).lower()} "
        f"view_alignment_confidence={align_txt} "
        f"{requested} "
        "(GPT/stub target: requested view is known, observed pose is NOT assumed identical; "
        "do not set camera_confidence=1.0)"
    )


def build_propose_prompt(
    *,
    prompt_body: str,
    task: TaskSpec,
    geometry: dict[str, Any],
    diagnosis: str = "",
    feature_note: str = "",
    extra: str = "",
    frozen_intent: EditIntent | None = None,
    closing: str = "",
) -> str:
    import json

    geo_txt = json.dumps(geometry, ensure_ascii=False, indent=2)
    if len(geo_txt) > 12000:
        geo_txt = geo_txt[:12000] + "\n…(truncated)"
    parts = [
        prompt_body.strip(),
        "",
        "## Task",
        f"task_id: {task.task_id}",
        "",
        "## Edit Description",
        task.instruction.strip() or "(empty)",
        "",
        "## STEP geometry summary",
        geo_txt,
    ]
    if feature_note.strip():
        parts.extend(["", "## FeatureProbe note", feature_note.strip()])
    if frozen_intent is not None:
        parts.extend(
            [
                "",
                "## Current frozen intent",
                json.dumps(frozen_intent.to_dict(), ensure_ascii=False, indent=2),
            ]
        )
    n_targets = sum(
        1
        for v in task.target_views
        if v.image_path and Path(v.image_path).is_file()
    )
    camera_lines = [
        "",
        "## Cameras",
        "The CAD base_view images follow"
        + (
            f", plus {n_targets} target_reference image(s)"
            if n_targets
            else " (no target_reference images; plan from CAD views + Edit Description)"
        )
        + ", plus any requested/previous candidate views.",
        "Available requested_views camera names: " + ", ".join(VIEW_CAMERAS),
        "Camera names are controller aliases, not assertions about the product's semantic front/left/top. "
        "In particular, front means the upright-render -Y axis view.",
        "If the current images are insufficient, return action=REQUEST_VIEWS with up to the allowed number of camera names.",
        "Each image has a requested_view_spec. VTK CAD views: requested == observed.",
        "Images use the upright_render frame, while FeatureProbe measurements and CadQuery use original_step. "
        "Use step_look_from and step_view_up when converting an image direction into a modeling axis.",
    ]
    if n_targets:
        camera_lines.extend(
            [
                "GPT/stub targets: expected view is known, but view_alignment_confidence is not 1.0.",
                "Do not treat a GPT framing difference as a geometry difference.",
            ]
        )
    parts.extend(camera_lines)
    if diagnosis.strip():
        parts.extend(["", "## Previous diagnosis (fix this layer)", diagnosis.strip()])
    if extra.strip():
        parts.extend(["", extra.strip()])
    parts.extend(
        [
            "",
            closing
            or "Return ONLY the JSON object (action + cadquery_code). Implement the frozen intent.",
        ]
    )
    return "\n".join(parts)


def multimodal_content(text: str, views: list[ReferenceView], *, detail: str = "high") -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for view in views:
        path = Path(view.image_path) if view.image_path else None
        if path is None or not path.is_file():
            continue
        parts.append({"type": "text", "text": "\n" + view_caption(view)})
        parts.append(image_part(path, detail=detail))
    return parts


def labeled_views(
    task: TaskSpec, supplemental_views: list[ReferenceView] | None = None
) -> list[ReferenceView]:
    views = list(task.base_views) + list(task.target_views) + list(supplemental_views or [])
    return [v for v in views if v.image_path and Path(v.image_path).is_file()]


def propose_intent(
    media: Any,
    *,
    settings: Any,
    task: TaskSpec,
    geometry: dict[str, Any],
    diagnosis: str = "",
    feature_note: str = "",
    extra: str = "",
    history_messages: list[dict[str, Any]] | None = None,
    supplemental_views: list[ReferenceView] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    body = load_cad_prompt(settings, "s2_intent.md")
    text = build_propose_prompt(
        prompt_body=body,
        task=task,
        geometry=geometry,
        diagnosis=diagnosis,
        feature_note=feature_note,
        extra=extra,
        closing=(
            "Return ONLY PLAN_INTENT JSON (target/change/relationship + what/where/how/why/constraints), or REQUEST_VIEWS "
            "JSON when another listed camera is required. No cadquery_code."
        ),
    )
    content = multimodal_content(text, labeled_views(task, supplemental_views))
    turn = media.chat([*(history_messages or []), {"role": "user", "content": content}])
    return (turn.content or ""), content


def propose_edit(
    media: Any,
    *,
    settings: Any,
    task: TaskSpec,
    geometry: dict[str, Any],
    diagnosis: str = "",
    feature_note: str = "",
    extra: str = "",
    frozen_intent: EditIntent | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    supplemental_views: list[ReferenceView] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    body = load_cad_prompt(settings, "s2_harness.md")
    text = build_propose_prompt(
        prompt_body=body,
        task=task,
        geometry=geometry,
        diagnosis=diagnosis,
        feature_note=feature_note,
        extra=extra,
        frozen_intent=frozen_intent,
        closing=(
            "Return ONLY one JSON action: APPLY_EDIT + cadquery_code when the current intent is "
            "supported by the evidence; REVISE_INTENT with the smallest intent_patch + "
            "revision_reason + revision_evidence when any intent field is wrong; or REQUEST_VIEWS "
            "when another listed camera is required first."
        ),
    )
    content = multimodal_content(text, labeled_views(task, supplemental_views))
    turn = media.chat([*(history_messages or []), {"role": "user", "content": content}])
    return (turn.content or ""), content
