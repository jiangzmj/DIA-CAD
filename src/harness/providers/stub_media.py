"""Stub media / LLM stand-in when no real API keys are configured."""

from __future__ import annotations

import json
import re
import struct
import zlib
from pathlib import Path
from typing import Any

from harness.providers.base import AssistantTurn


def _minimal_png(
    width: int = 128,
    height: int = 128,
    rgb: tuple[int, int, int] = (230, 230, 235),
) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    r, g, b = rgb
    row = b"\x00" + bytes([r, g, b]) * width
    raw = row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


class StubMediaProvider:
    """Offline stand-in for image/video generation and simple chat turns."""

    name = "stub_media"

    def __init__(self, workspace: Path | None = None):
        self.workspace = Path(workspace) if workspace else None
        self.meter = None

    def set_meter(self, meter) -> None:
        self.meter = meter

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AssistantTurn:
        last = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    last = content
                elif isinstance(content, list):
                    bits: list[str] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            bits.append(str(part.get("text") or ""))
                    last = "\n".join(bits)
                else:
                    last = str(content)
                break
        lower = last.lower()
        if "camera_distance_factor" in lower or (
            "occupancy" in lower and "factors" in lower
        ):
            from harness.pipeline.view_qc_adjust import (
                default_factors,
                heuristic_adjust_factors,
            )

            current = default_factors()
            qc_views: dict[str, Any] = {}
            try:
                m = re.search(
                    r"<<<QC_STATE>>>\s*(\{[\s\S]*?\})\s*<<<END_QC_STATE>>>",
                    last,
                )
                if not m:
                    m = re.search(
                        r'"current_factors"\s*:\s*\{[\s\S]*?"views"\s*:\s*\{[\s\S]*?\}\s*\}',
                        last,
                    )
                    blob_txt = ("{" + m.group(0) + "}") if m else None
                else:
                    blob_txt = m.group(1)
                if blob_txt:
                    blob = json.loads(blob_txt)
                    if isinstance(blob.get("current_factors"), dict):
                        current.update(
                            {str(k): float(v) for k, v in blob["current_factors"].items()}
                        )
                    if isinstance(blob.get("views"), dict):
                        qc_views = blob["views"]
            except Exception:  # noqa: BLE001
                pass
            factors = heuristic_adjust_factors({"views": qc_views}, current)
            payload = {"factors": factors, "notes": "stub QC heuristic"}
            return AssistantTurn(content=json.dumps(payload, indent=2), finish_reason="stop")
        if (
            "matches_description" in lower
            or "preserves_unchanged" in lower
            or "target render qc" in lower
            or "s1 — target" in lower
        ):
            payload = {
                "pass": True,
                "verdict": "pass",
                "matches_description": True,
                "preserves_unchanged": True,
                "camera_matches_base_view": True,
                "same_part_identity": True,
                "cad_consistent": True,
                "same_edit": True,
                "viewpoint_only": True,
                "evaluation": "stub pass",
                "issues": "",
                "fix_hint": "",
            }
            return AssistantTurn(content=json.dumps(payload, indent=2), finish_reason="stop")
        if (
            "only the intent json" in lower
            or "plan_intent" in lower
            or "s2 — plan intent" in lower
            or "s2 - plan intent" in lower
        ):
            payload = {
                "action": "PLAN_INTENT",
                "what": {
                    "feature": "imported_solid",
                    "parameters": {},
                },
                "where": {
                    "semantic_location": "body",
                    "feature_id": "unknown",
                },
                "how": {"operation": "modify_feature"},
                "why": "stub passthrough of the imported solid",
                "constraints": {
                    "keep": ["imported solid"],
                    "forbidden": ["do not replace with a placeholder box"],
                    "unknowns": [],
                },
            }
            return AssistantTurn(content=json.dumps(payload, indent=2), finish_reason="stop")
        if (
            "s2+ harness" in lower
            or "implement the frozen intent" in lower
            or "action apply_edit + cadquery_code" in lower
            or "frozen intent (implement this" in lower
        ):
            payload = {
                "action": "APPLY_EDIT",
                "from_commit": "base",
                "cadquery_code": (
                    "import cadquery as cq\n"
                    "result = cq.importers.importStep(INPUT_STEP)\n"
                ),
                "requested_views": [],
                "fallback_action": "REQUEST_REVIEW",
            }
            return AssistantTurn(content=json.dumps(payload, indent=2), finish_reason="stop")
        if "up_axis" in lower or "bottom_axis" in lower or "upright" in lower or "bottom_face" in lower:
            payload = {
                "object": "stub part",
                "top_face": "working top",
                "bottom_face": "base plate",
                "bottom_axis": "-Z",
                "already_upright": True,
                "imagine": "part sitting on its base, working top facing the sky",
                "reason": "stub: functional base already toward -Z",
            }
            return AssistantTurn(content=json.dumps(payload, indent=2), finish_reason="stop")
        if "cadquery_code" in lower or "edit_spec" in lower or "apply_edit" in lower:
            payload = {
                "action": "APPLY_EDIT",
                "from_commit": "base",
                "cadquery_code": (
                    "import cadquery as cq\n"
                    "result = cq.importers.importStep(INPUT_STEP)\n"
                ),
                "requested_views": [],
                "fallback_action": "REQUEST_REVIEW",
            }
            return AssistantTurn(content=json.dumps(payload, indent=2), finish_reason="stop")
        if "json" in lower or "edit.json" in lower or "schema" in lower:
            payload = {
                "version": 1,
                "edit_type": "modify",
                "operations": [
                    {
                        "op": "modify_feature",
                        "target": "body",
                        "params": {"from_stub": True},
                    }
                ],
                "notes": "stub S2 response",
            }
            return AssistantTurn(content=json.dumps(payload, indent=2), finish_reason="stop")
        if "cadquery" in lower or "edit.py" in lower or "code" in lower:
            code = (
                "import cadquery as cq\n"
                "result = cq.Workplane('XY').box(10, 10, 10)\n"
            )
            return AssistantTurn(content=f"```python\n{code}```", finish_reason="stop")
        if (
            "must_show" in lower
            or "target render direction" in lower
            or "base_reason" in lower
        ):
            payload = {
                "base_views": ["iso", "z_corner"],
                "base_view": "iso",
                "base_reasons": {
                    "iso": "stub overall shape",
                    "z_corner": "stub attachment region",
                },
                "base_reason": "stub default",
                "camera": "same as each base_view",
                "shading": "olive shaded technical, white background",
                "must_show": "edited features",
            }
            return AssistantTurn(content=json.dumps(payload, indent=2), finish_reason="stop")
        if "render" in lower or "image" in lower or "target" in lower:
            return AssistantTurn(
                content="Intent understood. Generating target render (stub).",
                finish_reason="stop",
            )
        return AssistantTurn(
            content="Stub intent: apply the described CAD edit.",
            finish_reason="stop",
        )

    def generate_image(
        self,
        prompt: str,
        out_path: Path | str,
        *,
        size: tuple[int, int] = (256, 256),
        **_kwargs: Any,
    ) -> dict[str, Any]:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(_minimal_png(size[0], size[1], (220, 225, 235)))
        return {"ok": True, "path": str(out_path.resolve()), "prompt": prompt[:200], "mode": "stub"}
