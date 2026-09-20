"""Pipeline stage implementations (stubs in Phase 1; real logic layered later)."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any, Callable

from harness.pipeline.artifacts import (
    DIR_OUTPUT,
    DIR_TARGET,
    DIR_VIEWS,
    REL_EDIT_JSON,
    REL_EDIT_PY,
    REL_OUTPUT_STEP,
    REL_OUTPUT_STL,
    REL_RESULT,
    rel_target_view,
    ArtifactStore,
)
from harness.pipeline.manifest import VIEW_NAMES, Manifest


class StageError(RuntimeError):
    """Hard failure that stops the pipeline."""


StageFn = Callable[[ArtifactStore, Manifest, dict[str, Any]], Manifest]


def _minimal_png(width: int = 8, height: int = 8, rgb: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    """Write a tiny valid RGB PNG (no external deps)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b""
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


def stage_ingest(store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]) -> Manifest:
    # Already done in ArtifactStore.init_run; keep as no-op checkpoint.
    manifest.stage = "step1_ingest"
    manifest.set_summary("Step1 ingest complete")
    store.persist(manifest)
    return manifest


def stage_cad_views(store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]) -> Manifest:
    """Phase 1 stub / Phase 2+ real render via ctx['cad_views']."""
    manifest.stage = "step2_views"
    renderer = ctx.get("cad_views")
    if renderer is not None:
        result = renderer(store, manifest, ctx)
        store.persist(result)
        return result

    views: dict[str, str] = {}
    png = _minimal_png()
    for name in VIEW_NAMES:
        path = store.write_bytes(f"{DIR_VIEWS}/{name}.png", png)
        views[name] = str(path.resolve())
    manifest.artifacts["views"] = views
    manifest.artifacts["views_dir"] = str(store.path(DIR_VIEWS).resolve())
    manifest.checks["views"] = {
        "hard": "skipped_stub",
        "detail": "Phase 1 stub PNGs; real QC in Phase 2",
    }
    manifest.set_summary(f"Wrote {len(views)} stub view PNGs → {DIR_VIEWS}/")
    store.persist(manifest)
    return manifest


def stage_edit_visualizer(
    store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]
) -> Manifest:
    manifest.stage = "step3_target_render"
    handler = ctx.get("edit_visualizer")
    if handler is not None:
        return handler(store, manifest, ctx)

    renders: dict[str, str] = {}
    for name in ("iso", "z_corner"):
        p = store.write_bytes(rel_target_view(name), _minimal_png(16, 16, (240, 240, 240)))
        renders[name] = str(p.resolve())
    manifest.artifacts["target_render"] = None
    manifest.artifacts["target_renders"] = renders
    manifest.artifacts["base_views"] = ["iso", "z_corner"]
    manifest.artifacts["base_view"] = "iso"
    manifest.checks["target_render"] = {"hard": "skipped_stub", "soft": None}
    manifest.sessions.setdefault("S1_edit_visualizer", "stub")
    manifest.set_summary(f"Stub target renders → {DIR_TARGET}/")
    store.persist(manifest)
    return manifest


def stage_edit_json(store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]) -> Manifest:
    manifest.stage = "step4_edit_json"
    handler = ctx.get("edit_structurer")
    if handler is not None:
        return handler(store, manifest, ctx)

    skeleton = {
        "version": 1,
        "edit_type": "modify",
        "operations": [],
        "notes": "Phase 1 stub schema skeleton",
    }
    path = store.write_text(
        REL_EDIT_JSON, json.dumps(skeleton, indent=2, ensure_ascii=False) + "\n"
    )
    manifest.artifacts["edit_json"] = str(path.resolve())
    manifest.checks["edit_json"] = {"hard": "skipped_stub", "schema": "skeleton"}
    manifest.sessions.setdefault("S2_edit_structurer", "stub")
    manifest.set_summary(f"Wrote stub edit.json → {REL_EDIT_JSON}")
    store.persist(manifest)
    return manifest


def stage_cadquery_code(
    store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]
) -> Manifest:
    manifest.stage = "step5_cadquery_code"
    handler = ctx.get("cad_coder")
    if handler is not None:
        return handler(store, manifest, ctx)

    code = (
        "# TODO: generated CadQuery script (Phase 1 stub)\n"
        "import cadquery as cq\n"
        "result = cq.Workplane('XY').box(10, 10, 10)\n"
    )
    path = store.write_text(REL_EDIT_PY, code)
    manifest.artifacts["edit_py"] = str(path.resolve())
    manifest.checks["edit_py"] = {"hard": "skipped_stub", "syntax": None}
    manifest.sessions.setdefault("S3_cad_coder", "stub")
    manifest.set_summary(f"Wrote stub edit.py → {REL_EDIT_PY}")
    store.persist(manifest)
    return manifest


def stage_cadquery_exec(
    store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]
) -> Manifest:
    manifest.stage = "step6_exec"
    handler = ctx.get("cadquery_exec")
    if handler is not None:
        return handler(store, manifest, ctx)

    step_path = store.write_text(REL_OUTPUT_STEP, "ISO-10303-21;\n/* stub STEP */\n")
    stl_path = store.write_text(REL_OUTPUT_STL, "solid stub\nendsolid stub\n")
    manifest.artifacts["output_step"] = str(step_path.resolve())
    manifest.artifacts["output_stl"] = str(stl_path.resolve())
    manifest.checks["exec"] = {"hard": "skipped_stub", "exit_code": 0}
    manifest.set_summary(f"Wrote stub outputs → {DIR_OUTPUT}/")
    store.persist(manifest)
    return manifest


def stage_edit_harness(
    store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]
) -> Manifest:
    """S2+: state-driven EditSpec + CadQuery + validate + known-camera render."""
    manifest.stage = "step4_edit_harness"
    handler = ctx.get("edit_harness")
    if handler is not None:
        return handler(store, manifest, ctx)

    from harness.pipeline.edit_harness.loop import DIR_HARNESS

    (store.root / DIR_HARNESS).mkdir(parents=True, exist_ok=True)
    skeleton = {
        "version": 1,
        "edit_type": "modify",
        "operations": [],
        "notes": "stub harness (no media)",
    }
    path = store.write_text(
        REL_EDIT_JSON, json.dumps(skeleton, indent=2, ensure_ascii=False) + "\n"
    )
    code = (
        "import cadquery as cq\n"
        "result = cq.importers.importStep(INPUT_STEP) "
        "if 'INPUT_STEP' in globals() else cq.Workplane('XY').box(10, 10, 10)\n"
    )
    py_path = store.write_text(REL_EDIT_PY, code)
    step_path = store.write_text(REL_OUTPUT_STEP, "ISO-10303-21;\n/* stub STEP */\n")
    stl_path = store.write_text(REL_OUTPUT_STL, "solid stub\nendsolid stub\n")
    png = _minimal_png(16, 16, (220, 220, 220))
    res_path = store.write_bytes(REL_RESULT, png)
    manifest.artifacts["edit_json"] = str(path.resolve())
    manifest.artifacts["edit_py"] = str(py_path.resolve())
    manifest.artifacts["output_step"] = str(step_path.resolve())
    manifest.artifacts["output_stl"] = str(stl_path.resolve())
    manifest.artifacts["result_render"] = str(res_path.resolve())
    manifest.checks["edit_harness"] = {"status": "SUCCESS", "hard": "skipped_stub"}
    manifest.set_summary(f"Stub edit harness → {DIR_HARNESS}/")
    store.persist(manifest)
    return manifest


def stage_result_verify(
    store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]
) -> Manifest:
    """Re-render output.step at saved orient+base_view; agent QC vs target."""
    manifest.stage = "step7_result_verify"
    handler = ctx.get("result_verify")
    if handler is not None:
        return handler(store, manifest, ctx)

    png = _minimal_png(16, 16, (220, 220, 220))
    path = store.write_bytes(REL_RESULT, png)
    manifest.artifacts["result_render"] = str(path.resolve())
    manifest.checks["result_qc"] = {
        "hard": "skipped_stub",
        "detail": "Phase 1 stub result verify",
    }
    manifest.sessions.setdefault("S4_result_qc", "stub")
    manifest.set_summary(f"Stub result verify → {REL_RESULT}")
    store.persist(manifest)
    return manifest


# Legacy linear S2–S4 (edit.json → code → exec → QC). Kept for --legacy-s2.
LEGACY_STAGE_ORDER: list[tuple[str, StageFn]] = [
    ("step1_ingest", stage_ingest),
    ("step2_views", stage_cad_views),
    ("step3_target_render", stage_edit_visualizer),
    ("step4_edit_json", stage_edit_json),
    ("step5_cadquery_code", stage_cadquery_code),
    ("step6_exec", stage_cadquery_exec),
    ("step7_result_verify", stage_result_verify),
]

# Default: S0/S1 unchanged, then a state-driven harness instead of a fixed S2–S4 loop.
STAGE_ORDER: list[tuple[str, StageFn]] = [
    ("step1_ingest", stage_ingest),
    ("step2_views", stage_cad_views),
    ("step3_target_render", stage_edit_visualizer),
    ("step4_edit_harness", stage_edit_harness),
]


def register_cad_tools(registry: Any | None = None) -> Any:
    """Optionally register CAD pipeline tools on a ToolRegistry.

    Not mounted on the main REPL by default — CLI may call this explicitly.
    """
    if registry is None:
        from harness.tools.registry import ToolRegistry

        registry = ToolRegistry()

    def render_cad_views(step_path: str, out_dir: str, max_retries: int = 3) -> str:
        from harness.pipeline.skills.cad_views import render_cad_views as _render

        result = _render(step_path, out_dir, max_retries=int(max_retries))
        return json.dumps(result, ensure_ascii=False)

    def check_view_qc_tool(path: str) -> str:
        from harness.pipeline.tools.view_qc import check_view_qc

        return json.dumps(check_view_qc(path).to_dict(), ensure_ascii=False)

    def run_cadquery_tool(edit_py: str, out_step: str, out_stl: str) -> str:
        from harness.pipeline.skills.cadquery_exec import run_cadquery_script

        return json.dumps(
            run_cadquery_script(edit_py, out_step, out_stl),
            ensure_ascii=False,
        )

    registry.register(
        {
            "type": "function",
            "function": {
                "name": "render_cad_views",
                "description": "Render 4 fixed VTK shaded views after upright orient (iso+6 ortho probes) + hard QC.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "step_path": {"type": "string"},
                        "out_dir": {"type": "string"},
                        "max_retries": {"type": "integer"},
                    },
                    "required": ["step_path", "out_dir"],
                },
            },
        },
        render_cad_views,
    )
    registry.register(
        {
            "type": "function",
            "function": {
                "name": "check_view_qc",
                "description": "Hard QC a view PNG: white 5px border, centered, occupancy≥60%.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        check_view_qc_tool,
    )
    registry.register(
        {
            "type": "function",
            "function": {
                "name": "run_cadquery",
                "description": "Execute edit.py to export output.step and output.stl.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "edit_py": {"type": "string"},
                        "out_step": {"type": "string"},
                        "out_stl": {"type": "string"},
                    },
                    "required": ["edit_py", "out_step", "out_stl"],
                },
            },
        },
        run_cadquery_tool,
    )
    return registry
