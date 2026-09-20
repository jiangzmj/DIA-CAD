"""Subprocess CadQuery execution with isolation. Not a security sandbox."""

from __future__ import annotations

import ast
import hashlib
import re
import shutil
import time
from pathlib import Path
from typing import Any

from harness.pipeline.edit_harness.spec import ExecutionResult
from harness.pipeline.skills.cadquery_exec import run_cadquery_script

BANNED_IMPORT_ROOTS = {"subprocess", "socket", "requests", "httpx"}
BANNED_CALLS = {"exec", "eval", "__import__"}

# Injected into every candidate script so the model can boolean assemblies
# without ``base.cut(tool)`` fragmenting unrelated solids.
CUT_OVERLAPPING_SNIPPET = '''
def cut_overlapping(base, tool):
    """Cut only solids whose bounding box overlaps the tool. Prefer this over base.cut(tool) on assemblies."""
    import cadquery as cq
    def _solids(shape):
        if shape is None:
            return []
        if hasattr(shape, "solids") and hasattr(shape, "vals"):
            try:
                return list(shape.solids().vals())
            except Exception:
                pass
        wp = cq.Workplane("XY").newObject([shape])
        try:
            return list(wp.solids().vals())
        except Exception:
            return [shape]
    def _as_solid(shape):
        solids = _solids(shape)
        if not solids:
            raise ValueError("cut_overlapping: tool has no solids")
        if len(solids) == 1:
            return solids[0]
        return cq.Compound.makeCompound(solids)
    def _overlap(a, b, tol=1e-6):
        return not (
            a.xmax < b.xmin - tol or b.xmax < a.xmin - tol
            or a.ymax < b.ymin - tol or b.ymax < a.ymin - tol
            or a.zmax < b.zmin - tol or b.zmax < a.zmin - tol
        )
    tool_s = _as_solid(tool)
    bb = tool_s.BoundingBox()
    keep = []
    for solid in _solids(base):
        if _overlap(solid.BoundingBox(), bb):
            keep.extend(_solids(solid.cut(tool_s)))
        else:
            keep.append(solid)
    if not keep:
        raise ValueError("cut_overlapping removed every solid")
    if len(keep) == 1:
        return cq.Workplane("XY").newObject(keep)
    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(keep)])
'''


def sha256_file(path: Path | str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lint_cadquery_code(code: str) -> dict[str, Any]:
    """Syntax + banned-module check. Empty code is a hard fail."""
    if not (code or "").strip():
        return {"ok": False, "error": "empty cadquery code", "failure_type": "INVALID_RESPONSE"}
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"ok": False, "error": str(exc), "failure_type": "CODE_ERROR"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED_IMPORT_ROOTS:
                    return {
                        "ok": False,
                        "error": f"banned import {alias.name}",
                        "failure_type": "INVALID_RESPONSE",
                    }
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in BANNED_IMPORT_ROOTS:
                return {
                    "ok": False,
                    "error": f"banned import {node.module}",
                    "failure_type": "INVALID_RESPONSE",
                }
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in BANNED_CALLS:
                return {
                    "ok": False,
                    "error": f"banned call {name}",
                    "failure_type": "INVALID_RESPONSE",
                }
    unstable = _unstable_selector_warnings(code)
    return {"ok": True, "error": None, "failure_type": None, "warnings": unstable}


def _unstable_selector_warnings(code: str) -> list[str]:
    warnings: list[str] = []
    if ".vals()[" in code or "faces[" in code or "solids().vals()[" in code:
        warnings.append(
            "Code uses index-based face/solid access; prefer type/axis/radius/bbox selectors."
        )
    return warnings


def prepare_script(
    code: str,
    dest: Path,
    *,
    input_step: Path | str,
) -> Path:
    """Write candidate code with INPUT_STEP and cut_overlapping injected."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    assigned = bool(re.search(r"(?m)^\s*INPUT_STEP\s*=", code))
    if not assigned:
        parts.append(f"INPUT_STEP = {str(Path(input_step).resolve())!r}")
        parts.append(
            "# Harness sets INPUT_STEP. Load the reference with cq.importers.importStep(INPUT_STEP)."
        )
    parts.append(CUT_OVERLAPPING_SNIPPET.strip())
    parts.append(code if code.endswith("\n") else code + "\n")
    dest.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return dest


def execute_cadquery(
    code: str,
    input_step_path: Path | str,
    output_directory: Path | str,
    *,
    timeout: float = 120.0,
) -> ExecutionResult:
    """Run generated code in a subprocess. Never overwrites a caller-held valid STEP."""
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_path = out_dir / "generated_cadquery.py"
    out_step = out_dir / "candidate.step"
    out_stl = out_dir / "candidate.stl"
    if out_step.exists():
        out_step.unlink()
    if out_stl.exists():
        out_stl.unlink()

    lint = lint_cadquery_code(code)
    if not lint["ok"]:
        prepare_script(code, code_path, input_step=input_step_path)
        return ExecutionResult(
            success=False,
            exit_code=None,
            stderr=str(lint["error"]),
            output_step_path=None,
            duration_seconds=0.0,
            failure_type=str(lint.get("failure_type") or "CODE_ERROR"),
            code_path=str(code_path.resolve()),
        )

    prepare_script(code, code_path, input_step=input_step_path)
    t0 = time.time()
    raw = run_cadquery_script(
        code_path,
        out_step,
        out_stl,
        timeout=timeout,
        cwd=out_dir,
    )
    duration = time.time() - t0
    ok = bool(raw.get("ok")) and out_step.is_file() and out_step.stat().st_size > 0
    failure: str | None = None
    if not ok:
        stderr = str(raw.get("stderr") or "")
        if "timeout" in stderr.lower():
            failure = "CODE_ERROR"
        elif not out_step.is_file() or out_step.stat().st_size == 0:
            failure = "BUILD_ERROR"
        else:
            failure = "CODE_ERROR"
        if out_step.exists() and (not ok):
            # Do not leave a truncated/invalid candidate as if it were usable.
            try:
                out_step.unlink()
            except OSError:
                pass
    return ExecutionResult(
        success=ok,
        exit_code=raw.get("exit_code"),
        stdout=str(raw.get("stdout") or ""),
        stderr=str(raw.get("stderr") or ""),
        output_step_path=str(out_step.resolve()) if ok else None,
        output_stl_path=str(out_stl.resolve()) if ok and out_stl.is_file() else None,
        duration_seconds=round(duration, 3),
        failure_type=failure,
        code_path=str(code_path.resolve()),
    )


def copy_valid_step(src: Path | str, dest: Path | str) -> Path:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_path)
    return dest_path
