"""cadquery_exec skill: run edit.py in a subprocess → output.step / output.stl."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any


RUNNER_TEMPLATE = textwrap.dedent(
    """\
    import runpy
    import sys
    from pathlib import Path

    edit_py = Path({edit_py!r})
    out_step = Path({out_step!r})
    out_stl = Path({out_stl!r})
    ns = runpy.run_path(str(edit_py))

    result = ns.get("result") or ns.get("solid") or ns.get("part") or ns.get("model")
    if result is None:
        raise SystemExit("edit.py must define result/solid/part/model")

    try:
        import cadquery as cq
    except ImportError as exc:
        raise SystemExit(f"cadquery required: {{exc}}") from exc

    # Accept Workplane or Shape.  CadQuery's exporter may silently export only
    # one object when a Workplane stack contains an imported compound plus
    # detached solids added with ``result.add(...)``.  Preserve every stack
    # object by normalizing multi-object Workplanes to one Compound.  This is
    # output plumbing only: the model still decides all CAD operations.
    shape = result
    if hasattr(result, "vals"):
        values = [value for value in result.vals() if isinstance(value, cq.Shape)]
        if len(values) == 1:
            shape = values[0]
        elif len(values) > 1:
            shape = cq.Compound.makeCompound(values)
    cq.exporters.export(shape, str(out_step))
    cq.exporters.export(shape, str(out_stl))
    print(f"exported {{out_step}} {{out_stl}}")
    """
)


def run_cadquery_script(
    edit_py: Path | str,
    out_step: Path | str,
    out_stl: Path | str,
    *,
    timeout: float = 120.0,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    edit_py = Path(edit_py)
    out_step = Path(out_step)
    out_stl = Path(out_stl)
    out_step.parent.mkdir(parents=True, exist_ok=True)

    runner = RUNNER_TEMPLATE.format(
        edit_py=str(edit_py.resolve()),
        out_step=str(out_step.resolve()),
        out_stl=str(out_stl.resolve()),
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            cwd=str(cwd) if cwd else str(edit_py.parent),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": exc.stdout or "",
            "stderr": f"timeout after {timeout}s",
            "output_step": str(out_step),
            "output_stl": str(out_stl),
        }

    ok = (
        proc.returncode == 0
        and out_step.is_file()
        and out_step.stat().st_size > 0
    )
    return {
        "ok": ok,
        "exit_code": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "output_step": str(out_step.resolve()) if out_step.exists() else str(out_step),
        "output_stl": str(out_stl.resolve()) if out_stl.exists() else str(out_stl),
    }
