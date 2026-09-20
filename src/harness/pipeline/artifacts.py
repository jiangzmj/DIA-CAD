"""Artifact directory layout helpers for CAD edit runs.

Canonical layout under ``workspace/artifacts/<run_id>/``::

    manifest.json          # run status, checks, absolute paths
    README.md              # human index of this run
    01_input/              # step1 — copied source
      input.step
      description.txt
    02_views/              # step2 — 4 PNGs after uprighting
                           # (iso / z_corner / x_corner / y_corner)
                           # plus _orient_probe_*.png (iso + 6 ortho, pre-upright)
    03_target_render/      # step3 — two target intent images
      target_<view>.png    # one per chosen base_view
    04_structure/          # step4 — structured edit JSON
      edit.json
    05_code/               # step5 — CadQuery script
      edit.py
    06_output/             # step6 — exported geometry
      output.step
      output.stl
    07_result_verify/      # step7 — same-pose result renders + QC
      result_<view>.png    # one per chosen base_view
      result.png           # copy of the first (compat)
    08_harness/            # S2+ state-driven loop (iterations + final report)
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from harness.pipeline.manifest import Manifest


# Stage-ordered directories (relative to run root).
DIR_INPUT = "01_input"
DIR_VIEWS = "02_views"
DIR_TARGET = "03_target_render"
DIR_STRUCTURE = "04_structure"
DIR_CODE = "05_code"
DIR_OUTPUT = "06_output"
DIR_RESULT = "07_result_verify"
DIR_HARNESS = "08_harness"

SUBDIRS = (
    DIR_INPUT,
    DIR_VIEWS,
    DIR_TARGET,
    DIR_STRUCTURE,
    DIR_CODE,
    DIR_OUTPUT,
    DIR_RESULT,
    DIR_HARNESS,
)

# Stable relative paths used by stages / handlers.
REL_INPUT_STEP = f"{DIR_INPUT}/input.step"
REL_DESCRIPTION = f"{DIR_INPUT}/description.txt"
REL_TARGET = f"{DIR_TARGET}/target.png"


def rel_target_view(view: str) -> str:
    return f"{DIR_TARGET}/target_{view}.png"


def rel_result_view(view: str) -> str:
    return f"{DIR_RESULT}/result_{view}.png"


REL_EDIT_JSON = f"{DIR_STRUCTURE}/edit.json"
REL_INTENT_JSON = f"{DIR_STRUCTURE}/intent.json"
REL_EDIT_PY = f"{DIR_CODE}/edit.py"
REL_FEATURE_PROBE = f"{DIR_CODE}/feature_probe.json"
REL_OUTPUT_STEP = f"{DIR_OUTPUT}/output.step"
REL_OUTPUT_STL = f"{DIR_OUTPUT}/output.stl"
REL_RESULT = f"{DIR_RESULT}/result.png"


LAYOUT_README = """# CAD edit run `{run_id}`

Pipeline artifacts are grouped by stage. Open folders in order.

| Dir | Stage | Contents |
|-----|-------|----------|
| `01_input/` | 1 ingest | `input.step`, `description.txt` |
| `02_views/` | 2 views | 4 PNGs after upright + `_orient_probe_*` (iso + 6 ortho) |
| `03_target_render/` | 3 intent | `target_<view>.png` ×2 (chosen base views) |
| `04_structure/` | 4 JSON | `edit.json` |
| `05_code/` | 5 code | `edit.py` (CadQuery) |
| `06_output/` | 6 export | `output.step`, `output.stl` |
| `07_result_verify/` | 7 verify | `result_<view>.png` (same pose as each base_view) |
| `08_harness/` | S2+ loop | iterations, transition_history, final report |

Machine-readable index: [`manifest.json`](manifest.json).
"""


class ArtifactStore:
    """Creates and updates ``workspace/artifacts/<run_id>/`` trees."""

    def __init__(self, artifacts_root: Path, run_id: str | None = None):
        self.artifacts_root = Path(artifacts_root)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.root = self.artifacts_root / self.run_id

    def init_run(
        self,
        input_step: Path,
        description_txt: Path,
    ) -> Manifest:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)

        step_dst = self.path(REL_INPUT_STEP)
        desc_dst = self.path(REL_DESCRIPTION)
        shutil.copy2(input_step, step_dst)
        shutil.copy2(description_txt, desc_dst)
        self.write_layout_readme()

        manifest = Manifest(
            run_id=self.run_id,
            root=str(self.root.resolve()),
            stage="step1_ingest",
            status="running",
            input_step=str(step_dst.resolve()),
            description_txt=str(desc_dst.resolve()),
            artifacts={
                "layout": {d: d for d in SUBDIRS},
                "input_step": str(step_dst.resolve()),
                "description_txt": str(desc_dst.resolve()),
                "views_dir": str(self.path(DIR_VIEWS).resolve()),
                "views": {},
                "target_render": None,
                "target_renders": {},
                "base_views": [],
                "edit_json": None,
                "edit_py": None,
                "output_step": None,
                "output_stl": None,
                "result_render": None,
            },
        )
        manifest.set_summary(
            f"Ingested input into {DIR_INPUT}/ under {self.root}"
        )
        manifest.save()
        return manifest

    def write_layout_readme(self) -> Path:
        return self.write_text("README.md", LAYOUT_README.format(run_id=self.run_id))

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def write_bytes(self, relative: str, data: bytes) -> Path:
        out = self.path(relative)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out

    def write_text(self, relative: str, text: str) -> Path:
        out = self.path(relative)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return out

    def read_text(self, relative: str) -> str:
        return self.path(relative).read_text(encoding="utf-8")

    def read_bytes(self, relative: str) -> bytes:
        return self.path(relative).read_bytes()

    def view_paths(self) -> dict[str, str]:
        from harness.pipeline.manifest import VIEW_NAMES

        return {
            name: str(self.path(DIR_VIEWS, f"{name}.png").resolve())
            for name in VIEW_NAMES
        }

    def persist(self, manifest: Manifest) -> None:
        manifest.root = str(self.root.resolve())
        manifest.save()


def read_artifact(root: Path, relative: str) -> Any:
    path = Path(root) / relative
    if path.suffix.lower() in {".json"}:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def write_artifact(root: Path, relative: str, content: str | bytes) -> Path:
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path
