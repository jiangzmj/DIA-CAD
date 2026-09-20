"""Discover single-run or batch CAD-edit input cases under ``input/``.

Supported layouts
-----------------
**Single (legacy)** — flat files under ``input/``::

    input/input.step
    input/description.txt

**Batch (long test)** — one subdirectory per case::

    input/
      01/
        input.step          # or any *.step / *.stp
        description.txt
      02/
        ...
      16/
        input.step
        description.txt

Case id = subdirectory name (used as default ``run_id``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

STEP_SUFFIXES = {".step", ".stp"}
DESC_NAMES = ("description.txt", "description.md", "edit.txt", "prompt.txt")


@dataclass(frozen=True)
class InputCase:
    """One pipeline run's STEP + description."""

    case_id: str
    input_step: Path
    description_txt: Path
    source_dir: Path

    def run_id(self, prefix: str | None = None) -> str:
        safe = re.sub(r"[^\w.\-]+", "_", self.case_id).strip("_") or "case"
        if prefix:
            pref = re.sub(r"[^\w.\-]+", "_", prefix).strip("_")
            return f"{pref}_{safe}" if pref else safe
        return safe


def _natural_key(name: str) -> tuple:
    """Sort ``01``, ``2``, ``10`` in human order."""
    parts = re.split(r"(\d+)", name)
    key: list = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


def _find_step(directory: Path) -> Path | None:
    preferred = directory / "input.step"
    if preferred.is_file():
        return preferred
    steps = sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in STEP_SUFFIXES
        ),
        key=lambda p: _natural_key(p.name),
    )
    return steps[0] if steps else None


def _find_description(directory: Path) -> Path | None:
    for name in DESC_NAMES:
        path = directory / name
        if path.is_file():
            return path
    txts = sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".txt", ".md"}
            and p.name.lower() not in {"readme.md", "readme.txt"}
        ),
        key=lambda p: _natural_key(p.name),
    )
    return txts[0] if txts else None


def discover_case_dir(directory: Path, *, case_id: str | None = None) -> InputCase | None:
    """Return a case if ``directory`` contains STEP + description, else None."""
    directory = directory.resolve()
    if not directory.is_dir():
        return None
    step = _find_step(directory)
    desc = _find_description(directory)
    if step is None or desc is None:
        return None
    cid = case_id if case_id is not None else directory.name
    return InputCase(
        case_id=cid,
        input_step=step,
        description_txt=desc,
        source_dir=directory,
    )


def discover_batch_cases(inputs_dir: Path) -> list[InputCase]:
    """Scan ``inputs_dir`` for case subdirectories (skips incomplete folders)."""
    inputs_dir = inputs_dir.resolve()
    if not inputs_dir.is_dir():
        raise FileNotFoundError(f"inputs dir not found: {inputs_dir}")

    cases: list[InputCase] = []
    for child in sorted(inputs_dir.iterdir(), key=lambda p: _natural_key(p.name)):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.lower() in {"__pycache__"}:
            continue
        case = discover_case_dir(child)
        if case is not None:
            cases.append(case)
    return cases


def resolve_inputs(
    inputs_dir: Path,
    *,
    batch: bool,
    input_step: Path | None = None,
    description_txt: Path | None = None,
) -> list[InputCase]:
    """Resolve one or many cases.

    - ``batch=True``: only subdirectory cases under ``inputs_dir``.
    - ``batch=False`` with explicit paths: single case from those paths.
    - ``batch=False`` without paths: prefer flat ``input.step``+``description.txt``;
      if missing but case subdirs exist, use those (auto batch).
    """
    inputs_dir = inputs_dir.resolve()

    if batch:
        cases = discover_batch_cases(inputs_dir)
        if not cases:
            raise FileNotFoundError(
                f"no complete cases under {inputs_dir} "
                "(need subdirs each with a .step/.stp and description.txt)"
            )
        return cases

    if input_step is not None and description_txt is not None:
        return [
            InputCase(
                case_id=input_step.parent.name
                if input_step.parent != inputs_dir
                else "single",
                input_step=input_step.resolve(),
                description_txt=description_txt.resolve(),
                source_dir=input_step.parent.resolve(),
            )
        ]

    flat = discover_case_dir(inputs_dir, case_id="single")
    if flat is not None:
        return [flat]

    cases = discover_batch_cases(inputs_dir)
    if cases:
        return cases

    raise FileNotFoundError(
        f"no inputs found under {inputs_dir}: expected flat "
        "input.step+description.txt, or case subdirs like 01/, 02/, ..."
    )
