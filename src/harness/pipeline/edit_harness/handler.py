"""Pipeline stage handler: S2+ edit harness."""

from __future__ import annotations

from typing import Any, Callable

from harness.config import Settings
from harness.pipeline.artifacts import ArtifactStore
from harness.pipeline.edit_harness.loop import run_edit_harness
from harness.pipeline.manifest import Manifest
from harness.pipeline.stages import StageError


def make_edit_harness_handler(settings: Settings) -> Callable[..., Manifest]:
    def handler(store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]) -> Manifest:
        ctx = dict(ctx)
        ctx.setdefault("settings", settings)
        result = run_edit_harness(store, manifest, ctx)
        status = str((result.checks.get("edit_harness") or {}).get("status") or "")
        if result.status == "failed" and status not in {"NEEDS_REVIEW", "UNSUPPORTED_EDIT"}:
            raise StageError(result.error or "edit harness failed")
        if status == "UNSUPPORTED_EDIT":
            raise StageError(result.error or "UNSUPPORTED_EDIT")
        return result

    return handler
