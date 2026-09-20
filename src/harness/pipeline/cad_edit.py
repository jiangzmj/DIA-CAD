"""CADEditPipeline: fixed stage orchestration (no free-agent routing)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from harness.pipeline.artifacts import ArtifactStore
from harness.pipeline.manifest import Manifest
from harness.pipeline.stages import (
    LEGACY_STAGE_ORDER,
    STAGE_ORDER,
    StageError,
    StageFn,
)


class CADEditPipeline:
    def __init__(
        self,
        artifacts_root: Path,
        ctx: dict[str, Any] | None = None,
        on_stage: Callable[[str, Manifest], None] | None = None,
    ):
        self.artifacts_root = Path(artifacts_root)
        self.ctx = ctx or {}
        self.on_stage = on_stage or (lambda _name, _m: None)

    def run(
        self,
        input_step: Path | str,
        description_txt: Path | str,
        run_id: str | None = None,
    ) -> Manifest:
        input_step = Path(input_step)
        description_txt = Path(description_txt)
        if not input_step.is_file():
            raise FileNotFoundError(f"input.step not found: {input_step}")
        if not description_txt.is_file():
            raise FileNotFoundError(f"description.txt not found: {description_txt}")

        store = ArtifactStore(self.artifacts_root, run_id=run_id)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        manifest = store.init_run(input_step, description_txt)
        print(
            f"[running] start run_id={manifest.run_id} root={manifest.root}",
            flush=True,
        )
        self.on_stage("step1_ingest", manifest)

        order = (
            LEGACY_STAGE_ORDER
            if self.ctx.get("use_legacy_s2")
            else STAGE_ORDER
        )
        for name, fn in order:
            if name == "step1_ingest":
                # init_run already covered ingest; still invoke for consistency
                try:
                    print(f"[running] entering {name}…", flush=True)
                    manifest = fn(store, manifest, self.ctx)
                    self.on_stage(name, manifest)
                except StageError as exc:
                    return self._fail(store, manifest, name, exc)
                continue
            try:
                print(f"[running] entering {name}…", flush=True)
                manifest = fn(store, manifest, self.ctx)
                self.on_stage(name, manifest)
            except StageError as exc:
                return self._fail(store, manifest, name, exc)
            except Exception as exc:  # noqa: BLE001 — surface as stage failure
                return self._fail(store, manifest, name, StageError(str(exc)))

        if manifest.status in {"needs_review", "failed"}:
            self._attach_usage(manifest)
            store.persist(manifest)
            print(f"[{manifest.status}] {manifest.handoff.get('summary', '')}", flush=True)
            return manifest

        if manifest.checks.get("edit_harness"):
            self._attach_usage(manifest)
            if not manifest.handoff.get("summary"):
                manifest.set_summary(
                    f"Pipeline complete. artifacts={manifest.root} "
                    f"output={manifest.artifacts.get('output_step')}"
                )
            store.persist(manifest)
            print(f"[done] {manifest.handoff.get('summary', '')}", flush=True)
            return manifest

        manifest.stage = "done"
        manifest.status = "done"
        self._attach_usage(manifest)
        manifest.set_summary(
            f"Pipeline complete. artifacts={manifest.root} "
            f"output={manifest.artifacts.get('output_step')}"
        )
        store.persist(manifest)
        print(f"[done] {manifest.handoff.get('summary', '')}", flush=True)
        return manifest

    def _attach_usage(self, manifest: Manifest) -> None:
        meter = self.ctx.get("token_meter")
        if meter is None:
            return
        snap = meter.snapshot() if hasattr(meter, "snapshot") else None
        if snap:
            manifest.checks["usage"] = snap

    def _fail(
        self,
        store: ArtifactStore,
        manifest: Manifest,
        stage: str,
        exc: StageError | Exception,
    ) -> Manifest:
        manifest.stage = stage
        manifest.status = "failed"
        manifest.error = str(exc)
        self._attach_usage(manifest)
        manifest.set_summary(f"Failed at {stage}: {exc}")
        store.persist(manifest)
        print(f"[failed] {stage} — {exc}", flush=True)
        return manifest


__all__ = ["CADEditPipeline", "StageError", "StageFn"]
