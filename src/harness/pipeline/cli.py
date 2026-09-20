"""CLI for CAD edit pipeline.

Usage:
  # Single: <project-root>/input/input.step + description.txt
  PYTHONPATH=src python -m harness.pipeline

  # Batch long-test: one subdir per case under input/
  #   input/01/{input.step,description.txt} ... input/16/...
  PYTHONPATH=src python -m harness.pipeline --batch
  PYTHONPATH=src python -m harness.pipeline --batch --run-id longtest

  PYTHONPATH=src python -m harness.pipeline.cli \\
    --input path/to/input.step --description path/to/description.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

from harness.config import load_settings
from harness.pipeline.cad_edit import CADEditPipeline
from harness.pipeline.inputs import InputCase, resolve_inputs
from harness.pipeline.runtime import build_pipeline_context
from harness.providers.stub_media import StubMediaProvider
from harness.providers.usage import TokenMeter, bind_meter, format_duration


def _resolve_under_root(path: str | Path, root: Path) -> Path:
    """Resolve a path; relative paths are under project root (settings.root)."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    return p


def build_parser(
    default_input: str = "input/input.step",
    default_description: str = "input/description.txt",
    default_inputs_dir: str = "input",
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run CAD edit pipeline")
    p.add_argument(
        "--input",
        default=None,
        help=f"Path to input.step (default: <root>/{default_input})",
    )
    p.add_argument(
        "--description",
        default=None,
        help=f"Path to description.txt (default: <root>/{default_description})",
    )
    p.add_argument(
        "--inputs-dir",
        default=None,
        help=f"Cases root (default: <root>/{default_inputs_dir}). "
        "Batch mode reads one case per subdirectory.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Run every complete case subdir under --inputs-dir "
        "(e.g. input/01 .. input/16)",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Fixed run id (single). In --batch, used as prefix: <run-id>_<case>",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="In batch mode, abort after the first failed case (default: continue)",
    )
    p.add_argument(
        "--artifacts-root",
        default=None,
        help="Override artifacts root (default: workspace/artifacts)",
    )
    p.add_argument(
        "--stub",
        action="store_true",
        help="Force Phase-1 stub stages only (no real cad_views / sessions / exec)",
    )
    p.add_argument(
        "--stub-media",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use stub media for image/video (default: config pipeline.stub_media)",
    )
    p.add_argument(
        "--register-tools",
        action="store_true",
        help="Register optional CAD tools on a throwaway registry (smoke)",
    )
    p.add_argument(
        "--legacy-s2",
        action="store_true",
        help="Use the old linear S2 JSON → S3 code → exec → S4 QC loop",
    )
    p.add_argument(
        "--list-cases",
        action="store_true",
        help="List discovered input cases and exit",
    )
    return p


def _print_layout(manifest) -> None:
    print(f"run_id={manifest.run_id}")
    print(f"status={manifest.status} stage={manifest.stage}")
    print(f"root={manifest.root}")
    print("layout:")
    for name in (
        "01_input",
        "02_views",
        "03_target_render",
        "04_structure",
        "05_code",
        "06_output",
        "07_result_verify",
        "08_harness",
    ):
        p = Path(manifest.root) / name
        mark = "✓" if p.is_dir() else "·"
        print(f"  {mark} {name}/")
    print("  · README.md  · manifest.json")
    if manifest.error:
        print(f"error={manifest.error}", file=sys.stderr)


USAGE_CSV_FIELDS = (
    "case_id",
    "run_id",
    "status",
    "stage",
    "ok",
    "elapsed_s",
    "time",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "calls",
    "api_calls",
    "estimated_calls",
    "estimated",
    "error",
    "root",
)


def _usage_row(result: dict) -> dict:
    elapsed = float(result.get("elapsed_s") or 0)
    error = result.get("error")
    if error is not None:
        error = " ".join(str(error).split())[:300]
    return {
        "case_id": result.get("case_id") or "",
        "run_id": result.get("run_id") or "",
        "status": result.get("status") or "",
        "stage": result.get("stage") or "",
        "ok": bool(result.get("ok")),
        "elapsed_s": round(elapsed, 1),
        "time": format_duration(elapsed),
        "total_tokens": int(result.get("total_tokens") or 0),
        "input_tokens": int(result.get("input_tokens") or 0),
        "output_tokens": int(result.get("output_tokens") or 0),
        "reasoning_tokens": int(result.get("reasoning_tokens") or 0),
        "calls": int(result.get("calls") or 0),
        "api_calls": int(result.get("api_calls") or 0),
        "estimated_calls": int(result.get("estimated_calls") or 0),
        "estimated": bool(result.get("estimated")),
        "error": error or "",
        "root": result.get("root") or "",
    }


def _write_usage_csv(path: Path, results: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_usage_row(r) for r in results]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=USAGE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path.resolve()


def _result_from_snap(
    *,
    case_id: str,
    run_id: str | None,
    status: str,
    stage: str,
    root: str | None,
    error: str | None,
    ok: bool,
    elapsed_s: float,
    snap: dict,
) -> dict:
    return {
        "case_id": case_id,
        "run_id": run_id,
        "status": status,
        "stage": stage,
        "root": root,
        "error": error,
        "ok": ok,
        "elapsed_s": round(elapsed_s, 1),
        "tokens": snap.get("total_tokens") or 0,
        "total_tokens": snap.get("total_tokens") or 0,
        "input_tokens": snap.get("input_tokens") or 0,
        "output_tokens": snap.get("output_tokens") or 0,
        "reasoning_tokens": snap.get("reasoning_tokens") or 0,
        "calls": snap.get("calls") or 0,
        "api_calls": snap.get("api_calls") or 0,
        "estimated_calls": snap.get("estimated_calls") or 0,
        "estimated": bool(snap.get("estimated")),
    }


def _print_run_footer(
    meter: TokenMeter | None,
    elapsed_s: float,
    *,
    csv_path: Path | None = None,
) -> None:
    print("任务已完成", flush=True)
    if meter is not None:
        print(meter.format_line(), flush=True)
    else:
        print("tokens=0 (no meter)", flush=True)
    print(f"time={format_duration(elapsed_s)}", flush=True)
    if csv_path is not None:
        print(f"csv={csv_path}", flush=True)


def _build_ctx(settings, args, pipe_cfg):
    use_real = not args.stub
    stub_media = (
        bool(args.stub_media)
        if args.stub_media is not None
        else bool(pipe_cfg.stub_media)
    )

    ctx = build_pipeline_context(settings, use_real=use_real)
    ctx["stub_media"] = stub_media
    if stub_media:
        ctx["media_provider"] = StubMediaProvider(settings.workspace)
    else:
        from harness.providers.gpt_media import GptMediaProvider

        try:
            ctx["media_provider"] = GptMediaProvider.from_settings(settings)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    from harness.agents.helper import HelperAgent

    ctx["helper"] = HelperAgent(
        ctx["media_provider"],
        workspace=settings.workspace,
        agent_id="cad_edit",
    )
    ctx["cad_views_max_retries"] = settings.pipeline.cad_views_max_retries
    ctx["s1_max_retries"] = settings.pipeline.s1_max_retries
    ctx["s3_max_retries"] = settings.pipeline.s3_max_retries
    ctx["result_qc_max_retries"] = settings.pipeline.result_qc_max_retries
    ctx["result_qc_exec_retries"] = settings.pipeline.result_qc_exec_retries
    ctx["view_size"] = settings.pipeline.view_size
    ctx["use_legacy_s2"] = bool(getattr(args, "legacy_s2", False)) or not bool(
        getattr(settings.pipeline, "use_edit_harness", True)
    )
    meter = ctx.get("token_meter") or TokenMeter()
    ctx["token_meter"] = meter
    bind_meter(ctx.get("media_provider"), meter)
    return ctx


def _run_one(
    pipe: CADEditPipeline,
    case: InputCase,
    *,
    run_id: str | None,
) -> dict:
    t0 = time.time()
    meter = pipe.ctx.get("token_meter")
    if meter is not None:
        meter.reset()
        bind_meter(pipe.ctx.get("media_provider"), meter)
    print(
        f"\n======== case={case.case_id} run_id={run_id or '(auto)'} "
        f"step={case.input_step.name} ========",
        flush=True,
    )
    try:
        manifest = pipe.run(
            case.input_step,
            case.description_txt,
            run_id=run_id,
        )
        _print_layout(manifest)
        elapsed = time.time() - t0
        snap = meter.snapshot() if meter is not None else {}
        ok = manifest.status == "done" and not manifest.error
        result = _result_from_snap(
            case_id=case.case_id,
            run_id=manifest.run_id,
            status=manifest.status,
            stage=manifest.stage,
            root=str(manifest.root),
            error=manifest.error,
            ok=ok,
            elapsed_s=elapsed,
            snap=snap,
        )
        try:
            _write_usage_csv(Path(manifest.root) / "usage.csv", [result])
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not write run usage csv: {exc}", file=sys.stderr)
        _print_run_footer(meter, elapsed)
        return result
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"case={case.case_id} FAILED: {exc}", file=sys.stderr, flush=True)
        elapsed = time.time() - t0
        snap = meter.snapshot() if meter is not None else {}
        result = _result_from_snap(
            case_id=case.case_id,
            run_id=run_id,
            status="failed",
            stage="exception",
            root=None,
            error=str(exc),
            ok=False,
            elapsed_s=elapsed,
            snap=snap,
        )
        _print_run_footer(meter, elapsed)
        return result


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    pipe_cfg = settings.pipeline
    args = build_parser(
        default_input=pipe_cfg.default_input,
        default_description=pipe_cfg.default_description,
        default_inputs_dir=pipe_cfg.input_dir,
    ).parse_args(argv)

    inputs_dir = _resolve_under_root(
        args.inputs_dir or pipe_cfg.input_dir,
        settings.root,
    )

    explicit_input = (
        _resolve_under_root(args.input, settings.root) if args.input else None
    )
    explicit_desc = (
        _resolve_under_root(args.description, settings.root)
        if args.description
        else None
    )
    if (explicit_input is None) ^ (explicit_desc is None):
        print(
            "error: --input and --description must be provided together",
            file=sys.stderr,
        )
        return 2

    try:
        cases = resolve_inputs(
            inputs_dir,
            batch=bool(args.batch),
            input_step=explicit_input,
            description_txt=explicit_desc,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list_cases:
        print(f"inputs_dir={inputs_dir}")
        print(f"cases={len(cases)}")
        for case in cases:
            rid = case.run_id(args.run_id if args.batch or len(cases) > 1 else None)
            print(
                f"  - {case.case_id}: step={case.input_step} "
                f"desc={case.description_txt} run_id={rid}"
            )
        return 0

    artifacts_root = (
        Path(args.artifacts_root)
        if args.artifacts_root
        else settings.workspace / settings.pipeline.artifacts_root
    )

    if args.register_tools:
        from harness.pipeline.stages import register_cad_tools
        from harness.tools.registry import ToolRegistry

        reg = ToolRegistry()
        register_cad_tools(reg)
        print(f"registered_tools={reg.names()}")

    def on_stage(name: str, manifest) -> None:
        print(
            f"[{manifest.status}] {name} — {manifest.handoff.get('summary', '')}",
            flush=True,
        )

    ctx = _build_ctx(settings, args, pipe_cfg)
    pipe = CADEditPipeline(artifacts_root, ctx=ctx, on_stage=on_stage)

    multi = len(cases) > 1 or args.batch
    results: list[dict] = []

    for case in cases:
        if multi:
            run_id: str | None = case.run_id(args.run_id)
        else:
            # Single: honor --run-id; otherwise let ArtifactStore mint one.
            # If the case came from a named subdir, use that name as run_id.
            if args.run_id:
                run_id = args.run_id
            elif case.case_id != "single":
                run_id = case.case_id
            else:
                run_id = None

        result = _run_one(pipe, case, run_id=run_id)
        results.append(result)
        if multi and (not result["ok"]) and args.stop_on_error:
            print("stop-on-error: aborting remaining cases", flush=True)
            break

    usage_csv = artifacts_root / "usage.csv"
    try:
        usage_csv = _write_usage_csv(usage_csv, results)
        print(f"csv={usage_csv}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not write usage csv: {exc}", file=sys.stderr)

    if multi:
        summary_path = artifacts_root / "_batch_summary.json"
        summary = {
            "inputs_dir": str(inputs_dir),
            "total": len(results),
            "passed": sum(1 for r in results if r["ok"]),
            "failed": sum(1 for r in results if not r["ok"]),
            "results": results,
        }
        try:
            artifacts_root.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not write batch summary: {exc}", file=sys.stderr)
        print(
            f"\n======== batch done passed={summary['passed']}/{summary['total']} "
            f"summary={summary_path} ========",
            flush=True,
        )
        for r in results:
            mark = "OK" if r["ok"] else "FAIL"
            print(
                f"  [{mark}] {r['case_id']} run_id={r['run_id']} "
                f"stage={r.get('stage')} ({r.get('elapsed_s')}s) "
                f"tokens={r.get('tokens')}",
                flush=True,
            )
        return 0 if summary["failed"] == 0 else 1

    # single
    r = results[0]
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
