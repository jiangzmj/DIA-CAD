"""Offline replay of four-layer gates against old-pipeline artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.pipeline.edit_harness.compare import (
    compare_before_after,
    compare_images,
    estimate_view_alignment,
    visual_gates,
)
from harness.pipeline.edit_harness.inspect import inspect_step
from harness.pipeline.edit_harness.operations import edit_class_from_legacy
from harness.pipeline.edit_harness.spec import EditSpec
from harness.pipeline.edit_harness.validate import (
    _max_bbox_rel,
    compare_geometry,
    validate_step,
)

SKIP_PREFIXES = ("harness0818", "_")


@dataclass
class GateThresholds:
    silhouette_iou_min: float = 0.55
    silhouette_iou_floor: float = 0.40
    cross_renderer_iou_min: float = 0.28
    cross_renderer_iou_floor: float = 0.15
    preservation_bbox_rel_tol: float = 0.50
    preservation_bbox_rel_tol_replace: float = 1.20
    volume_explode_ratio: float = 8.0
    volume_explode_ratio_global: float = 12.0
    wrecked_iou_min: float = 0.25
    view_alignment_min: float = 0.40
    region_change_iou_min: float = 0.12
    region_edge_iou_min: float = 0.12
    region_mask_max_frac: float = 0.25
    region_mask_min_frac: float = 0.001
    noop_silhouette_iou_min: float = 0.997
    noop_pixel_mae_max: float = 0.008

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass
class ArtifactRun:
    run_id: str
    root: Path
    status: str
    weight: str  # high | low | skip
    qc_verdict: str | None
    input_step: Path
    output_step: Path | None
    edit_json: Path | None
    views: dict[str, Path] = field(default_factory=dict)
    targets: dict[str, Path] = field(default_factory=dict)
    results: dict[str, Path] = field(default_factory=dict)
    ops: list[str] = field(default_factory=list)
    edit_class_name: str = "replace"


def _exists(path: Path | str | None) -> bool:
    if not path:
        return False
    p = Path(path)
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _qc_verdict(manifest: dict[str, Any]) -> str | None:
    qc = ((manifest.get("checks") or {}).get("result_qc") or {}).get("qc") or {}
    verdict = qc.get("verdict")
    if verdict in {"pass", "accept", "reject", "fail"}:
        return str(verdict)
    if qc.get("pass") is True:
        return "pass"
    if qc.get("pass") is False:
        return "reject"
    return None


def _weight(status: str, verdict: str | None, has_output: bool) -> str:
    if status == "done" and verdict in {"pass", "accept"} and has_output:
        return "high"
    if has_output:
        return "low"
    return "skip"


def legacy_edit_spec(edit_json: dict[str, Any] | None, _cls: str = "replace") -> EditSpec:
    ops = []
    for item in (edit_json or {}).get("operations") or []:
        if isinstance(item, dict) and item.get("op"):
            ops.append(str(item["op"]))
    op = ops[0] if ops else str((edit_json or {}).get("edit_type") or "modify_feature")
    return EditSpec(
        operation=op,
        edit_type=str((edit_json or {}).get("edit_type") or "modify"),
        canonical_operation=None,
        expected_changes=ops,
    )


def discover_runs(artifacts_root: Path | str) -> list[ArtifactRun]:
    root = Path(artifacts_root)
    runs: list[ArtifactRun] = []
    for man_path in sorted(root.glob("*/manifest.json")):
        name = man_path.parent.name
        if name.startswith(SKIP_PREFIXES) or name.startswith("_"):
            continue
        data = json.loads(man_path.read_text(encoding="utf-8"))
        arts = data.get("artifacts") or {}
        inp = Path(str(data.get("input_step") or arts.get("input_step") or man_path.parent / "01_input/input.step"))
        out_raw = arts.get("output_step") or (man_path.parent / "06_output/output.step")
        out = Path(str(out_raw)) if out_raw else None
        edit_raw = arts.get("edit_json") or (man_path.parent / "04_structure/edit.json")
        edit_p = Path(str(edit_raw)) if edit_raw else None
        views = {k: Path(str(v)) for k, v in dict(arts.get("views") or {}).items() if _exists(v)}
        targets = {k: Path(str(v)) for k, v in dict(arts.get("target_renders") or {}).items() if _exists(v)}
        results = {}
        for k, v in dict(arts.get("result_renders") or {}).items():
            if _exists(v):
                results[k] = Path(str(v))
        rdir = man_path.parent / "07_result_verify"
        if rdir.is_dir():
            for png in rdir.glob("result_*.png"):
                key = png.stem.replace("result_", "", 1)
                results.setdefault(key, png)
        edit_json = None
        ops: list[str] = []
        if edit_p and edit_p.is_file():
            try:
                edit_json = json.loads(edit_p.read_text(encoding="utf-8"))
                ops = [
                    str(item.get("op"))
                    for item in (edit_json.get("operations") or [])
                    if isinstance(item, dict) and item.get("op")
                ]
            except (OSError, json.JSONDecodeError):
                edit_json = None
        cls = edit_class_from_legacy(edit_json) if edit_json else "replace"
        has_out = _exists(out)
        verdict = _qc_verdict(data)
        runs.append(
            ArtifactRun(
                run_id=str(data.get("run_id") or name),
                root=man_path.parent,
                status=str(data.get("status") or ""),
                weight=_weight(str(data.get("status") or ""), verdict, has_out),
                qc_verdict=verdict,
                input_step=inp,
                output_step=out if has_out else None,
                edit_json=edit_p if edit_p and edit_p.is_file() else None,
                views=views,
                targets=targets,
                results=results,
                ops=ops,
                edit_class_name=cls,
            )
        )
    return runs


def replay_run(run: ArtifactRun, thresholds: GateThresholds | None = None) -> dict[str, Any]:
    thr = thresholds or GateThresholds()
    rec: dict[str, Any] = {
        "run_id": run.run_id,
        "weight": run.weight,
        "status": run.status,
        "qc_verdict": run.qc_verdict,
        "ops": list(run.ops),
        "edit_class": run.edit_class_name,
        "geometry_ok": None,
        "preservation_ok": None,
        "intent_ok": None,
        "visual_ok": None,
        "passed": False,
        "skip_reason": None,
    }
    if run.weight == "skip" or run.output_step is None or not _exists(run.input_step):
        rec["skip_reason"] = "missing_step"
        return rec
    edit_json = None
    if run.edit_json and run.edit_json.is_file():
        try:
            edit_json = json.loads(run.edit_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            edit_json = None
    spec = legacy_edit_spec(edit_json, run.edit_class_name)
    base_report = inspect_step(run.input_step)
    geom = validate_step(
        run.output_step,
        base_report=base_report,
        volume_explode_ratio=thr.volume_explode_ratio,
        volume_explode_ratio_global=thr.volume_explode_ratio_global,
        edit_class_name=run.edit_class_name,
    )
    rec["geometry_ok"] = bool(geom.get("ok"))
    rec["volume_ratio"] = None
    rec["diag_ratio"] = None
    rec["max_bbox_rel"] = None
    if geom.get("ok") and base_report.get("ok"):
        cand = geom.get("report") or {}
        bv = float(base_report.get("volume_mm3") or 0.0) or 1e-6
        cv = float(cand.get("volume_mm3") or 0.0)
        rec["volume_ratio"] = round(max(cv / bv, bv / max(cv, 1e-9)), 4)
        bd = float(base_report.get("diag_mm") or 0.0) or 1e-6
        cd = float(cand.get("diag_mm") or 0.0)
        rec["diag_ratio"] = round(max(cd / bd, bd / max(cd, 1e-9)), 4)
    if not geom.get("ok"):
        rec["geometry_error"] = geom.get("error")
        rec["passed"] = False
        return rec

    comparison = compare_geometry(
        run.input_step,
        run.output_step,
        spec,
        bbox_rel_tol=thr.preservation_bbox_rel_tol,
        bbox_rel_tol_replace=thr.preservation_bbox_rel_tol_replace,
        edit_class_name=run.edit_class_name,
    )
    rec["preservation_ok"] = bool(comparison.get("preservation_ok"))
    rec["intent_ok"] = bool(comparison.get("intent_ok"))
    rec["max_bbox_rel"] = round(
        _max_bbox_rel(comparison.get("base") or {}, comparison.get("candidate") or {}),
        4,
    )
    rec["intent_checks"] = (comparison.get("intent_constraints") or {}).get("checks")
    rec["preservation_checks"] = (comparison.get("preservation_constraints") or {}).get("checks")

    visual = _visual_for_run(run, thr)
    rec["visual"] = visual
    rec["visual_ok"] = bool(visual.get("passed")) if visual.get("available") else None
    rec["mean_target_iou"] = visual.get("mean_target_iou")
    rec["min_target_iou"] = visual.get("min_target_iou")
    rec["min_before_iou"] = visual.get("min_before_iou")
    rec["view_alignment"] = visual.get("view_alignment")
    rec["same_renderer"] = visual.get("same_renderer")
    rec["images_missing"] = visual.get("images_missing")

    rec["passed"] = bool(
        rec["geometry_ok"]
        and rec["preservation_ok"]
        and rec["intent_ok"]
        and (rec["visual_ok"] is True or rec["visual_ok"] is None)
    )
    rec["passed_geom_layers"] = bool(rec["geometry_ok"] and rec["preservation_ok"] and rec["intent_ok"])
    return rec


def _visual_for_run(run: ArtifactRun, thr: GateThresholds) -> dict[str, Any]:
    views = sorted(set(run.targets) | set(run.results))
    if not views:
        views = sorted(run.views)
    per_view: list[dict[str, Any]] = []
    vs_before: list[dict[str, Any]] = []
    alignments: list[float] = []
    images_missing = False
    for name in views:
        target = run.targets.get(name)
        result = run.results.get(name)
        before = run.views.get(name)
        if result is None or target is None or not result.is_file() or not target.is_file():
            images_missing = True
            continue
        source = "gpt_img2img"
        alignment = 0.99
        if before and before.is_file():
            est = estimate_view_alignment(before, target)
            if est.get("ok"):
                alignment = min(float(est.get("view_alignment_confidence") or 0.0), 0.99)
                alignments.append(alignment)
            vs_before.append(compare_before_after(before, result, view_id=name))
        cmp = compare_images(
            result,
            target,
            before_image=str(before) if before and before.is_file() else None,
            view_id=name,
            view_alignment_confidence=alignment,
            image_source=source,
            expected_view_known=True,
            region_mask_max_frac=thr.region_mask_max_frac,
            region_mask_min_frac=thr.region_mask_min_frac,
        )
        per_view.append(cmp)
    if not per_view:
        return {
            "available": False,
            "passed": False,
            "images_missing": True,
            "reason": "no comparable PNG pair",
        }
    gates = visual_gates(
        per_view,
        vs_before,
        same_renderer_iou_min=thr.silhouette_iou_min,
        cross_renderer_iou_min=thr.cross_renderer_iou_min,
        wrecked_iou_min=thr.wrecked_iou_min,
        view_alignment_min=thr.view_alignment_min,
        same_renderer_iou_floor=thr.silhouette_iou_floor,
        cross_renderer_iou_floor=thr.cross_renderer_iou_floor,
        edit_class_name=run.edit_class_name,
        region_change_iou_min=thr.region_change_iou_min,
        region_edge_iou_min=thr.region_edge_iou_min,
        noop_silhouette_iou_min=thr.noop_silhouette_iou_min,
        noop_pixel_mae_max=thr.noop_pixel_mae_max,
    )
    ious = [float(v.get("silhouette_iou") or 0.0) for v in per_view if v.get("ok")]
    before_ious = [float(v.get("silhouette_iou") or 0.0) for v in vs_before if v.get("ok")]
    return {
        "available": True,
        "passed": bool(gates.get("passed")),
        "failure_type": gates.get("failure_type"),
        "layers": gates.get("layers"),
        "mean_target_iou": round(sum(ious) / len(ious), 4) if ious else None,
        "min_target_iou": round(min(ious), 4) if ious else None,
        "min_before_iou": round(min(before_ious), 4) if before_ious else None,
        "view_alignment": round(min(alignments), 4) if alignments else None,
        "same_renderer": bool(gates.get("layers", {}).get("edit_region_vs_target", {}).get("same_renderer")),
        "images_missing": images_missing,
        "n_views": len(per_view),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    high = [r for r in results if r.get("weight") == "high"]
    low = [r for r in results if r.get("weight") == "low"]

    def _rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        n = len(vals)
        ok = sum(1 for v in vals if v)
        return {"n": n, "ok": ok, "rate": round(ok / n, 4) if n else None}

    def _nums(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        xs = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        return {
            "n": len(xs),
            "p10": _percentile(xs, 10),
            "p50": _percentile(xs, 50),
            "p90": _percentile(xs, 90),
            "p95": _percentile(xs, 95),
            "p99": _percentile(xs, 99),
            "min": min(xs) if xs else None,
            "max": max(xs) if xs else None,
        }

    return {
        "n_high": len(high),
        "n_low": len(low),
        "high": {
            "geometry": _rate(high, "geometry_ok"),
            "preservation": _rate(high, "preservation_ok"),
            "intent": _rate(high, "intent_ok"),
            "visual": _rate(high, "visual_ok"),
            "all_gates": _rate(high, "passed"),
            "geom_layers": _rate(high, "passed_geom_layers"),
        },
        "low": {
            "geometry": _rate(low, "geometry_ok"),
            "all_gates": _rate(low, "passed"),
        },
        "high_metrics": {
            "volume_ratio": _nums(high, "volume_ratio"),
            "diag_ratio": _nums(high, "diag_ratio"),
            "max_bbox_rel": _nums(high, "max_bbox_rel"),
            "mean_target_iou": _nums(high, "mean_target_iou"),
            "min_target_iou": _nums(high, "min_target_iou"),
            "min_before_iou": _nums(high, "min_before_iou"),
            "view_alignment": _nums(high, "view_alignment"),
        },
        "failures_high": [
            {
                "run_id": r["run_id"],
                "edit_class": r.get("edit_class"),
                "ops": r.get("ops"),
                "geometry_ok": r.get("geometry_ok"),
                "preservation_ok": r.get("preservation_ok"),
                "intent_ok": r.get("intent_ok"),
                "visual_ok": r.get("visual_ok"),
                "mean_target_iou": r.get("mean_target_iou"),
                "min_target_iou": r.get("min_target_iou"),
                "view_alignment": r.get("view_alignment"),
                "geometry_error": r.get("geometry_error"),
            }
            for r in high
            if not r.get("passed")
        ],
    }


def suggest_thresholds(summary: dict[str, Any]) -> dict[str, Any]:
    m = summary.get("high_metrics") or {}

    def g(name: str, key: str, default: float) -> float:
        val = ((m.get(name) or {}).get(key))
        return float(val) if isinstance(val, (int, float)) else default

    vol_p95 = g("volume_ratio", "p95", 8.0)
    vol_p99 = g("volume_ratio", "p99", 8.0)
    vol_max = g("volume_ratio", "max", 8.0)
    bbox_p90 = g("max_bbox_rel", "p90", 0.45)
    mean_p10 = g("mean_target_iou", "p10", 0.22)
    min_p10 = g("min_target_iou", "p10", 0.12)
    align_p10 = g("view_alignment", "p10", 0.40)
    before_p10 = g("min_before_iou", "p10", 0.25)

    # Geometry: cover ~P95 of pass, cap so DESTROY-scale still fails.
    local_explode = min(max(8.0, math.ceil(vol_p95 * 10) / 10.0 + 0.5), 16.0)
    global_explode = min(max(local_explode, math.ceil(vol_p99 * 10) / 10.0 + 1.0, vol_max * 1.05), 48.0)

    # Visual: ~P10 of pass so ~90% clear the bar; floors a bit below.
    # GPT targets are cross-renderer; do not raise the bar to VTK-vs-VTK.
    cross_min = max(0.18, min(mean_p10 * 0.92, 0.32))
    cross_floor = max(0.12, min(min_p10 * 0.85, cross_min))
    same_min = 0.55
    same_floor = 0.40
    align_min = max(0.35, min(align_p10 * 0.9, 0.40))
    wrecked = max(0.18, min(before_p10 * 0.6, 0.25))

    local_bbox = min(max(0.40, bbox_p90 if bbox_p90 < 0.6 else 0.50), 0.55)
    replace_bbox = min(max(0.90, bbox_p90 * 1.15 if bbox_p90 < 2 else 1.15), 1.25)

    return {
        "silhouette_iou_min": round(same_min, 4),
        "silhouette_iou_floor": round(same_floor, 4),
        "cross_renderer_iou_min": round(cross_min, 4),
        "cross_renderer_iou_floor": round(cross_floor, 4),
        "preservation_bbox_rel_tol": round(local_bbox, 4),
        "preservation_bbox_rel_tol_replace": round(replace_bbox, 4),
        "volume_explode_ratio": round(local_explode, 4),
        "volume_explode_ratio_global": round(global_explode, 4),
        "wrecked_iou_min": round(wrecked, 4),
        "view_alignment_min": round(align_min, 4),
        "calibrated_on": "artifacts_longtest_20260815",
        "thresholds_are_placeholders": False,
    }


def run_calibration(
    artifacts_root: Path | str,
    thresholds: GateThresholds | None = None,
) -> dict[str, Any]:
    runs = discover_runs(artifacts_root)
    thr = thresholds or GateThresholds()
    results: list[dict[str, Any]] = []
    todo = [run for run in runs if run.weight != "skip"]
    for i, run in enumerate(todo, 1):
        print(f"[calibrate] {i}/{len(todo)} {run.run_id} class={run.edit_class_name}", flush=True)
        results.append(replay_run(run, thr))
    summary = summarize(results)
    suggested = suggest_thresholds(summary)
    return {
        "thresholds_used": thr.to_dict(),
        "suggested": suggested,
        "summary": summary,
        "n_discovered": len(runs),
        "n_replayed": len(results),
        "results": results,
    }
