"""Layered visual comparison. Never collapse to a single score."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[misc, assignment]

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[misc, assignment]


WHITE = 250


def estimate_view_alignment(
    cad_image: Path | str,
    target_image: Path | str,
) -> dict[str, Any]:
    """Estimate whether a target still shares the CAD view's camera.

    Uses unmodified-region proxies (centroid, bbox, orientation). Silhouette
    IoU is recorded but not treated as camera proof — a large intended edit
    also lowers IoU.
    """
    cad_path = Path(cad_image)
    tgt_path = Path(target_image)
    empty = {
        "view_alignment_confidence": 0.0,
        "centroid_shift": None,
        "bbox_iou": None,
        "orientation_cos": None,
        "silhouette_iou": None,
        "ok": False,
    }
    if Image is None or np is None or not cad_path.is_file() or not tgt_path.is_file():
        return empty
    cad = _load_rgb(cad_path)
    tgt = _load_rgb(tgt_path)
    h = min(cad.shape[0], tgt.shape[0])
    w = min(cad.shape[1], tgt.shape[1])
    cad = _resize(cad, w, h)
    tgt = _resize(tgt, w, h)
    cad_mask = _silhouette(cad)
    tgt_mask = _silhouette(tgt)
    mc = _mask_moments(cad_mask)
    mt = _mask_moments(tgt_mask)
    if mc is None or mt is None:
        return {**empty, "error": "empty silhouette"}
    diag = float(max(math.hypot(w, h), 1.0))
    shift = math.hypot(mc["cx"] - mt["cx"], mc["cy"] - mt["cy"]) / diag
    bbox_iou = _bbox_iou(mc["bbox"], mt["bbox"])
    ori = abs(float(mc["axis"][0] * mt["axis"][0] + mc["axis"][1] * mt["axis"][1]))
    sil_iou = _iou(cad_mask, tgt_mask)
    centroid_score = math.exp(-shift / 0.08)
    alignment = 0.45 * centroid_score + 0.35 * bbox_iou + 0.20 * ori
    return {
        "ok": True,
        "view_alignment_confidence": round(float(max(0.0, min(1.0, alignment))), 4),
        "centroid_shift": round(shift, 4),
        "bbox_iou": round(bbox_iou, 4),
        "orientation_cos": round(ori, 4),
        "silhouette_iou": round(sil_iou, 4),
        "estimated_observed_view_spec": {
            "centroid_px": [round(mt["cx"], 2), round(mt["cy"], 2)],
            "bbox_px": [int(x) for x in mt["bbox"]],
            "axis": [round(float(mt["axis"][0]), 4), round(float(mt["axis"][1]), 4)],
        },
    }


def compare_images(
    candidate_image: Path | str,
    reference_image: Path | str,
    *,
    edit_mask: Path | str | None = None,
    before_image: Path | str | None = None,
    mask_dir: Path | str | None = None,
    camera_confidence: float | None = None,
    view_id: str = "",
    requested_view_spec: dict[str, Any] | None = None,
    view_alignment_confidence: float | None = None,
    image_source: str = "unknown",
    expected_view_known: bool = True,
    region_mask_max_frac: float = 0.25,
    region_mask_min_frac: float = 0.001,
) -> dict[str, Any]:
    cand_path = Path(candidate_image)
    ref_path = Path(reference_image)
    alignment = (
        float(view_alignment_confidence)
        if view_alignment_confidence is not None
        else (float(camera_confidence) if camera_confidence is not None else 1.0)
    )
    if image_source in {"gpt_img2img", "stub"} and alignment >= 1.0:
        alignment = 0.99
    reported_cam = alignment if image_source != "vtk" else 1.0
    if camera_confidence is not None and image_source == "vtk":
        reported_cam = float(camera_confidence)

    if Image is None or np is None:
        return {
            "view_id": view_id,
            "ok": False,
            "error": "Pillow/numpy not installed",
            "camera_confidence": reported_cam,
            "view_alignment_confidence": alignment,
            "requested_view_spec": requested_view_spec,
            "image_source": image_source,
            "expected_view_known": expected_view_known,
        }
    if not cand_path.is_file() or not ref_path.is_file():
        return {
            "view_id": view_id,
            "ok": False,
            "error": "missing candidate or reference image",
            "camera_confidence": reported_cam,
            "view_alignment_confidence": alignment,
            "requested_view_spec": requested_view_spec,
            "image_source": image_source,
            "expected_view_known": expected_view_known,
            "silhouette_iou": 0.0,
        }

    cand = _load_rgb(cand_path)
    ref = _load_rgb(ref_path)
    h = min(cand.shape[0], ref.shape[0])
    w = min(cand.shape[1], ref.shape[1])
    cand = _resize(cand, w, h)
    ref = _resize(ref, w, h)

    cand_mask = _silhouette(cand)
    ref_mask = _silhouette(ref)
    iou = _iou(cand_mask, ref_mask)
    pixel_mae = float(np.mean(np.abs(cand.astype(np.float32) - ref.astype(np.float32))) / 255.0)
    edge = _edge_l1(cand, ref)

    edit_score = None
    unchanged_score = None
    if edit_mask and Path(edit_mask).is_file():
        mask = _resize(_silhouette(_load_rgb(Path(edit_mask))), w, h)
        edit_score = _masked_similarity(cand, ref, mask)
        unchanged_score = _masked_similarity(cand, ref, ~mask)

    vtk_like_ref = _looks_like_vtk_cad(ref)
    vtk_like_cand = _looks_like_vtk_cad(cand)

    out: dict[str, Any] = {
        "view_id": view_id,
        "ok": True,
        "camera_confidence": float(reported_cam),
        "view_alignment_confidence": round(float(alignment), 4),
        "requested_view_spec": requested_view_spec,
        "image_source": image_source,
        "expected_view_known": bool(expected_view_known),
        "silhouette_iou": round(iou, 4),
        "pixel_mae": round(pixel_mae, 4),
        "edge_distance": round(edge, 4),
        "edit_region_score": None if edit_score is None else round(edit_score, 4),
        "unchanged_region_score": None if unchanged_score is None else round(unchanged_score, 4),
        "same_renderer": bool(vtk_like_ref and vtk_like_cand),
        "reference_looks_like_vtk": vtk_like_ref,
        "candidate_looks_like_vtk": vtk_like_cand,
        "candidate_path": str(cand_path.resolve()),
        "reference_path": str(ref_path.resolve()),
        "change_iou": None,
        "region_edge_iou": None,
        "region_reliable": False,
        "region_unreliable": False,
        "needed_frac": None,
        "did_change_frac": None,
        "needed_mask_path": None,
        "did_change_mask_path": None,
    }
    if before_image:
        before_path = Path(before_image)
        if before_path.is_file():
            region = _region_metrics(
                before_path,
                cand_path,
                ref_path,
                mask_dir=Path(mask_dir) if mask_dir else None,
                view_id=view_id,
                max_frac=region_mask_max_frac,
                min_frac=region_mask_min_frac,
            )
            out.update(region)
    return out


def compare_before_after(
    before_image: Path | str,
    result_image: Path | str,
    *,
    view_id: str = "",
) -> dict[str, Any]:
    """How much the result moved vs the original CAD view (same known camera)."""
    cmp = compare_images(
        result_image,
        before_image,
        camera_confidence=1.0,
        view_id=view_id,
    )
    cmp["role"] = "result_vs_before"
    return cmp


def visual_gates(
    per_view: list[dict[str, Any]],
    vs_before: list[dict[str, Any]],
    *,
    same_renderer_iou_min: float,
    cross_renderer_iou_min: float,
    wrecked_iou_min: float,
    view_alignment_min: float = 0.40,
    same_renderer_iou_floor: float | None = None,
    cross_renderer_iou_floor: float | None = None,
    edit_class_name: str | None = None,
    region_change_iou_min: float = 0.12,
    region_edge_iou_min: float = 0.12,
    noop_silhouette_iou_min: float = 0.997,
    noop_pixel_mae_max: float = 0.008,
) -> dict[str, Any]:
    """Layered pass/fail. Camera first, then wreckage, then target match.

    Additive/subtractive edits use *region* IoU (needed vs did-change, plus
    edge IoU inside the needed mask). Whole-image silhouette IoU is recorded
    but does not decide FINISH for those classes — a missed pocket barely
    changes the outer outline.

    GPT targets never pass by claiming camera_confidence=1.0. They pass the
    camera layer only when the requested view is known *and* unmodified-region
    alignment is above a placeholder threshold.

    Wreckage (result vs before) only applies to local_modify edits.
    """
    if not per_view:
        return {
            "passed": True,
            "failure_type": None,
            "reason": None,
            "skipped_target_match": True,
            "layers": {
                "camera": {"passed": True, "mode": "skipped"},
                "unchanged_region": {"passed": True, "mode": "skipped"},
                "edit_region_vs_target": {
                    "passed": True,
                    "mode": "skipped",
                    "reason": "no_target_renders",
                },
            },
            "mean_target_iou": 0.0,
        }

    alignments = [
        float(
            v.get("view_alignment_confidence")
            if v.get("view_alignment_confidence") is not None
            else (v.get("camera_confidence") or 0.0)
        )
        for v in per_view
    ]
    min_align = min(alignments) if alignments else 0.0
    expected_known = all(bool(v.get("expected_view_known", True)) for v in per_view)
    sources = [str(v.get("image_source") or "unknown") for v in per_view]
    gpt_like = any(s in {"gpt_img2img", "stub"} for s in sources)
    camera_ok = expected_known and min_align >= float(view_alignment_min)
    same_renderer = (not gpt_like) and all(bool(v.get("same_renderer")) for v in per_view if v.get("ok"))
    ious = [float(v.get("silhouette_iou") or 0.0) for v in per_view if v.get("ok")]
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    min_iou = min(ious) if ious else 0.0
    iou_min = same_renderer_iou_min if same_renderer else cross_renderer_iou_min
    if same_renderer:
        iou_floor = same_renderer_iou_floor if same_renderer_iou_floor is not None else min(iou_min, 0.40)
    else:
        iou_floor = (
            cross_renderer_iou_floor if cross_renderer_iou_floor is not None else min(iou_min, 0.12)
        )
    silhouette_ok = bool(ious) and mean_iou >= iou_min and min_iou >= float(iou_floor)

    before_ious = [float(v.get("silhouette_iou") or 0.0) for v in vs_before if v.get("ok")]
    before_maes = [float(v.get("pixel_mae") or 0.0) for v in vs_before if v.get("ok")]
    min_before = min(before_ious) if before_ious else 1.0
    max_before_mae = max(before_maes) if before_maes else 0.0
    wreckage_applies = (edit_class_name or "local_modify") == "local_modify"
    wrecked = wreckage_applies and min_before < wrecked_iou_min
    unchanged_ok = not wrecked
    uses_region = edit_class_name in {"additive", "subtractive"}
    noop = bool(before_ious) and min_before >= float(noop_silhouette_iou_min) and max_before_mae <= float(
        noop_pixel_mae_max
    )

    reliable = [v for v in per_view if v.get("ok") and v.get("region_reliable")]
    region_mode = "silhouette"
    change_ious = [float(v["change_iou"]) for v in reliable if v.get("change_iou") is not None]
    edge_ious = [float(v["region_edge_iou"]) for v in reliable if v.get("region_edge_iou") is not None]
    mean_change = sum(change_ious) / len(change_ious) if change_ious else 0.0
    mean_edge = sum(edge_ious) / len(edge_ious) if edge_ious else 0.0
    min_change = min(change_ious) if change_ious else 0.0
    min_edge = min(edge_ious) if edge_ious else 0.0

    if uses_region and reliable:
        region_mode = "region"
        target_ok = (
            mean_change >= float(region_change_iou_min)
            and min_change >= float(region_change_iou_min) * 0.5
            and mean_edge >= float(region_edge_iou_min)
            and min_edge >= float(region_edge_iou_min) * 0.5
        )
        score_iou = mean_change
    elif uses_region:
        region_mode = "unreliable_fallback"
        if before_ious:
            target_ok = not noop
            score_iou = 0.0 if noop else mean_iou
        else:
            region_mode = "silhouette_fallback"
            target_ok = silhouette_ok
            score_iou = mean_iou
    else:
        target_ok = silhouette_ok
        score_iou = mean_iou

    layers = {
        "camera": {
            "passed": camera_ok,
            "expected_view_known": expected_known,
            "view_alignment_confidence": round(min_align, 4),
            "view_alignment_min": float(view_alignment_min),
            "image_sources": sources,
            "gpt_target_not_camera_1": gpt_like,
            "confidence": round(min_align, 4),
        },
        "unchanged_region": {
            "passed": unchanged_ok,
            "min_result_vs_before_iou": round(min_before, 4) if before_ious else None,
            "wrecked": wrecked,
            "wreckage_applies": wreckage_applies,
            "edit_class": edit_class_name,
            "noop": noop,
        },
        "edit_region_vs_target": {
            "passed": target_ok,
            "mode": region_mode,
            "mean_silhouette_iou": round(mean_iou, 4),
            "min_silhouette_iou": round(min_iou, 4),
            "mean_change_iou": round(mean_change, 4) if change_ious else None,
            "min_change_iou": round(min_change, 4) if change_ious else None,
            "mean_region_edge_iou": round(mean_edge, 4) if edge_ious else None,
            "min_region_edge_iou": round(min_edge, 4) if edge_ious else None,
            "reliable_views": [v.get("view_id") for v in reliable],
            "threshold": (
                float(region_change_iou_min) if region_mode == "region" else iou_min
            ),
            "floor": (
                float(region_change_iou_min) * 0.5 if region_mode == "region" else float(iou_floor)
            ),
            "region_change_iou_min": float(region_change_iou_min),
            "region_edge_iou_min": float(region_edge_iou_min),
            "threshold_is_placeholder": False,
            "same_renderer": same_renderer,
            "per_view": {v.get("view_id"): v.get("silhouette_iou") for v in per_view},
            "per_view_change_iou": {v.get("view_id"): v.get("change_iou") for v in per_view},
        },
        "overall_similarity": {
            "mean_pixel_mae": round(
                sum(float(v.get("pixel_mae") or 0.0) for v in per_view) / max(len(per_view), 1),
                4,
            )
        },
    }
    passed = camera_ok and unchanged_ok and target_ok
    failure = None
    if not camera_ok:
        failure = "VIEW_UNCERTAIN"
    elif wrecked:
        failure = "PRESERVATION_VIOLATION"
    elif not target_ok:
        failure = "VISUAL_MISMATCH"
    reason = None
    if not passed:
        if uses_region and noop:
            reason = "VISUAL_MISMATCH: no-op vs before; needed edit region unchanged"
        elif uses_region and region_mode == "region":
            reason = (
                "VISUAL_MISMATCH: region IoU "
                f"change={mean_change:.3f} edge={mean_edge:.3f}"
            )
        else:
            reason = failure
    return {
        "passed": passed,
        "failure_type": failure,
        "reason": reason,
        "layers": layers,
        "mean_target_iou": round(float(score_iou), 4),
    }


def _region_metrics(
    before_path: Path,
    result_path: Path,
    target_path: Path,
    *,
    mask_dir: Path | None,
    view_id: str,
    max_frac: float,
    min_frac: float,
) -> dict[str, Any]:
    empty = {
        "change_iou": None,
        "region_edge_iou": None,
        "region_reliable": False,
        "region_unreliable": False,
        "needed_frac": None,
        "did_change_frac": None,
        "needed_mask_path": None,
        "did_change_mask_path": None,
    }
    if Image is None or np is None:
        return empty
    before = _load_rgb(before_path)
    result = _load_rgb(result_path)
    target = _load_rgb(target_path)
    h = min(before.shape[0], result.shape[0], target.shape[0])
    w = min(before.shape[1], result.shape[1], target.shape[1])
    before = _resize(before, w, h)
    result = _resize(result, w, h)
    target = _resize(target, w, h)

    sil = _silhouette(before)
    obj = float(sil.sum()) or 1.0
    needed = _dilate(_edge_map(target) ^ _edge_map(before), radius=5) & sil
    pixel_delta = np.abs(before.astype(np.float32) - result.astype(np.float32)).mean(axis=2) > 10.0
    did = _dilate(pixel_delta | (_edge_map(before) ^ _edge_map(result)), radius=4)
    did = did & (_silhouette(before) | _silhouette(result))

    needed_frac = float(needed.sum()) / obj
    did_frac = float(did.sum()) / obj
    unreliable = needed_frac > float(max_frac)
    empty_needed = needed_frac < float(min_frac)
    reliable = (not unreliable) and (not empty_needed)

    change_iou = _iou(needed, did) if not empty_needed else None
    er = _edge_map(result) & needed
    et = _edge_map(target) & needed
    region_edge_iou = _iou(er, et) if needed.sum() else None

    needed_path = None
    did_path = None
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
        stem = view_id or "view"
        needed_path = str((mask_dir / f"{stem}_needed.png").resolve())
        did_path = str((mask_dir / f"{stem}_did_change.png").resolve())
        _save_mask(needed, needed_path)
        _save_mask(did, did_path)

    return {
        "change_iou": None if change_iou is None else round(float(change_iou), 4),
        "region_edge_iou": None if region_edge_iou is None else round(float(region_edge_iou), 4),
        "region_reliable": bool(reliable),
        "region_unreliable": bool(unreliable),
        "needed_frac": round(needed_frac, 4),
        "did_change_frac": round(did_frac, 4),
        "needed_mask_path": needed_path,
        "did_change_mask_path": did_path,
    }


def _load_rgb(path: Path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _resize(arr, width: int, height: int):
    if arr.shape[1] == width and arr.shape[0] == height:
        return arr
    img = Image.fromarray(arr)
    return np.asarray(img.resize((width, height), Image.Resampling.BILINEAR))


def _silhouette(arr) -> Any:
    # Near-white background used by the VTK renderer and S1 targets.
    return np.any(arr < WHITE, axis=2)


def _iou(a, b) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def _edge_map(arr, thresh: float = 18.0):
    gx = np.abs(arr[:, 1:, :].astype(np.int16) - arr[:, :-1, :].astype(np.int16))
    gy = np.abs(arr[1:, :, :].astype(np.int16) - arr[:-1, :, :].astype(np.int16))
    gx = np.pad(gx, ((0, 0), (0, 1), (0, 0)))
    gy = np.pad(gy, ((0, 1), (0, 0), (0, 0)))
    mag = gx.mean(axis=2) + gy.mean(axis=2)
    return mag > thresh


def _dilate(mask, radius: int = 4):
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            if dy < 0:
                shifted[dy:, :] = False
            elif dy > 0:
                shifted[:dy, :] = False
            if dx < 0:
                shifted[:, dx:] = False
            elif dx > 0:
                shifted[:, :dx] = False
            out = out | shifted
    return out


def _save_mask(mask, path: str) -> None:
    if Image is None:
        return
    img = Image.fromarray((mask.astype("uint8") * 255))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _edge_l1(a, b) -> float:
    ea = _edge_map(a)
    eb = _edge_map(b)
    # Hamming on edge maps, in pixels — a cheap Chamfer stand-in.
    return float(np.abs(ea.astype(np.int16) - eb.astype(np.int16)).mean() * max(a.shape[0], a.shape[1]))


def _masked_similarity(a, b, mask) -> float:
    if mask.sum() == 0:
        return 1.0
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2) / 255.0
    return float(1.0 - diff[mask].mean())


def _mask_moments(mask) -> dict[str, Any] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    cx = float(xs.mean())
    cy = float(ys.mean())
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    xs_c = xs.astype(np.float64) - cx
    ys_c = ys.astype(np.float64) - cy
    cov_xx = float((xs_c * xs_c).mean())
    cov_yy = float((ys_c * ys_c).mean())
    cov_xy = float((xs_c * ys_c).mean())
    theta = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
    axis = (math.cos(theta), math.sin(theta))
    return {"cx": cx, "cy": cy, "bbox": (x0, y0, x1, y1), "axis": axis}


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 1.0
    return float(inter) / float(union)


def _looks_like_vtk_cad(arr) -> bool:
    """VTK CAD views: mostly white bg + olive-gray foreground."""
    bg = np.all(arr >= WHITE, axis=2)
    bg_frac = float(bg.mean())
    if bg_frac < 0.15 or bg_frac > 0.95:
        return False
    fg = arr[~bg]
    if fg.size == 0:
        return False
    mean = fg.mean(axis=0)
    # olive-ish, not saturated photo colors
    return bool(mean[1] >= mean[2] - 8 and mean[0] < 180 and mean[1] < 190)
