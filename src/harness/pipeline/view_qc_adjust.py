"""Agent / heuristic advisors that retune per-view camera_distance_factor after QC fail."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from harness.pipeline.manifest import VIEW_NAMES

AdjustFn = Callable[
    [dict[str, Any], dict[str, float], dict[str, str], int],
    dict[str, float],
]

FACTOR_MIN = 0.55
FACTOR_MAX = 3.5
DEFAULT_FACTOR = 1.18
# ResetCamera frames tightly; modest factor adds white margin.
DEFAULT_FACTORS = {name: 1.18 for name in VIEW_NAMES}


def default_factors() -> dict[str, float]:
    return {n: float(DEFAULT_FACTORS.get(n, DEFAULT_FACTOR)) for n in VIEW_NAMES}


def clamp_factor(value: float) -> float:
    return max(FACTOR_MIN, min(FACTOR_MAX, float(value)))


def _qc_summary(qc: dict[str, Any]) -> str:
    views = qc.get("views") or {}
    fails = []
    for name in VIEW_NAMES:
        meta = views.get(name) or {}
        if meta.get("ok"):
            continue
        detail = meta.get("detail") or "fail"
        fails.append(f"{name}: {detail}")
    if not fails:
        return "all views pass"
    return "; ".join(fails[:12])


def enrich_qc_detail(qc: dict[str, Any]) -> dict[str, Any]:
    """Attach a human-readable ``detail`` string for StageError / logs."""
    out = dict(qc)
    out["detail"] = _qc_summary(qc)
    return out


def heuristic_adjust_factors(
    qc: dict[str, Any],
    factors: dict[str, float],
) -> dict[str, float]:
    """Per-view zoom: clip → zoom out (wins); else small/off-center → zoom in."""
    views = qc.get("views") or {}
    new = {n: clamp_factor(factors.get(n, DEFAULT_FACTORS.get(n, DEFAULT_FACTOR))) for n in VIEW_NAMES}
    for name in VIEW_NAMES:
        meta = views.get(name) or {}
        if meta.get("ok"):
            continue
        f = new[name]
        # Clipping always wins — never zoom in on a clipped view.
        if meta.get("white_border") is False:
            new[name] = clamp_factor(f * 1.18)
            continue
        # Centering alone cannot be fixed by zoom for diagonal silhouettes —
        # only nudge when occupancy is also low.
        occ = float(meta.get("occupancy") or 0.0)
        if occ < 0.18:
            new[name] = clamp_factor(f * 0.86)
        elif occ < 0.26:
            new[name] = clamp_factor(f * 0.90)
        elif meta.get("centered") is False and occ >= 0.28:
            # Keep framing; asymmetric diagonal views often sit slightly off-center.
            continue
        elif meta.get("centered") is False:
            new[name] = clamp_factor(f * 0.96)
        else:
            new[name] = clamp_factor(f * 0.94)
    return new


def parse_factor_json(text: str, current: dict[str, float]) -> dict[str, float] | None:
    """Parse agent JSON into a full ``{view_name: float}`` factor map."""
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    data: Any = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("factors"), dict):
        data = data["factors"]

    out: dict[str, float] = {n: current.get(n, DEFAULT_FACTOR) for n in VIEW_NAMES}
    found = 0
    for key, val in data.items():
        name = str(key).strip()
        if name.endswith(".png"):
            name = name[: -len(".png")]
        if name not in VIEW_NAMES:
            continue
        try:
            if isinstance(val, dict):
                val = val.get("camera_distance_factor", val.get("factor"))
            out[name] = clamp_factor(float(val))
            found += 1
        except (TypeError, ValueError):
            return None
    if found == 0:
        return None
    return out


def make_agent_view_qc_adjuster(
    *,
    helper: Any,
    prompt: str,
    session_append: Callable[[dict[str, Any]], None] | None = None,
    parent_session_id: str | None = None,
) -> AdjustFn:
    """Build an adjust_fn that asks the helper sub-agent, then falls back to heuristic."""

    def adjust(
        qc: dict[str, Any],
        factors: dict[str, float],
        paths: dict[str, str],
        attempt: int,
    ) -> dict[str, float]:
        failing = [
            n
            for n in VIEW_NAMES
            if not ((qc.get("views") or {}).get(n) or {}).get("ok")
        ]
        attach_names = failing or list(VIEW_NAMES)
        images = [Path(paths[n]) for n in attach_names if n in paths and Path(paths[n]).is_file()]

        payload = {
            "attempt": attempt,
            "current_factors": {n: round(factors.get(n, DEFAULT_FACTOR), 4) for n in VIEW_NAMES},
            "qc_detail": _qc_summary(qc),
            "views": {
                n: {
                    "ok": (qc.get("views") or {}).get(n, {}).get("ok"),
                    "white_border": (qc.get("views") or {}).get(n, {}).get("white_border"),
                    "centered": (qc.get("views") or {}).get(n, {}).get("centered"),
                    "occupancy": (qc.get("views") or {}).get(n, {}).get("occupancy"),
                    "detail": (qc.get("views") or {}).get(n, {}).get("detail"),
                }
                for n in VIEW_NAMES
            },
        }
        extra = (
            f"<<<QC_STATE>>>\n"
            f"{json.dumps(payload, indent=2)}\n"
            f"<<<END_QC_STATE>>>\n\n"
            "Return ONLY the JSON object with factors for all four views "
            "(iso, z_corner, x_corner, y_corner)."
        )
        task_prompt = f"{prompt}\n\n{extra}"

        try:
            result = helper.run(
                task_prompt,
                images=images,
                label=f"view_qc_attempt_{attempt}",
                parent_session_id=parent_session_id,
                append=session_append,
            )
            reply = result.content if result.ok else ""
            if not result.ok:
                return heuristic_adjust_factors(qc, factors)
        except Exception:  # noqa: BLE001
            return heuristic_adjust_factors(qc, factors)

        parsed = parse_factor_json(reply, factors)
        if parsed is None:
            return heuristic_adjust_factors(qc, factors)
        if all(abs(parsed[n] - factors.get(n, DEFAULT_FACTOR)) < 1e-6 for n in VIEW_NAMES):
            return heuristic_adjust_factors(qc, factors)
        return parsed

    return adjust
