"""Parse S1 target-render QC helper replies."""

from __future__ import annotations

import json
import re
from typing import Any


def _first_json_object(text: str) -> dict[str, Any] | None:
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
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
    return data if isinstance(data, dict) else None


def _tri_bool(val: Any) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and val in (0, 1):
        return bool(val)
    if isinstance(val, str):
        key = val.strip().lower()
        if key in {"true", "yes", "1", "pass", "ok", "accept"}:
            return True
        if key in {"false", "no", "0", "fail", "reject"}:
            return False
    return bool(val)


def _copy_text_fields(data: dict[str, Any], out: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in data and data[key] is not None:
            out[key] = str(data[key])


def parse_target_qc_json(
    text: str,
    *,
    require_view_gates: bool = False,
) -> dict[str, Any] | None:
    """Parse helper QC reply into ``{pass, matches_description, ...}``.

    ``require_view_gates`` (S1 target QC): pass also needs explicit
    ``camera_matches_base_view`` and ``same_part_identity``. Missing those
    fields fail closed so a bare ``pass: true`` cannot rubber-stamp a restyle.

    Returns None if unusable.
    """
    data = _first_json_object(text)
    if data is None:
        return None

    matches = data.get("matches_description")
    preserves = data.get("preserves_unchanged")
    if matches is None and "matches" in data:
        matches = data["matches"]
    if preserves is None and "preserves" in data:
        preserves = data["preserves"]

    camera = data.get("camera_matches_base_view")
    if camera is None:
        camera = data.get("camera_match")
    identity = data.get("same_part_identity")
    if identity is None:
        identity = data.get("no_feature_redesign")
    if identity is None:
        identity = data.get("same_body")

    passed = data.get("pass")
    if passed is None and "ok" in data:
        passed = data["ok"]
    if passed is None and isinstance(data.get("verdict"), str):
        passed = data["verdict"].strip().lower() in {"pass", "ok", "accept", "yes"}

    matches_b = _tri_bool(matches)
    preserves_b = _tri_bool(preserves)
    camera_b = _tri_bool(camera)
    identity_b = _tri_bool(identity)

    if passed is None:
        if matches_b is not None and preserves_b is not None:
            passed = bool(matches_b and preserves_b)
        elif require_view_gates:
            passed = False
        else:
            return None

    passed_b = bool(_tri_bool(passed))
    if matches_b is not None and preserves_b is not None:
        passed_b = bool(matches_b and preserves_b and passed_b)

    if require_view_gates:
        # Fail closed: the helper must explicitly affirm camera + identity.
        if camera_b is not True or identity_b is not True:
            passed_b = False
        if matches_b is not True or preserves_b is not True:
            passed_b = False

    elif camera_b is False or identity_b is False:
        passed_b = False

    out: dict[str, Any] = {
        "pass": passed_b,
        "matches_description": True if matches_b is None else matches_b,
        "preserves_unchanged": True if preserves_b is None else preserves_b,
        "camera_matches_base_view": camera_b,
        "same_part_identity": identity_b,
    }
    _copy_text_fields(
        data,
        out,
        ("issues", "fix_hint", "notes", "reason", "evaluation", "verdict"),
    )
    return out


_PAIR_VIEWS = ("iso", "z_corner", "x_corner", "y_corner")
_PAIR_ALIASES = {
    "iso": "iso",
    "isometric": "iso",
    "z_corner": "z_corner",
    "zcorner": "z_corner",
    "x_corner": "x_corner",
    "xcorner": "x_corner",
    "y_corner": "y_corner",
    "ycorner": "y_corner",
}


def _pair_view_name(raw: Any) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    name = _PAIR_ALIASES.get(key)
    return name if name in _PAIR_VIEWS else None


def parse_target_pair_qc_json(text: str) -> dict[str, Any] | None:
    """Parse S1 pair QC: same 3D edit from two cameras, still the CAD part."""
    data = _first_json_object(text)
    if data is None:
        return None

    same = data.get("same_edit")
    if same is None:
        same = data.get("same_object")
    if same is None:
        same = data.get("consistent")
    viewpoint = data.get("viewpoint_only")
    if viewpoint is None:
        viewpoint = data.get("viewpoint_difference_only")
    if viewpoint is None:
        viewpoint = data.get("camera_only")
    cad = data.get("cad_consistent")
    if cad is None:
        cad = data.get("each_matches_cad")
    if cad is None:
        cad = data.get("same_part_identity")

    passed = data.get("pass")
    if passed is None and "ok" in data:
        passed = data["ok"]
    if passed is None and isinstance(data.get("verdict"), str):
        passed = data["verdict"].strip().lower() in {"pass", "ok", "accept", "yes"}

    same_b = _tri_bool(same)
    view_b = _tri_bool(viewpoint)
    cad_b = _tri_bool(cad)
    if passed is None:
        if same_b is not None and view_b is not None:
            passed = bool(same_b and view_b)
        else:
            return None
    matches = data.get("matches_description")
    if matches is None:
        matches = data.get("matches")
    matches_b = _tri_bool(matches)

    passed_b = bool(_tri_bool(passed))
    if same_b is not None and view_b is not None:
        passed_b = bool(same_b and view_b and passed_b)
    # Two independently restyled targets can still look like "the same edit".
    # Pass only when each is still the CAD part + described delta.
    if cad_b is not True:
        passed_b = False
    if matches_b is False:
        passed_b = False

    both_fail_b = _tri_bool(data.get("both_fail"))
    if both_fail_b is True:
        passed_b = False

    keep = _pair_view_name(
        data.get("keep_view") or data.get("good_view") or data.get("keep")
    )
    retry = _pair_view_name(
        data.get("retry_view") or data.get("bad_view") or data.get("fix_view")
    )
    retry_raw = data.get("retry_views")
    if retry_raw is None:
        retry_raw = data.get("regenerate")
    retry_list: list[str] = []
    if isinstance(retry_raw, str):
        retry_raw = [retry_raw]
    if isinstance(retry_raw, list):
        seen: set[str] = set()
        for item in retry_raw:
            name = _pair_view_name(item)
            if name and name not in seen:
                retry_list.append(name)
                seen.add(name)
    if retry is None and retry_list:
        retry = retry_list[0]
    if retry and retry not in retry_list:
        retry_list = [retry, *[v for v in retry_list if v != retry]]

    view_pass = _parse_view_pass_map(data)
    if any(flag is False for flag in view_pass.values()):
        passed_b = False
    if passed_b:
        both_fail_b = False
    elif both_fail_b is True:
        keep = None
        retry = None
        if not retry_list:
            retry_list = []

    out: dict[str, Any] = {
        "pass": passed_b,
        "same_edit": True if same_b is None else same_b,
        "viewpoint_only": True if view_b is None else view_b,
        "cad_consistent": cad_b,
        "matches_description": matches_b,
        "both_fail": True if both_fail_b is True else False,
        "view_pass": view_pass,
        "keep_view": keep,
        "retry_view": retry,
        "retry_views": retry_list,
    }
    _copy_text_fields(data, out, ("issues", "fix_hint", "notes", "reason", "evaluation"))
    return out


def _parse_view_pass_map(data: dict[str, Any]) -> dict[str, bool]:
    """Per-camera pass flags from ``view_pass`` / ``views`` / passed|failed lists."""
    out: dict[str, bool] = {}
    raw = data.get("view_pass")
    if raw is None:
        raw = data.get("views")
    if isinstance(raw, dict):
        for key, val in raw.items():
            name = _pair_view_name(key)
            if not name:
                continue
            if isinstance(val, dict):
                flag = val.get("pass")
                if flag is None:
                    flag = val.get("ok")
                parsed = _tri_bool(flag)
            else:
                parsed = _tri_bool(val)
            if parsed is not None:
                out[name] = parsed
    for key, flag in (("passed_views", True), ("failed_views", False)):
        raw_list = data.get(key)
        if isinstance(raw_list, str):
            raw_list = [raw_list]
        if isinstance(raw_list, list):
            for item in raw_list:
                name = _pair_view_name(item)
                if name:
                    out[name] = flag
    return out


def pair_qc_view_verdicts(
    parsed: dict[str, Any] | None,
    base_views: list[str],
) -> dict[str, bool]:
    """Which of the two cameras the pair QC accepted."""
    views = [v for v in base_views if v]
    data = parsed or {}
    if data.get("pass"):
        return {v: True for v in views}

    explicit = dict(data.get("view_pass") or {})
    out: dict[str, bool] = {}
    for view in views:
        flag = explicit.get(view)
        if flag is not None:
            out[view] = bool(flag)

    if data.get("both_fail"):
        return {v: False for v in views}

    retry_list = [v for v in (data.get("retry_views") or []) if v in views]
    if views and set(retry_list) >= set(views):
        return {v: False for v in views}

    keep = data.get("keep_view")
    retry = data.get("retry_view")
    if keep in views and keep not in out:
        out[str(keep)] = True
    if retry in views and retry not in out:
        out[str(retry)] = False
    for view in retry_list:
        out.setdefault(view, False)
    for view in views:
        out.setdefault(view, False)
    return out


def resolve_pair_keep_retry(
    parsed: dict[str, Any] | None,
    base_views: list[str],
) -> tuple[str, str]:
    """Pick ``(keep_view, retry_view)`` from a failed pair QC."""
    views = [v for v in base_views if v]
    if not views:
        return "iso", "z_corner"
    if len(views) == 1:
        return views[0], views[0]
    data = parsed or {}
    keep = data.get("keep_view")
    retry = data.get("retry_view")
    retries = [v for v in (data.get("retry_views") or []) if v in views]
    if keep not in views:
        keep = None
    if retry not in views:
        retry = retries[0] if retries else None
    if keep and retry and keep != retry:
        return str(keep), str(retry)
    if retry and retry in views:
        keep = next(v for v in views if v != retry)
        return keep, str(retry)
    if keep and keep in views:
        retry = next(v for v in views if v != keep)
        return str(keep), retry
    verdicts = pair_qc_view_verdicts(data, views)
    passed = [v for v in views if verdicts.get(v)]
    failed = [v for v in views if not verdicts.get(v)]
    if len(passed) == 1 and failed:
        return passed[0], failed[0]
    return views[0], views[1]


def decide_s1_pair_followup(
    parsed: dict[str, Any] | None,
    base_views: list[str],
    *,
    round_index: int,
    max_rounds: int = 2,
) -> dict[str, Any]:
    """Next S1 action after a pair QC round.

    Two QC rounds total. After the first fail: regenerate the failed camera(s).
    After the second fail: keep only the passing camera, or drop both so the
    harness proceeds from CAD views + description alone.
    """
    views = [v for v in base_views if v]
    data = parsed or {}
    last = int(round_index) >= max(1, int(max_rounds)) - 1
    verdicts = pair_qc_view_verdicts(data, views)
    passed_views = [v for v in views if verdicts.get(v)]
    failed_views = [v for v in views if not verdicts.get(v)]

    def _pack(action: str, keep: list[str], retry: list[str]) -> dict[str, Any]:
        return {
            "action": action,
            "keep_views": keep,
            "retry_views": retry,
            "view_pass": verdicts,
            "last_round": last,
        }

    if data.get("pass") and len(passed_views) >= 2:
        return _pack("accept_both", list(views), [])

    if len(passed_views) >= 2:
        keep, retry = resolve_pair_keep_retry(data, views)
        if last:
            return _pack("keep_one", [keep], [])
        return _pack("retry_one", [keep], [retry])

    if len(passed_views) == 1:
        keep = passed_views[0]
        retry = (
            failed_views[0]
            if failed_views
            else next((v for v in views if v != keep), keep)
        )
        if last:
            return _pack("keep_one", [keep], [])
        return _pack("retry_one", [keep], [retry] if retry != keep else [])

    if last:
        return _pack("drop_all", [], [])
    return _pack("retry_both", [], list(views))


def parse_result_qc_json(text: str) -> dict[str, Any] | None:
    """Parse S5 result-verify QC; reject replies must include ``evaluation``."""
    parsed = parse_target_qc_json(text, require_view_gates=False)
    if parsed is None:
        return None
    if not parsed.get("pass"):
        evaluation = str(parsed.get("evaluation") or "").strip()
        if not evaluation:
            # Fallbacks some models use instead of evaluation.
            for key in ("reason", "notes", "issues", "fix_hint"):
                alt = str(parsed.get(key) or "").strip()
                if alt:
                    evaluation = alt
                    break
        if not evaluation:
            return None
        parsed["evaluation"] = evaluation
        parsed["verdict"] = str(parsed.get("verdict") or "reject")
    else:
        if "evaluation" not in parsed:
            parsed["evaluation"] = str(
                parsed.get("notes") or parsed.get("reason") or "pass"
            )
        parsed["verdict"] = str(parsed.get("verdict") or "pass")
    return parsed
