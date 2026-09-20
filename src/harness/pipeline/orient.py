"""Parse upright-orientation helper replies (bottom face → up axis)."""

from __future__ import annotations

import json
import re
from typing import Any

VALID_UP_AXES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

_AXIS_VEC = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}

_OPPOSITE = {
    "+X": "-X",
    "-X": "+X",
    "+Y": "-Y",
    "-Y": "+Y",
    "+Z": "-Z",
    "-Z": "+Z",
}


def normalize_up_axis(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().upper().replace(" ", "")
    aliases = {
        "X": "+X",
        "+X": "+X",
        "POSX": "+X",
        "X+": "+X",
        "-X": "-X",
        "NEGX": "-X",
        "X-": "-X",
        "Y": "+Y",
        "+Y": "+Y",
        "POSY": "+Y",
        "Y+": "+Y",
        "-Y": "-Y",
        "NEGY": "-Y",
        "Y-": "-Y",
        "Z": "+Z",
        "+Z": "+Z",
        "POSZ": "+Z",
        "Z+": "+Z",
        "-Z": "-Z",
        "NEGZ": "-Z",
        "Z-": "-Z",
    }
    key = key.replace("WORLD", "").replace("AXIS", "").replace("_", "")
    return aliases.get(key)


def opposite_axis(axis: str) -> str:
    key = normalize_up_axis(axis) or "+Z"
    return _OPPOSITE[key]


def parse_orient_json(text: str) -> dict[str, Any] | None:
    """Parse orient reply into ``{up_axis, bottom_axis, object, ...}``.

    Prefers ``bottom_axis`` (human ground direction); falls back to
    legacy ``up_axis``. Returns None if unusable.
    """
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    data: Any = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            axis = normalize_up_axis(raw.split()[0] if raw.split() else raw)
            if axis is None:
                return None
            return {"up_axis": axis, "bottom_axis": opposite_axis(axis), "source_field": "bare"}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None

    bottom = None
    for key in ("bottom_axis", "bottom", "down_axis", "down"):
        if key in data:
            bottom = normalize_up_axis(str(data[key]))
            if bottom:
                break

    up = None
    for key in ("up_axis", "up", "axis", "upAxis"):
        if key in data:
            up = normalize_up_axis(str(data[key]))
            if up:
                break

    if bottom and not up:
        up = opposite_axis(bottom)
    if up and not bottom:
        bottom = opposite_axis(up)
    if not up:
        return None

    out: dict[str, Any] = {
        "up_axis": up,
        "bottom_axis": bottom or opposite_axis(up),
        "source_field": "bottom_axis" if "bottom_axis" in data or "bottom" in data else "up_axis",
    }
    for key in ("object", "top_face", "bottom_face", "imagine", "reason"):
        if key in data and data[key] is not None:
            out[key] = str(data[key])
    if "already_upright" in data:
        out["already_upright"] = bool(data["already_upright"])
    return out


def parse_up_axis_json(text: str) -> str | None:
    """Extract up axis from model JSON (legacy helper)."""
    parsed = parse_orient_json(text)
    if parsed is None:
        return None
    return str(parsed["up_axis"])


def axis_vector(up_axis: str) -> tuple[float, float, float]:
    axis = normalize_up_axis(up_axis) or "+Z"
    return _AXIS_VEC[axis]
