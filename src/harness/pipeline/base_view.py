"""Parse S1 base_view choice (which CAD PNG to edit from)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from harness.pipeline.manifest import VIEW_NAMES

_VALID = set(VIEW_NAMES)

# Complementary second camera when the model only names one view.
_COMPLEMENT = {
    "iso": "z_corner",
    "z_corner": "iso",
    "x_corner": "z_corner",
    "y_corner": "iso",
}


def normalize_base_view(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "iso": "iso",
        "isometric": "iso",
        "z_corner": "z_corner",
        "zcorner": "z_corner",
        "x_corner": "x_corner",
        "xcorner": "x_corner",
        "y_corner": "y_corner",
        "ycorner": "y_corner",
    }
    return aliases.get(key)


def complementary_base_view(primary: str | None) -> str:
    """A second, different corner that usually covers another side of the edit."""
    base = normalize_base_view(primary) or "iso"
    nxt = _COMPLEMENT.get(base, "z_corner")
    if nxt == base:
        nxt = next((n for n in VIEW_NAMES if n != base), "iso")
    return nxt


def _json_object(text: str) -> dict[str, Any] | None:
    raw = str(text).strip()
    if not raw:
        return None
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


def _dedupe_views(items: list[str | None]) -> list[str]:
    out: list[str] = []
    for item in items:
        key = normalize_base_view(item) if item is not None else None
        if key and key in _VALID and key not in out:
            out.append(key)
    return out


def parse_base_view(text: str) -> str | None:
    """Extract a single ``base_view`` from JSON (or bare name) in model text."""
    views = parse_base_views(text, count=1)
    return views[0] if views else None


def parse_base_views(text: str, *, count: int = 2) -> list[str]:
    """Extract up to ``count`` distinct base views from model JSON/text.

    Accepts ``base_views: [...]``, ``base_view`` + ``base_view_2``, or a single
    ``base_view``. Pads with a complementary camera when fewer than ``count``.
    """
    want = max(1, int(count))
    found: list[str] = []
    data = _json_object(text) if text else None
    if isinstance(data, dict):
        raw_list = data.get("base_views")
        if raw_list is None:
            raw_list = data.get("base_view_list")
        if isinstance(raw_list, list):
            found.extend(_dedupe_views([str(x) for x in raw_list]))
        elif isinstance(raw_list, str):
            found.extend(_dedupe_views([raw_list]))
        for key in ("base_view", "base_view_2", "secondary_view", "second_base_view"):
            if key in data and data[key] is not None:
                found.extend(_dedupe_views([str(data[key])]))
        found = _dedupe_views(found)

    if not found and text:
        # Prefer explicit base_view= tokens, then first valid view name.
        raw = str(text)
        for name in VIEW_NAMES:
            if re.search(
                rf"base_views?[\"'\s:=]+{re.escape(name)}",
                raw,
                flags=re.IGNORECASE,
            ):
                found.append(name)
        if not found:
            one = None
            for name in VIEW_NAMES:
                if re.search(rf"\b{re.escape(name)}\b", raw, flags=re.IGNORECASE):
                    one = name
                    break
            if one:
                found.append(one)

    found = _dedupe_views(found)
    while len(found) < want:
        nxt = complementary_base_view(found[-1] if found else "iso")
        if nxt in found:
            nxt = next((n for n in VIEW_NAMES if n not in found), None)
            if nxt is None:
                break
        found.append(nxt)
    return found[:want]


def resolve_base_views(artifacts: dict[str, Any] | None, *, count: int = 2) -> list[str]:
    """Read ``base_views`` / ``base_view`` from a manifest artifacts dict."""
    arts = artifacts or {}
    found: list[str] = []
    raw_list = arts.get("base_views")
    if isinstance(raw_list, list):
        found.extend(_dedupe_views([str(x) for x in raw_list]))
    elif isinstance(raw_list, str):
        found.extend(_dedupe_views([raw_list]))
    one = arts.get("base_view")
    if one:
        found.extend(_dedupe_views([str(one)]))
    found = _dedupe_views(found)
    if not found:
        found = parse_base_views("", count=count)
    while len(found) < count:
        nxt = complementary_base_view(found[-1] if found else "iso")
        if nxt in found:
            nxt = next((n for n in VIEW_NAMES if n not in found), None)
            if nxt is None:
                break
        found.append(nxt)
    return found[:count]


def order_views_for_base(
    views: dict[str, Any] | None,
    base_view: str | None,
) -> list[Path]:
    """Paths with ``base_view`` first, then the other three in VIEW_NAMES order."""
    if not views:
        return []
    base = normalize_base_view(base_view) or "iso"
    if base not in _VALID:
        base = "iso"
    ordered: list[Path] = []
    seen: set[str] = set()
    for key in (base, *VIEW_NAMES):
        if key in seen or key not in views:
            continue
        val = views[key]
        if isinstance(val, str) and Path(val).is_file():
            ordered.append(Path(val))
            seen.add(key)
    return ordered
