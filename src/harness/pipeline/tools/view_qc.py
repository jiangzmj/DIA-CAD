"""Hard QC for CAD view PNGs: white border, centered model, occupancy ≥28%.

White outer border must come from the render (model fully inside the frame),
not from post-painted margins. Occupancy is the axis-aligned bounding-box
footprint of foreground pixels over the image area.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.pipeline.manifest import VIEW_NAMES

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[misc, assignment]


BORDER_PX = 5
MIN_OCCUPANCY = 0.28
CENTER_TOLERANCE = 0.15  # diagonal silhouettes of asymmetric parts are often off-center


@dataclass
class ViewQCResult:
    path: str
    ok: bool
    white_border: bool
    centered: bool
    occupancy: float
    detail: str = ""
    ink_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "white_border": self.white_border,
            "centered": self.centered,
            "occupancy": round(self.occupancy, 4),
            "ink_ratio": round(self.ink_ratio, 4),
            "detail": self.detail,
        }


def _is_white(pixel: tuple[int, ...] | int, threshold: int = 250) -> bool:
    if isinstance(pixel, int):
        return pixel >= threshold
    if len(pixel) >= 3:
        return pixel[0] >= threshold and pixel[1] >= threshold and pixel[2] >= threshold
    return False


def _foreground_mask(img: "Image.Image", threshold: int = 250) -> list[list[bool]]:
    """True where pixel is non-white (model / content)."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    px = rgb.load()
    return [[not _is_white(px[x, y], threshold) for x in range(w)] for y in range(h)]


def check_view_qc(
    path: Path | str,
    *,
    border_px: int = BORDER_PX,
    min_occupancy: float = MIN_OCCUPANCY,
    center_tol: float = CENTER_TOLERANCE,
) -> ViewQCResult:
    path = Path(path)
    if Image is None:
        return ViewQCResult(
            str(path),
            False,
            False,
            False,
            0.0,
            "Pillow not installed",
        )
    if not path.is_file():
        return ViewQCResult(str(path), False, False, False, 0.0, "file missing")

    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:  # noqa: BLE001
        return ViewQCResult(str(path), False, False, False, 0.0, f"decode error: {exc}")

    w, h = img.size
    if w < border_px * 2 + 2 or h < border_px * 2 + 2:
        return ViewQCResult(str(path), False, False, False, 0.0, f"too small {w}x{h}")

    rgb = img.convert("RGB")
    px = rgb.load()

    # Outer border_px rings must be white
    white_border = True
    for y in range(h):
        for x in range(w):
            on_border = x < border_px or y < border_px or x >= w - border_px or y >= h - border_px
            if on_border and not _is_white(px[x, y]):
                white_border = False
                break
        if not white_border:
            break

    mask = _foreground_mask(rgb)
    fg_coords = [(x, y) for y in range(h) for x in range(w) if mask[y][x]]
    total = w * h
    ink_ratio = len(fg_coords) / total if total else 0.0

    if not fg_coords:
        return ViewQCResult(
            str(path),
            False,
            white_border,
            False,
            0.0,
            "no foreground pixels",
            ink_ratio=0.0,
        )

    xs = [c[0] for c in fg_coords]
    ys = [c[1] for c in fg_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # 占比 = model AABB footprint (works for wireframe + solid edge-highlight).
    occupancy = ((max_x - min_x + 1) * (max_y - min_y + 1)) / total if total else 0.0
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    centered = abs(cx - (w - 1) / 2.0) <= center_tol * w and abs(cy - (h - 1) / 2.0) <= center_tol * h

    ok = white_border and centered and occupancy >= min_occupancy
    parts = []
    if not white_border:
        parts.append("border not white")
    if not centered:
        parts.append(f"not centered (cx={cx:.1f},cy={cy:.1f})")
    if occupancy < min_occupancy:
        parts.append(f"occupancy {occupancy:.2%} < {min_occupancy:.0%}")
    detail = "pass" if ok else "; ".join(parts)
    return ViewQCResult(str(path), ok, white_border, centered, occupancy, detail, ink_ratio=ink_ratio)


def check_views_dir(
    views_dir: Path | str,
    names: tuple[str, ...] = VIEW_NAMES,
) -> dict[str, Any]:
    views_dir = Path(views_dir)
    results: dict[str, Any] = {}
    all_ok = True
    missing: list[str] = []
    for name in names:
        p = views_dir / f"{name}.png"
        if not p.is_file():
            missing.append(name)
            all_ok = False
            results[name] = {"ok": False, "detail": "missing"}
            continue
        r = check_view_qc(p)
        results[name] = r.to_dict()
        if not r.ok:
            all_ok = False
    return {
        "ok": all_ok and not missing,
        "missing": missing,
        "views": results,
        "hard": "pass" if all_ok and not missing else "fail",
        "detail": (
            "all views pass"
            if all_ok and not missing
            else "; ".join(
                f"{n}: {(results.get(n) or {}).get('detail') or 'fail'}"
                for n in names
                if n in missing or not (results.get(n) or {}).get("ok")
            )
        ),
    }
