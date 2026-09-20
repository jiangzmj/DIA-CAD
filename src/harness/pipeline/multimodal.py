"""Build OpenAI-compatible multimodal user content (text + images)."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from harness.pipeline.manifest import VIEW_NAMES


def _mime(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def image_data_url(path: Path | str) -> str:
    path = Path(path)
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{_mime(path)};base64,{b64}"


def image_part(path: Path | str, *, detail: str = "low") -> dict[str, Any]:
    """One OpenAI vision ``image_url`` content part."""
    return {
        "type": "image_url",
        "image_url": {"url": image_data_url(path), "detail": detail},
    }


def existing_images(paths: list[Path | str] | None) -> list[Path]:
    out: list[Path] = []
    for p in paths or []:
        path = Path(p)
        if path.is_file():
            out.append(path)
    return out


def collect_view_paths(views: dict[str, Any] | None) -> list[Path]:
    """Ordered view PNGs from manifest.artifacts['views'].

    Prefers the fixed slots ``iso`` / ``z_corner`` / ``x_corner`` / ``y_corner``, then any extras.
    """
    if not views:
        return []
    preferred = list(VIEW_NAMES)
    paths: list[Path] = []
    seen: set[str] = set()
    for key in preferred:
        if key in seen or key not in views:
            continue
        val = views[key]
        if isinstance(val, str) and Path(val).is_file():
            paths.append(Path(val))
            seen.add(key)
    for key in sorted(views.keys()):
        if key in seen:
            continue
        val = views[key]
        if isinstance(val, str) and Path(val).is_file():
            paths.append(Path(val))
            seen.add(key)
    return paths


def user_content(
    text: str,
    images: list[Path | str] | None = None,
    *,
    detail: str = "low",
    max_images: int | None = None,
    caption_images: bool = True,
) -> str | list[dict[str, Any]]:
    """OpenAI chat ``content``: plain string or multimodal parts list.

    When ``caption_images`` is True, each image is preceded by a text part
    ``[stem]`` so the model knows which view it is (filenames are not otherwise visible).
    """
    imgs = existing_images(list(images or []))
    if max_images is not None:
        imgs = imgs[: max(0, max_images)]
    if not imgs:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for path in imgs:
        if caption_images:
            parts.append({"type": "text", "text": f"\n[{path.stem}]"})
        parts.append(image_part(path, detail=detail))
    return parts
