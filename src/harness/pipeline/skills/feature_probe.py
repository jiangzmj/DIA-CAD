"""Cheap STEP topology probe + optional one-instance reconstruct smoke.

Used by FeatureProbeAgent. No LLM. CadQuery required for a real probe.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


def _cq():
    try:
        import cadquery as cq
    except ImportError:  # pragma: no cover
        return None
    return cq


def inspect_step(step_path: Path | str) -> dict[str, Any]:
    """Count solids / cylindrical faces. Cheap; no mutation."""
    cq = _cq()
    path = Path(step_path)
    if cq is None:
        return {"ok": False, "error": "cadquery not installed", "n_solids": 0}
    if not path.is_file():
        return {"ok": False, "error": f"missing STEP: {path}", "n_solids": 0}

    wp = cq.importers.importStep(str(path.resolve()))
    solids = wp.solids().vals()
    n_solids = len(solids)
    cyl_faces = 0
    cyl_on_largest = 0
    small_solids = 0

    if solids:
        largest = max(
            solids,
            key=lambda s: max(
                s.BoundingBox().xlen,
                s.BoundingBox().ylen,
                s.BoundingBox().zlen,
            ),
        )
        large_span = max(
            largest.BoundingBox().xlen,
            largest.BoundingBox().ylen,
            largest.BoundingBox().zlen,
        )
        small_solids = 0
        for solid in solids:
            bb = solid.BoundingBox()
            span = max(bb.xlen, bb.ylen, bb.zlen)
            if solid is not largest and span < 0.55 * large_span:
                small_solids += 1
        for face in largest.Faces():
            try:
                if face.geomType() == "CYLINDER":
                    cyl_on_largest += 1
            except Exception:  # noqa: BLE001
                pass

    for solid in solids:
        for face in solid.Faces():
            try:
                if face.geomType() == "CYLINDER":
                    cyl_faces += 1
            except Exception:  # noqa: BLE001
                pass

    fused_likely = n_solids == 1 and cyl_on_largest > 0
    missing_in_brep = cyl_faces == 0 and n_solids <= 1

    return {
        "ok": True,
        "path": str(path.resolve()),
        "n_solids": n_solids,
        "n_cyl_faces": cyl_faces,
        "n_cyl_on_largest": cyl_on_largest,
        "n_small_solids": small_solids,
        "fused_feature_likely": fused_likely,
        "no_matching_brep_feature": missing_in_brep,
    }


def smoke_reconstruct_one(
    step_path: Path | str,
    *,
    out_step: Path | str | None = None,
) -> dict[str, Any]:
    """Instantiate **one** extra copy from measured cylinder faces, then union.

    Proves the fused-feature path without patterning the whole parent.
    """
    cq = _cq()
    path = Path(step_path)
    if cq is None:
        return {"ok": False, "error": "cadquery not installed"}
    if not path.is_file():
        return {"ok": False, "error": f"missing STEP: {path}"}

    wp = cq.importers.importStep(str(path.resolve()))
    solid = max(
        wp.solids().vals(),
        key=lambda s: max(
            s.BoundingBox().xlen,
            s.BoundingBox().ylen,
            s.BoundingBox().zlen,
        ),
    )
    cyl = None
    for face in solid.Faces():
        try:
            if face.geomType() == "CYLINDER":
                bb = face.BoundingBox()
                cyl = {
                    "radius": 0.5 * max(bb.xlen, bb.ylen),
                    "height": max(bb.zlen, 1.0),
                    "cx": 0.5 * (bb.xmin + bb.xmax),
                    "cy": 0.5 * (bb.ymin + bb.ymax),
                    "zmin": bb.zmin,
                }
                break
        except Exception:  # noqa: BLE001
            continue
    if cyl is None:
        return {
            "ok": False,
            "error": "no cylindrical face on parent; reconstruct-one not applicable",
            "skipped": True,
        }

    offset = max(2.2 * cyl["radius"], 1.0)
    copy = cq.Solid.makeCylinder(
        max(cyl["radius"], 0.1),
        max(cyl["height"], 0.1),
        cq.Vector(cyl["cx"] + offset, cyl["cy"], cyl["zmin"]),
        cq.Vector(0, 0, 1),
    )
    try:
        result = cq.Workplane("XY").newObject([solid]).union(
            cq.Workplane("XY").newObject([copy]),
            clean=False,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"union failed: {exc}"}

    dest = Path(out_step) if out_step else Path(tempfile.mkdtemp()) / "smoke_one.step"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        cq.exporters.export(result, str(dest))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"export failed: {exc}"}
    ok = dest.is_file() and dest.stat().st_size > 0
    return {
        "ok": ok,
        "method": "measure_faces_union_one",
        "out_step": str(dest.resolve()) if dest.exists() else str(dest),
        "offset": offset,
    }
