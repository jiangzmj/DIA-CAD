"""Deterministic STEP geometry summary. Feature ids are geometric, not face indices."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

MAX_FEATURES = 40


def inspect_step(
    step_path: Path | str,
    *,
    include_features: bool = True,
) -> dict[str, Any]:
    """Return a structured B-rep summary. Never uses faces[i] as an id."""
    path = Path(step_path)
    if not path.is_file():
        return {"ok": False, "error": f"missing STEP: {path}", "units": "mm"}

    try:
        import cadquery as cq
    except ImportError:
        return {"ok": False, "error": "cadquery not installed", "units": "mm"}

    try:
        wp = cq.importers.importStep(str(path.resolve()))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"import failed: {exc}", "units": "mm"}

    solids = wp.solids().vals() if hasattr(wp, "solids") else []
    shells = wp.shells().vals() if hasattr(wp, "shells") else []

    volume = 0.0
    area = 0.0
    bbox = None
    solid_metrics: list[dict[str, Any]] = []
    for solid in solids:
        solid_volume = 0.0
        try:
            solid_volume = float(solid.Volume())
            volume += solid_volume
        except Exception:  # noqa: BLE001
            pass
        try:
            area += float(solid.Area())
        except Exception:  # noqa: BLE001
            pass
        bb = _bbox_dict(solid)
        bbox = _union_bbox(bbox, bb)
        try:
            center_of_mass = solid.Center()
            solid_center = [
                float(center_of_mass.x),
                float(center_of_mass.y),
                float(center_of_mass.z),
            ]
        except Exception:  # noqa: BLE001
            solid_center = [
                0.5 * (bb["x"][0] + bb["x"][1]),
                0.5 * (bb["y"][0] + bb["y"][1]),
                0.5 * (bb["z"][0] + bb["z"][1]),
            ]
        solid_metrics.append(
            {
                "center_mm": [round(v, 5) for v in solid_center],
                "span_mm": [
                    round(bb["x"][1] - bb["x"][0], 5),
                    round(bb["y"][1] - bb["y"][0], 5),
                    round(bb["z"][1] - bb["z"][0], 5),
                ],
                "volume_mm3": round(solid_volume, 5),
            }
        )

    if bbox is None:
        try:
            bbox = _bbox_dict(wp.val())
        except Exception:  # noqa: BLE001
            bbox = {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "diag": 0.0,
            }

    center = (
        0.5 * (bbox["x"][0] + bbox["x"][1]),
        0.5 * (bbox["y"][0] + bbox["y"][1]),
        0.5 * (bbox["z"][0] + bbox["z"][1]),
    )

    features: list[dict[str, Any]] = []
    plane_normals: list[tuple[float, float, float]] = []
    face_count = 0
    edge_count = 0
    if include_features:
        faces = wp.faces().vals() if hasattr(wp, "faces") else []
        edges = wp.edges().vals() if hasattr(wp, "edges") else []
        face_count = len(faces)
        edge_count = len(edges)
        for face in faces:
            feat = _face_feature(face)
            if feat is None:
                continue
            features.append(feat)
            if feat.get("surface_type") == "plane" and feat.get("normal"):
                plane_normals.append(tuple(feat["normal"]))  # type: ignore[arg-type]
        features.sort(key=lambda f: float(f.get("area_mm2") or 0.0), reverse=True)
        features = features[:MAX_FEATURES]
        for feat in features:
            feat["feature_id"] = _feature_id(feat)
    else:
        face_count = 0
        edge_count = 0

    return {
        "ok": True,
        "path": str(path.resolve()),
        "units": "mm",
        "bounding_box": {
            "x": [round(bbox["x"][0], 4), round(bbox["x"][1], 4)],
            "y": [round(bbox["y"][0], 4), round(bbox["y"][1], 4)],
            "z": [round(bbox["z"][0], 4), round(bbox["z"][1], 4)],
        },
        "center": [round(c, 4) for c in center],
        "diag_mm": round(float(bbox.get("diag") or 0.0), 4),
        "volume_mm3": round(volume, 4),
        "area_mm2": round(area, 4),
        "solid_count": len(solids),
        "solid_metrics": solid_metrics,
        "shell_count": len(shells),
        "face_count": face_count,
        "edge_count": edge_count,
        "major_normals": _cluster_normals(plane_normals),
        "features": features,
        "selector_policy": (
            "Identify faces by type + normal/axis + radius + bbox, "
            "never by array index or .vals()[i]."
        ),
    }


def _bbox_dict(shape: Any) -> dict[str, Any]:
    bb = shape.BoundingBox()
    dx = float(bb.xmax - bb.xmin)
    dy = float(bb.ymax - bb.ymin)
    dz = float(bb.zmax - bb.zmin)
    return {
        "x": [float(bb.xmin), float(bb.xmax)],
        "y": [float(bb.ymin), float(bb.ymax)],
        "z": [float(bb.zmin), float(bb.zmax)],
        "diag": math.sqrt(dx * dx + dy * dy + dz * dz),
    }


def _union_bbox(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    if a is None:
        return b
    if b is None:
        return a
    x = [min(a["x"][0], b["x"][0]), max(a["x"][1], b["x"][1])]
    y = [min(a["y"][0], b["y"][0]), max(a["y"][1], b["y"][1])]
    z = [min(a["z"][0], b["z"][0]), max(a["z"][1], b["z"][1])]
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    return {"x": x, "y": y, "z": z, "diag": math.sqrt(dx * dx + dy * dy + dz * dz)}


def _face_feature(face: Any) -> dict[str, Any] | None:
    try:
        kind = str(face.geomType() or "OTHER").lower()
    except Exception:  # noqa: BLE001
        kind = "other"
    try:
        area = float(face.Area())
    except Exception:  # noqa: BLE001
        area = 0.0
    bb = _bbox_dict(face)
    feat: dict[str, Any] = {
        "surface_type": kind,
        "area_mm2": round(area, 4),
        "bbox": {
            "x": [round(bb["x"][0], 4), round(bb["x"][1], 4)],
            "y": [round(bb["y"][0], 4), round(bb["y"][1], 4)],
            "z": [round(bb["z"][0], 4), round(bb["z"][1], 4)],
        },
        "z_range": [round(bb["z"][0], 4), round(bb["z"][1], 4)],
    }
    if kind == "cylinder":
        cyl = _cylinder_params(face, bb)
        feat.update(cyl)
    elif kind == "plane":
        feat["normal"] = _plane_normal(face)
    elif kind == "cone":
        feat.update(_cone_params(face, bb))
    return feat


def _cylinder_params(face: Any, bb: dict[str, Any]) -> dict[str, Any]:
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface

        ad = BRepAdaptor_Surface(face.wrapped)
        cyl = ad.Cylinder()
        axis = cyl.Axis()
        loc = axis.Location()
        direction = axis.Direction()
        axis_t = _unit((float(direction.X()), float(direction.Y()), float(direction.Z())))
        return {
            "axis": [round(v, 6) for v in axis_t],
            "origin": [round(float(loc.X()), 4), round(float(loc.Y()), 4), round(float(loc.Z()), 4)],
            "radius_mm": round(float(cyl.Radius()), 4),
        }
    except Exception:  # noqa: BLE001
        spans = {
            "x": bb["x"][1] - bb["x"][0],
            "y": bb["y"][1] - bb["y"][0],
            "z": bb["z"][1] - bb["z"][0],
        }
        axis_name = max(spans, key=spans.get)
        axis = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis_name]
        rad_spans = [spans[k] for k in spans if k != axis_name]
        radius = 0.5 * min(rad_spans) if rad_spans else 0.0
        return {
            "axis": list(axis),
            "origin": [
                round(0.5 * (bb["x"][0] + bb["x"][1]), 4),
                round(0.5 * (bb["y"][0] + bb["y"][1]), 4),
                round(0.5 * (bb["z"][0] + bb["z"][1]), 4),
            ],
            "radius_mm": round(radius, 4),
        }


def _cone_params(face: Any, bb: dict[str, Any]) -> dict[str, Any]:
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface

        ad = BRepAdaptor_Surface(face.wrapped)
        cone = ad.Cone()
        axis = cone.Axis()
        direction = axis.Direction()
        return {
            "axis": [round(v, 6) for v in _unit((float(direction.X()), float(direction.Y()), float(direction.Z())))],
            "radius_mm": round(float(cone.RefRadius()), 4),
        }
    except Exception:  # noqa: BLE001
        return {"axis": [0.0, 0.0, 1.0], "radius_mm": round(0.5 * min(bb["x"][1] - bb["x"][0], bb["y"][1] - bb["y"][0]), 4)}


def _plane_normal(face: Any) -> list[float]:
    try:
        n = face.normalAt()
        vec = (float(n.x), float(n.y), float(n.z))
        return [round(v, 6) for v in _unit(vec)]
    except Exception:  # noqa: BLE001
        return [0.0, 0.0, 1.0]


def _unit(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(v * v for v in vec)) or 1.0
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _feature_id(feat: dict[str, Any]) -> str:
    kind = str(feat.get("surface_type") or "face")
    if kind == "cylinder":
        axis = feat.get("axis") or [0, 0, 1]
        origin = feat.get("origin") or [0, 0, 0]
        radius = feat.get("radius_mm") or 0
        z0, z1 = feat.get("z_range") or [0, 0]
        return (
            f"cyl_r{float(radius):.2f}_"
            f"a{axis[0]:.1f}{axis[1]:.1f}{axis[2]:.1f}_"
            f"o{float(origin[0]):.1f}{float(origin[1]):.1f}{float(origin[2]):.1f}_"
            f"z{float(z0):.1f}-{float(z1):.1f}"
        ).replace("-", "m")
    if kind == "plane":
        n = feat.get("normal") or [0, 0, 1]
        bb = feat.get("bbox") or {}
        z = (bb.get("z") or [0, 0])[1]
        return f"plane_n{n[0]:.1f}{n[1]:.1f}{n[2]:.1f}_z{float(z):.1f}".replace("-", "m")
    bb = feat.get("bbox") or {}
    return f"{kind}_a{float(feat.get('area_mm2') or 0):.1f}_z{(bb.get('z') or [0, 0])[0]:.1f}"


def _cluster_normals(normals: list[tuple[float, float, float]], *, limit: int = 6) -> list[list[float]]:
    clusters: list[tuple[float, float, float]] = []
    for n in normals:
        if any(abs(n[0] * c[0] + n[1] * c[1] + n[2] * c[2]) > 0.98 for c in clusters):
            continue
        clusters.append(n)
        if len(clusters) >= limit:
            break
    return [[round(v, 4) for v in c] for c in clusters]
