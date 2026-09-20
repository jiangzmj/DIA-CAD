"""Unified, deterministic STEP feature probe.

The probe reports geometric facts and recognizer candidates.  It deliberately
does not recommend CadQuery operations or boolean strategies; those remain an
MLLM decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "feature-probe/v3"
MAX_PROMPT_CANDIDATES = 48


def _round(value: Any, digits: int = 5) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _vec(value: Any, digits: int = 5) -> list[float]:
    attrs = (("x", "y", "z"), ("X", "Y", "Z"))
    for names in attrs:
        try:
            vals = []
            for name in names:
                item = getattr(value, name)
                vals.append(float(item() if callable(item) else item))
            return [round(v, digits) for v in vals]
        except (AttributeError, TypeError, ValueError):
            continue
    return [0.0, 0.0, 0.0]


def _bbox(shape: Any) -> dict[str, list[float]]:
    box = shape.BoundingBox()
    return {
        "x": [round(float(box.xmin), 5), round(float(box.xmax), 5)],
        "y": [round(float(box.ymin), 5), round(float(box.ymax), 5)],
        "z": [round(float(box.zmin), 5), round(float(box.zmax), 5)],
    }


def _span(box: dict[str, list[float]]) -> list[float]:
    return [round(box[a][1] - box[a][0], 5) for a in ("x", "y", "z")]


def _fingerprint(record: dict[str, Any], prefix: str) -> str:
    stable = {
        k: record.get(k)
        for k in ("surface_type", "curve_type", "area_mm2", "length_mm", "center_mm", "bbox", "radius_mm")
        if k in record
    }
    raw = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"


def _face_record(face: Any) -> dict[str, Any]:
    try:
        surface_type = str(face.geomType() or "other").lower()
    except Exception:  # noqa: BLE001
        surface_type = "other"
    rec: dict[str, Any] = {
        "surface_type": surface_type,
        "area_mm2": _round(face.Area()),
        "center_mm": _vec(face.Center()),
        "bbox": _bbox(face),
    }
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface

        adaptor = BRepAdaptor_Surface(face.wrapped)
        if surface_type == "cylinder":
            geom = adaptor.Cylinder()
            rec.update(
                radius_mm=_round(geom.Radius()),
                axis=_vec(geom.Axis().Direction(), 6),
                axis_origin_mm=_vec(geom.Axis().Location()),
            )
        elif surface_type == "cone":
            geom = adaptor.Cone()
            rec.update(
                radius_mm=_round(geom.RefRadius()),
                semi_angle_deg=_round(float(geom.SemiAngle()) * 180.0 / 3.141592653589793),
                axis=_vec(geom.Axis().Direction(), 6),
                axis_origin_mm=_vec(geom.Axis().Location()),
            )
        elif surface_type == "sphere":
            geom = adaptor.Sphere()
            rec.update(radius_mm=_round(geom.Radius()), center_of_curvature_mm=_vec(geom.Location()))
        elif surface_type == "torus":
            geom = adaptor.Torus()
            rec.update(
                major_radius_mm=_round(geom.MajorRadius()),
                minor_radius_mm=_round(geom.MinorRadius()),
                axis=_vec(geom.Axis().Direction(), 6),
                axis_origin_mm=_vec(geom.Axis().Location()),
            )
        elif surface_type == "plane":
            geom = adaptor.Plane()
            rec.update(normal=_vec(geom.Axis().Direction(), 6), plane_origin_mm=_vec(geom.Location()))
    except Exception as exc:  # noqa: BLE001
        rec["parameter_warning"] = type(exc).__name__
    return rec


def _edge_record(edge: Any) -> dict[str, Any]:
    try:
        curve_type = str(edge.geomType() or "other").lower()
    except Exception:  # noqa: BLE001
        curve_type = "other"
    rec: dict[str, Any] = {
        "curve_type": curve_type,
        "length_mm": _round(edge.Length()),
        "center_mm": _vec(edge.Center()),
        "bbox": _bbox(edge),
    }
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Curve

        adaptor = BRepAdaptor_Curve(edge.wrapped)
        if curve_type == "circle":
            circle = adaptor.Circle()
            rec.update(
                radius_mm=_round(circle.Radius()),
                axis=_vec(circle.Axis().Direction(), 6),
                center_of_curvature_mm=_vec(circle.Location()),
            )
        elif curve_type == "line":
            line = adaptor.Line()
            rec.update(
                direction=_vec(line.Direction(), 6),
                line_origin_mm=_vec(line.Location()),
            )
        elif curve_type == "ellipse":
            ellipse = adaptor.Ellipse()
            rec.update(
                major_radius_mm=_round(ellipse.MajorRadius()),
                minor_radius_mm=_round(ellipse.MinorRadius()),
                axis=_vec(ellipse.Axis().Direction(), 6),
            )
    except Exception as exc:  # noqa: BLE001
        rec["parameter_warning"] = type(exc).__name__
    return rec


def _sort_key(rec: dict[str, Any]) -> tuple[Any, ...]:
    return (
        rec.get("surface_type") or rec.get("curve_type") or "",
        -(float(rec.get("area_mm2") or rec.get("length_mm") or 0.0)),
        tuple(rec.get("center_mm") or []),
        json.dumps(rec.get("bbox") or {}, sort_keys=True),
    )


def probe_ocp(step_path: Path | str) -> dict[str, Any]:
    """Enumerate solids, faces, edges and face adjacency with stable geometric ids."""
    started = time.perf_counter()
    path = Path(step_path)
    if not path.is_file():
        return {"ok": False, "error": f"missing STEP: {path}", "elapsed_s": 0.0}
    try:
        import cadquery as cq
    except ImportError:
        return {"ok": False, "error": "cadquery not installed", "elapsed_s": 0.0}
    try:
        workplane = cq.importers.importStep(str(path.resolve()))
        raw_solids = list(workplane.solids().vals())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"STEP import failed: {exc}", "elapsed_s": round(time.perf_counter() - started, 4)}

    prepared: list[tuple[dict[str, Any], Any]] = []
    for solid in raw_solids:
        try:
            rec = {
                "volume_mm3": _round(solid.Volume()),
                "area_mm2": _round(solid.Area()),
                "center_mm": _vec(solid.Center()),
                "bbox": _bbox(solid),
            }
        except Exception as exc:  # noqa: BLE001
            rec = {"error": str(exc), "volume_mm3": 0.0, "area_mm2": 0.0, "center_mm": [0, 0, 0], "bbox": {}}
        prepared.append((rec, solid))
    prepared.sort(key=lambda item: (-(float(item[0].get("volume_mm3") or 0.0)), _sort_key(item[0])))

    solids: list[dict[str, Any]] = []
    faces: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    adjacency: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    largest_volume = float(prepared[0][0].get("volume_mm3") or 0.0) if prepared else 0.0
    valid_boxes = [item[0].get("bbox") or {} for item in prepared]
    assembly_bbox = {
        axis: [
            min(float(box[axis][0]) for box in valid_boxes if axis in box),
            max(float(box[axis][1]) for box in valid_boxes if axis in box),
        ]
        for axis in ("x", "y", "z")
    } if valid_boxes else {axis: [0.0, 0.0] for axis in ("x", "y", "z")}
    assembly_span = _span(assembly_bbox)
    shape_by_solid_id: dict[str, Any] = {}

    for solid_idx, (solid_rec, solid) in enumerate(prepared):
        solid_id = f"solid_{solid_idx:03d}_{_fingerprint(solid_rec, 's').split('_')[-1]}"
        face_items: list[tuple[dict[str, Any], Any]] = []
        for face in solid.Faces():
            try:
                face_items.append((_face_record(face), face))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{solid_id}: face skipped: {exc}")
        face_items.sort(key=lambda item: _sort_key(item[0]))

        edge_by_runtime_key: dict[int, tuple[dict[str, Any], Any]] = {}
        for _rec, face in face_items:
            for edge in face.Edges():
                key = int(edge.hashCode())
                if key not in edge_by_runtime_key:
                    try:
                        edge_by_runtime_key[key] = (_edge_record(edge), edge)
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"{solid_id}: edge skipped: {exc}")
        edge_items = sorted(edge_by_runtime_key.items(), key=lambda item: _sort_key(item[1][0]))
        edge_ids: dict[int, str] = {}
        for edge_idx, (runtime_key, (edge_rec, _edge)) in enumerate(edge_items):
            edge_id = f"{solid_id}_edge_{edge_idx:04d}_{_fingerprint(edge_rec, 'e').split('_')[-1]}"
            edge_ids[runtime_key] = edge_id
            edges.append({"edge_id": edge_id, "solid_id": solid_id, **edge_rec})
            if edge_rec.get("curve_type") in {"line", "circle", "ellipse"}:
                candidates.append(
                    {
                        "candidate_id": f"ocp_{edge_id}",
                        "kind": f"{edge_rec.get('curve_type')}_edge",
                        "source": "ocp",
                        "confidence": 0.35,
                        "solid_ids": [solid_id],
                        "face_ids": [],
                        "edge_ids": [edge_id],
                        "geometry": {
                            k: edge_rec.get(k)
                            for k in (
                                "length_mm",
                                "radius_mm",
                                "direction",
                                "axis",
                                "center_mm",
                                "center_of_curvature_mm",
                                "bbox",
                            )
                            if edge_rec.get(k) is not None
                        },
                        "reason": "analytic edge candidate; semantic role and fillet suitability are not inferred",
                    }
                )

        face_ids: list[str] = []
        edge_to_faces: dict[str, list[str]] = {}
        for face_idx, (face_rec, face) in enumerate(face_items):
            face_id = f"{solid_id}_face_{face_idx:04d}_{_fingerprint(face_rec, 'f').split('_')[-1]}"
            local_edges = []
            for edge in face.Edges():
                edge_id = edge_ids.get(int(edge.hashCode()))
                if edge_id:
                    local_edges.append(edge_id)
                    edge_to_faces.setdefault(edge_id, []).append(face_id)
            face_ids.append(face_id)
            full_face = {"face_id": face_id, "solid_id": solid_id, "edge_ids": sorted(set(local_edges)), **face_rec}
            faces.append(full_face)
            if face_rec.get("surface_type") in {"cylinder", "cone", "sphere", "torus"}:
                candidates.append(
                    {
                        "candidate_id": f"ocp_{face_id}",
                        "kind": f"{face_rec.get('surface_type')}_surface",
                        "source": "ocp",
                        "confidence": 0.45,
                        "solid_ids": [solid_id],
                        "face_ids": [face_id],
                        "geometry": {k: face_rec.get(k) for k in ("radius_mm", "major_radius_mm", "minor_radius_mm", "axis", "axis_origin_mm", "bbox") if face_rec.get(k) is not None},
                        "reason": "analytic surface candidate; semantic role is not inferred",
                    }
                )
        for edge_id, linked in sorted(edge_to_faces.items()):
            adjacency.append({"edge_id": edge_id, "face_ids": sorted(set(linked)), "kind": "shared_edge" if len(set(linked)) > 1 else "boundary_edge"})

        ratio = float(solid_rec.get("volume_mm3") or 0.0) / max(largest_volume, 1e-12)
        detached = len(prepared) > 1
        solid_bbox = solid_rec.get("bbox") or {"x": [0, 0], "y": [0, 0], "z": [0, 0]}
        solid_span = _span(solid_bbox)
        bbox_volume = max(solid_span[0] * solid_span[1] * solid_span[2], 1.0e-12)
        relative_center = []
        boundary_sides = []
        for axis_idx, axis in enumerate(("x", "y", "z")):
            low, high = assembly_bbox[axis]
            axis_span = max(high - low, 1.0e-12)
            center = float((solid_rec.get("center_mm") or [0, 0, 0])[axis_idx])
            relative_center.append(round((center - 0.5 * (low + high)) / axis_span, 6))
            tolerance = max(0.01 * axis_span, 1.0e-4)
            if abs(float(solid_bbox[axis][0]) - low) <= tolerance:
                boundary_sides.append(f"{axis}-")
            if abs(float(solid_bbox[axis][1]) - high) <= tolerance:
                boundary_sides.append(f"{axis}+")
        solid_facts = {
            "solid_id": solid_id,
            **solid_rec,
            "span_mm": solid_span,
            "bbox_fill_ratio": round(abs(float(solid_rec.get("volume_mm3") or 0.0)) / bbox_volume, 6),
            "relative_center": relative_center,
            "boundary_sides": boundary_sides,
            "relative_volume": round(ratio, 6),
            "detached_from_other_solids": detached,
            "face_ids": face_ids,
            "edge_ids": [edge_ids[k] for k in edge_ids],
        }
        solids.append(solid_facts)
        shape_by_solid_id[solid_id] = solid
        if detached:
            candidates.append(
                {
                    "candidate_id": f"ocp_{solid_id}",
                    "kind": "detached_solid",
                    "source": "ocp",
                    "confidence": 1.0,
                    "solid_ids": [solid_id],
                    "face_ids": face_ids,
                    "geometry": {
                        key: solid_facts.get(key)
                        for key in (
                            "bbox",
                            "center_mm",
                            "span_mm",
                            "volume_mm3",
                            "area_mm2",
                            "bbox_fill_ratio",
                            "relative_center",
                            "boundary_sides",
                        )
                    },
                    "reason": "STEP contains multiple top-level solids; purpose is not inferred",
                }
            )

    # Record rotation-tolerant geometric siblings.  This is measurement-only
    # evidence for copy/pattern tasks; it does not assign a semantic role.
    def _near(a: float, b: float, rel: float = 0.015, absolute: float = 1.0e-4) -> bool:
        return abs(a - b) <= max(absolute, rel * max(abs(a), abs(b), 1.0))

    for solid in solids:
        spans = sorted(float(value) for value in (solid.get("span_mm") or []))
        similar: list[str] = []
        for other in solids:
            if other["solid_id"] == solid["solid_id"]:
                continue
            other_spans = sorted(float(value) for value in (other.get("span_mm") or []))
            if (
                _near(float(solid.get("volume_mm3") or 0.0), float(other.get("volume_mm3") or 0.0))
                and _near(float(solid.get("area_mm2") or 0.0), float(other.get("area_mm2") or 0.0))
                and len(spans) == len(other_spans)
                and all(_near(a, b) for a, b in zip(spans, other_spans))
            ):
                similar.append(str(other["solid_id"]))
        solid["similar_solid_ids"] = similar[:16]

    # Check the largest bbox-overlap pairs for true B-Rep interference.  Exact
    # intersection volume is especially useful for collision-removal intent.
    overlap_pairs: list[tuple[float, str, str]] = []
    for idx, left in enumerate(solids):
        for right in solids[idx + 1 :]:
            overlap = []
            for axis in ("x", "y", "z"):
                lo = max(float(left["bbox"][axis][0]), float(right["bbox"][axis][0]))
                hi = min(float(left["bbox"][axis][1]), float(right["bbox"][axis][1]))
                overlap.append(max(0.0, hi - lo))
            bbox_overlap_volume = overlap[0] * overlap[1] * overlap[2]
            if bbox_overlap_volume > 1.0e-8:
                overlap_pairs.append(
                    (bbox_overlap_volume, str(left["solid_id"]), str(right["solid_id"]))
                )
    overlap_pairs.sort(reverse=True)
    solid_relations: list[dict[str, Any]] = []
    for bbox_overlap_volume, left_id, right_id in overlap_pairs[:64]:
        exact_volume = 0.0
        exact_ok = True
        try:
            common = shape_by_solid_id[left_id].intersect(shape_by_solid_id[right_id])
            exact_volume = abs(float(common.Volume())) if common is not None else 0.0
        except Exception as exc:  # noqa: BLE001
            exact_ok = False
            warnings.append(f"{left_id}/{right_id}: intersection probe failed: {type(exc).__name__}")
        solid_relations.append(
            {
                "solid_ids": [left_id, right_id],
                "bbox_overlap_volume_mm3": round(bbox_overlap_volume, 5),
                "exact_intersection_volume_mm3": round(exact_volume, 5) if exact_ok else None,
                "intersects": exact_ok and exact_volume > 1.0e-5,
            }
        )
    solid_relations.sort(
        key=lambda item: (
            -float(item.get("exact_intersection_volume_mm3") or 0.0),
            -float(item.get("bbox_overlap_volume_mm3") or 0.0),
        )
    )

    return {
        "ok": True,
        "elapsed_s": round(time.perf_counter() - started, 4),
        "model": {
            "path": str(path.resolve()),
            "units": "mm",
            "solid_count": len(solids),
            "face_count": len(faces),
            "edge_count": len(edges),
            "is_multi_solid": len(solids) > 1,
            "bounding_box": assembly_bbox,
            "span_mm": assembly_span,
        },
        "solids": solids,
        "solid_relations": solid_relations,
        "faces": faces,
        "edges": edges,
        "adjacency": adjacency,
        "feature_candidates": candidates,
        "warnings": warnings,
    }


def resolve_palmetto_engine(explicit: Path | str | None = None) -> Path | None:
    values = [explicit, os.getenv("PALMETTO_ENGINE_PATH"), shutil.which("palmetto_engine")]
    for value in values:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    return None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _collect_palmetto_features(value: Any, *, trail: str = "root") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, list):
        for idx, item in enumerate(value):
            if isinstance(item, dict):
                kind = item.get("type") or item.get("feature_type") or item.get("kind") or item.get("name")
                face_ids = item.get("face_ids") or item.get("faces") or item.get("faceIds")
                if kind or face_ids:
                    found.append({"trail": f"{trail}[{idx}]", **item})
                else:
                    found.extend(_collect_palmetto_features(item, trail=f"{trail}[{idx}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"features", "holes", "shafts", "fillets", "chamfers", "cavities", "thin_walls"}:
                found.extend(_collect_palmetto_features(item, trail=f"{trail}.{key}"))
            elif isinstance(item, (dict, list)):
                found.extend(_collect_palmetto_features(item, trail=f"{trail}.{key}"))
    return found


def run_palmetto(
    step_path: Path | str,
    *,
    engine_path: Path | str | None = None,
    timeout_s: float = 120.0,
    modules: str = "all",
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run the optional native sidecar and normalize recognized features."""
    engine = resolve_palmetto_engine(engine_path)
    if engine is None:
        return {"ok": False, "available": False, "skipped": True, "error": "palmetto_engine not found", "elapsed_s": 0.0, "feature_candidates": []}
    step = Path(step_path).resolve()
    owned_tmp: tempfile.TemporaryDirectory[str] | None = None
    if output_dir is None:
        owned_tmp = tempfile.TemporaryDirectory(prefix="palmetto_probe_")
        out = Path(owned_tmp.name)
    else:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
    command = [str(engine), "--input", str(step), "--outdir", str(out.resolve()), "--modules", modules]
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        if owned_tmp:
            owned_tmp.cleanup()
        return {"ok": False, "available": True, "skipped": False, "error": str(exc), "elapsed_s": round(time.perf_counter() - started, 4), "feature_candidates": [], "command": command}

    artifacts: dict[str, Any] = {}
    for name in ("features.json", "aag.json", "topology.json", "meta.json", "assembly.json"):
        file = out / name
        if file.is_file():
            artifacts[name] = _load_json(file)
    raw_features: list[dict[str, Any]] = []
    feature_payload = artifacts.get("features.json")
    if feature_payload is not None:
        raw_features.extend(_collect_palmetto_features(feature_payload, trail="features.json"))
    candidates = []
    seen: set[str] = set()
    for idx, item in enumerate(raw_features):
        kind = str(item.get("type") or item.get("feature_type") or item.get("kind") or item.get("name") or "recognized_feature").lower()
        external_faces = item.get("face_ids") or item.get("faces") or item.get("faceIds") or []
        if not isinstance(external_faces, list):
            external_faces = [external_faces]
        signature = json.dumps([kind, external_faces, item.get("center"), item.get("radius")], sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(
            {
                "candidate_id": f"palmetto_{idx:04d}_{hashlib.sha1(signature.encode()).hexdigest()[:8]}",
                "kind": kind,
                "source": "palmetto",
                "confidence": _round(item.get("confidence")) or 0.7,
                "solid_ids": [],
                "face_ids": [],
                "external_face_ids": external_faces,
                "geometry": {k: v for k, v in item.items() if k not in {"trail", "type", "feature_type", "kind", "name", "face_ids", "faces", "faceIds"}},
                "reason": "Palmetto recognizer candidate; external face ids are not assumed to equal OCP ids",
            }
        )
    result = {
        "ok": proc.returncode == 0,
        "available": True,
        "skipped": False,
        "engine_path": str(engine),
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "elapsed_s": round(time.perf_counter() - started, 4),
        "artifacts": sorted(artifacts),
        "aag_summary": {
            "node_count": len((artifacts.get("aag.json") or {}).get("nodes") or [])
            if isinstance(artifacts.get("aag.json"), dict)
            else 0,
            "link_count": len((artifacts.get("aag.json") or {}).get("links") or [])
            if isinstance(artifacts.get("aag.json"), dict)
            else 0,
        },
        "feature_candidates": candidates,
    }
    if owned_tmp:
        owned_tmp.cleanup()
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_feature_probe(
    step_path: Path | str,
    *,
    cache_dir: Path | str | None = None,
    palmetto_engine: Path | str | None = None,
    palmetto_enabled: bool = True,
    palmetto_timeout_s: float = 120.0,
    force: bool = False,
) -> dict[str, Any]:
    """Build/cache the unified contract consumed by the MLLM harness."""
    path = Path(step_path)
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "ok": False, "error": f"missing STEP: {path}"}
    engine = resolve_palmetto_engine(palmetto_engine) if palmetto_enabled else None
    engine_tag = str(engine) if engine else ("missing" if palmetto_enabled else "disabled")
    key_raw = f"{SCHEMA_VERSION}:{_sha256(path)}:{engine_tag}"
    cache_key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
    cache_path = Path(cache_dir) / f"{cache_key}.json" if cache_dir else None
    if cache_path and cache_path.is_file() and not force:
        cached = _load_json(cache_path)
        if isinstance(cached, dict):
            cached["cache"] = {"hit": True, "key": cache_key, "path": str(cache_path.resolve())}
            return cached

    ocp = probe_ocp(path)
    palmetto = (
        run_palmetto(path, engine_path=engine, timeout_s=palmetto_timeout_s)
        if palmetto_enabled
        else {"ok": False, "available": False, "skipped": True, "error": "disabled", "elapsed_s": 0.0, "feature_candidates": []}
    )
    candidates = list(ocp.get("feature_candidates") or []) + list(palmetto.get("feature_candidates") or [])
    warnings = list(ocp.get("warnings") or [])
    if palmetto_enabled and not palmetto.get("ok"):
        warnings.append(f"Palmetto unavailable/failed: {palmetto.get('error') or palmetto.get('stderr') or 'unknown error'}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "ok": bool(ocp.get("ok")),
        "input": {"path": str(path.resolve()), "sha256": _sha256(path)},
        "model": ocp.get("model") or {},
        "solids": ocp.get("solids") or [],
        "solid_relations": ocp.get("solid_relations") or [],
        "faces": ocp.get("faces") or [],
        "edges": ocp.get("edges") or [],
        "adjacency": ocp.get("adjacency") or [],
        "feature_candidates": candidates,
        "sources": {
            "ocp": {"ok": ocp.get("ok"), "elapsed_s": ocp.get("elapsed_s"), "error": ocp.get("error")},
            "palmetto": palmetto,
        },
        "warnings": warnings,
        "policy": {
            "facts_only": True,
            "tool_does_not_choose_cadquery": True,
            "candidate_not_ground_truth": True,
            "external_face_ids_are_namespaced": True,
        },
        "cache": {"hit": False, "key": cache_key, "path": str(cache_path.resolve()) if cache_path else None},
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def compact_probe_for_prompt(report: dict[str, Any]) -> dict[str, Any]:
    """Keep the prompt bounded while preserving ids and candidate evidence."""
    solids = []
    for solid in report.get("solids") or []:
        solids.append({k: solid.get(k) for k in ("solid_id", "volume_mm3", "area_mm2", "center_mm", "bbox", "span_mm", "bbox_fill_ratio", "relative_center", "boundary_sides", "relative_volume", "similar_solid_ids", "detached_from_other_solids")})
    all_candidates = list(report.get("feature_candidates") or [])
    ocp = [c for c in all_candidates if c.get("source") == "ocp"]
    palmetto = [c for c in all_candidates if c.get("source") == "palmetto"]

    def ocp_rank(item: dict[str, Any]) -> tuple[Any, ...]:
        kind = str(item.get("kind") or "")
        priority = 0 if kind == "detached_solid" else (1 if kind.endswith("_edge") else 2)
        length = float((item.get("geometry") or {}).get("length_mm") or 0.0)
        return priority, -length, kind, str(item.get("candidate_id") or "")

    ocp.sort(key=ocp_rank)
    palmetto.sort(
        key=lambda item: (
            -float(item.get("confidence") or 0.0),
            str(item.get("kind") or ""),
            str(item.get("candidate_id") or ""),
        )
    )
    ocp_limit = MAX_PROMPT_CANDIDATES // 2
    candidates = ocp[:ocp_limit] + palmetto[: MAX_PROMPT_CANDIDATES - ocp_limit]

    def compact_candidate(item: dict[str, Any]) -> dict[str, Any]:
        return {
            k: item.get(k)
            for k in (
                "candidate_id",
                "kind",
                "source",
                "confidence",
                "solid_ids",
                "face_ids",
                "edge_ids",
                "external_face_ids",
                "geometry",
            )
            if item.get(k) not in (None, [], {})
        }

    candidates = [compact_candidate(c) for c in candidates]
    return {
        "schema_version": report.get("schema_version"),
        "ok": report.get("ok"),
        "model": report.get("model"),
        "solids": solids,
        "solid_relations": list(report.get("solid_relations") or [])[:24],
        "feature_candidates": candidates,
        "candidate_count": len(all_candidates),
        "prompt_candidate_count": len(candidates),
        "adjacency_count": len(report.get("adjacency") or []),
        "sources": {
            "ocp": (report.get("sources") or {}).get("ocp"),
            "palmetto": {
                k: ((report.get("sources") or {}).get("palmetto") or {}).get(k)
                for k in ("ok", "available", "skipped", "elapsed_s", "error", "returncode")
            },
        },
        "warnings": list(report.get("warnings") or [])[:10],
        "policy": report.get("policy"),
    }
