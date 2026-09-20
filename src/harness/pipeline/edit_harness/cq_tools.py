"""STEP commit DAG plus optional CadQuery primitives.

EditHarness does not execute these primitives. The MLLM writes CadQuery;
``GeomRepo`` snapshots each successful script. ``cut_overlapping`` is the
assembly-safe boolean helper injected into that script.

``run_tool_trace`` remains for unit tests of the library, not the live loop.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

ALLOWED_TOOLS = (
    "import_base",
    "list_solids",
    "list_faces",
    "list_edges",
    "box",
    "rounded_box",
    "cylinder",
    "cone",
    "sphere",
    "polygon",
    "copy",
    "transform",
    "scale",
    "mirror",
    "union",
    "cut",
    "intersect",
    "keep",
    "hole",
    "fillet",
    "chamfer",
    "pad",
    "shell",
    "draft",
    "remove_blends",
    "linear_pattern",
    "circular_pattern",
    "text",
)

MAX_TRACE = 20


def _cq():
    import cadquery as cq

    return cq


class GeomRepo:
    """Named STEP snapshots with parent pointers."""

    def __init__(self, root: Path | str, base_step: Path | str):
        self.root = Path(root)
        self.commits_dir = self.root / "commits"
        self.commits_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "log.json"
        self.seq = 0
        if self.log_path.is_file():
            data = json.loads(self.log_path.read_text(encoding="utf-8"))
            self.commits: dict[str, dict[str, Any]] = dict(data.get("commits") or {})
            self.head = str(data.get("head") or "base")
            self.seq = int(data.get("seq") or 0)
        else:
            self.commits = {}
            self.head = "base"
            dest = self._step_path("base")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base_step, dest)
            self.commits["base"] = {
                "id": "base",
                "parent": None,
                "op": "import_base",
                "args": {},
                "step": str(dest.resolve()),
                "bodies": {},
                "error": None,
            }
            self._save()

    def _step_path(self, commit_id: str) -> Path:
        return self.commits_dir / commit_id / "model.step"

    def _save(self) -> None:
        self.log_path.write_text(
            json.dumps(
                {"head": self.head, "seq": self.seq, "commits": self.commits},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def get_step(self, commit_id: str | None = None) -> Path:
        cid = commit_id or self.head or "base"
        rec = self.commits.get(cid)
        if rec is None:
            raise KeyError(f"unknown commit {cid}")
        return Path(str(rec["step"]))

    def bodies_dir(self, commit_id: str) -> Path:
        return self.commits_dir / commit_id / "bodies"

    def commit(
        self,
        *,
        parent: str,
        op: str,
        args: dict[str, Any],
        step_path: Path | str,
        metrics: dict[str, Any] | None = None,
        bodies_dir: Path | str | None = None,
        error: str | None = None,
    ) -> str:
        self.seq += 1
        cid = f"c{self.seq:03d}"
        dest = self._step_path(cid)
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = Path(step_path)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        bodies: dict[str, str] = {}
        if bodies_dir and Path(bodies_dir).is_dir():
            bdest = self.bodies_dir(cid)
            if bdest.exists():
                shutil.rmtree(bdest)
            shutil.copytree(bodies_dir, bdest)
            for path in sorted(bdest.glob("*.step")):
                bodies[path.stem] = str(path.resolve())
        rec = {
            "id": cid,
            "parent": parent,
            "op": op,
            "args": args,
            "step": str(dest.resolve()),
            "bodies": bodies,
            "metrics": metrics or {},
            "error": error,
        }
        self.commits[cid] = rec
        if error is None:
            self.head = cid
        self._save()
        return cid


def run_tool_trace(
    repo: GeomRepo,
    tools: list[dict[str, Any]],
    *,
    from_commit: str | None = None,
    work_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute a short tool sequence. Snapshots after each successful step."""
    if not isinstance(tools, list) or not tools:
        return {"ok": False, "error": "tool_trace is empty", "steps": [], "commit": from_commit or "base"}
    if len(tools) > MAX_TRACE:
        return {
            "ok": False,
            "error": f"tool_trace longer than {MAX_TRACE}",
            "steps": [],
            "commit": from_commit or "base",
        }
    parent = from_commit or "base"
    try:
        start = repo.get_step(parent)
    except KeyError as exc:
        return {"ok": False, "error": str(exc), "steps": [], "commit": parent}

    cq = _cq()
    session = _Session(cq, start)
    session.load_bodies(repo.bodies_dir(parent))
    work = Path(work_dir) if work_dir else repo.root / "work"
    work.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    last_commit = parent

    for idx, raw in enumerate(tools):
        if not isinstance(raw, dict):
            return _fail(steps, last_commit, f"tool {idx} is not an object")
        op = str(raw.get("op") or "").strip().lower()
        args = dict(raw.get("args") or {})
        for key, value in raw.items():
            if key in {"op", "args"}:
                continue
            args.setdefault(key, value)
        if op not in ALLOWED_TOOLS:
            return _fail(steps, last_commit, f"unknown tool {op}")
        try:
            info = session.apply(op, args)
        except Exception as exc:  # noqa: BLE001
            snap = work / f"fail_{idx}.step"
            try:
                session.export(snap)
            except Exception:  # noqa: BLE001
                snap = None
            steps.append(
                {
                    "index": idx,
                    "op": op,
                    "args": args,
                    "ok": False,
                    "error": str(exc),
                    "commit": last_commit,
                }
            )
            return {
                "ok": False,
                "error": f"tool {op} failed: {exc}",
                "steps": steps,
                "commit": last_commit,
                "step_path": str(repo.get_step(last_commit).resolve()),
            }
        snap = work / f"step_{idx:02d}.step"
        session.export(snap)
        bodies_dir = work / f"step_{idx:02d}_bodies"
        body_names = session.dump_bodies(bodies_dir)
        metrics = session.metrics()
        metrics["bodies"] = body_names
        cid = repo.commit(
            parent=last_commit,
            op=op,
            args=args,
            step_path=snap,
            metrics=metrics,
            bodies_dir=bodies_dir,
        )
        rec = {
            "index": idx,
            "op": op,
            "args": args,
            "ok": True,
            "commit": cid,
            "parent": last_commit,
            "metrics": metrics,
            "info": info,
            "bodies": body_names,
        }
        steps.append(rec)
        last_commit = cid

    return {
        "ok": True,
        "error": None,
        "steps": steps,
        "commit": last_commit,
        "step_path": str(repo.get_step(last_commit).resolve()),
    }


def _fail(steps: list[dict[str, Any]], commit: str, error: str) -> dict[str, Any]:
    return {"ok": False, "error": error, "steps": steps, "commit": commit}


class _Session:
    def __init__(self, cq: Any, step_path: Path):
        self.cq = cq
        imported = cq.importers.importStep(str(step_path.resolve()))
        self.result = imported
        self.named: dict[str, Any] = {"result": imported}

    def apply(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_op_{op}")
        return handler(args)

    def export(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.cq.exporters.export(self.result, str(dest))

    def dump_bodies(self, dest_dir: Path) -> list[str]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        for name, shape in self.named.items():
            if name == "result":
                continue
            path = dest_dir / f"{name}.step"
            try:
                self.cq.exporters.export(shape, str(path))
                names.append(name)
            except Exception:  # noqa: BLE001
                try:
                    self.cq.exporters.export(_compound(self.cq, _solids(self.cq, shape)), str(path))
                    names.append(name)
                except Exception:  # noqa: BLE001
                    continue
        return names

    def load_bodies(self, bodies_dir: Path) -> None:
        if not bodies_dir.is_dir():
            return
        for path in sorted(bodies_dir.glob("*.step")):
            self.named[path.stem] = self.cq.importers.importStep(str(path))

    def metrics(self) -> dict[str, Any]:
        solids = _solids(self.cq, self.result)
        volume = 0.0
        for solid in solids:
            try:
                volume += float(solid.Volume())
            except Exception:  # noqa: BLE001
                pass
        return {"solid_count": len(solids), "volume_mm3": round(volume, 4)}

    def _op_import_base(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"named": list(self.named)}

    def _op_list_solids(self, args: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for i, solid in enumerate(_solids(self.cq, self.result)):
            bb = solid.BoundingBox()
            sid = f"solid_v{float(solid.Volume()):.1f}_x{bb.xmin:.1f}-{bb.xmax:.1f}_y{bb.ymin:.1f}-{bb.ymax:.1f}"
            rows.append(
                {
                    "id": sid.replace("-", "m"),
                    "volume_mm3": round(float(solid.Volume()), 4),
                    "bbox": {
                        "x": [round(bb.xmin, 4), round(bb.xmax, 4)],
                        "y": [round(bb.ymin, 4), round(bb.ymax, 4)],
                        "z": [round(bb.zmin, 4), round(bb.zmax, 4)],
                    },
                    "rank_volume": i,
                }
            )
        rows.sort(key=lambda r: float(r["volume_mm3"]), reverse=True)
        for i, row in enumerate(rows):
            row["rank_volume"] = i
        return {"solids": rows[:40]}

    def _op_box(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("id") or "box")
        dx, dy, dz = _triple(args, ("dx", "dy", "dz"), (1.0, 1.0, 1.0))
        origin = _vec(args.get("origin"), (0.0, 0.0, 0.0))
        solid = self.cq.Solid.makeBox(dx, dy, dz, self.cq.Vector(*origin))
        self.named[name] = solid
        return {"id": name}

    def _op_cylinder(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("id") or "cylinder")
        radius = float(args.get("radius") or args.get("radius_mm") or 1.0)
        height = float(args.get("height") or args.get("height_mm") or 1.0)
        origin = _vec(args.get("origin") or args.get("start"), (0.0, 0.0, 0.0))
        axis = _vec(args.get("axis") or args.get("direction"), (0.0, 0.0, 1.0))
        solid = self.cq.Solid.makeCylinder(radius, height, self.cq.Vector(*origin), self.cq.Vector(*axis))
        self.named[name] = solid
        return {"id": name}

    def _op_sphere(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("id") or "sphere")
        radius = float(args.get("radius") or args.get("radius_mm") or 1.0)
        center = _vec(args.get("center") or args.get("origin"), (0.0, 0.0, 0.0))
        solid = self.cq.Solid.makeSphere(radius, self.cq.Vector(*center))
        self.named[name] = solid
        return {"id": name}

    def _op_copy(self, args: dict[str, Any]) -> dict[str, Any]:
        src = str(args.get("src") or args.get("target") or "")
        name = str(args.get("id") or f"{src}_copy")
        self.named[name] = _copy_shape(self.cq, self._get(src))
        return {"id": name, "src": src}

    def _op_transform(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("target") or args.get("id") or "")
        shape = self._get(name)
        translate = args.get("translate")
        if translate is not None:
            vec = _vec(translate, (0.0, 0.0, 0.0))
            if hasattr(shape, "translate"):
                shape = shape.translate(vec)
            else:
                shape = self.cq.Workplane("XY").newObject([shape]).translate(vec).val()
        rotate = args.get("rotate")
        if isinstance(rotate, dict):
            origin = _vec(rotate.get("origin"), (0.0, 0.0, 0.0))
            axis = _vec(rotate.get("axis"), (0.0, 0.0, 1.0))
            angle = float(rotate.get("angle_deg") or rotate.get("angle") or 0.0)
            end = (origin[0] + axis[0], origin[1] + axis[1], origin[2] + axis[2])
            shape = shape.rotate(origin, end, angle)
        self.named[name] = shape
        if name == "result":
            self.result = self.cq.Workplane("XY").newObject(_solids(self.cq, shape))
        return {"id": name}

    def _op_union(self, args: dict[str, Any]) -> dict[str, Any]:
        dest = str(args.get("id") or "").strip()
        raw_bodies = args.get("bodies") or args.get("parts") or args.get("tools")
        if isinstance(raw_bodies, str):
            raw_bodies = [raw_bodies]
        names: list[str] = []
        if isinstance(raw_bodies, list) and raw_bodies:
            names = [str(x).strip() for x in raw_bodies if str(x).strip()]
        else:
            target = str(args.get("target") or "").strip()
            tool_name = str(args.get("tool") or "").strip()
            names = [n for n in (target, tool_name) if n]
        if len(names) < 2:
            raise ValueError("union needs target+tool or args.bodies with ≥2 names")
        fused = _as_solid(self.cq, self._get(names[0]))
        for name in names[1:]:
            fused = fused.fuse(_as_solid(self.cq, self._get(name)))
        out = dest or names[0]
        self.named[out] = fused
        if out == "result":
            self.result = self.cq.Workplane("XY").newObject(_solids(self.cq, fused))
        return {"id": out, "bodies": names}

    def _op_intersect(self, args: dict[str, Any]) -> dict[str, Any]:
        target = str(args.get("target") or "result")
        tool_name = str(args.get("tool") or "")
        a = self._get(target)
        b = self._get(tool_name)
        common = _as_solid(self.cq, a).intersect(_as_solid(self.cq, b))
        self._store(target, common)
        return {"target": target, "tool": tool_name}

    def _op_cut(self, args: dict[str, Any]) -> dict[str, Any]:
        target_sel = str(args.get("target") or "solids_overlapping:tool")
        tool_name = str(args.get("tool") or "")
        tool = self._get(tool_name)
        if target_sel.startswith("solids_overlapping:"):
            self.result = cut_overlapping(self.result, tool)
        else:
            tool_s = _as_solid(self.cq, tool)
            keep: list[Any] = []
            for solid in _solids(self.cq, self.result):
                keep.extend(_solids(self.cq, solid.cut(tool_s)))
            self.result = _compound(self.cq, keep)
        self.named["result"] = self.result
        return {"tool": tool_name, "solids_after": len(_solids(self.cq, self.result))}

    def _op_keep(self, args: dict[str, Any]) -> dict[str, Any]:
        solids = _solids(self.cq, self.result)
        min_vol = args.get("min_volume_mm3")
        overlapping = args.get("overlapping")
        ref_bb = None
        if overlapping:
            ref_bb = _as_solid(self.cq, self._get(str(overlapping))).BoundingBox()
        kept = []
        for solid in solids:
            if min_vol is not None and float(solid.Volume()) < float(min_vol):
                continue
            if ref_bb is not None and not _bbox_overlap(solid.BoundingBox(), ref_bb):
                continue
            kept.append(solid)
        if not kept:
            raise ValueError("keep() removed every solid")
        self.result = _compound(self.cq, kept)
        self.named["result"] = self.result
        return {"solids_after": len(kept)}

    def _op_list_faces(self, args: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for face in _faces(self.cq, self._target_shape(args)):
            bb = face.BoundingBox()
            rec = {
                "geom": str(face.geomType()),
                "area_mm2": round(float(face.Area()), 4) if hasattr(face, "Area") else None,
                "bbox": _bbox_dict(bb),
            }
            if rec["geom"] == "CYLINDER":
                rec["radius_mm"] = _face_radius(face)
            rows.append(rec)
        rows.sort(key=lambda r: float(r.get("area_mm2") or 0.0), reverse=True)
        return {"faces": rows[:40]}

    def _op_list_edges(self, args: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for edge in _edges(self.cq, self._target_shape(args)):
            bb = edge.BoundingBox()
            rec = {
                "geom": str(edge.geomType()),
                "length_mm": round(float(edge.Length()), 4),
                "bbox": _bbox_dict(bb),
            }
            if rec["geom"] == "CIRCLE":
                rec["radius_mm"] = _edge_radius(edge)
            rows.append(rec)
        rows.sort(key=lambda r: float(r["length_mm"]), reverse=True)
        return {"edges": rows[:60]}

    def _op_rounded_box(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("id") or "rounded_box")
        dx, dy, dz = _triple(args, ("dx", "dy", "dz"), (1.0, 1.0, 1.0))
        origin = _vec(args.get("origin"), (0.0, 0.0, 0.0))
        radius = float(args.get("corner_radius") or args.get("radius") or 0.0)
        solid = self.cq.Solid.makeBox(dx, dy, dz, self.cq.Vector(*origin))
        if radius > 1e-6:
            axis = _vec(args.get("round_axis") or args.get("fillet_axis"), (0.0, 1.0, 0.0))
            edges = _filter_edges(_edges(self.cq, solid), {"parallel_to": axis})
            if not edges:
                raise ValueError("rounded_box: no edges parallel to round_axis")
            solid = solid.fillet(radius, edges)
        self.named[name] = solid
        return {"id": name, "corner_radius": radius}

    def _op_cone(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("id") or "cone")
        r1 = float(args.get("radius1") or args.get("r1") or 1.0)
        r2 = float(args.get("radius2") or args.get("r2") or 0.0)
        height = float(args.get("height") or args.get("height_mm") or 1.0)
        origin = _vec(args.get("origin") or args.get("start"), (0.0, 0.0, 0.0))
        axis = _vec(args.get("axis") or args.get("direction"), (0.0, 0.0, 1.0))
        solid = self.cq.Solid.makeCone(r1, r2, height, self.cq.Vector(*origin), self.cq.Vector(*axis))
        self.named[name] = solid
        return {"id": name}

    def _op_polygon(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("id") or "polygon")
        n_sides = int(args.get("n_sides") or args.get("sides") or 6)
        diameter = float(args.get("diameter") or args.get("diameter_mm") or 0.0)
        if diameter <= 0:
            radius = float(args.get("radius") or args.get("radius_mm") or 1.0)
            diameter = 2.0 * radius
        height = float(args.get("height") or args.get("height_mm") or 1.0)
        origin = _vec(args.get("origin"), (0.0, 0.0, 0.0))
        axis = _vec(args.get("axis") or args.get("direction"), (0.0, 0.0, 1.0))
        inscribed = bool(args.get("inscribed", True))
        solid = (
            self.cq.Workplane("XY")
            .polygon(n_sides, diameter, circumscribed=not inscribed)
            .extrude(height)
            .val()
        )
        solid = _align_along_axis(self.cq, solid, origin, axis)
        self.named[name] = solid
        return {"id": name, "n_sides": n_sides, "diameter": diameter}

    def _op_scale(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("target") or args.get("id") or "result")
        factor = float(args.get("factor") or args.get("scale") or 1.0)
        origin = _vec(args.get("origin"), (0.0, 0.0, 0.0))
        shape = self._get(name)
        moved = _translate_shape(self.cq, shape, (-origin[0], -origin[1], -origin[2]))
        if hasattr(moved, "scale"):
            scaled = moved.scale(factor)
        else:
            scaled = _as_solid(self.cq, moved).scale(factor)
        scaled = _translate_shape(self.cq, scaled, origin)
        self._set_shape(name, scaled)
        return {"id": name, "factor": factor}

    def _op_mirror(self, args: dict[str, Any]) -> dict[str, Any]:
        src = str(args.get("src") or args.get("target") or "result")
        dest = str(args.get("id") or src)
        plane = str(args.get("plane") or "YZ").upper()
        origin = _vec(args.get("origin"), (0.0, 0.0, 0.0))
        combine = bool(args.get("combine") if args.get("combine") is not None else args.get("union", True))
        shape = self._get(src)
        wp = self.cq.Workplane("XY").newObject(_solids(self.cq, shape))
        mirrored = wp.mirror(plane, origin, union=False)
        if combine and src == "result":
            fused = _as_solid(self.cq, shape).fuse(_as_solid(self.cq, mirrored))
            self._set_shape("result", fused)
            if dest not in {"result", src}:
                self.named[dest] = mirrored
            return {"id": "result", "plane": plane, "combined": True}
        self._set_shape(dest, mirrored)
        if combine and dest != "result":
            fused = _as_solid(self.cq, self.result).fuse(_as_solid(self.cq, mirrored))
            self._set_shape("result", fused)
        return {"id": dest, "plane": plane, "combined": combine}

    def _op_hole(self, args: dict[str, Any]) -> dict[str, Any]:
        diameter = float(args.get("diameter") or args.get("diameter_mm") or 0.0)
        if diameter <= 0:
            diameter = 2.0 * float(args.get("radius") or args.get("radius_mm") or 1.0)
        origin = _vec(args.get("origin") or args.get("center"), (0.0, 0.0, 0.0))
        axis = _vec(args.get("axis") or args.get("direction"), (0.0, 0.0, -1.0))
        depth = args.get("depth") or args.get("depth_mm")
        height = float(depth) if depth is not None else max(_diag(self.result) * 2.0, 10.0)
        start = (
            origin[0] - axis[0] * 0.05 * height,
            origin[1] - axis[1] * 0.05 * height,
            origin[2] - axis[2] * 0.05 * height,
        )
        cutter = self.cq.Solid.makeCylinder(diameter / 2.0, height * 1.1, self.cq.Vector(*start), self.cq.Vector(*axis))
        tool_name = str(args.get("id") or "hole_cutter")
        self.named[tool_name] = cutter
        return self._op_cut({"target": f"solids_overlapping:{tool_name}", "tool": tool_name})

    def _op_fillet(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("target") or "result")
        radius = float(args.get("radius") or args.get("radius_mm") or 0.0)
        if radius <= 0:
            raise ValueError("fillet radius must be > 0")
        n = _edge_edit_solids(self.cq, self._get(name), args, kind="fillet", size=radius)
        self._set_shape(name, _compound(self.cq, n["solids"]))
        return {"id": name, "edges": n["n_edges"], "solids_changed": n["n_changed"]}

    def _op_chamfer(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("target") or "result")
        d1 = float(args.get("distance") or args.get("length") or args.get("d1") or args.get("size") or 0.0)
        d2 = args.get("d2") or args.get("length2")
        if d1 <= 0:
            raise ValueError("chamfer distance must be > 0")
        n = _edge_edit_solids(
            self.cq,
            self._get(name),
            args,
            kind="chamfer",
            size=d1,
            size2=float(d2) if d2 is not None else None,
        )
        self._set_shape(name, _compound(self.cq, n["solids"]))
        return {"id": name, "edges": n["n_edges"], "solids_changed": n["n_changed"]}

    def _op_pad(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("target") or "result")
        distance = float(args.get("distance") or args.get("distance_mm") or args.get("thickness") or 0.0)
        if abs(distance) < 1e-9:
            raise ValueError("pad distance must be non-zero")
        if _parse_bbox(args) is None and args.get("normal") is None:
            raise ValueError("pad needs bbox and/or normal to pick faces")
        faces = _filter_faces(_faces(self.cq, self._get(name)), args)
        if not faces:
            raise ValueError("pad: no faces matched")
        fused = _as_solid(self.cq, self._get(name))
        used = 0
        for face in faces:
            try:
                chunk = face.thicken(distance)
            except Exception:  # noqa: BLE001
                continue
            if distance >= 0:
                fused = fused.fuse(chunk)
            else:
                fused = fused.cut(chunk)
            used += 1
        if used == 0:
            raise ValueError("pad: thicken failed on matched faces")
        self._set_shape(name, fused)
        return {"id": name, "faces": used, "distance": distance}

    def _op_shell(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("target") or "result")
        thickness = float(args.get("thickness") or args.get("thickness_mm") or 1.0)
        wp = self.cq.Workplane("XY").newObject(_solids(self.cq, self._get(name)))
        selector = _box_selector(self.cq, args)
        if selector is not None:
            wp = wp.faces(selector)
        shelled = wp.shell(thickness)
        self._set_shape(name, shelled)
        return {"id": name, "thickness": thickness}

    def _op_draft(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("target") or "result")
        angle = float(args.get("angle_deg") or args.get("angle") or 0.0)
        if abs(angle) < 1e-9:
            raise ValueError("draft angle_deg must be non-zero")
        pull = _vec(args.get("direction") or args.get("pull"), (0.0, 0.0, 1.0))
        hinge_origin = _vec(args.get("hinge_origin") or args.get("origin"), (0.0, 0.0, 0.0))
        hinge_normal = _vec(args.get("hinge_normal") or args.get("normal"), pull)
        faces = _filter_faces(_faces(self.cq, self._get(name)), args)
        if not faces:
            raise ValueError("draft: no faces matched")
        from math import radians

        from OCP.BRepOffsetAPI import BRepOffsetAPI_DraftAngle
        from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Ax3

        solid = _as_solid(self.cq, self._get(name))
        draft = BRepOffsetAPI_DraftAngle(solid.wrapped)
        plane = gp_Pln(gp_Ax3(gp_Pnt(*hinge_origin), gp_Dir(*hinge_normal)))
        direction = gp_Dir(*pull)
        ang = radians(angle)
        for face in faces:
            draft.Add(face.wrapped, direction, ang, plane, True)
        if not draft.AddDone():
            raise ValueError("draft: OCC rejected one or more faces")
        draft.Build()
        if not draft.IsDone():
            raise ValueError("draft: OCC Build failed")
        self._set_shape(name, self.cq.Shape.cast(draft.Shape()))
        return {"id": name, "faces": len(faces), "angle_deg": angle}

    def _op_remove_blends(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("target") or "result")
        max_r = args.get("max_radius_mm") or args.get("max_radius")
        faces = _filter_faces(_faces(self.cq, self._get(name)), args)
        blends = []
        for face in faces:
            g = str(face.geomType()).upper()
            if g not in {"CYLINDER", "TORUS", "CONE", "SPHERE"}:
                continue
            rad = _face_radius(face)
            if max_r is not None and rad is not None and rad > float(max_r) + 1e-6:
                continue
            blends.append(face)
        if not blends:
            raise ValueError("remove_blends: no cylindrical/toroidal faces matched")
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing

        solid = _as_solid(self.cq, self._get(name))
        algo = BRepAlgoAPI_Defeaturing()
        algo.SetShape(solid.wrapped)
        for face in blends:
            algo.AddFaceToRemove(face.wrapped)
        algo.Build()
        if not algo.IsDone():
            raise ValueError("remove_blends: OCC Defeaturing failed")
        self._set_shape(name, self.cq.Shape.cast(algo.Shape()))
        return {"id": name, "faces_removed": len(blends)}

    def _op_linear_pattern(self, args: dict[str, Any]) -> dict[str, Any]:
        src = str(args.get("src") or args.get("target") or "")
        count = int(args.get("count") or 2)
        offset = _vec(args.get("offset") or args.get("spacing"), (10.0, 0.0, 0.0))
        dest = str(args.get("id") or f"{src}_pattern")
        combine = bool(args.get("combine") if args.get("combine") is not None else True)
        if count < 2:
            raise ValueError("linear_pattern count must be >= 2")
        base = self._get(src)
        copies = [_copy_shape(self.cq, base)]
        for i in range(1, count):
            copies.append(
                _translate_shape(
                    self.cq,
                    _copy_shape(self.cq, base),
                    (offset[0] * i, offset[1] * i, offset[2] * i),
                )
            )
        fused = _fuse_many(self.cq, copies)
        self.named[dest] = fused
        if combine:
            self._set_shape("result", _as_solid(self.cq, self.result).fuse(_as_solid(self.cq, fused)))
        return {"id": dest, "count": count, "combined": combine}

    def _op_circular_pattern(self, args: dict[str, Any]) -> dict[str, Any]:
        src = str(args.get("src") or args.get("target") or "")
        count = int(args.get("count") or 2)
        origin = _vec(args.get("origin") or args.get("axis_origin"), (0.0, 0.0, 0.0))
        axis = _vec(args.get("axis") or args.get("direction"), (0.0, 0.0, 1.0))
        dest = str(args.get("id") or f"{src}_pattern")
        combine = bool(args.get("combine") if args.get("combine") is not None else True)
        if count < 2:
            raise ValueError("circular_pattern count must be >= 2")
        base = self._get(src)
        end = (origin[0] + axis[0], origin[1] + axis[1], origin[2] + axis[2])
        copies = [_copy_shape(self.cq, base)]
        step = 360.0 / float(count)
        for i in range(1, count):
            piece = _copy_shape(self.cq, base)
            if hasattr(piece, "rotate"):
                piece = piece.rotate(origin, end, step * i)
            else:
                piece = self.cq.Workplane("XY").newObject(_solids(self.cq, piece)).rotate(origin, end, step * i)
            copies.append(piece)
        fused = _fuse_many(self.cq, copies)
        self.named[dest] = fused
        if combine:
            self._set_shape("result", _as_solid(self.cq, self.result).fuse(_as_solid(self.cq, fused)))
        return {"id": dest, "count": count, "combined": combine}

    def _op_text(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("id") or "text")
        txt = str(args.get("text") or args.get("txt") or "")
        if not txt:
            raise ValueError("text is empty")
        size = float(args.get("size") or args.get("fontsize") or 10.0)
        height = float(args.get("height") or args.get("distance") or 1.0)
        origin = _vec(args.get("origin"), (0.0, 0.0, 0.0))
        combine = str(args.get("combine") or "a")
        solid = self.cq.Workplane("XY").text(txt, size, height, combine=False, font="Arial").val()
        solid = _translate_shape(self.cq, solid, origin)
        self.named[name] = solid
        if combine in {"a", "union", "true", "True"}:
            self._set_shape("result", _as_solid(self.cq, self.result).fuse(_as_solid(self.cq, solid)))
        elif combine in {"s", "cut"}:
            self._set_shape("result", _as_solid(self.cq, self.result).cut(_as_solid(self.cq, solid)))
        return {"id": name, "text": txt, "combine": combine}

    def _target_shape(self, args: dict[str, Any]) -> Any:
        return self._get(str(args.get("target") or "result"))

    def _set_shape(self, name: str, shape: Any) -> None:
        self.named[name] = shape
        if name == "result":
            self.result = _compound(self.cq, _solids(self.cq, shape))

    def _get(self, name: str) -> Any:
        if name in self.named:
            return self.named[name]
        if name == "result":
            return self.result
        raise KeyError(f"unknown body {name}")

    def _store(self, name: str, shape: Any) -> None:
        self.named[name] = shape
        if name == "result":
            self.result = self.cq.Workplane("XY").newObject(_solids(self.cq, shape))


def _solids(cq: Any, shape: Any) -> list[Any]:
    if shape is None:
        return []
    if hasattr(shape, "solids") and hasattr(shape, "vals"):
        try:
            return list(shape.solids().vals())
        except Exception:  # noqa: BLE001
            pass
    wp = cq.Workplane("XY").newObject([shape])
    try:
        return list(wp.solids().vals())
    except Exception:  # noqa: BLE001
        return [shape]


def _as_solid(cq: Any, shape: Any) -> Any:
    if shape is None:
        raise ValueError("empty shape")
    if hasattr(shape, "wrapped"):
        solids = _solids(cq, shape)
        if len(solids) == 1:
            return solids[0]
        if solids:
            return cq.Compound.makeCompound(solids)
        return shape
    solids = _solids(cq, shape)
    if not solids:
        raise ValueError("no solids")
    if len(solids) == 1:
        return solids[0]
    return cq.Compound.makeCompound(solids)


def _compound(cq: Any, solids: list[Any]) -> Any:
    if not solids:
        raise ValueError("no solids")
    if len(solids) == 1:
        return cq.Workplane("XY").newObject(solids)
    return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])


def _copy_shape(cq: Any, shape: Any) -> Any:
    if hasattr(shape, "val"):
        try:
            val = shape.val()
            if hasattr(val, "copy"):
                return val.copy()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(shape, "copy"):
        try:
            return shape.copy()
        except Exception:  # noqa: BLE001
            pass
    solids = _solids(cq, shape)
    copied = [s.copy() if hasattr(s, "copy") else s for s in solids]
    return _compound(cq, copied)


def _bbox_overlap(a: Any, b: Any, tol: float = 1e-6) -> bool:
    return not (
        a.xmax < b.xmin - tol
        or b.xmax < a.xmin - tol
        or a.ymax < b.ymin - tol
        or b.ymax < a.ymin - tol
        or a.zmax < b.zmin - tol
        or b.zmax < a.zmin - tol
    )


def cut_overlapping(base: Any, tool: Any) -> Any:
    """Cut only solids whose bounding box overlaps the tool.

    Use this instead of ``base.cut(tool)`` on assemblies: a whole-body cut
    can fragment unrelated solids.
    """
    cq = _cq()
    tool_s = _as_solid(cq, tool)
    bb = tool_s.BoundingBox()
    keep: list[Any] = []
    for solid in _solids(cq, base):
        if _bbox_overlap(solid.BoundingBox(), bb):
            keep.extend(_solids(cq, solid.cut(tool_s)))
        else:
            keep.append(solid)
    if not keep:
        raise ValueError("cut_overlapping removed every solid")
    return _compound(cq, keep)


def _vec(raw: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    return default


def _triple(args: dict[str, Any], keys: tuple[str, str, str], default: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        float(args.get(keys[0], default[0])),
        float(args.get(keys[1], default[1])),
        float(args.get(keys[2], default[2])),
    )


def _bbox_dict(bb: Any) -> dict[str, list[float]]:
    return {
        "x": [round(bb.xmin, 4), round(bb.xmax, 4)],
        "y": [round(bb.ymin, 4), round(bb.ymax, 4)],
        "z": [round(bb.zmin, 4), round(bb.zmax, 4)],
    }


def _parse_bbox(args: dict[str, Any]) -> dict[str, tuple[float, float]] | None:
    raw = args.get("bbox") or args.get("region")
    if not isinstance(raw, dict):
        return None
    out: dict[str, tuple[float, float]] = {}
    for axis in ("x", "y", "z"):
        pair = raw.get(axis) or raw.get(f"{axis}lim")
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            out[axis] = (float(pair[0]), float(pair[1]))
            continue
        lo = raw.get(f"{axis}min")
        hi = raw.get(f"{axis}max")
        if lo is not None and hi is not None:
            out[axis] = (float(lo), float(hi))
    return out or None


def _bb_overlaps_region(bb: Any, region: dict[str, tuple[float, float]], tol: float = 1e-6) -> bool:
    for axis, (lo, hi) in region.items():
        amin = float(getattr(bb, f"{axis}min"))
        amax = float(getattr(bb, f"{axis}max"))
        if amax < lo - tol or amin > hi + tol:
            return False
    return True


def _edges(cq: Any, shape: Any) -> list[Any]:
    wp = cq.Workplane("XY").newObject(_solids(cq, shape))
    try:
        return list(wp.edges().vals())
    except Exception:  # noqa: BLE001
        return []


def _faces(cq: Any, shape: Any) -> list[Any]:
    wp = cq.Workplane("XY").newObject(_solids(cq, shape))
    try:
        return list(wp.faces().vals())
    except Exception:  # noqa: BLE001
        return []


def _filter_edges(edges: list[Any], args: dict[str, Any]) -> list[Any]:
    region = _parse_bbox(args)
    axis = args.get("parallel_to") or args.get("axis")
    kind = str(args.get("kind") or args.get("geom") or "").upper()
    min_len = args.get("min_length_mm") or args.get("min_length")
    max_len = args.get("max_length_mm") or args.get("max_length")
    out = []
    for edge in edges:
        if region and not _bb_overlaps_region(edge.BoundingBox(), region):
            continue
        geom = str(edge.geomType()).upper()
        if kind in {"LINE", "CIRCLE"} and geom != kind:
            continue
        length = float(edge.Length())
        if min_len is not None and length < float(min_len):
            continue
        if max_len is not None and length > float(max_len):
            continue
        if axis is not None and not _edge_parallel(edge, _vec(axis, (0.0, 0.0, 1.0))):
            continue
        out.append(edge)
    return out


def _filter_faces(faces: list[Any], args: dict[str, Any]) -> list[Any]:
    region = _parse_bbox(args)
    kind = str(args.get("kind") or args.get("geom") or "").upper()
    normal = args.get("normal")
    min_area = args.get("min_area_mm2")
    out = []
    for face in faces:
        if region and not _bb_overlaps_region(face.BoundingBox(), region):
            continue
        geom = str(face.geomType()).upper()
        if kind and kind not in {"", "ALL"} and geom != kind:
            continue
        if min_area is not None and hasattr(face, "Area") and float(face.Area()) < float(min_area):
            continue
        if normal is not None:
            try:
                n = face.normalAt()
                want = _vec(normal, (0.0, 0.0, 1.0))
                dot = abs(n.x * want[0] + n.y * want[1] + n.z * want[2])
                if dot < 0.9:
                    continue
            except Exception:  # noqa: BLE001
                continue
        out.append(face)
    return out


def _edge_parallel(edge: Any, axis: tuple[float, float, float], tol: float = 0.12) -> bool:
    try:
        start = edge.startPoint()
        end = edge.endPoint()
        vx, vy, vz = end.x - start.x, end.y - start.y, end.z - start.z
        mag = (vx * vx + vy * vy + vz * vz) ** 0.5
        if mag < 1e-9:
            return False
        ax, ay, az = axis
        am = (ax * ax + ay * ay + az * az) ** 0.5 or 1.0
        dot = abs(vx * ax + vy * ay + vz * az) / (mag * am)
        return dot >= 1.0 - tol
    except Exception:  # noqa: BLE001
        return False


def _edge_edit_solids(
    cq: Any,
    shape: Any,
    args: dict[str, Any],
    *,
    kind: str,
    size: float,
    size2: float | None = None,
) -> dict[str, Any]:
    solids = _solids(cq, shape)
    out = []
    n_edges = 0
    n_changed = 0
    for solid in solids:
        matched = _filter_edges(_edges(cq, solid), args)
        if not matched:
            out.append(solid)
            continue
        try:
            if kind == "fillet":
                out.append(solid.fillet(size, matched))
            else:
                out.append(solid.chamfer(size, size2, matched))
            n_edges += len(matched)
            n_changed += 1
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{kind} failed on {len(matched)} edges: {exc}") from exc
    if n_changed == 0:
        raise ValueError(f"{kind}: no edges matched the selector")
    return {"solids": out, "n_edges": n_edges, "n_changed": n_changed}


def _translate_shape(cq: Any, shape: Any, vec: tuple[float, float, float]) -> Any:
    if hasattr(shape, "translate"):
        return shape.translate(vec)
    return cq.Workplane("XY").newObject(_solids(cq, shape)).translate(vec)


def _align_along_axis(cq: Any, solid: Any, origin: tuple[float, float, float], axis: tuple[float, float, float]) -> Any:
    ax, ay, az = axis
    if abs(ax) < 1e-6 and abs(ay) < 1e-6 and az >= 0:
        return _translate_shape(cq, solid, origin)
    from math import acos, degrees

    z = cq.Vector(0, 0, 1)
    target = cq.Vector(*axis)
    if target.Length < 1e-9:
        target = z
    rot_axis = z.cross(target)
    angle = degrees(acos(max(-1.0, min(1.0, z.normalized().dot(target.normalized())))))
    wp = cq.Workplane("XY").newObject([solid])
    if rot_axis.Length > 1e-9 and abs(angle) > 1e-6:
        wp = wp.rotate((0, 0, 0), (rot_axis.x, rot_axis.y, rot_axis.z), angle)
    return wp.translate(origin).val()


def _fuse_many(cq: Any, shapes: list[Any]) -> Any:
    fused = _as_solid(cq, shapes[0])
    for shape in shapes[1:]:
        fused = fused.fuse(_as_solid(cq, shape))
    return fused


def _diag(shape: Any) -> float:
    try:
        bb = shape.val().BoundingBox() if hasattr(shape, "val") else shape.BoundingBox()
        return float((bb.xlen**2 + bb.ylen**2 + bb.zlen**2) ** 0.5)
    except Exception:  # noqa: BLE001
        return 100.0


def _face_radius(face: Any) -> float | None:
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface

        surf = BRepAdaptor_Surface(face.wrapped)
        if str(face.geomType()).upper() == "CYLINDER":
            return float(surf.Cylinder().Radius())
        if str(face.geomType()).upper() == "SPHERE":
            return float(surf.Sphere().Radius())
        if str(face.geomType()).upper() == "TORUS":
            return float(surf.Torus().MinorRadius())
    except Exception:  # noqa: BLE001
        return None
    return None


def _edge_radius(edge: Any) -> float | None:
    try:
        if hasattr(edge, "radius"):
            return float(edge.radius())
        from OCP.BRepAdaptor import BRepAdaptor_Curve

        curve = BRepAdaptor_Curve(edge.wrapped)
        if str(edge.geomType()).upper() == "CIRCLE":
            return float(curve.Circle().Radius())
    except Exception:  # noqa: BLE001
        return None
    return None


def _box_selector(cq: Any, args: dict[str, Any]) -> Any | None:
    region = _parse_bbox(args)
    if not region:
        return None
    p1 = (
        region.get("x", (-1e9, 1e9))[0],
        region.get("y", (-1e9, 1e9))[0],
        region.get("z", (-1e9, 1e9))[0],
    )
    p2 = (
        region.get("x", (-1e9, 1e9))[1],
        region.get("y", (-1e9, 1e9))[1],
        region.get("z", (-1e9, 1e9))[1],
    )
    return cq.selectors.BoxSelector(p1, p2)


def format_trace_for_prompt(trace_result: dict[str, Any] | None, *, limit: int = 8) -> str:
    if not trace_result:
        return ""
    steps = list(trace_result.get("steps") or [])[-limit:]
    lines = ["## Previous tool_trace (reuse from_commit; do not restart from scratch)"]
    lines.append(f"head_commit: {trace_result.get('commit')}")
    if trace_result.get("error"):
        lines.append(f"error: {trace_result.get('error')}")
    for step in steps:
        lines.append(
            json.dumps(
                {k: step.get(k) for k in ("index", "op", "args", "ok", "commit", "error", "metrics")},
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)
