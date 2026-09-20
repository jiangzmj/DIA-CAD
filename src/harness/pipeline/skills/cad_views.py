"""cad_views skill: STEP → identify bottom → upright → 4 fixed views + QC.

Pipeline:
  .STEP → tessellate → 7 probe PNGs (iso + 6 orthographic) → agent: bottom / upright
       → rotate so bottom → −Z → VTK render 4 corner look-froms
       → hard QC loop on camera_distance_factor

Final cameras = unit-box corners looking at AABB center.
Probe cameras = one iso + front/back/left/right/top/bottom along raw STEP axes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from harness.pipeline.manifest import VIEW_NAMES
from harness.pipeline.tools.view_qc import check_views_dir

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[misc, assignment]

try:
    import cadquery as cq
except ImportError:  # pragma: no cover
    cq = None  # type: ignore[misc, assignment]

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[misc, assignment]

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
except ImportError:  # pragma: no cover
    vtk = None  # type: ignore[misc, assignment]
    numpy_to_vtk = None  # type: ignore[misc, assignment]


# Named, controller-owned cameras. The first four remain the pipeline's standard
# views; the six axis views are available to the edit harness on demand.
# Direction = camera − center (then normalized in render).
# After uprighting, +Z is natural up. ViewUp = +Z except when look ≈ ±Z.
VIEW_CAMERAS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "iso": ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
    "z_corner": ((-1.0, -1.0, 1.0), (0.0, 0.0, 1.0)),
    "x_corner": ((1.0, -1.0, -1.0), (0.0, 0.0, 1.0)),
    "y_corner": ((-1.0, 1.0, -1.0), (0.0, 0.0, 1.0)),
    "front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "bottom": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}

# Orient probes: iso + six faces along raw STEP axes (pre-upright).
# Orthographic: front/back along ±Y, left/right along ±X, top/bottom along ±Z.
# ViewUp = +Z except when looking along ±Z (then +Y).
PROBE_CAMERAS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "iso": ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
    "front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "bottom": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}

PROBE_VIEW_NAME = "_orient_probe"
PROBE_NAMES = tuple(PROBE_CAMERAS.keys())

# SolidWorks-like olive gray + dark feature edges.
OLIVE = (0.35, 0.38, 0.30)
EDGE = (0.10, 0.10, 0.10)
BG = (255, 255, 255)

DEFAULT_SIZE = 1024
TESSELLATION_TOL = 0.03
TESSELLATION_ANG_TOL = 0.12
VTK_RENDER_SIZE = 1400


def _deps_ok() -> tuple[bool, str]:
    if cq is None:
        return False, "cadquery not installed"
    if np is None:
        return False, "numpy not installed"
    if vtk is None or numpy_to_vtk is None:
        return False, "vtk not installed"
    if Image is None:
        return False, "Pillow not installed"
    return True, "ok"


def _probe_x_display(display: str) -> bool:
    try:
        proc = subprocess.run(
            ["xdpyinfo", "-display", display],
            capture_output=True,
            timeout=3,
            check=False,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        # No xdpyinfo — assume the candidate is usable.
        return True
    except Exception:  # noqa: BLE001
        return False


def _discover_x_display() -> str | None:
    """Find a live X display when DISPLAY is unset (common in plain SSH/TTY shells)."""
    candidates: list[str] = []
    env = os.environ.get("DISPLAY")
    if env:
        candidates.append(env)
    # Prefer sockets under /tmp/.X11-unix (X0 → :0, X1 → :1, …).
    xdir = Path("/tmp/.X11-unix")
    if xdir.is_dir():
        for sock in sorted(xdir.glob("X*")):
            name = sock.name  # X1
            if len(name) > 1 and name[1:].isdigit():
                candidates.append(f":{name[1:]}")
    for extra in (":0", ":1"):
        if extra not in candidates:
            candidates.append(extra)
    seen: set[str] = set()
    for display in candidates:
        if display in seen:
            continue
        seen.add(display)
        if _probe_x_display(display):
            return display
    return None


def _gl_display_ok() -> tuple[bool, str]:
    """VTK wheels here use X OpenGL; a broken DISPLAY aborts the process."""
    display = os.environ.get("DISPLAY")
    if display and _probe_x_display(display):
        return True, "ok"
    found = _discover_x_display()
    if found:
        os.environ["DISPLAY"] = found
        return True, f"ok (auto DISPLAY={found})"
    if not display:
        return False, "DISPLAY unset (use a desktop session or xvfb-run)"
    return False, f"bad X display {display} (xdpyinfo failed)"


def _render_cad_views_via_xvfb(
    step_path: Path,
    out_dir: Path,
    *,
    max_retries: int,
    size: int,
) -> dict[str, Any]:
    """Re-enter ``render_cad_views`` under ``xvfb-run -a``."""
    xvfb = shutil.which("xvfb-run")
    if not xvfb:
        return {
            "ok": False,
            "paths": {n: str((out_dir / f"{n}.png").resolve()) for n in VIEW_NAMES},
            "backend": "gl_unavailable:no_xvfb",
            "attempts": [],
            "qc": {
                "ok": False,
                "hard": "fail",
                "detail": (
                    "DISPLAY broken and xvfb-run not found; "
                    "install with: sudo apt-get install -y xvfb"
                ),
            },
        }

    src_root = Path(__file__).resolve().parents[3]  # .../src
    with tempfile.TemporaryDirectory(prefix="cad_views_xvfb_") as tmp:
        result_path = Path(tmp) / "result.json"
        script = Path(tmp) / "worker.py"
        script.write_text(
            "\n".join(
                [
                    "import json, os, sys",
                    f"sys.path.insert(0, {str(src_root)!r})",
                    "os.environ['HARNESS_UNDER_XVFB'] = '1'",
                    "from harness.pipeline.skills.cad_views import render_cad_views",
                    f"r = render_cad_views({str(step_path)!r}, {str(out_dir)!r}, "
                    f"max_retries={int(max_retries)}, size={int(size)})",
                    f"open({str(result_path)!r}, 'w', encoding='utf-8').write("
                    "json.dumps(r))",
                ]
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["HARNESS_UNDER_XVFB"] = "1"
        env["PYTHONPATH"] = (
            f"{src_root}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(src_root)
        )
        proc = subprocess.run(
            [xvfb, "-a", sys.executable, str(script)],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
            check=False,
        )
        if proc.returncode != 0 or not result_path.is_file():
            detail = (proc.stderr or proc.stdout or "xvfb worker failed")[:2000]
            return {
                "ok": False,
                "paths": {n: str((out_dir / f"{n}.png").resolve()) for n in VIEW_NAMES},
                "backend": "gl_xvfb_failed",
                "attempts": [],
                "qc": {"ok": False, "hard": "fail", "detail": detail},
            }
        data = json.loads(result_path.read_text(encoding="utf-8"))
        data["backend"] = f"{data.get('backend', 'cadquery_vtk_shaded')}+xvfb"
        return data


def tessellate_shape(
    step_path: Path | str,
    *,
    tol: float = TESSELLATION_TOL,
    ang_tol: float = TESSELLATION_ANG_TOL,
) -> tuple[Any, Any]:
    """STEP → (numpy points, triangle indices)."""
    if cq is None or np is None:
        raise RuntimeError("cadquery and numpy are required")
    shape = cq.importers.importStep(str(step_path)).val()
    verts, faces = shape.tessellate(tol, ang_tol)
    pts = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)
    tris = np.array([[a, b, c] for a, b, c in faces], dtype=np.int64)
    return pts, tris


def build_polydata(pts, tris):
    """NumPy mesh → vtkPolyData."""
    if vtk is None or numpy_to_vtk is None:
        raise RuntimeError("vtk is required")
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(pts, deep=True))
    cells = vtk.vtkCellArray()
    for a, b, c in tris:
        cell = vtk.vtkTriangle()
        cell.GetPointIds().SetId(0, int(a))
        cell.GetPointIds().SetId(1, int(b))
        cell.GetPointIds().SetId(2, int(c))
        cells.InsertNextCell(cell)
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetPolys(cells)
    return pd


def render_view(
    pd,
    view_name: str,
    view_dir: tuple[float, float, float],
    view_up: tuple[float, float, float],
    center: tuple[float, float, float],
    diag: float,
    *,
    camera_distance_factor: float = 1.15,
    render_size: int = VTK_RENDER_SIZE,
):
    """Render one camera view; return vtkImageData."""
    if vtk is None:
        raise RuntimeError("vtk is required")

    norms = vtk.vtkPolyDataNormals()
    norms.SetInputData(pd)
    norms.SplittingOn()
    norms.SetFeatureAngle(25.0)
    norms.ConsistencyOn()
    norms.AutoOrientNormalsOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(norms.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*OLIVE)
    # Higher ambient + softer key/fill so recessed faces stay readable
    # instead of near-black (still keep mild diffuse shading for depth).
    prop.SetAmbient(0.62)
    prop.SetDiffuse(0.72)
    prop.SetSpecular(0.12)
    prop.SetSpecularPower(24)
    prop.SetInterpolationToPhong()

    ren = vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)
    ren.TwoSidedLightingOn()

    key = vtk.vtkLight()
    key.SetLightTypeToSceneLight()
    key.SetPosition(
        center[0] + diag * 1.2,
        center[1] + diag * 1.6,
        center[2] + diag * 2.0,
    )
    key.SetFocalPoint(*center)
    key.SetIntensity(0.70)
    ren.AddLight(key)

    fill = vtk.vtkLight()
    fill.SetLightTypeToSceneLight()
    fill.SetPosition(
        center[0] + diag * 1.8,
        center[1] - diag * 1.2,
        center[2] + diag * 0.4,
    )
    fill.SetFocalPoint(*center)
    fill.SetIntensity(0.55)
    ren.AddLight(fill)

    # Camera-follow fill so the front of each view never collapses to black.
    head = vtk.vtkLight()
    head.SetLightTypeToHeadlight()
    head.SetIntensity(0.35)
    ren.AddLight(head)

    cam = ren.GetActiveCamera()
    # Normalize look-from so ResetCamera can frame from this corner.
    length = (view_dir[0] ** 2 + view_dir[1] ** 2 + view_dir[2] ** 2) ** 0.5 or 1.0
    vd = (view_dir[0] / length, view_dir[1] / length, view_dir[2] / length)
    cam.ParallelProjectionOn()
    cam.SetFocalPoint(*center)
    cam.SetPosition(
        center[0] + vd[0] * diag,
        center[1] + vd[1] * diag,
        center[2] + vd[2] * diag,
    )
    cam.SetViewUp(*view_up)

    fe = vtk.vtkFeatureEdges()
    fe.SetInputConnection(norms.GetOutputPort())
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOn()
    fe.SetFeatureAngle(25.0)
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()

    silhouette = vtk.vtkPolyDataSilhouette()
    silhouette.SetInputData(pd)
    silhouette.SetCamera(cam)
    silhouette.SetEnableFeatureAngle(0)

    append = vtk.vtkAppendPolyData()
    append.AddInputConnection(fe.GetOutputPort())
    append.AddInputConnection(silhouette.GetOutputPort())

    edge_mapper = vtk.vtkPolyDataMapper()
    edge_mapper.SetInputConnection(append.GetOutputPort())
    edge_mapper.ScalarVisibilityOff()

    edge_actor = vtk.vtkActor()
    edge_actor.SetMapper(edge_mapper)
    edge_actor.GetProperty().SetColor(*EDGE)
    edge_actor.GetProperty().SetLineWidth(2.0)

    ren.AddActor(actor)
    ren.AddActor(edge_actor)

    # Frame actors from the chosen corner, then zoom out by distance factor
    # (larger factor → more white margin / smaller occupancy).
    ren.ResetCamera()
    # ResetCamera can roll the camera; re-assert world-up so +Z stays image-up.
    cam.SetViewUp(*view_up)
    cam.OrthogonalizeViewUp()
    factor = max(float(camera_distance_factor), 1e-6)
    cam.SetParallelScale(cam.GetParallelScale() * factor)
    cam.SetClippingRange(diag * 0.05, diag * 10.0)

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(int(render_size), int(render_size))
    rw.SetMultiSamples(8)
    rw.AddRenderer(ren)
    rw.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetScale(1)
    w2i.ReadFrontBufferOff()
    w2i.Update()
    return w2i.GetOutput()


def save_vtk_png(vtk_image, path: Path | str) -> None:
    if vtk is None:
        raise RuntimeError("vtk is required")
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(vtk_image)
    writer.Write()


def rotation_aligning_up(
    up_vec: tuple[float, float, float],
    *,
    target: tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    """3×3 rotation matrix mapping ``up_vec`` → ``target`` (right-handed)."""
    if np is None:
        raise RuntimeError("numpy is required")
    a = np.asarray(up_vec, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3, dtype=np.float64)
    a = a / na
    b = b / nb
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-10:
        # 180°: pick any orthogonal axis
        axis = np.cross(a, (1.0, 0.0, 0.0))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, (0.0, 1.0, 0.0))
        axis = axis / np.linalg.norm(axis)
        # Rodrigues with θ=π: R = -I + 2 axis⊗axis
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    kmat = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + kmat + kmat @ kmat * ((1.0 - dot) / (s * s))


def apply_upright_rotation(pts, up_axis: str):
    """Rotate point cloud so ``up_axis`` becomes +Z. Returns (pts, rotation 3×3)."""
    from harness.pipeline.orient import axis_vector

    if np is None:
        raise RuntimeError("numpy is required")
    R = rotation_aligning_up(axis_vector(up_axis))
    return (pts @ R.T), R


def render_probe_png(
    pd,
    out_path: Path,
    center: tuple[float, float, float],
    diag: float,
    *,
    direction: tuple[float, float, float],
    view_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    size: int = DEFAULT_SIZE,
    camera_distance_factor: float = 1.25,
    tag: str = "probe",
) -> Path:
    """One-off probe used to decide upright orientation."""
    if Image is None:
        raise RuntimeError("Pillow is required")
    image = render_view(
        pd,
        f"{PROBE_VIEW_NAME}_{tag}",
        direction,
        view_up,
        center,
        diag,
        camera_distance_factor=camera_distance_factor,
        render_size=VTK_RENDER_SIZE,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cad_probe_") as tmp:
        raw = Path(tmp) / f"{tag}_raw.png"
        save_vtk_png(image, raw)
        src = Image.open(raw).convert("RGB")
        if src.size != (size, size):
            src = src.resize((size, size), Image.Resampling.LANCZOS)
        src.save(out_path, format="PNG")
    return out_path


def render_one_view_png(
    pd,
    name: str,
    out_path: Path,
    center: tuple[float, float, float],
    diag: float,
    *,
    size: int = DEFAULT_SIZE,
    camera_distance_factor: float = 1.2,
    vtk_size: int = VTK_RENDER_SIZE,
) -> Path:
    """VTK shade one view and write ``out_path`` (no painted border / no crop-fit)."""
    if name not in VIEW_CAMERAS:
        raise ValueError(f"unknown view: {name}")
    if Image is None:
        raise RuntimeError("Pillow is required")
    direction, up = VIEW_CAMERAS[name]
    image = render_view(
        pd,
        name,
        direction,
        up,
        center,
        diag,
        camera_distance_factor=camera_distance_factor,
        render_size=vtk_size,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cad_views_") as tmp:
        raw = Path(tmp) / f"{name}_raw.png"
        save_vtk_png(image, raw)
        src = Image.open(raw).convert("RGB")
        if src.size != (size, size):
            src = src.resize((size, size), Image.Resampling.LANCZOS)
        src.save(out_path, format="PNG")
    return out_path


def _render_oriented_view_via_xvfb(
    step_path: Path,
    out_path: Path,
    *,
    up_axis: str,
    view_name: str,
    camera_distance_factor: float,
    size: int,
) -> dict[str, Any]:
    """Re-enter ``render_oriented_view_png`` under ``xvfb-run -a``."""
    xvfb = shutil.which("xvfb-run")
    if not xvfb:
        return {
            "ok": False,
            "path": str(out_path.resolve()),
            "backend": "gl_unavailable:no_xvfb",
            "detail": (
                "DISPLAY broken and xvfb-run not found; "
                "install with: sudo apt-get install -y xvfb"
            ),
        }

    src_root = Path(__file__).resolve().parents[3]  # .../src
    with tempfile.TemporaryDirectory(prefix="cad_result_xvfb_") as tmp:
        result_path = Path(tmp) / "result.json"
        script = Path(tmp) / "worker.py"
        script.write_text(
            "\n".join(
                [
                    "import json, os, sys",
                    f"sys.path.insert(0, {str(src_root)!r})",
                    "os.environ['HARNESS_UNDER_XVFB'] = '1'",
                    "from harness.pipeline.skills.cad_views import render_oriented_view_png",
                    f"r = render_oriented_view_png("
                    f"{str(step_path)!r}, {str(out_path)!r}, "
                    f"up_axis={up_axis!r}, view_name={view_name!r}, "
                    f"camera_distance_factor={float(camera_distance_factor)}, "
                    f"size={int(size)})",
                    f"open({str(result_path)!r}, 'w', encoding='utf-8').write("
                    "json.dumps(r))",
                ]
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["HARNESS_UNDER_XVFB"] = "1"
        env["PYTHONPATH"] = (
            f"{src_root}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(src_root)
        )
        proc = subprocess.run(
            [xvfb, "-a", sys.executable, str(script)],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
            check=False,
        )
        if proc.returncode != 0 or not result_path.is_file():
            detail = (proc.stderr or proc.stdout or "xvfb worker failed")[:2000]
            return {
                "ok": False,
                "path": str(out_path.resolve()),
                "backend": "gl_xvfb_failed",
                "detail": detail,
            }
        data = json.loads(result_path.read_text(encoding="utf-8"))
        backend = data.get("backend", "cadquery_vtk_shaded")
        data["backend"] = f"{backend}+xvfb"
        return data


def render_oriented_view_png(
    step_path: Path | str,
    out_path: Path | str,
    *,
    up_axis: str = "+Z",
    view_name: str = "iso",
    camera_distance_factor: float = 1.18,
    size: int = DEFAULT_SIZE,
) -> dict[str, Any]:
    """Tessellate STEP, apply saved upright axis, render one named view.

    Does **not** re-run orient probes/agents. Uses the same ``VIEW_CAMERAS``
    framing as ``02_views`` so result QC matches the chosen ``base_view``.
    """
    from harness.pipeline.orient import normalize_up_axis
    from harness.pipeline.view_qc_adjust import clamp_factor

    step_path = Path(step_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if view_name not in VIEW_CAMERAS:
        return {
            "ok": False,
            "path": str(out_path.resolve()),
            "backend": "bad_view",
            "detail": f"unknown view_name={view_name!r}",
            "up_axis": up_axis,
            "view_name": view_name,
        }

    ok_deps, dep_msg = _deps_ok()
    if not ok_deps:
        return {
            "ok": False,
            "path": str(out_path.resolve()),
            "backend": f"missing_dep:{dep_msg}",
            "detail": dep_msg,
            "up_axis": up_axis,
            "view_name": view_name,
        }

    ok_gl, gl_msg = _gl_display_ok()
    if not ok_gl:
        if not os.environ.get("HARNESS_UNDER_XVFB"):
            return _render_oriented_view_via_xvfb(
                step_path,
                out_path,
                up_axis=up_axis,
                view_name=view_name,
                camera_distance_factor=camera_distance_factor,
                size=size,
            )
        return {
            "ok": False,
            "path": str(out_path.resolve()),
            "backend": f"gl_unavailable:{gl_msg}",
            "detail": f"{gl_msg} (already under xvfb)",
            "up_axis": up_axis,
            "view_name": view_name,
        }

    if not step_path.is_file():
        return {
            "ok": False,
            "path": str(out_path.resolve()),
            "backend": "missing_step",
            "detail": f"missing {step_path}",
            "up_axis": up_axis,
            "view_name": view_name,
        }

    axis = normalize_up_axis(up_axis) or "+Z"
    factor = clamp_factor(float(camera_distance_factor))
    try:
        pts, tris = tessellate_shape(step_path)
        if len(pts) == 0 or len(tris) == 0:
            raise RuntimeError("empty tessellation")
        if axis != "+Z":
            pts, _rot = apply_upright_rotation(pts, axis)
        pd = build_polydata(pts, tris)
        pd.ComputeBounds()
        b = pd.GetBounds()
        center = ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
        diag = max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 1e-6)
        render_one_view_png(
            pd,
            view_name,
            out_path,
            center,
            diag,
            size=size,
            camera_distance_factor=factor,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "path": str(out_path.resolve()),
            "backend": f"render_error:{exc}",
            "detail": str(exc),
            "up_axis": axis,
            "view_name": view_name,
            "camera_distance_factor": factor,
        }

    return {
        "ok": out_path.is_file(),
        "path": str(out_path.resolve()),
        "backend": "cadquery_vtk_shaded",
        "detail": None,
        "up_axis": axis,
        "view_name": view_name,
        "camera_distance_factor": factor,
    }


def render_cad_views(
    step_path: Path | str,
    out_dir: Path | str,
    *,
    max_retries: int = 3,
    size: int = DEFAULT_SIZE,
    adjust_fn: Callable[..., dict[str, float]] | None = None,
    initial_factors: dict[str, float] | None = None,
    orient_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Probe → upright mesh → render 4 fixed views with QC retries.

    ``orient_fn(probe_paths, extents) -> up_axis`` chooses which current axis
    becomes +Z (e.g. ``\"+Y\"``). When omitted, keeps ``\"+Z\"``.
    ``probe_paths`` is the seven probe PNGs in ``PROBE_NAMES`` order
    (iso + front/back/left/right/top/bottom).
    ``extents`` is ``{\"X\":…, \"Y\":…, \"Z\":…}``.

    ``adjust_fn(qc, factors, paths, attempt) -> new_factors`` may be an agent advisor.
    When omitted, a per-view heuristic zoom is used.
    """
    from harness.pipeline.view_qc_adjust import (
        DEFAULT_FACTOR,
        clamp_factor,
        default_factors,
        heuristic_adjust_factors,
    )

    step_path = Path(step_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_deps, dep_msg = _deps_ok()
    if not ok_deps:
        return {
            "ok": False,
            "paths": {n: str((out_dir / f"{n}.png").resolve()) for n in VIEW_NAMES},
            "backend": f"missing_dep:{dep_msg}",
            "attempts": [],
            "qc": {"ok": False, "hard": "fail", "detail": dep_msg},
        }
    ok_gl, gl_msg = _gl_display_ok()
    if not ok_gl:
        if not os.environ.get("HARNESS_UNDER_XVFB"):
            return _render_cad_views_via_xvfb(
                step_path,
                out_dir,
                max_retries=max_retries,
                size=size,
            )
        return {
            "ok": False,
            "paths": {n: str((out_dir / f"{n}.png").resolve()) for n in VIEW_NAMES},
            "backend": f"gl_unavailable:{gl_msg}",
            "attempts": [],
            "qc": {
                "ok": False,
                "hard": "fail",
                "detail": f"{gl_msg} (already under xvfb)",
            },
        }
    if not step_path.is_file():
        return {
            "ok": False,
            "paths": {n: str((out_dir / f"{n}.png").resolve()) for n in VIEW_NAMES},
            "backend": "missing_step",
            "attempts": [],
            "qc": {"ok": False, "hard": "fail", "detail": f"missing {step_path}"},
        }

    backend = "cadquery_vtk_shaded"
    orient_meta: dict[str, Any] = {"up_axis": "+Z", "source": "default"}
    try:
        pts, tris = tessellate_shape(step_path)
        if len(pts) == 0 or len(tris) == 0:
            raise RuntimeError("empty tessellation")
        pd = build_polydata(pts, tris)
        pd.ComputeBounds()
        b = pd.GetBounds()
        center = ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
        diag = max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 1e-6)
        extents = {
            "X": float(b[1] - b[0]),
            "Y": float(b[3] - b[2]),
            "Z": float(b[5] - b[4]),
        }
        orient_meta["extents"] = {k: round(v, 4) for k, v in extents.items()}

        # --- iso + 6 orthographic probes on raw STEP → choose up → rotate ---
        probe_paths: list[Path] = []
        probe_map: dict[str, str] = {}
        for name in PROBE_NAMES:
            direction, view_up = PROBE_CAMERAS[name]
            out_probe = out_dir / f"{PROBE_VIEW_NAME}_{name}.png"
            render_probe_png(
                pd,
                out_probe,
                center,
                diag,
                direction=direction,
                view_up=view_up,
                size=size,
                tag=name,
            )
            probe_paths.append(out_probe)
            probe_map[name] = str(out_probe.resolve())
        # Alias: first probe (iso) for older tooling / README.
        probe_alias = out_dir / f"{PROBE_VIEW_NAME}.png"
        try:
            shutil.copy2(probe_paths[0], probe_alias)
        except Exception:  # noqa: BLE001
            pass

        up_axis = "+Z"
        if orient_fn is not None:
            try:
                chosen = orient_fn(probe_paths, extents)
                if chosen:
                    up_axis = str(chosen)
                    orient_meta["source"] = "agent"
            except TypeError:
                # Backward-compatible: orient_fn(probe_path) only.
                try:
                    chosen = orient_fn(probe_paths[0])  # type: ignore[misc, call-arg]
                    if chosen:
                        up_axis = str(chosen)
                        orient_meta["source"] = "agent"
                except Exception as exc:  # noqa: BLE001
                    orient_meta["orient_error"] = str(exc)
                    orient_meta["source"] = "default_after_error"
            except Exception as exc:  # noqa: BLE001
                orient_meta["orient_error"] = str(exc)
                orient_meta["source"] = "default_after_error"
        from harness.pipeline.orient import normalize_up_axis

        up_axis = normalize_up_axis(up_axis) or "+Z"
        orient_meta["up_axis"] = up_axis
        orient_meta["probe"] = str(probe_alias.resolve())
        orient_meta["probes"] = probe_map

        if up_axis != "+Z":
            pts, rot = apply_upright_rotation(pts, up_axis)
            orient_meta["rotation"] = rot.round(6).tolist()
            pd = build_polydata(pts, tris)
            pd.ComputeBounds()
            b = pd.GetBounds()
            center = ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
            diag = max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 1e-6)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "paths": {n: str((out_dir / f"{n}.png").resolve()) for n in VIEW_NAMES},
            "backend": f"tessellate_error:{exc}",
            "attempts": [],
            "qc": {"ok": False, "hard": "fail", "detail": str(exc)},
            "orient": orient_meta,
        }

    last_qc: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    factors: dict[str, float] = default_factors()
    if initial_factors:
        for k, v in initial_factors.items():
            if str(k) in VIEW_NAMES:
                factors[str(k)] = clamp_factor(float(v))

    advisor = adjust_fn or (
        lambda qc, fac, _paths, _attempt: heuristic_adjust_factors(qc, fac)
    )

    for attempt in range(max_retries + 1):
        paths: dict[str, str] = {}
        for name in VIEW_NAMES:
            out = out_dir / f"{name}.png"
            render_one_view_png(
                pd,
                name,
                out,
                center,
                diag,
                size=size,
                camera_distance_factor=factors.get(name, DEFAULT_FACTOR),
            )
            paths[name] = str(out.resolve())
        qc = check_views_dir(out_dir)
        action = "accept" if qc["ok"] else "adjust"
        attempts.append(
            {
                "attempt": attempt,
                "factors": {k: round(v, 4) for k, v in factors.items()},
                "action": action,
                "hard": qc["hard"],
                "detail": qc.get("detail"),
                "advisor": "agent" if adjust_fn is not None else "heuristic",
            }
        )
        last_qc = qc
        if qc["ok"]:
            return {
                "ok": True,
                "paths": paths,
                "backend": backend,
                "attempts": attempts,
                "qc": qc,
                "factors": factors,
                "orient": orient_meta,
            }
        if attempt >= max_retries:
            break
        try:
            factors = advisor(qc, factors, paths, attempt)
        except Exception:  # noqa: BLE001
            factors = heuristic_adjust_factors(qc, factors)
        factors = {
            n: clamp_factor(float(factors.get(n, DEFAULT_FACTOR))) for n in VIEW_NAMES
        }

    return {
        "ok": False,
        "paths": {n: str((out_dir / f"{n}.png").resolve()) for n in VIEW_NAMES},
        "backend": backend,
        "attempts": attempts,
        "qc": last_qc,
        "factors": factors,
        "orient": orient_meta,
    }
