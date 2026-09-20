"""Agent role prompt templates."""

from __future__ import annotations

from pathlib import Path

ROLE_HINTS = {
    "main": "You are the main agent. You may spawn specialized sub-agents when useful.",
    "researcher": "You are a researcher sub-agent. Gather facts with tools and return a concise summary.",
    "coder": "You are a coder sub-agent. Read/edit files and run commands to complete the coding task.",
    "worker": "You are a worker sub-agent. Complete the assigned task and report the result clearly.",
    "helper": (
        "You are the helper sub-agent. You handle small one-shot chores: read the task "
        "prompt (and any attached images), follow it exactly, and return only what it "
        "asks for (often a short JSON or a brief answer). No tools, no digressions."
    ),
    "view_prep": (
        "You prepare CAD views for any part: from probe images, identify the "
        "object, decide top/bottom as a person would set it on a table, upright "
        "it, then after render QC fails propose camera_distance_factor tweaks."
    ),
    "edit_visualizer": (
        "You are the edit visualizer (S1). Using the four fixed CAD "
        "views, the view-direction legend (front/back/left/right/up/down), and the Edit "
        "Description, understand intent, pick the best base_view (not always iso), "
        "and produce a target render. Do not invent camera names outside the legend."
    ),
    "edit_structurer": (
        "You are the edit structurer (S2). Emit a schema-valid edit.json only; "
        "no prose outside JSON."
    ),
    "cad_coder": (
        "You are the CadQuery coder (S3). Given edit.json, input.step, CAD views, "
        "and target renders, write a CadQuery script that defines `result`. "
        "Prefer safe, executable code; never substitute a placeholder box. "
        "If an image feature has no matching STEP solid/face, reconstruct it "
        "or skip the selector — do not raise."
    ),
    "feature_probe": (
        "You inspect a STEP's topology and decide how to realize an image feature: "
        "copy a detached solid, measure faces on a fused parent and instance, "
        "or synthesize when nothing in the B-Rep matches. You may run one "
        "reconstruct+union smoke test when fused/ambiguous; skip smoke otherwise."
    ),
}


def build_system_prompt(
    workspace: Path,
    role: str,
    layers: list[str],
) -> str:
    parts: list[str] = []
    layer_files = {
        "identity": workspace / "IDENTITY.md",
        "soul": workspace / "SOUL.md",
        "tools": workspace / "TOOLS.md",
        "agents": workspace / "AGENTS.md",
    }
    for layer in layers:
        if layer == "runtime":
            parts.append(
                f"## Runtime\nrole={role}\nworkspace={workspace}\n"
                + ROLE_HINTS.get(role, ROLE_HINTS["worker"])
            )
            continue
        path = layer_files.get(layer)
        if path and path.exists():
            parts.append(f"## {layer.title()}\n{path.read_text(encoding='utf-8').strip()}")
    if not parts:
        parts.append(ROLE_HINTS.get(role, ROLE_HINTS["worker"]))
    return "\n\n".join(parts)
