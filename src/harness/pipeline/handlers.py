"""Stage handlers wired from CLI/runtime (Phase 2/3 real paths + stubs)."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Callable

from harness.config import Settings
from harness.pipeline.artifacts import (
    DIR_CODE,
    DIR_TARGET,
    DIR_VIEWS,
    REL_EDIT_JSON,
    REL_EDIT_PY,
    REL_FEATURE_PROBE,
    REL_OUTPUT_STEP,
    REL_OUTPUT_STL,
    REL_RESULT,
    REL_TARGET,
    ArtifactStore,
    rel_result_view,
    rel_target_view,
)
from harness.pipeline.manifest import Manifest
from harness.pipeline.schema import skeleton_edit_json, validate_edit_json
from harness.pipeline.stages import StageError
from harness.providers.stub_media import StubMediaProvider


def _progress(msg: str) -> None:
    """Immediate stdout progress (stages can run a long time without on_stage)."""
    print(f"[progress] {msg}", flush=True)


def _run_feature_probe(
    store: ArtifactStore, manifest: Manifest, settings: Settings | None = None
) -> dict[str, Any]:
    """Inspect input.step and return facts/candidates without modeling advice."""
    from harness.agents.feature_probe import FeatureProbeAgent
    from harness.pipeline.feature_probe import compact_probe_for_prompt

    step = manifest.input_step
    if not step:
        return {}
    probe_cfg = getattr(getattr(settings, "pipeline", None), "feature_probe", None)
    root = getattr(settings, "root", Path.cwd())
    cache_name = str(getattr(probe_cfg, "cache_dir", "feature_probe_cache"))
    cache_dir = Path(cache_name)
    if not cache_dir.is_absolute():
        cache_dir = Path(root) / cache_dir
    try:
        report = FeatureProbeAgent().run(
            step,
            cache_dir=cache_dir,
            palmetto_engine=getattr(probe_cfg, "palmetto_engine_path", "") or None,
            palmetto_enabled=bool(getattr(probe_cfg, "palmetto_enabled", True)),
            palmetto_timeout_s=float(
                getattr(probe_cfg, "palmetto_timeout_seconds", 120.0)
            ),
        )
    except Exception as exc:  # noqa: BLE001
        report = {
            "ok": False,
            "plan": {"action": "skip", "run_smoke": False, "reason": str(exc)},
            "coder_note": f"FeatureProbe skipped: {exc}",
        }
    unified = report.get("inspect") if isinstance(report.get("inspect"), dict) else {}
    prompt_probe = compact_probe_for_prompt(unified)
    compact = {
        "ok": report.get("ok"),
        "schema_version": unified.get("schema_version"),
        "model": prompt_probe.get("model"),
        "candidate_count": prompt_probe.get("candidate_count"),
        "sources": prompt_probe.get("sources"),
        "plan": report.get("plan"),
        "coder_note": report.get("coder_note"),
        "smoke": None,
    }
    smoke = report.get("smoke")
    if isinstance(smoke, dict):
        compact["smoke"] = {
            "ok": smoke.get("ok"),
            "skipped": smoke.get("skipped"),
            "method": smoke.get("method"),
            "error": smoke.get("error"),
        }
    path = store.write_text(
        REL_FEATURE_PROBE,
        json.dumps(unified or compact, indent=2, ensure_ascii=False),
    )
    manifest.artifacts["feature_probe"] = str(path.resolve())
    manifest.checks["feature_probe"] = compact
    return report


def _feature_probe_block(report: dict[str, Any] | None) -> str:
    if not report:
        return ""
    note = str(report.get("coder_note") or "").strip()
    if not note:
        return ""
    return (
        "\nFeatureProbe (topology only; not a selector list):\n"
        f"{note}\n"
        "If the image shows a feature with no matching STEP entity, reconstruct "
        "or skip — do not force a selector until it raises.\n"
    )


def _clip(text: Any, n: int = 320) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _pipeline_cfg(settings: Settings) -> dict[str, Any]:
    cfg = getattr(settings, "pipeline", None)
    if cfg is None:
        return {}
    if hasattr(cfg, "__dict__"):
        return {
            "artifacts_root": getattr(cfg, "artifacts_root", None),
            "cad_views_max_retries": getattr(cfg, "cad_views_max_retries", 3),
            "s1_max_retries": getattr(cfg, "s1_max_retries", 2),
            "s3_max_retries": getattr(
                cfg, "s3_max_retries", getattr(cfg, "s4_max_retries", 2)
            ),
            "result_qc_max_retries": getattr(cfg, "result_qc_max_retries", 5),
            "result_qc_exec_retries": getattr(cfg, "result_qc_exec_retries", 1),
            "stub_media": getattr(cfg, "stub_media", False),
            "prompts_dir": getattr(cfg, "prompts_dir", None),
            "view_size": getattr(cfg, "view_size", 512),
            "image_model": getattr(cfg, "image_model", "gpt-image-1"),
        }
    return dict(cfg)


def _load_prompt(settings: Settings, name: str) -> str:
    cfg = _pipeline_cfg(settings)
    prompts_dir = cfg.get("prompts_dir")
    if prompts_dir:
        path = Path(prompts_dir) / name
    else:
        path = settings.workspace / "prompts" / "cad_edit" / name
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return f"(placeholder prompt: {name})"


def _edit_description_only(raw: str) -> str:
    """Prefer the Edit Description body; drop request-header metadata when present."""
    text = (raw or "").strip()
    if not text:
        return ""
    m = re.search(r"Edit Description:\s*(.*)\Z", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        body = m.group(1).strip()
        if body:
            return body
    return text


def _media(settings: Settings) -> Any:
    """Prefer GPT media when configured; otherwise stub."""
    if getattr(settings.pipeline, "stub_media", False):
        return StubMediaProvider(settings.workspace)
    try:
        from harness.providers.gpt_media import GptMediaProvider

        return GptMediaProvider.from_settings(settings)
    except Exception:  # noqa: BLE001
        return StubMediaProvider(settings.workspace)


def make_cad_views_handler(settings: Settings) -> Callable[..., Manifest]:
    cfg = _pipeline_cfg(settings)

    def handler(store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]) -> Manifest:
        from harness.agents.helper import HelperAgent
        from harness.pipeline.skills.cad_views import render_cad_views
        from harness.pipeline.view_qc_adjust import make_agent_view_qc_adjuster
        from harness.session.store import SessionStore

        max_retries = int(ctx.get("cad_views_max_retries", cfg.get("cad_views_max_retries", 3)))
        size = int(ctx.get("view_size", cfg.get("view_size", 512)))
        stub_media = bool(ctx.get("stub_media", cfg.get("stub_media", False)))
        media = ctx.get("media_provider") or _media(settings)
        helper = ctx.get("helper") or HelperAgent(
            media,
            workspace=settings.workspace,
            agent_id="cad_edit",
        )

        store_sess = SessionStore(settings.workspace, agent_id="cad_edit")
        sid = store_sess.create_session(label="S0_view_qc", agent_role="view_prep")
        manifest.sessions["S0_view_qc"] = sid

        prompt_qc = _load_prompt(settings, "s0_view_qc.md")
        prompt_orient = _load_prompt(settings, "s0_orient.md")
        desc_raw = ""
        try:
            desc_raw = Path(manifest.description_txt).read_text(encoding="utf-8")
        except OSError:
            desc_raw = ""
        edit_desc = _edit_description_only(desc_raw)

        def session_append(entry: dict[str, Any]) -> None:
            store_sess.append_transcript(sid, entry)

        def orient_fn(probe_paths, extents: dict[str, float] | None = None) -> str:
            from harness.pipeline.orient import parse_orient_json

            if isinstance(probe_paths, (str, Path)):
                images = [Path(probe_paths)]
            else:
                images = [Path(p) for p in probe_paths]
            ext = extents or {}
            ext_line = ""
            if ext:
                ext_line = (
                    "\nCurrent AABB extents (raw STEP axes, factual only):\n"
                    f"- X: {ext.get('X', 0):.3f}\n"
                    f"- Y: {ext.get('Y', 0):.3f}\n"
                    f"- Z: {ext.get('Z', 0):.3f}\n"
                )
            desc_block = ""
            if edit_desc:
                desc_block = (
                    "\n## Edit Description (context for upright only — do not edit)\n"
                    f"{edit_desc}\n"
                )
            task = (
                f"{prompt_orient}{ext_line}{desc_block}\n"
                "Use the 7 probe images + Edit Description to identify the object "
                "and how a person would set it on a table. "
                "Top/bottom = everyday placement, not image-down in the probe. "
                "Description helps naming faces (base/back/top); do not perform the edit. "
                "Return ONLY the JSON object."
            )
            result = helper.run(
                task,
                images=images,
                label="S0_orient",
                parent_session_id=sid,
                append=session_append,
                detail="high",
            )
            parsed = parse_orient_json(result.content or "") if result.ok else None
            if parsed is None:
                return "+Z"
            # Stash richer orient fields on the closure for the handler to read.
            orient_fn.last_parsed = parsed  # type: ignore[attr-defined]
            return str(parsed["up_axis"])

        orient_fn.last_parsed = None  # type: ignore[attr-defined]

        adjust_fn = make_agent_view_qc_adjuster(
            helper=helper,
            prompt=prompt_qc,
            session_append=session_append,
            parent_session_id=sid,
        )

        step = Path(manifest.input_step)
        result = render_cad_views(
            step,
            store.path(DIR_VIEWS),
            max_retries=max_retries,
            size=size,
            adjust_fn=adjust_fn,
            orient_fn=orient_fn,
        )
        manifest.artifacts["views"] = result["paths"]
        manifest.artifacts["views_dir"] = str(store.path(DIR_VIEWS).resolve())
        if result.get("factors"):
            manifest.artifacts["view_factors"] = result["factors"]
        orient_info = dict(result.get("orient") or {})
        extra_orient = getattr(orient_fn, "last_parsed", None)
        if isinstance(extra_orient, dict):
            for k, v in extra_orient.items():
                orient_info.setdefault(k, v)
        if orient_info:
            manifest.artifacts["orient"] = orient_info
        hard = "pass" if result["ok"] else "fail"
        manifest.checks["views"] = {
            "hard": hard,
            "backend": result.get("backend"),
            "attempts": result.get("attempts"),
            "qc": result.get("qc"),
            "orient": orient_info,
            "session": sid,
            "stub_media": stub_media,
            "helper": "helper",
        }
        if not result["ok"]:
            detail = (result.get("qc") or {}).get("detail") or result.get("backend")
            manifest.set_summary(f"cad_views QC failed after retries: {detail}")
            store.persist(manifest)
            raise StageError(f"cad_views hard QC failed: {detail}")
        n_att = len(result.get("attempts") or [])
        up = orient_info.get("up_axis", "+Z")
        obj = orient_info.get("object") or "part"
        manifest.set_summary(
            f"Rendered {len(result['paths'])} views → {DIR_VIEWS}/ "
            f"via {result.get('backend')} "
            f"(upright {obj!r} up={up}, {n_att} QC attempt(s))"
        )
        store.persist(manifest)
        return manifest

    return handler


def _collect_target_paths(manifest: Manifest) -> list[Path]:
    """Chosen target renders in base_views order."""
    from harness.pipeline.base_view import resolve_base_views

    renders = dict(manifest.artifacts.get("target_renders") or {})
    views = resolve_base_views(manifest.artifacts, count=2)
    paths: list[Path] = []
    seen: set[str] = set()
    for view in views:
        val = renders.get(view)
        if val and Path(str(val)).is_file():
            paths.append(Path(str(val)))
            seen.add(str(Path(str(val)).resolve()))
    for val in renders.values():
        p = Path(str(val))
        key = str(p.resolve()) if p.is_file() else ""
        if key and key not in seen:
            paths.append(p)
            seen.add(key)
    return paths


VISION_DETAIL = "high"


def _collect_edit_ref_images(manifest: Manifest) -> tuple[list[Path], list[Path]]:
    """CAD views + S1 target renders."""
    from harness.pipeline.multimodal import collect_view_paths

    view_paths = collect_view_paths(manifest.artifacts.get("views"))
    target_paths = _collect_target_paths(manifest)
    return view_paths, target_paths


def _read_optional_text(path: Any) -> str:
    if not path:
        return ""
    p = Path(str(path))
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def _vision_content(text: str, images: list[Path]) -> Any:
    from harness.pipeline.multimodal import user_content

    return user_content(text, images, max_images=None, detail=VISION_DETAIL)


def _prefer_runnable_code(candidate: str, previous: str) -> str:
    """Keep previous AST-valid code instead of a placeholder box."""
    if _ast_check(candidate)["ok"]:
        return candidate
    if previous and _ast_check(previous)["ok"]:
        return previous
    return candidate


def _best_target_view(
    base_views: list[str],
    per_view: dict[str, dict[str, Any]],
) -> str | None:
    """Choose the strongest available S1 target from per-view QC evidence."""
    gate_names = (
        "matches_description",
        "preserves_unchanged",
        "same_part_identity",
        "camera_matches_base_view",
    )
    ranked: list[tuple[tuple[int, int, int], str]] = []
    for index, view in enumerate(base_views):
        result = per_view.get(view) or {}
        path = result.get("path")
        if not path or not Path(str(path)).is_file():
            continue
        qc = result.get("qc") or {}
        gate_score = sum(qc.get(name) is True for name in gate_names)
        # A fully accepted image always wins.  Otherwise use the number of
        # passed gates, retaining base-view order as the deterministic tie-break.
        rank = (1 if result.get("ok") else 0, gate_score, -index)
        ranked.append((rank, view))
    return max(ranked)[1] if ranked else None


def make_edit_visualizer_handler(settings: Settings) -> Callable[..., Manifest]:
    """S1: design intent → two target renders, then pair QC (no per-image QC)."""

    def handler(store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]) -> Manifest:
        from harness.agents.helper import HelperAgent
        from harness.pipeline.base_view import order_views_for_base, parse_base_views
        from harness.pipeline.multimodal import collect_view_paths, user_content
        from harness.pipeline.target_qc import (
            decide_s1_pair_followup,
            parse_target_pair_qc_json,
        )
        from harness.session.store import SessionStore

        cfg = _pipeline_cfg(settings)
        media = ctx.get("media_provider") or _media(settings)
        stub_media = bool(ctx.get("stub_media", cfg.get("stub_media", False)))
        max_retries = int(ctx.get("s1_max_retries", cfg.get("s1_max_retries", 2)))

        helper = ctx.get("helper") or HelperAgent(
            media,
            workspace=settings.workspace,
            agent_id="cad_edit",
        )

        store_sess = SessionStore(settings.workspace, agent_id="cad_edit")
        sid = store_sess.create_session(label="S1_edit_visualizer", agent_role="edit_visualizer")
        manifest.sessions["S1_edit_visualizer"] = sid

        desc = Path(manifest.description_txt).read_text(encoding="utf-8")
        p_intent = _load_prompt(settings, "s1_intent.md")
        p_render = _load_prompt(settings, "s1_render.md")
        p_pair = _load_prompt(settings, "s1_target_pair_qc.md")
        p_legend = _load_prompt(settings, "view_legend.md")
        views_map = dict(manifest.artifacts.get("views") or {})
        view_paths = collect_view_paths(views_map)

        # --- Turn 1: design intent with description + legend + fixed views ---
        intent_text = (
            f"{p_intent}\n\n## View direction legend\n\n{p_legend}\n\n"
            f"Description:\n{desc}"
        )
        intent_content = user_content(intent_text, view_paths)
        store_sess.append_transcript(
            sid,
            {
                "type": "user",
                "content": intent_text,
                "images": [str(p) for p in view_paths],
            },
        )
        turn1 = media.chat([{"role": "user", "content": intent_content}])
        store_sess.append_transcript(
            sid,
            {"type": "assistant", "content": turn1.content or ""},
        )

        # --- Turn 2: two base_views + render direction (text only) ---
        render_text = (
            f"{p_render}\n\n## View direction legend\n\n{p_legend}\n\n"
            "Pick TWO distinct base_views that together show the attachment region. "
            "Edit Description 'from the top' ≠ legend up / screen-up."
        )
        store_sess.append_transcript(
            sid,
            {"type": "user", "content": render_text},
        )
        turn2 = media.chat(
            [
                {"role": "user", "content": intent_content},
                {"role": "assistant", "content": turn1.content or ""},
                {"role": "user", "content": render_text},
            ]
        )
        store_sess.append_transcript(
            sid,
            {"type": "assistant", "content": turn2.content or ""},
        )

        base_views = parse_base_views(turn2.content or "", count=2)
        if len(base_views) < 2:
            base_views = parse_base_views(turn1.content or "", count=2)
        manifest.artifacts["base_views"] = base_views
        manifest.artifacts["base_view"] = base_views[0]

        def generate_one(
            base_view: str,
            *,
            guide_from: str | None = None,
            guide_path: Path | None = None,
            extra_fix: str = "",
        ) -> dict[str, Any]:
            ref_paths = list(order_views_for_base(views_map, base_view) or view_paths)
            out = store.path(rel_target_view(base_view))
            guide_block = ""
            if guide_from and guide_path is not None and Path(guide_path).is_file():
                ref_paths.append(Path(guide_path))
                guide_block = (
                    f"\nGUIDE TARGET from camera [{guide_from}] is attached last. "
                    "That image is the accepted 3D edit. "
                    f"Redraw the SAME edit from the [{base_view}] camera only. "
                    "Match attachment, new-feature shape, and unchanged CAD body. "
                    "If the guide restyled unmentioned features, restore CAD identity "
                    "and keep only the described delta.\n"
                )
            base_img_prompt = (
                f"{p_legend}\n\n"
                f"PRIMARY BASE IMAGE: [{base_view}] (first attached reference). "
                "That PNG is the part identity and the camera. "
                "Keep the same angle, part pose, body, and unmentioned features. "
                "Other attached CAD views are supporting identity only.\n"
                "Apply ONLY the Edit Description delta. "
                "The Intent JSON is a hint for that delta; if it restyles unmentioned "
                "features, ignore the restyle and keep the CAD appearance. "
                "Copy/mirror/pattern must duplicate CAD source features, not substitute "
                "a different feature type.\n"
                "Edit Description words like 'from the top' mean attachment on the part's "
                "top interface — not 'extrude vertically toward the top of the image'.\n"
                f"{guide_block}\n"
                f"Edit Description:\n{desc}\n\n"
                f"Intent (delta hint only):\n{turn1.content or ''}\n\n"
                f"Render direction:\n{turn2.content or p_render}\n\n"
                f"This run is ONLY the [{base_view}] camera. "
                f"Chosen base_views for the pair: {base_views}."
            )

            if extra_fix:
                base_img_prompt = (
                    f"{base_img_prompt}\n\n"
                    f"Previous pair QC rejected this camera. Fix these issues:\n{extra_fix}"
                )

            attempts: list[dict[str, Any]] = []
            img_mode = "stub"
            gen_tries = max(1, min(2, max_retries))
            for attempt in range(gen_tries):
                if stub_media or not hasattr(media, "generate_image"):
                    media.generate_image(base_img_prompt, out)
                    img_mode = "stub"
                else:
                    result = media.generate_image(
                        base_img_prompt,
                        out,
                        reference_images=ref_paths,
                    )
                    img_mode = result.get("mode", "gpt_images")
                ok = out.is_file()
                attempts.append(
                    {
                        "attempt": attempt,
                        "action": "generated" if ok else "fail",
                        "hard": "pass" if ok else "fail",
                        "detail": None if ok else "target render missing",
                        "base_view": base_view,
                        "guided_by": guide_from,
                    }
                )
                if ok:
                    return {
                        "ok": True,
                        "path": str(out.resolve()),
                        "mode": img_mode,
                        "attempts": attempts,
                    }
            return {
                "ok": False,
                "path": str(out.resolve()) if out.is_file() else None,
                "mode": img_mode,
                "attempts": attempts,
            }

        def _synthetic_pair_fail(issues: str) -> dict[str, Any]:
            return {
                "pass": False,
                "cad_consistent": False,
                "same_edit": False,
                "viewpoint_only": False,
                "matches_description": False,
                "both_fail": True,
                "view_pass": {v: False for v in base_views},
                "issues": issues,
                "fix_hint": (
                    "Regenerate both cameras from the CAD PNGs; "
                    "apply only the Edit Description delta."
                ),
                "keep_view": None,
                "retry_view": None,
                "retry_views": list(base_views),
            }

        def run_pair_qc(attempt: int) -> dict[str, Any]:
            images: list[Path] = []
            for view in base_views:
                before = views_map.get(view)
                if before and Path(str(before)).is_file():
                    images.append(Path(str(before)))
                tgt = target_renders.get(view)
                if tgt and Path(str(tgt)).is_file():
                    images.append(Path(str(tgt)))
            task = (
                f"{p_pair}\n\n## View direction legend\n\n{p_legend}\n\n"
                f"Edit Description:\n{desc}\n\n"
                f"base_views={base_views}\n"
                "Ground truth = each CAD before PNG + Edit Description. "
                "Judge whether the two targets correspond (same 3D edit) "
                "AND each matches the Edit Description. "
                "Two matching restyles still fail.\n"
                "Images per camera, in order: CAD before, then that camera's target. "
                "If a camera's target is missing, that view fails. "
                "Return ONLY the JSON object."
            )
            qc_result = helper.run(
                task,
                images=images,
                label=f"S1_target_pair_qc_{attempt}",
                detail="high",
            )
            if qc_result.session_id:
                manifest.sessions[f"S1_target_pair_qc_{attempt}"] = qc_result.session_id
            parsed = (
                parse_target_pair_qc_json(qc_result.content or "")
                if qc_result.ok
                else None
            )
            if parsed is None:
                return _synthetic_pair_fail("unparseable pair QC reply")
            return parsed

        def _record_view(view: str, one: dict[str, Any]) -> None:
            prev = list(per_view.get(view, {}).get("attempts") or [])
            one = dict(one)
            one["attempts"] = prev + list(one.get("attempts") or [])
            per_view[view] = one
            nonlocal img_mode
            img_mode = str(one.get("mode") or img_mode)
            path = one.get("path")
            if path and Path(str(path)).is_file():
                target_renders[view] = str(path)
            else:
                target_renders.pop(view, None)

        def _pair_fix_hint(parsed: dict[str, Any]) -> str:
            return str(
                parsed.get("fix_hint")
                or parsed.get("issues")
                or (
                    "Restore unmentioned CAD features; apply only the "
                    "Edit Description delta from this camera."
                    if parsed.get("cad_consistent") is False
                    else "Match the kept target's 3D edit from this camera."
                )
            )

        legacy = store.path(REL_TARGET)
        if legacy.is_file():
            try:
                legacy.unlink()
            except OSError:
                pass
        manifest.artifacts["target_render"] = None

        target_renders: dict[str, str] = {}
        per_view: dict[str, Any] = {}
        img_mode = "stub"
        for base_view in base_views:
            _progress(f"S1 target img2 for base_view={base_view}")
            _record_view(base_view, generate_one(base_view))

        pair_attempts: list[dict[str, Any]] = []
        last_pair: dict[str, Any] | None = None
        qc_rounds = max(1, max_retries)
        selected_views: list[str] = []
        pair_ok = False
        drop_all = False
        anchored_keep: str | None = None
        fallback_trigger: str | None = None
        fallback_reason: str | None = None

        for pair_i in range(qc_rounds):
            missing = [
                v
                for v in base_views
                if not (
                    target_renders.get(v) and Path(str(target_renders[v])).is_file()
                )
            ]
            _progress(
                f"S1 pair QC attempt {pair_i} (same edit + description, two cameras)"
            )
            if len(missing) == len(base_views):
                parsed = _synthetic_pair_fail("both target renders missing")
            else:
                parsed = run_pair_qc(pair_i)
                if missing:
                    parsed["pass"] = False
                    view_pass = dict(parsed.get("view_pass") or {})
                    for view in missing:
                        view_pass[view] = False
                    parsed["view_pass"] = view_pass
            last_pair = parsed
            decision = decide_s1_pair_followup(
                parsed,
                base_views,
                round_index=pair_i,
                max_rounds=qc_rounds,
            )
            rec = {
                "attempt": pair_i,
                "action": decision["action"],
                "pass": bool(parsed.get("pass")),
                "cad_consistent": parsed.get("cad_consistent"),
                "same_edit": parsed.get("same_edit"),
                "viewpoint_only": parsed.get("viewpoint_only"),
                "matches_description": parsed.get("matches_description"),
                "view_pass": decision.get("view_pass"),
                "keep_view": parsed.get("keep_view"),
                "retry_view": parsed.get("retry_view"),
                "keep_views": decision.get("keep_views"),
                "retry_views": decision.get("retry_views"),
                "issues": parsed.get("issues"),
                "fix_hint": parsed.get("fix_hint"),
            }
            pair_attempts.append(rec)
            action = str(decision["action"])
            if action == "accept_both":
                pair_ok = True
                selected_views = list(base_views)
                break
            if action == "keep_one":
                selected_views = list(decision.get("keep_views") or [])
                fallback_trigger = "pair_qc_one"
                fallback_reason = str(
                    parsed.get("issues") or parsed.get("fix_hint") or "one target passed"
                )
                _progress(
                    "S1 pair QC: keep "
                    f"{selected_views[0] if selected_views else '?'} only"
                )
                break
            if action == "drop_all":
                if (
                    anchored_keep
                    and target_renders.get(anchored_keep)
                    and Path(str(target_renders[anchored_keep])).is_file()
                ):
                    selected_views = [anchored_keep]
                    fallback_trigger = "pair_qc_one"
                    fallback_reason = str(
                        parsed.get("issues")
                        or parsed.get("fix_hint")
                        or f"repair failed; keeping {anchored_keep}"
                    )
                    _progress(
                        f"S1 pair QC: repair failed, keep {anchored_keep} only"
                    )
                    break
                drop_all = True
                selected_views = []
                fallback_trigger = "pair_qc_drop"
                fallback_reason = str(
                    parsed.get("issues")
                    or parsed.get("fix_hint")
                    or "both targets failed pair QC twice"
                )
                _progress("S1 pair QC: drop both targets; harness proceeds without them")
                break
            if action == "retry_one":
                keep_list = list(decision.get("keep_views") or [])
                retry_list = list(decision.get("retry_views") or [])
                if not keep_list or not retry_list:
                    drop_all = True
                    selected_views = []
                    fallback_trigger = "pair_qc_drop"
                    fallback_reason = "pair QC retry_one missing keep/retry"
                    break
                keep, retry = keep_list[0], retry_list[0]
                anchored_keep = keep
                keep_path = Path(str(target_renders.get(keep) or ""))
                _progress(f"S1 pair reject: keep={keep} retry={retry} (guide repair)")
                pair_attempts[-1]["keep_view"] = keep
                pair_attempts[-1]["retry_view"] = retry
                _record_view(
                    retry,
                    generate_one(
                        retry,
                        guide_from=keep,
                        guide_path=keep_path if keep_path.is_file() else None,
                        extra_fix=_pair_fix_hint(parsed),
                    ),
                )
                continue
            if action == "retry_both":
                _progress("S1 pair reject: both failed, regenerate both cameras")
                hint = _pair_fix_hint(parsed)
                for view in base_views:
                    _record_view(view, generate_one(view, extra_fix=hint))
                continue

        manifest.artifacts["target_render"] = None
        selected_fallback_view: str | None = None
        if drop_all or (not pair_ok and not selected_views):
            target_renders = {}
            drop_all = True
            manifest.artifacts["base_views"] = base_views
            manifest.artifacts["base_view"] = base_views[0]
        elif pair_ok:
            manifest.artifacts["base_views"] = base_views
            manifest.artifacts["base_view"] = base_views[0]
        else:
            keep = selected_views[0]
            keep_path = target_renders.get(keep)
            if keep_path and Path(str(keep_path)).is_file():
                selected_fallback_view = keep
                target_renders = {keep: str(keep_path)}
                manifest.artifacts["base_view"] = keep
                manifest.artifacts["base_views"] = [keep]
            else:
                target_renders = {}
                drop_all = True
                fallback_trigger = "pair_qc_drop"
                fallback_reason = f"kept view {keep} is missing on disk"
                manifest.artifacts["base_views"] = base_views
                manifest.artifacts["base_view"] = base_views[0]
        manifest.artifacts["target_renders"] = target_renders
        n_att = sum(len(per_view.get(v, {}).get("attempts") or []) for v in base_views)

        if drop_all:
            summary_extra = "; dropped both targets, harness uses CAD+description"
        elif selected_fallback_view:
            summary_extra = f"; selected {selected_fallback_view} as single target"
        else:
            summary_extra = ""
        manifest.checks["target_render"] = {
            "hard": "pass",
            "mode": img_mode,
            "session": sid,
            "view_count": len(view_paths),
            "base_view": (
                selected_fallback_view or base_views[0]
            ),
            "base_views": (
                [selected_fallback_view]
                if selected_fallback_view
                else list(base_views)
            ),
            "attempts": {v: per_view.get(v, {}).get("attempts") for v in base_views},
            "qc": last_pair,
            "pair_qc": last_pair,
            "pair_attempts": pair_attempts,
            "pair_fallback": {
                "used": bool(selected_fallback_view or drop_all),
                "selected_view": selected_fallback_view,
                "dropped_all": drop_all,
                "trigger": fallback_trigger,
                "reason": fallback_reason,
            },
            "helper": "helper",
        }
        manifest.set_summary(
            f"S1 intent→img2 ×{len(base_views)} → {DIR_TARGET}/ "
            f"(bases={','.join(base_views)}, {img_mode}, "
            f"{n_att} renders, {len(pair_attempts)} pair QC{summary_extra})"
        )
        store.persist(manifest)
        return manifest

    return handler


def make_edit_structurer_handler(settings: Settings) -> Callable[..., Manifest]:
    def handler(store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]) -> Manifest:
        from harness.session.store import SessionStore

        media = ctx.get("media_provider") or _media(settings)

        store_sess = SessionStore(settings.workspace, agent_id="cad_edit")
        sid = store_sess.create_session(label="S2_edit_structurer", agent_role="edit_structurer")
        manifest.sessions["S2_edit_structurer"] = sid

        desc = Path(manifest.description_txt).read_text(encoding="utf-8")
        prompt = _load_prompt(settings, "s2_json.md")
        view_paths, target_paths = _collect_edit_ref_images(manifest)
        images = view_paths + target_paths

        user_msg = (
            f"{prompt}\n\nDescription:\n{desc}\n\n"
            f"Attached: {len(view_paths)} CAD views + {len(target_paths)} target renders "
            f"(one per chosen base_view).\n"
            "Return ONLY a JSON object matching the edit schema."
        )
        content = _vision_content(user_msg, images)
        store_sess.append_transcript(
            sid,
            {
                "type": "user",
                "content": user_msg,
                "images": [str(p) for p in images],
            },
        )
        turn = media.chat([{"role": "user", "content": content}])
        store_sess.append_transcript(sid, {"type": "assistant", "content": turn.content or ""})

        raw = turn.content or ""
        data: Any
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    data = skeleton_edit_json(desc)
            else:
                data = skeleton_edit_json(desc)

        check = validate_edit_json(data)
        if not check["ok"]:
            # One repair attempt with skeleton merge
            data = skeleton_edit_json(desc)
            check = validate_edit_json(data)

        path = store.write_text(
            REL_EDIT_JSON, json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        )
        manifest.artifacts["edit_json"] = str(path.resolve())
        manifest.checks["edit_json"] = {
            "hard": check["hard"],
            "errors": check.get("errors"),
            "session": sid,
            "image_count": len(images),
        }
        if not check["ok"]:
            raise StageError(f"edit.json schema failed: {check['errors']}")
        manifest.set_summary(f"S2 wrote schema-valid → {REL_EDIT_JSON}")
        store.persist(manifest)
        return manifest

    return handler

def _extract_python_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


def _ast_check(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"ok": False, "error": str(exc)}
    # Soft ban dangerous calls
    banned = {"exec", "eval", "os", "subprocess", "socket"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in banned - {"os"}:
                    # allow nothing from subprocess/socket
                    if alias.name.split(".")[0] in {"subprocess", "socket"}:
                        return {"ok": False, "error": f"banned import {alias.name}"}
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in {"subprocess", "socket"}:
                return {"ok": False, "error": f"banned import {node.module}"}
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in {"exec", "eval", "__import__"}:
                return {"ok": False, "error": f"banned call {name}"}
    return {"ok": True, "error": None}


def make_cad_coder_handler(settings: Settings) -> Callable[..., Manifest]:
    def handler(store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]) -> Manifest:
        from harness.session.store import SessionStore

        media = ctx.get("media_provider") or _media(settings)
        store_sess = SessionStore(settings.workspace, agent_id="cad_edit")
        sid = store_sess.create_session(label="S3_cad_coder", agent_role="cad_coder")
        manifest.sessions["S3_cad_coder"] = sid

        edit_json = _read_optional_text(manifest.artifacts.get("edit_json")) or "{}"
        desc = _read_optional_text(manifest.description_txt)
        prompt = _load_prompt(settings, "s3_cadquery.md")
        view_paths, target_paths = _collect_edit_ref_images(manifest)
        images = view_paths + target_paths
        probe = _run_feature_probe(store, manifest, settings)
        user_msg = (
            f"{prompt}\n\nedit.json:\n{edit_json}\n\n"
            f"Edit Description:\n{desc}\n\n"
            f"input.step path: {manifest.input_step}\n"
            f"{_feature_probe_block(probe)}"
            f"Attached: {len(view_paths)} CAD views + {len(target_paths)} target renders "
            f"(intended edit). Match the targets; load/edit the STEP; define `result`.\n"
            "Do not replace the imported part with a placeholder box.\n"
            "Return a CadQuery script defining `result`."
        )
        content = _vision_content(user_msg, images)
        store_sess.append_transcript(
            sid,
            {
                "type": "user",
                "content": user_msg,
                "images": [str(p) for p in images],
            },
        )
        turn = media.chat([{"role": "user", "content": content}])
        store_sess.append_transcript(sid, {"type": "assistant", "content": turn.content or ""})

        code = _extract_python_code(turn.content or "")
        check = _ast_check(code)
        if not check["ok"]:
            raise StageError(f"edit.py AST check failed: {check['error']}")

        path = store.write_text(REL_EDIT_PY, code)
        manifest.artifacts["edit_py"] = str(path.resolve())
        manifest.checks["edit_py"] = {
            "hard": "pass",
            "syntax": check,
            "session": sid,
            "image_count": len(images),
        }
        manifest.set_summary(f"S3 wrote AST-valid → {REL_EDIT_PY}")
        store.persist(manifest)
        return manifest

    return handler


def make_cadquery_exec_handler(settings: Settings) -> Callable[..., Manifest]:
    def handler(store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]) -> Manifest:
        from harness.pipeline.skills.cadquery_exec import run_cadquery_script
        from harness.session.store import SessionStore

        cfg = _pipeline_cfg(settings)
        max_retries = int(
            ctx.get(
                "s3_max_retries",
                ctx.get("s4_max_retries", cfg.get("s3_max_retries", 2)),
            )
        )
        edit_py = Path(manifest.artifacts.get("edit_py") or store.path(REL_EDIT_PY))
        out_step = store.path(REL_OUTPUT_STEP)
        out_stl = store.path(REL_OUTPUT_STL)
        edit_json = _read_optional_text(manifest.artifacts.get("edit_json"))
        desc = _read_optional_text(manifest.description_txt)
        view_paths, target_paths = _collect_edit_ref_images(manifest)
        images = view_paths + target_paths
        media = ctx.get("media_provider") or _media(settings)
        store_sess = SessionStore(settings.workspace, agent_id="cad_edit")
        sid = str(manifest.sessions.get("S3_cad_coder") or "")
        if not sid:
            sid = store_sess.create_session(label="S3_cad_coder", agent_role="cad_coder")
            manifest.sessions["S3_cad_coder"] = sid

        attempts: list[dict[str, Any]] = []
        for attempt in range(max_retries + 1):
            result = run_cadquery_script(edit_py, out_step, out_stl, cwd=store.root)
            attempts.append(
                {
                    "attempt": attempt,
                    "ok": result["ok"],
                    "exit_code": result["exit_code"],
                    "stderr": (result.get("stderr") or "")[:500],
                }
            )
            if result["ok"]:
                manifest.artifacts["output_step"] = result["output_step"]
                manifest.artifacts["output_stl"] = result["output_stl"]
                manifest.checks["exec"] = {
                    "hard": "pass",
                    "exit_code": result["exit_code"],
                    "attempts": attempts,
                }
                manifest.set_summary("cadquery_exec exported output.step/stl")
                store.persist(manifest)
                return manifest

            if attempt >= max_retries:
                break

            prev_code = _read_optional_text(edit_py)
            err = result.get("stderr") or result.get("stdout") or "exec failed"
            probe_note = ""
            cached = manifest.checks.get("feature_probe")
            if isinstance(cached, dict):
                probe_note = _feature_probe_block(cached)
            repair_text = (
                "Fix this CadQuery script so it runs AND still performs the intended "
                "edit. Do not replace the imported part with a placeholder box.\n\n"
                f"input.step path: {manifest.input_step}\n\n"
                f"{probe_note}"
                f"Edit Description:\n{desc}\n\n"
                f"edit.json:\n{edit_json}\n\n"
                "=== RUNTIME ERROR (must fix) ===\n"
                f"{err}\n\n"
                f"Attached: {len(view_paths)} CAD views + {len(target_paths)} "
                "target renders of the intended edit.\n\n"
                "=== Previous edit.py ===\n"
                f"{prev_code}\n\n"
                "Return a CadQuery script defining `result`. "
                "Return ONLY python (or a python fence)."
            )
            content = _vision_content(repair_text, images)
            store_sess.append_transcript(
                sid,
                {
                    "type": "user",
                    "content": repair_text,
                    "images": [str(p) for p in images],
                    "kind": "exec_repair",
                    "attempt": attempt,
                },
            )
            turn = media.chat([{"role": "user", "content": content}])
            store_sess.append_transcript(
                sid,
                {
                    "type": "assistant",
                    "content": turn.content or "",
                    "kind": "exec_repair",
                    "attempt": attempt,
                },
            )
            code = _prefer_runnable_code(
                _extract_python_code(turn.content or ""),
                prev_code,
            )
            if code == prev_code:
                _progress(
                    f"S3 exec repair {attempt}: AST-invalid coder reply — keeping previous code"
                )
            store.write_text(REL_EDIT_PY, code)
            edit_py = store.path(REL_EDIT_PY)
            manifest.artifacts["edit_py"] = str(edit_py.resolve())

        manifest.artifacts["output_step"] = str(out_step)
        manifest.artifacts["output_stl"] = str(out_stl)
        manifest.checks["exec"] = {
            "hard": "fail",
            "exit_code": attempts[-1]["exit_code"] if attempts else -1,
            "attempts": attempts,
        }
        manifest.set_summary("cadquery_exec failed after S3 retries")
        store.persist(manifest)
        raise StageError("cadquery_exec failed after retries")

    return handler


def make_result_verify_handler(settings: Settings) -> Callable[..., Manifest]:
    """S4: re-render output.step at two base views + helper QC; 5 attempts total."""

    def handler(store: ArtifactStore, manifest: Manifest, ctx: dict[str, Any]) -> Manifest:
        import shutil

        from harness.agents.helper import HelperAgent
        from harness.pipeline.base_view import resolve_base_views
        from harness.pipeline.skills.cad_views import render_oriented_view_png
        from harness.pipeline.skills.cadquery_exec import run_cadquery_script
        from harness.pipeline.target_qc import parse_result_qc_json
        from harness.pipeline.view_qc_adjust import DEFAULT_FACTOR, clamp_factor
        from harness.session.store import SessionStore

        cfg = _pipeline_cfg(settings)
        media = ctx.get("media_provider") or _media(settings)
        helper = ctx.get("helper") or HelperAgent(
            media,
            workspace=settings.workspace,
            agent_id="cad_edit",
        )
        max_retries = int(
            ctx.get("result_qc_max_retries", cfg.get("result_qc_max_retries", 5))
        )
        n_attempts = max(1, max_retries)
        size = int(ctx.get("view_size", cfg.get("view_size", 512)))

        store_sess = SessionStore(settings.workspace, agent_id="cad_edit")
        sid = store_sess.create_session(label="S4_result_qc", agent_role="result_qc")
        manifest.sessions["S4_result_qc"] = sid

        # S1 may keep both targets, only one, or none (pair QC dropped
        # both). Respect that explicit selection here; otherwise preserve
        # the normal two-view default.
        selected_views = manifest.artifacts.get("base_views")
        if isinstance(selected_views, list) and selected_views:
            base_views = [str(v) for v in selected_views]
        else:
            base_views = resolve_base_views(manifest.artifacts, count=2)
        base_view = base_views[0]
        orient = dict(manifest.artifacts.get("orient") or {})
        up_axis = str(orient.get("up_axis") or "+Z")
        factors = dict(manifest.artifacts.get("view_factors") or {})
        view_factor = {
            v: clamp_factor(float(factors.get(v, DEFAULT_FACTOR))) for v in base_views
        }

        views = dict(manifest.artifacts.get("views") or {})
        target_renders = {
            str(k): Path(str(v))
            for k, v in dict(manifest.artifacts.get("target_renders") or {}).items()
            if v and Path(str(v)).is_file()
        }

        before_paths: dict[str, Path] = {}
        target_paths: dict[str, Path] = {}
        for view in base_views:
            before = Path(str(views.get(view) or ""))
            if not before.is_file():
                raise StageError(f"result verify missing before view: {view}")
            before_paths[view] = before
            tgt = target_renders.get(view)
            if tgt is not None:
                target_paths[view] = tgt

        output_step = Path(
            str(manifest.artifacts.get("output_step") or store.path(REL_OUTPUT_STEP))
        )
        out_stl = Path(
            str(manifest.artifacts.get("output_stl") or store.path(REL_OUTPUT_STL))
        )
        result_path = store.path(REL_RESULT)
        result_paths: dict[str, Path] = {
            v: store.path(rel_result_view(v)) for v in base_views
        }

        if not output_step.is_file():
            raise StageError("result verify missing output.step")

        desc = Path(manifest.description_txt).read_text(encoding="utf-8")
        p_qc = _load_prompt(settings, "s4_result_qc.md")
        edit_py_path = Path(str(manifest.artifacts.get("edit_py") or store.path(REL_EDIT_PY)))

        attempts: list[dict[str, Any]] = []
        last_qc: dict[str, Any] | None = None

        def _current_edit_py() -> str:
            if edit_py_path.is_file():
                try:
                    return edit_py_path.read_text(encoding="utf-8")
                except OSError:
                    return ""
            return ""

        _progress(
            f"S4 result QC start session={sid} base_views={base_views} "
            f"up_axis={up_axis} max_attempts={n_attempts} "
            f"exec_retries={cfg.get('result_qc_exec_retries', 1)} "
            f"qc_context=fresh"
        )

        def _run_result_qc(
            *,
            attempt: int,
            images: list[Path],
        ) -> tuple[dict[str, Any], Any]:
            """Fresh QC turn (no prior-attempt history) against current edit.py."""
            code = _current_edit_py()
            qc_prompt = (
                f"{p_qc}\n\n"
                f"base_views={base_views}\nup_axis={up_axis}\n"
                f"attempt={attempt + 1}/{n_attempts} — judge THIS script only.\n\n"
                f"Edit Description:\n{desc}\n\n"
                "=== Current edit.py ===\n"
                f"{code}\n\n"
                "Images per camera, in order: "
                "[before], [target], [result] "
                f"for each of {base_views}.\n"
                "If reject: fix_hint must be English "
                "'<this code is wrong>; it should <the operation to use>'. "
                "Return ONLY the JSON object."
            )
            store_sess.append_transcript(
                sid,
                {
                    "type": "user",
                    "content": qc_prompt,
                    "images": [str(p) for p in images],
                    "attempt": attempt,
                    "shared_qc": False,
                },
            )
            qc_result = helper.run(
                qc_prompt,
                images=images,
                label=f"S4_result_qc_{attempt}",
                parent_session_id=sid,
                detail="high",
            )
            store_sess.append_transcript(
                sid,
                {
                    "type": "assistant",
                    "content": qc_result.content or "",
                    "attempt": attempt,
                    "ok": qc_result.ok,
                    "helper_session": qc_result.session_id,
                },
            )

            parsed = parse_result_qc_json(qc_result.content or "")
            if parsed is None:
                _progress(
                    f"S4 attempt {attempt}: QC reply unparseable — re-asking…"
                )
                retry_prompt = (
                    "Return ONLY the required JSON. On reject, set fix_hint to "
                    "one English sentence: what this edit.py does wrong, and "
                    "what it should do instead."
                )
                store_sess.append_transcript(
                    sid,
                    {
                        "type": "user",
                        "content": retry_prompt,
                        "attempt": attempt,
                    },
                )
                qc_retry = helper.run(
                    retry_prompt,
                    images=images,
                    label=f"S4_result_qc_{attempt}",
                    session_id=qc_result.session_id,
                    history=list(qc_result.history or []),
                    detail="high",
                )
                store_sess.append_transcript(
                    sid,
                    {
                        "type": "assistant",
                        "content": qc_retry.content or "",
                        "attempt": attempt,
                        "helper_session": qc_retry.session_id,
                    },
                )
                parsed = parse_result_qc_json(qc_retry.content or "")
                qc_result = qc_retry

            if parsed is None:
                parsed = {
                    "pass": False,
                    "verdict": "reject",
                    "matches_description": False,
                    "preserves_unchanged": False,
                    "evaluation": (
                        "The edit is not visible or the solid is broken; "
                        "it should apply the Edit Description boolean on the "
                        "imported STEP and keep unrelated faces."
                    ),
                    "issues": "unparseable QC reply",
                    "fix_hint": (
                        "The edit is not visible or the solid is broken; "
                        "it should apply the Edit Description boolean on the "
                        "imported STEP and keep unrelated faces."
                    ),
                }
            hint = str(parsed.get("fix_hint") or parsed.get("issues") or "").strip()
            if hint:
                parsed["fix_hint"] = hint
                parsed["issues"] = hint
                parsed["evaluation"] = hint
            return parsed, qc_result

        for attempt in range(n_attempts):
            _progress(
                f"S4 attempt {attempt + 1}/{n_attempts}: render output.step → "
                f"result_<view>.png (views={base_views})"
            )
            render_ok = True
            render_backends: dict[str, Any] = {}
            render_detail = ""
            for view in base_views:
                dest = result_paths[view]
                render = render_oriented_view_png(
                    output_step,
                    dest,
                    up_axis=up_axis,
                    view_name=view,
                    camera_distance_factor=view_factor[view],
                    size=size,
                )
                render_backends[view] = render.get("backend")
                if not render.get("ok") or not dest.is_file():
                    render_ok = False
                    render_detail = str(
                        render.get("detail") or render.get("backend") or "render failed"
                    )
                    _progress(
                        f"S4 attempt {attempt}: RENDER FAIL {view} — "
                        f"{_clip(render_detail)}"
                    )
                    break
                _progress(
                    f"S4 attempt {attempt + 1}/{n_attempts}: render ok {view} "
                    f"backend={render.get('backend')}"
                )

            if render_ok:
                first_result = result_paths[base_views[0]]
                try:
                    shutil.copy2(first_result, result_path)
                except Exception:  # noqa: BLE001
                    pass
                manifest.artifacts["result_render"] = str(result_path.resolve())
                manifest.artifacts["result_renders"] = {
                    v: str(result_paths[v].resolve()) for v in base_views
                }

            if not render_ok:
                attempts.append(
                    {
                        "attempt": attempt,
                        "pass": False,
                        "render_ok": False,
                        "issues": render_detail,
                        "evaluation": (
                            f"Could not re-render output.step: {render_detail}"
                        ),
                        "fix_hint": (
                            "edit.py did not export a usable STEP; "
                            "it should define result and export a non-empty solid."
                        ),
                    }
                )
                last_qc = attempts[-1]
                if attempt >= n_attempts - 1:
                    break
                evaluation = last_qc["evaluation"]
                fix_hint = last_qc["fix_hint"]
            else:
                _progress(
                    f"S4 attempt {attempt + 1}/{n_attempts}: asking QC helper "
                    f"(fresh, against current edit.py)…"
                )
                images: list[Path] = []
                for view in base_views:
                    images.append(before_paths[view])
                    tgt = target_paths.get(view)
                    if tgt is not None:
                        images.append(tgt)
                    images.append(result_paths[view])
                parsed, qc_result = _run_result_qc(
                    attempt=attempt,
                    images=images,
                )

                attempt_rec = {
                    "attempt": attempt,
                    "pass": bool(parsed.get("pass")),
                    "verdict": parsed.get("verdict"),
                    "matches_description": parsed.get("matches_description"),
                    "preserves_unchanged": parsed.get("preserves_unchanged"),
                    "evaluation": parsed.get("evaluation"),
                    "issues": parsed.get("issues"),
                    "fix_hint": parsed.get("fix_hint"),
                    "render_ok": True,
                    "render_backend": render_backends,
                    "helper_session": qc_result.session_id,
                    "qc_shared_context": False,
                    "base_views": list(base_views),
                }
                attempts.append(attempt_rec)
                last_qc = parsed

                verdict = "PASS" if parsed.get("pass") else "REJECT"
                _progress(
                    f"S4 attempt {attempt + 1}/{n_attempts}: QC {verdict} "
                    f"matches={parsed.get('matches_description')} "
                    f"preserves={parsed.get('preserves_unchanged')}"
                )
                _progress(
                    f"S4 attempt {attempt + 1}/{n_attempts}: "
                    f"fix_hint={_clip(parsed.get('fix_hint'), 320)}"
                )

                if parsed.get("pass"):
                    if qc_result.session_id:
                        try:
                            store_sess.update_meta(qc_result.session_id, status="done")
                        except Exception:  # noqa: BLE001
                            pass
                    manifest.artifacts["result_render"] = str(result_path.resolve())
                    manifest.artifacts["result_renders"] = {
                        v: str(result_paths[v].resolve()) for v in base_views
                    }
                    manifest.checks["result_qc"] = {
                        "hard": "pass",
                        "session": sid,
                        "helper_session": qc_result.session_id,
                        "qc_shared_context": False,
                        "base_view": base_view,
                        "base_views": list(base_views),
                        "up_axis": up_axis,
                        "camera_distance_factor": view_factor,
                        "attempts": attempts,
                        "qc": parsed,
                        "helper": "helper",
                    }
                    manifest.set_summary(
                        f"S4 result QC pass (bases={','.join(base_views)}, "
                        f"attempt={attempt + 1}/{n_attempts})"
                    )
                    store.persist(manifest)
                    _progress(
                        f"S4 result QC PASS on attempt {attempt + 1}/{n_attempts}"
                    )
                    return manifest

                evaluation = str(parsed.get("fix_hint") or parsed.get("issues") or "")
                fix_hint = evaluation

            if attempt >= n_attempts - 1:
                _progress(
                    f"S4 exhausted attempts ({n_attempts}); no more repairs"
                )
                break

            # Reject → rewrite edit.py (with evaluation) and re-exec until it runs,
            # before spending another visual QC cycle on stale output.step.
            prev_code = ""
            if edit_py_path.is_file():
                prev_code = edit_py_path.read_text(encoding="utf-8")

            exec_retries = int(
                ctx.get(
                    "result_qc_exec_retries",
                    cfg.get("result_qc_exec_retries", 0),
                )
            )
            working_code = prev_code
            last_stderr = ""
            repairs: list[dict[str, Any]] = []
            exec_ok = False

            for repair_i in range(exec_retries + 1):
                _progress(
                    f"S4 attempt {attempt + 1}/{n_attempts}: reject→coder repair "
                    f"{repair_i + 1}/{exec_retries + 1}…"
                )
                stderr_block = ""
                if last_stderr:
                    stderr_block = (
                        f"\nRuntime error: {last_stderr}\n"
                        "This call failed; it should select a valid face/solid "
                        "and not raise.\n"
                    )
                probe_cached = manifest.checks.get("feature_probe")
                probe_note = _feature_probe_block(
                    probe_cached if isinstance(probe_cached, dict) else None
                )
                repair_prompt = (
                    "Fix this CadQuery script.\n\n"
                    f"{fix_hint}\n"
                    f"{stderr_block}\n"
                    f"input.step: {manifest.input_step}\n"
                    f"{probe_note}\n"
                    "=== Current edit.py ===\n"
                    f"{prev_code}\n\n"
                    "Return ONLY python defining `result`. "
                    "Do not replace the imported part with a placeholder box. "
                    "Do not raise RuntimeError when a selector is empty; "
                    "widen the face/solid search instead. "
                    "If a feature to duplicate is fused into a parent solid, "
                    "measure its faces on that solid and union only new instances; "
                    "do not translate the whole parent. "
                    "If the image shows a feature with no STEP counterpart, "
                    "reconstruct it or skip that selector; do not force a search."
                )
                store_sess.append_transcript(
                    sid,
                    {
                        "type": "user",
                        "content": repair_prompt,
                        "kind": "reject_to_coder",
                        "attempt": attempt,
                        "repair": repair_i,
                    },
                )
                turn = media.chat(
                    [{"role": "user", "content": repair_prompt}]
                )
                store_sess.append_transcript(
                    sid,
                    {
                        "type": "assistant",
                        "content": turn.content or "",
                        "kind": "coder_repair",
                        "attempt": attempt,
                        "repair": repair_i,
                    },
                )
                code = _prefer_runnable_code(
                    _extract_python_code(turn.content or ""),
                    prev_code,
                )
                if not _ast_check(code)["ok"]:
                    _progress(
                        f"S4 attempt {attempt} repair {repair_i}: "
                        "AST-invalid coder reply — skipping write"
                    )
                    continue
                store.write_text(REL_EDIT_PY, code)
                edit_py_path = store.path(REL_EDIT_PY)
                manifest.artifacts["edit_py"] = str(edit_py_path.resolve())
                prev_code = code
                _progress(
                    f"S4 attempt {attempt} repair {repair_i}: "
                    f"wrote edit.py ({len(code)} chars) — exec…"
                )

                exec_result = run_cadquery_script(
                    edit_py_path, output_step, out_stl, cwd=store.root
                )
                err_text = (
                    (exec_result.get("stderr") or "")
                    or (exec_result.get("stdout") or "")
                )[:2000]
                repairs.append(
                    {
                        "repair": repair_i,
                        "ok": bool(exec_result.get("ok")),
                        "exit_code": exec_result.get("exit_code"),
                        "stderr": err_text[:800],
                    }
                )
                store_sess.append_transcript(
                    sid,
                    {
                        "type": "system_note",
                        "content": (
                            f"re-exec attempt={attempt} repair={repair_i}: "
                            f"ok={exec_result.get('ok')} "
                            f"exit={exec_result.get('exit_code')}"
                        ),
                        "stderr": err_text[:800],
                    },
                )
                if exec_result.get("ok"):
                    manifest.artifacts["output_step"] = exec_result["output_step"]
                    manifest.artifacts["output_stl"] = exec_result["output_stl"]
                    output_step = Path(exec_result["output_step"])
                    out_stl = Path(exec_result["output_stl"])
                    exec_ok = True
                    last_stderr = ""
                    _progress(
                        f"S4 attempt {attempt} repair {repair_i}: "
                        "exec OK — will re-render & QC"
                    )
                    break

                last_stderr = err_text or "cadquery_exec failed with empty stderr"
                _progress(
                    f"S4 attempt {attempt} repair {repair_i}: "
                    f"exec FAIL exit={exec_result.get('exit_code')} "
                    f"— {_clip(last_stderr, 240)}"
                )

            attempts[-1]["exec_ok"] = exec_ok
            attempts[-1]["repairs"] = repairs
            if last_stderr:
                attempts[-1]["exec_stderr"] = last_stderr[:800]
            if not exec_ok:
                _progress(
                    f"S4 attempt {attempt + 1}/{n_attempts}: "
                    "repairs failed — restore last working edit.py"
                )
                if working_code:
                    store.write_text(REL_EDIT_PY, working_code)
                    edit_py_path = store.path(REL_EDIT_PY)
                    manifest.artifacts["edit_py"] = str(edit_py_path.resolve())
                err_hint = (
                    "The rewrite raised RuntimeError / failed to run; "
                    "it should not raise, widen face/solid selection, "
                    "and still apply the Edit Description on the import."
                )
                attempts[-1]["evaluation"] = err_hint
                last_qc = {
                    **(last_qc or {}),
                    "evaluation": err_hint,
                    "issues": err_hint,
                    "fix_hint": err_hint,
                }
                if attempt >= n_attempts - 1:
                    break
                fix_hint = err_hint
                continue

        detail = ""
        if last_qc:
            detail = str(
                last_qc.get("fix_hint")
                or last_qc.get("issues")
                or last_qc.get("evaluation")
                or "result QC failed"
            )
        last_helper = None
        if attempts:
            last_helper = attempts[-1].get("helper_session")
        if last_helper:
            try:
                store_sess.update_meta(str(last_helper), status="done")
            except Exception:  # noqa: BLE001
                pass
        manifest.artifacts["result_render"] = (
            str(result_path.resolve()) if result_path.is_file() else None
        )
        manifest.artifacts["result_renders"] = {
            v: str(result_paths[v].resolve())
            for v in base_views
            if result_paths[v].is_file()
        }
        manifest.checks["result_qc"] = {
            "hard": "fail",
            "session": sid,
            "helper_session": last_helper,
            "qc_shared_context": False,
            "base_view": base_view,
            "base_views": list(base_views),
            "up_axis": up_axis,
            "camera_distance_factor": view_factor,
            "attempts": attempts,
            "qc": last_qc,
            "helper": "helper",
        }
        manifest.set_summary(f"S4 result QC failed: {detail[:240]}")
        store.persist(manifest)
        _progress(f"S4 result QC FAILED after {len(attempts)} attempt(s)")
        raise StageError(f"result verify QC failed: {detail}")

    return handler
