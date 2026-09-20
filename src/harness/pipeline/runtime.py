"""Wire real or stub stage handlers into pipeline ctx."""

from __future__ import annotations

from typing import Any

from harness.config import Settings


def build_pipeline_context(settings: Settings, use_real: bool = True) -> dict[str, Any]:
    """Build ``ctx`` handlers for CADEditPipeline.

    When ``use_real`` is False, returns empty ctx (all stage stubs).
    When True, attaches Phase 2/3 handlers that fall back to stubs on missing deps.
    """
    stub_media = bool(getattr(settings.pipeline, "stub_media", False))
    ctx: dict[str, Any] = {
        "settings": settings,
        "stub_media": stub_media,
        "cad_views_max_retries": settings.pipeline.cad_views_max_retries,
        "s1_max_retries": settings.pipeline.s1_max_retries,
        "s3_max_retries": settings.pipeline.s3_max_retries,
        "result_qc_max_retries": settings.pipeline.result_qc_max_retries,
        "result_qc_exec_retries": settings.pipeline.result_qc_exec_retries,
        "view_size": settings.pipeline.view_size,
        "image_model": settings.pipeline.image_model,
    }
    if not use_real:
        return ctx

    from harness.pipeline.handlers import (
        make_cad_coder_handler,
        make_cad_views_handler,
        make_cadquery_exec_handler,
        make_edit_structurer_handler,
        make_edit_visualizer_handler,
        make_result_verify_handler,
    )
    from harness.providers.stub_media import StubMediaProvider
    from harness.providers.usage import TokenMeter, bind_meter

    if stub_media:
        ctx["media_provider"] = StubMediaProvider(settings.workspace)
    else:
        from harness.providers.gpt_media import GptMediaProvider

        ctx["media_provider"] = GptMediaProvider.from_settings(settings)

    from harness.agents.helper import HelperAgent

    ctx["helper"] = HelperAgent(
        ctx["media_provider"],
        workspace=settings.workspace,
        agent_id="cad_edit",
    )
    meter = TokenMeter()
    ctx["token_meter"] = meter
    bind_meter(ctx.get("media_provider"), meter)
    ctx["cad_views"] = make_cad_views_handler(settings)
    ctx["edit_visualizer"] = make_edit_visualizer_handler(settings)
    ctx["edit_structurer"] = make_edit_structurer_handler(settings)
    ctx["cad_coder"] = make_cad_coder_handler(settings)
    ctx["cadquery_exec"] = make_cadquery_exec_handler(settings)
    ctx["result_verify"] = make_result_verify_handler(settings)
    from harness.pipeline.edit_harness.handler import make_edit_harness_handler

    ctx["edit_harness"] = make_edit_harness_handler(settings)
    ctx["use_legacy_s2"] = not bool(getattr(settings.pipeline, "use_edit_harness", True))
    return ctx
