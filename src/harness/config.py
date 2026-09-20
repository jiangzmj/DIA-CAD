"""Load config.yaml + .env into a single settings object."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ContextSettings:
    max_messages: int = 80
    max_tool_chars: int = 20000
    compact_keep_recent: float = 0.2
    compact_summarize_oldest: float = 0.5
    handoff_default: str = "summary"
    handoff_window: int = 12
    system_layers: list[str] = field(
        default_factory=lambda: ["identity", "soul", "tools", "runtime"]
    )


@dataclass
class AgentSettings:
    max_tool_iterations: int = 25
    max_depth: int = 2
    max_child_turns: int = 20


@dataclass
class HttpToolSettings:
    timeout_seconds: float = 30.0
    max_response_chars: int = 50000
    allow_private_hosts: bool = False


@dataclass
class ToolSettings:
    http_request: HttpToolSettings = field(default_factory=HttpToolSettings)
    max_bash_output_chars: int = 50000


@dataclass
class ProviderSettings:
    primary: str = "openai_compat"
    fallback: str = "openai_compat"
    model: str = "gpt-4.1-mini"
    chatgpt_model: str = "gpt-5.6-sol"
    chatgpt_reasoning_effort: str = "high"
    openai_api_key: str = ""
    openai_base_url: str | None = None
    chatgpt_auth_path: Path | None = None
    proxy: str | None = None
    auto_detect_proxy: bool = True


@dataclass
class HarnessSettings:
    """Budgets and gates for the S2+ edit harness (not a fixed S2–S4 loop)."""

    max_edit_rounds: int = 5
    max_code_repairs: int = 3
    max_mllm_calls: int = 12
    max_view_searches: int = 3
    max_views_per_search: int = 3
    memory_max_messages: int = 8
    memory_max_chars: int = 24000
    memory_summary_chars: int = 6000
    exec_timeout_seconds: float = 120.0
    silhouette_iou_min: float = 0.55
    silhouette_iou_floor: float = 0.40
    cross_renderer_iou_min: float = 0.28
    cross_renderer_iou_floor: float = 0.15
    preservation_bbox_rel_tol: float = 0.50
    preservation_bbox_rel_tol_replace: float = 1.20
    volume_explode_ratio: float = 8.0
    volume_explode_ratio_global: float = 12.0
    wrecked_iou_min: float = 0.25
    view_alignment_min: float = 0.40
    region_change_iou_min: float = 0.12
    region_edge_iou_min: float = 0.12
    region_mask_max_frac: float = 0.25
    region_mask_min_frac: float = 0.001
    noop_silhouette_iou_min: float = 0.997
    noop_pixel_mae_max: float = 0.008
    thresholds_are_placeholders: bool = True
    calibrated_on: str = ""


@dataclass
class FeatureProbeSettings:
    enabled: bool = True
    cache_dir: str = "feature_probe_cache"
    palmetto_enabled: bool = True
    palmetto_engine_path: str = ""
    palmetto_timeout_seconds: float = 120.0


@dataclass
class PipelineSettings:
    artifacts_root: str = "artifacts"
    input_dir: str = "input"  # repo-root folder for STEP + description defaults
    default_input: str = "input/input.step"
    default_description: str = "input/description.txt"
    cad_views_max_retries: int = 3
    s1_max_retries: int = 2  # S1 pair-QC rounds (generate both, QC, maybe repair once)
    s3_max_retries: int = 2  # CadQuery coder / exec rewrite attempts
    result_qc_max_retries: int = 5  # 5 QC attempts total
    result_qc_exec_retries: int = 1  # each rewrite may retry exec once
    stub_media: bool = False
    image_model: str = "gpt-image-1"
    prompts_dir: str | None = None  # default: workspace/prompts/cad_edit
    view_size: int = 512
    use_edit_harness: bool = True
    harness: HarnessSettings = field(default_factory=HarnessSettings)
    feature_probe: FeatureProbeSettings = field(default_factory=FeatureProbeSettings)


@dataclass
class Settings:
    root: Path = ROOT
    workspace: Path = ROOT / "workspace"
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    context: ContextSettings = field(default_factory=ContextSettings)
    tools: ToolSettings = field(default_factory=ToolSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    proxy: str | None = None


def _as_path(value: str | Path | None, default: Path) -> Path:
    if not value:
        return default
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_settings(
    config_path: Path | None = None,
    env_path: Path | None = None,
) -> Settings:
    load_dotenv(env_path or (ROOT / ".env"), override=False)

    cfg_file = config_path or (ROOT / "config.yaml")
    raw: dict[str, Any] = {}
    if cfg_file.exists():
        raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}

    ctx_raw = raw.get("context") or {}
    agent_raw = raw.get("agent") or {}
    tools_raw = raw.get("tools") or {}
    http_raw = tools_raw.get("http_request") or {}
    prov_raw = raw.get("provider") or {}
    pipe_raw = raw.get("pipeline") or {}
    cad_views_raw = pipe_raw.get("cad_views") or {}
    harness_raw = pipe_raw.get("harness") or {}
    if not isinstance(harness_raw, dict):
        harness_raw = {}
    feature_probe_raw = pipe_raw.get("feature_probe") or {}
    if not isinstance(feature_probe_raw, dict):
        feature_probe_raw = {}

    workspace = _as_path(raw.get("workspace"), ROOT / "workspace")

    primary = os.getenv("DEFAULT_PROVIDER") or prov_raw.get("primary") or "openai_compat"
    model = os.getenv("MODEL_ID") or prov_raw.get("model") or "gpt-5.6"
    chatgpt_model = os.getenv("CHATGPT_MODEL") or prov_raw.get("chatgpt_model") or "gpt-5.6-sol"
    chatgpt_reasoning_effort = (
        os.getenv("CHATGPT_REASONING_EFFORT")
        or prov_raw.get("chatgpt_reasoning_effort")
        or "high"
    )
    api_key = os.getenv("OPENAI_API_KEY") or ""
    base_url = os.getenv("OPENAI_BASE_URL") or None
    if base_url == "":
        base_url = None
    image_model = (
        os.getenv("IMAGE_MODEL")
        or pipe_raw.get("image_model")
        or "gpt-image-1"
    )

    auth_env = os.getenv("CHATGPT_AUTH_PATH")
    auth_path = Path(auth_env).expanduser() if auth_env else Path.home() / ".codex" / "auth.json"

    net_raw = raw.get("network") or {}
    # Only an explicit yaml/CLI pin. Live HTTPS_PROXY and Clash/Tailscale
    # are resolved per request in harness.net.resolve_proxy.
    proxy = net_raw.get("proxy") or None
    auto_detect_proxy = bool(net_raw.get("auto_detect_proxy", True))
    if isinstance(proxy, str) and proxy.strip() == "":
        proxy = None

    settings = Settings(
        root=ROOT,
        workspace=workspace,
        proxy=proxy,
        provider=ProviderSettings(
            primary=primary,
            fallback=prov_raw.get("fallback") or "openai_compat",
            model=model,
            chatgpt_model=chatgpt_model,
            chatgpt_reasoning_effort=str(chatgpt_reasoning_effort),
            openai_api_key=api_key,
            openai_base_url=base_url,
            chatgpt_auth_path=auth_path,
            proxy=proxy,
            auto_detect_proxy=auto_detect_proxy,
        ),
        agent=AgentSettings(
            max_tool_iterations=int(agent_raw.get("max_tool_iterations", 25)),
            max_depth=int(agent_raw.get("max_depth", 2)),
            max_child_turns=int(agent_raw.get("max_child_turns", 20)),
        ),
        context=ContextSettings(
            max_messages=int(ctx_raw.get("max_messages", 80)),
            max_tool_chars=int(ctx_raw.get("max_tool_chars", 20000)),
            compact_keep_recent=float(ctx_raw.get("compact_keep_recent", 0.2)),
            compact_summarize_oldest=float(ctx_raw.get("compact_summarize_oldest", 0.5)),
            handoff_default=str(ctx_raw.get("handoff_default", "summary")),
            handoff_window=int(ctx_raw.get("handoff_window", 12)),
            system_layers=list(ctx_raw.get("system_layers") or ["identity", "soul", "tools", "runtime"]),
        ),
        tools=ToolSettings(
            http_request=HttpToolSettings(
                timeout_seconds=float(http_raw.get("timeout_seconds", 30)),
                max_response_chars=int(http_raw.get("max_response_chars", 50000)),
                allow_private_hosts=bool(http_raw.get("allow_private_hosts", False)),
            ),
            max_bash_output_chars=int(tools_raw.get("max_bash_output_chars", 50000)),
        ),
        pipeline=PipelineSettings(
            artifacts_root=str(pipe_raw.get("artifacts_root") or "artifacts"),
            input_dir=str(pipe_raw.get("input_dir") or "input"),
            default_input=str(
                pipe_raw.get("default_input") or "input/input.step"
            ),
            default_description=str(
                pipe_raw.get("default_description") or "input/description.txt"
            ),
            cad_views_max_retries=int(
                cad_views_raw.get("max_retries", pipe_raw.get("cad_views_max_retries", 3))
            ),
            s1_max_retries=int(pipe_raw.get("s1_max_retries", 2)),
            s3_max_retries=int(
                pipe_raw.get("s3_max_retries", pipe_raw.get("s4_max_retries", 2))
            ),
            result_qc_max_retries=int(pipe_raw.get("result_qc_max_retries", 5)),
            result_qc_exec_retries=int(pipe_raw.get("result_qc_exec_retries", 1)),
            stub_media=bool(pipe_raw.get("stub_media", False)),
            image_model=str(image_model),
            prompts_dir=pipe_raw.get("prompts_dir"),
            view_size=int(cad_views_raw.get("size", pipe_raw.get("view_size", 512))),
            use_edit_harness=bool(pipe_raw.get("use_edit_harness", True)),
            harness=HarnessSettings(
                max_edit_rounds=int(harness_raw.get("max_edit_rounds", 5)),
                max_code_repairs=int(harness_raw.get("max_code_repairs", 3)),
                max_mllm_calls=int(harness_raw.get("max_mllm_calls", 12)),
                max_view_searches=int(harness_raw.get("max_view_searches", 3)),
                max_views_per_search=int(harness_raw.get("max_views_per_search", 3)),
                memory_max_messages=int(harness_raw.get("memory_max_messages", 8)),
                memory_max_chars=int(harness_raw.get("memory_max_chars", 24000)),
                memory_summary_chars=int(harness_raw.get("memory_summary_chars", 6000)),
                exec_timeout_seconds=float(harness_raw.get("exec_timeout_seconds", 120.0)),
                silhouette_iou_min=float(harness_raw.get("silhouette_iou_min", 0.55)),
                silhouette_iou_floor=float(harness_raw.get("silhouette_iou_floor", 0.40)),
                cross_renderer_iou_min=float(harness_raw.get("cross_renderer_iou_min", 0.28)),
                cross_renderer_iou_floor=float(harness_raw.get("cross_renderer_iou_floor", 0.15)),
                preservation_bbox_rel_tol=float(
                    harness_raw.get("preservation_bbox_rel_tol", 0.50)
                ),
                preservation_bbox_rel_tol_replace=float(
                    harness_raw.get("preservation_bbox_rel_tol_replace", 1.20)
                ),
                volume_explode_ratio=float(harness_raw.get("volume_explode_ratio", 8.0)),
                volume_explode_ratio_global=float(
                    harness_raw.get("volume_explode_ratio_global", 12.0)
                ),
                wrecked_iou_min=float(harness_raw.get("wrecked_iou_min", 0.25)),
                view_alignment_min=float(harness_raw.get("view_alignment_min", 0.40)),
                region_change_iou_min=float(harness_raw.get("region_change_iou_min", 0.12)),
                region_edge_iou_min=float(harness_raw.get("region_edge_iou_min", 0.12)),
                region_mask_max_frac=float(harness_raw.get("region_mask_max_frac", 0.25)),
                region_mask_min_frac=float(harness_raw.get("region_mask_min_frac", 0.001)),
                noop_silhouette_iou_min=float(harness_raw.get("noop_silhouette_iou_min", 0.997)),
                noop_pixel_mae_max=float(harness_raw.get("noop_pixel_mae_max", 0.008)),
                thresholds_are_placeholders=bool(
                    harness_raw.get("thresholds_are_placeholders", True)
                ),
                calibrated_on=str(harness_raw.get("calibrated_on") or ""),
            ),
            feature_probe=FeatureProbeSettings(
                enabled=bool(feature_probe_raw.get("enabled", True)),
                cache_dir=str(feature_probe_raw.get("cache_dir") or "feature_probe_cache"),
                palmetto_enabled=bool(feature_probe_raw.get("palmetto_enabled", True)),
                palmetto_engine_path=str(feature_probe_raw.get("palmetto_engine_path") or ""),
                palmetto_timeout_seconds=float(
                    feature_probe_raw.get("palmetto_timeout_seconds", 120.0)
                ),
            ),
        ),
    )
    return settings
