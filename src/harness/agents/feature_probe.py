"""FeatureProbeAgent: expose STEP facts and candidates without modeling it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

from harness.pipeline.feature_probe import build_feature_probe, compact_probe_for_prompt


class FeatureProbeAgent:
    """Thin compatibility wrapper around the deterministic unified probe."""

    role = "feature_probe"

    def decide(self, inspect: dict[str, Any]) -> dict[str, Any]:
        """Describe evidence availability; never choose a CadQuery strategy."""
        if not inspect.get("ok"):
            return {"action": "report_unavailable", "run_smoke": False, "reason": inspect.get("error") or "inspect failed"}
        model = inspect.get("model") or {}
        candidates = list(inspect.get("feature_candidates") or [])
        kinds = sorted({str(c.get("kind") or "unknown") for c in candidates})
        return {
            "action": "report_candidates",
            "run_smoke": False,
            "reason": "Geometry facts and candidates are available; the MLLM must choose the target and modeling method.",
            "solid_count": int(model.get("solid_count") or 0),
            "candidate_count": len(candidates),
            "candidate_kinds": kinds,
        }

    def run(
        self,
        step_path: Path | str,
        *,
        force_smoke: bool = False,
        smoke_out: Path | str | None = None,
        cache_dir: Path | str | None = None,
        palmetto_engine: Path | str | None = None,
        palmetto_enabled: bool = True,
        palmetto_timeout_s: float = 120.0,
        force: bool = False,
    ) -> dict[str, Any]:
        del smoke_out
        inspect = build_feature_probe(
            step_path,
            cache_dir=cache_dir,
            palmetto_engine=palmetto_engine,
            palmetto_enabled=palmetto_enabled,
            palmetto_timeout_s=palmetto_timeout_s,
            force=force,
        )
        plan = self.decide(inspect)
        smoke: dict[str, Any] | None = None
        if force_smoke:
            smoke = {
                "ok": False,
                "skipped": True,
                "error": "reconstruction smoke disabled: probes may not choose or execute modeling strategies",
            }
        return {
            "ok": bool(inspect.get("ok")),
            "inspect": inspect,
            "plan": plan,
            "smoke": smoke,
            "coder_note": self.to_coder_note(plan, inspect),
        }

    @staticmethod
    def to_coder_note(plan: dict[str, Any], inspect: dict[str, Any] | None = None) -> str:
        compact = compact_probe_for_prompt(inspect or {})
        return (
            "FeatureProbe facts/candidates (not ground truth; do not infer a required boolean strategy):\n"
            + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        )
