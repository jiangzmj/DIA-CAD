"""Accumulate LLM token usage for a pipeline run (API numbers, else estimate)."""

from __future__ import annotations

from typing import Any


def parse_api_usage(data: Any) -> dict[str, int] | None:
    """Normalize Responses / Chat Completions ``usage`` blobs."""
    if not isinstance(data, dict):
        return None
    candidates: list[Any] = [
        data.get("usage"),
        (data.get("response") or {}).get("usage")
        if isinstance(data.get("response"), dict)
        else None,
        data,
    ]
    usage = None
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        if any(
            k in cand
            for k in (
                "input_tokens",
                "prompt_tokens",
                "output_tokens",
                "completion_tokens",
                "total_tokens",
            )
        ):
            usage = cand
            break
    if not isinstance(usage, dict):
        return None
    input_t = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or 0
    )
    output_t = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or 0
    )
    details = (
        usage.get("output_tokens_details")
        or usage.get("input_tokens_details")
        or usage.get("completion_tokens_details")
        or {}
    )
    if not isinstance(details, dict):
        details = {}
    reasoning = (
        details.get("reasoning_tokens")
        or usage.get("reasoning_tokens")
        or 0
    )
    total = usage.get("total_tokens")
    try:
        input_t = int(input_t or 0)
        output_t = int(output_t or 0)
        reasoning = int(reasoning or 0)
        total = int(total) if total is not None else input_t + output_t
    except (TypeError, ValueError):
        return None
    if input_t <= 0 and output_t <= 0 and total <= 0:
        return None
    return {
        "input_tokens": input_t,
        "output_tokens": output_t,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
    }


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _image_tokens(detail: str | None) -> int:
    # OpenAI high-detail ~4 tiles on 1024²; low is a flat 85.
    if (detail or "high").lower() == "low":
        return 85
    return 765


def estimate_messages(messages: list[dict[str, Any]] | None) -> tuple[int, int]:
    """Return (text_tokens, image_tokens) for a chat request."""
    text_t = 0
    image_t = 0
    for msg in messages or []:
        content = msg.get("content")
        if isinstance(content, str):
            text_t += estimate_text_tokens(content)
            continue
        if not isinstance(content, list):
            text_t += estimate_text_tokens(str(content or ""))
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in {"text", "input_text", "output_text"}:
                text_t += estimate_text_tokens(str(part.get("text") or ""))
            elif ptype in {"image_url", "input_image"}:
                image = part.get("image_url")
                detail = part.get("detail")
                if isinstance(image, dict):
                    detail = image.get("detail") or detail
                image_t += _image_tokens(detail)
    return text_t, image_t


class TokenMeter:
    """One meter per pipeline run. Thread-unsafe; the CLI is sequential."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.api_calls = 0
        self.estimated_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0

    def record(
        self,
        *,
        data: Any = None,
        turn: Any = None,
        messages: list[dict[str, Any]] | None = None,
        output_text: str | None = None,
    ) -> None:
        usage = None
        if turn is not None:
            usage = parse_api_usage(getattr(turn, "usage", None))
            if usage is None:
                usage = parse_api_usage(getattr(turn, "raw", None))
            if output_text is None:
                output_text = getattr(turn, "content", None)
        if usage is None:
            usage = parse_api_usage(data)
        self.calls += 1
        if usage:
            self.api_calls += 1
            inp = usage["input_tokens"]
            out = usage["output_tokens"]
            if inp == 0 and out == 0:
                out = usage["total_tokens"]
            self.input_tokens += inp
            self.output_tokens += out
            self.reasoning_tokens += usage["reasoning_tokens"]
            return
        self.estimated_calls += 1
        text_in, image_in = estimate_messages(messages)
        out = estimate_text_tokens(output_text or "")
        reason = estimate_text_tokens(getattr(turn, "reasoning", None) or "")
        self.input_tokens += text_in + image_in
        self.output_tokens += out + reason
        self.reasoning_tokens += reason

    def snapshot(self) -> dict[str, Any]:
        total = self.input_tokens + self.output_tokens
        estimated = self.estimated_calls > 0 and self.api_calls == 0
        mixed = self.estimated_calls > 0 and self.api_calls > 0
        return {
            "calls": self.calls,
            "api_calls": self.api_calls,
            "estimated_calls": self.estimated_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": total,
            "estimated": estimated,
            "mixed": mixed,
        }

    def format_line(self) -> str:
        snap = self.snapshot()
        prefix = "tokens≈" if snap["estimated"] else "tokens="
        parts = [
            f"input={snap['input_tokens']}",
            f"output={snap['output_tokens']}",
        ]
        if snap["reasoning_tokens"]:
            parts.append(f"reasoning={snap['reasoning_tokens']}")
        parts.append(f"calls={snap['calls']}")
        if snap["estimated"]:
            parts.append("estimated")
        elif snap["mixed"]:
            parts.append(
                f"{snap['estimated_calls']}/{snap['calls']} estimated"
            )
        return f"{prefix}{snap['total_tokens']} ({', '.join(parts)})"


def format_duration(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def bind_meter(media: Any, meter: TokenMeter | None) -> None:
    if media is None:
        return
    setter = getattr(media, "set_meter", None)
    if callable(setter):
        setter(meter)
        return
    chat = getattr(media, "_chat", None)
    if chat is not None:
        chat.meter = meter
