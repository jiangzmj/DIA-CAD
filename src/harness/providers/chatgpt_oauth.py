"""ChatGPT Plus OAuth provider via Codex backend (personal/local use)."""

from __future__ import annotations

import json
import ssl
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from harness.auth.token_store import ensure_fresh_tokens, refresh_chatgpt_tokens
from harness.net import resolve_proxy
from harness.providers.base import AssistantTurn, ToolCall

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
IMAGE_GEN_ATTEMPTS = 3


def _is_transient_http(exc: BaseException) -> bool:
    """Proxy drops, TLS EOF, and reset connections are worth retrying."""
    if isinstance(exc, (httpx.TransportError, httpx.RemoteProtocolError, ssl.SSLError)):
        return True
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "unexpected_eof",
            "ssleof",
            "ssl:",
            "eof occurred",
            "connection reset",
            "broken pipe",
            "server disconnected",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "503",
            "502",
            "connection refused",
        )
    )


def _chat_tools_to_responses(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    out: list[dict[str, Any]] = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        out.append(
            {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def _content_to_response_parts(content: Any, *, role: str) -> list[dict[str, Any]]:
    """Map OpenAI chat content (str | multimodal parts) → Responses API parts."""
    text_type = "input_text" if role in {"system", "user"} else "output_text"
    if content is None:
        return [{"type": text_type, "text": ""}]
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    if not isinstance(content, list):
        return [{"type": text_type, "text": str(content)}]

    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append({"type": text_type, "text": str(part)})
            continue
        ptype = part.get("type")
        if ptype == "text":
            parts.append({"type": text_type, "text": part.get("text") or ""})
        elif ptype == "image_url":
            image = part.get("image_url") or {}
            url = image.get("url") if isinstance(image, dict) else None
            if not url:
                continue
            # Responses / Codex: input_image with data URL or http(s) URL.
            img_part: dict[str, Any] = {"type": "input_image", "image_url": url}
            detail = image.get("detail") if isinstance(image, dict) else None
            if detail:
                img_part["detail"] = detail
            parts.append(img_part)
        else:
            # Pass through unknown structured parts when already Responses-shaped.
            if ptype in {"input_text", "output_text", "input_image"}:
                parts.append(part)
            else:
                parts.append({"type": text_type, "text": json.dumps(part, ensure_ascii=False)})
    return parts or [{"type": text_type, "text": ""}]


def _messages_to_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI chat messages to Responses API input items."""
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            items.append(
                {
                    "role": "system",
                    "content": _content_to_response_parts(content, role="system"),
                }
            )
        elif role == "user":
            items.append(
                {
                    "role": "user",
                    "content": _content_to_response_parts(content, role="user"),
                }
            )
        elif role == "assistant":
            if msg.get("tool_calls"):
                if content:
                    items.append(
                        {
                            "role": "assistant",
                            "content": _content_to_response_parts(content, role="assistant"),
                        }
                    )
                for tc in msg["tool_calls"]:
                    fn = tc.get("function") or {}
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments") or "{}",
                        }
                    )
            else:
                items.append(
                    {
                        "role": "assistant",
                        "content": _content_to_response_parts(content, role="assistant"),
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id"),
                    "output": content if isinstance(content, str) else json.dumps(content),
                }
            )
    return items


def _output_items(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    output = data.get("output")
    if not isinstance(output, list):
        return []
    return [item for item in output if isinstance(item, dict)]


def _item_id(item: dict[str, Any]) -> str | None:
    for key in ("id", "call_id", "item_id"):
        val = item.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _normalize_image_item(item: dict[str, Any]) -> dict[str, Any]:
    """Unwrap ``response.image_generation_call.completed`` into a call item."""
    itype = str(item.get("type") or "")
    if itype == "response.image_generation_call.completed":
        inner = item.get("item")
        if isinstance(inner, dict):
            merged = dict(inner)
            if merged.get("result") in (None, "") and item.get("result") is not None:
                merged["result"] = item["result"]
            merged.setdefault("type", "image_generation_call")
            return merged
        out = dict(item)
        out["type"] = "image_generation_call"
        return out
    return item


def _append_unique_item(bucket: list[dict[str, Any]], item: dict[str, Any]) -> None:
    item_id = _item_id(item)
    if item_id and any(_item_id(existing) == item_id for existing in bucket):
        # Prefer the later copy (usually has result bytes).
        for i, existing in enumerate(bucket):
            if _item_id(existing) == item_id:
                bucket[i] = item
                return
        return
    bucket.append(item)


def _absorb_output_item(
    item: dict[str, Any],
    *,
    text_parts: list[str],
    reasoning_parts: list[str],
    tool_calls: list[dict[str, Any]],
    image_items: list[dict[str, Any]],
) -> None:
    itype = item.get("type")
    if itype == "function_call":
        _append_unique_item(tool_calls, item)
    elif itype in {
        "image_generation_call",
        "image_generation",
        "response.image_generation_call.completed",
    }:
        _append_unique_item(image_items, _normalize_image_item(item))
    elif itype == "reasoning":
        if any(part.strip() for part in reasoning_parts):
            return
        summary = item.get("summary") or item.get("content") or []
        if isinstance(summary, str) and summary.strip():
            reasoning_parts.append(summary)
        elif isinstance(summary, list):
            for part in summary:
                if isinstance(part, dict):
                    t = part.get("text") or part.get("summary") or ""
                    if t:
                        reasoning_parts.append(str(t))
                elif isinstance(part, str) and part.strip():
                    reasoning_parts.append(part)
    elif itype in {"message", "output_message"}:
        for c in item.get("content") or []:
            if not isinstance(c, dict):
                continue
            if c.get("type") in ("output_text", "text"):
                t = (c.get("text") or "").strip()
                if t and not any(part.strip() for part in text_parts):
                    text_parts.append(t)
            if c.get("type") in ("output_image", "image"):
                _append_unique_item(image_items, c)


def _finalize_stream_result(
    *,
    text_parts: list[str],
    reasoning_parts: list[str],
    tool_calls: list[dict[str, Any]],
    image_items: list[dict[str, Any]],
    final: dict[str, Any],
) -> dict[str, Any]:
    """Merge streamed deltas with the completed payload (images often land only there)."""
    if isinstance(final, dict):
        for item in _output_items(final):
            _absorb_output_item(
                item,
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
                tool_calls=tool_calls,
                image_items=image_items,
            )
        # Some Codex payloads nest the real output under response.output.
        nested = final.get("response")
        if isinstance(nested, dict):
            for item in _output_items(nested):
                _absorb_output_item(
                    item,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                    tool_calls=tool_calls,
                    image_items=image_items,
                )

    reasoning_text = "".join(reasoning_parts).strip() or None
    usage = None
    if isinstance(final, dict):
        usage = final.get("usage")
        if usage is None and isinstance(final.get("response"), dict):
            usage = final["response"].get("usage")

    if text_parts or tool_calls or image_items or reasoning_text:
        out_items: list[dict[str, Any]] = []
        if reasoning_text:
            out_items.append(
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": reasoning_text}],
                }
            )
        if text_parts:
            out_items.append(
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "".join(text_parts)}
                    ],
                }
            )
        out_items.extend(tool_calls)
        out_items.extend(image_items)
        result: dict[str, Any] = {"output": out_items, "_reasoning": reasoning_text}
        if usage:
            result["usage"] = usage
        return result
    if final:
        return final
    return {"output": []}


def _consume_sse_stream(resp: httpx.Response) -> dict[str, Any]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    image_items: list[dict[str, Any]] = []
    final: dict[str, Any] = {}

    for line in resp.iter_lines():
        if not line:
            continue
        if line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue
        event_data = line[5:].strip()
        if event_data == "[DONE]":
            break
        try:
            payload = json.loads(event_data)
        except json.JSONDecodeError:
            continue
        etype = payload.get("type") or ""
        if etype == "response.output_text.delta":
            text_parts.append(payload.get("delta") or "")
        elif etype == "response.output_text.done":
            # Prefer the finalized text for this item (avoid double-counting deltas)
            text = payload.get("text")
            if text:
                text_parts = [text]
        elif etype in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            reasoning_parts.append(payload.get("delta") or "")
        elif etype in {
            "response.reasoning_summary_text.done",
            "response.reasoning_text.done",
        }:
            text = payload.get("text")
            if text:
                reasoning_parts = [text]
        elif etype == "response.output_item.done":
            item = payload.get("item") or {}
            if isinstance(item, dict):
                _absorb_output_item(
                    item,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                    tool_calls=tool_calls,
                    image_items=image_items,
                )
        elif etype == "response.image_generation_call.completed":
            _append_unique_item(image_items, _normalize_image_item(payload))
        elif etype == "response.completed":
            # Do not break: image bytes often arrive after / only inside this event.
            final = payload.get("response") or payload
            if isinstance(final, dict):
                for item in _output_items(final):
                    _absorb_output_item(
                        item,
                        text_parts=text_parts,
                        reasoning_parts=reasoning_parts,
                        tool_calls=tool_calls,
                        image_items=image_items,
                    )
        elif "response" in payload and isinstance(payload.get("response"), dict):
            final = payload["response"]

    return _finalize_stream_result(
        text_parts=text_parts,
        reasoning_parts=reasoning_parts,
        tool_calls=tool_calls,
        image_items=image_items,
        final=final,
    )


def _maybe_b64_image(val: Any) -> bytes | None:
    """Decode PNG/JPEG bytes from a string or nested result object."""
    import base64

    if val is None:
        return None
    if isinstance(val, dict):
        for key in ("b64_json", "image_base64", "image", "result", "data", "url"):
            blob = _maybe_b64_image(val.get(key))
            if blob:
                return blob
        return None
    if isinstance(val, list):
        for part in val:
            blob = _maybe_b64_image(part)
            if blob:
                return blob
        return None
    if not isinstance(val, str) or not val:
        return None
    raw = val.strip()
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        blob = base64.b64decode(raw, validate=False)
    except Exception:  # noqa: BLE001
        return None
    if blob.startswith(b"\x89PNG") or blob.startswith(b"\xff\xd8"):
        return blob
    # Codex sometimes returns raw (uncompressed) PNG without a long prefix check.
    if len(blob) >= 64 and (b"IHDR" in blob[:32] or blob[:4] == b"\x89PNG"):
        return blob
    return None


def _extract_images_from_response(data: dict[str, Any]) -> list[bytes]:
    """Pull PNG/JPEG bytes from Responses image_generation / output_image items."""
    images: list[bytes] = []
    seen: set[bytes] = set()

    def _add(blob: bytes | None) -> None:
        if not blob or blob in seen:
            return
        seen.add(blob)
        images.append(blob)

    items = list(_output_items(data))
    if isinstance(data, dict) and isinstance(data.get("response"), dict):
        items.extend(_output_items(data["response"]))

    for item in items:
        item = _normalize_image_item(item)
        itype = str(item.get("type") or "")
        if itype in {
            "image_generation_call",
            "image_generation",
            "output_image",
            "image",
        } or "image_generation" in itype:
            for key in ("result", "b64_json", "image_base64", "image"):
                _add(_maybe_b64_image(item.get(key)))
            for c in item.get("content") or []:
                if isinstance(c, dict):
                    _add(
                        _maybe_b64_image(
                            c.get("b64_json")
                            or c.get("image_base64")
                            or c.get("data")
                            or c.get("result")
                            or c.get("image")
                        )
                    )
        if itype in {"message", "output_message"}:
            for c in item.get("content") or []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") in {"output_image", "image", "image_generation_call"}:
                    url = c.get("image_url")
                    _add(
                        _maybe_b64_image(
                            c.get("b64_json")
                            or c.get("image_base64")
                            or (url.get("url") if isinstance(url, dict) else url)
                            or c.get("data")
                        )
                    )
    return images


def _turn_from_response(data: dict[str, Any]) -> AssistantTurn:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    if isinstance(data.get("_reasoning"), str) and data["_reasoning"].strip():
        reasoning_parts.append(data["_reasoning"])

    output = data.get("output") or data.get("choices") or []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "reasoning":
                summary = item.get("summary") or item.get("content") or []
                if isinstance(summary, str) and summary.strip():
                    reasoning_parts.append(summary)
                elif isinstance(summary, list):
                    for part in summary:
                        if isinstance(part, dict):
                            t = part.get("text") or part.get("summary") or ""
                            if t:
                                reasoning_parts.append(str(t))
                        elif isinstance(part, str) and part.strip():
                            reasoning_parts.append(part)
            elif itype in ("message", "output_message"):
                for c in item.get("content") or []:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        content_parts.append(c.get("text") or "")
            elif itype == "function_call":
                args_raw = item.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {"_raw": args_raw}
                tool_calls.append(
                    ToolCall(
                        id=item.get("call_id")
                        or item.get("id")
                        or f"call_{uuid.uuid4().hex[:8]}",
                        name=item.get("name") or "",
                        arguments=args if isinstance(args, dict) else {"value": args},
                    )
                )
            elif itype == "text":
                content_parts.append(item.get("text") or "")

    if not content_parts and not tool_calls and "choices" in data:
        msg = data["choices"][0].get("message") or {}
        content_parts.append(msg.get("content") or "")
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            tool_calls.append(
                ToolCall(id=tc.get("id"), name=fn.get("name"), arguments=args)
            )

    finish = "tool_calls" if tool_calls else "stop"
    reasoning = "\n".join(p for p in reasoning_parts if p).strip() or None
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    return AssistantTurn(
        content="".join(content_parts) or None,
        tool_calls=tool_calls,
        finish_reason=finish,
        raw=data,
        reasoning=reasoning,
        usage=usage,
    )


class ChatGPTOAuthProvider:
    name = "chatgpt_oauth"

    def __init__(
        self,
        auth_path: Path,
        model: str = "gpt-5.4",
        proxy: str | None = None,
        auto_detect_proxy: bool = True,
        reasoning_effort: str = "high",
    ):
        self.auth_path = auth_path
        self.model = model
        self.reasoning_effort = (reasoning_effort or "high").strip().lower()
        self._explicit_proxy = proxy
        self._auto_detect_proxy = auto_detect_proxy
        self._tokens = ensure_fresh_tokens(auth_path, proxy=self.proxy)
        self._timeout = httpx.Timeout(connect=15.0, read=180.0, write=60.0, pool=15.0)
        self.meter = None

    @property
    def proxy(self) -> str | None:
        return resolve_proxy(self._explicit_proxy, auto_detect=self._auto_detect_proxy)

    def set_meter(self, meter) -> None:
        self.meter = meter

    def _reasoning_body(self) -> dict[str, Any]:
        # Codex rejects summary=none; allowed: concise | detailed | auto.
        return {"effort": self.reasoning_effort, "summary": "detailed"}

    def _client(self) -> httpx.Client:
        # Decide proxy per request so Clash / Tailscale / TUN can change live.
        # trust_env=False: a stale HTTPS_PROXY from process start must not win.
        kwargs: dict[str, Any] = {"timeout": self._timeout, "trust_env": False}
        proxy = self.proxy
        if proxy:
            kwargs["proxy"] = proxy
        return httpx.Client(**kwargs)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._tokens.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=v1",
            "originator": "codex_cli_rs",
            "User-Agent": "codex_cli_rs",
        }
        if self._tokens.account_id:
            headers["ChatGPT-Account-ID"] = self._tokens.account_id
        return headers

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AssistantTurn:
        body: dict[str, Any] = {
            "model": self.model,
            "input": _messages_to_input(messages),
            "stream": True,
            "store": False,
            "reasoning": self._reasoning_body(),
        }
        resp_tools = _chat_tools_to_responses(tools)
        if resp_tools:
            body["tools"] = resp_tools
        data = self._post_responses(body)
        turn = _turn_from_response(data)
        if self.meter is not None:
            self.meter.record(turn=turn, data=data, messages=messages)
        return turn

    def generate_image(
        self,
        prompt: str,
        out_path: Path | str,
        *,
        reference_images: list[Path | str] | None = None,
        size: str = "1024x1024",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Same Codex OAuth model as chat: attach refs + request a generated image."""
        from harness.pipeline.multimodal import user_content

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        refs = [Path(p) for p in (reference_images or []) if Path(p).is_file()]
        text = (
            "Using the attached CAD reference views, generate ONE target render image. "
            "Return an image (not only text).\n"
            "The FIRST attached image is the PRIMARY BASE: same camera, same part identity. "
            "Apply only the described delta; do not restyle unmentioned features. "
            "Later images are supporting references only.\n"
            f"Size hint: {size}.\n\n{prompt}"
        )
        content = user_content(text, refs, detail="high")
        body: dict[str, Any] = {
            "model": self.model,
            "input": _messages_to_input([{"role": "user", "content": content}]),
            "stream": True,
            "store": False,
            "reasoning": self._reasoning_body(),
            "tools": [{"type": "image_generation"}],
        }
        user_msgs = [{"role": "user", "content": content}]
        images: list[bytes] = []
        last_exc: BaseException | None = None
        for attempt in range(IMAGE_GEN_ATTEMPTS):
            req = dict(body)
            if attempt == IMAGE_GEN_ATTEMPTS - 1:
                req.pop("tools", None)
            try:
                # Outer loop owns the 3 image-gen chances; do not stack HTTP retries.
                data = self._post_responses(req, max_attempts=1)
                if self.meter is not None:
                    self.meter.record(data=data, messages=user_msgs)
                images = _extract_images_from_response(data)
                if images:
                    break
                last_exc = RuntimeError(
                    "ChatGPT OAuth image generation returned no image bytes "
                    f"(model={self.model}). Stream completed without an extractable PNG/JPEG."
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                images = []
                if not _is_transient_http(exc):
                    raise
            if images:
                break
            if attempt < IMAGE_GEN_ATTEMPTS - 1:
                time.sleep(1.5 * (attempt + 1))
        if not images:
            raise RuntimeError(
                "ChatGPT OAuth image generation failed after "
                f"{IMAGE_GEN_ATTEMPTS} attempts (model={self.model}): {last_exc}"
            ) from last_exc
        out_path.write_bytes(images[0])
        return {
            "ok": True,
            "path": str(out_path.resolve()),
            "mode": "chatgpt_oauth",
            "model": self.model,
            "refs": len(refs),
        }

    def _post_responses(
        self, body: dict[str, Any], *, max_attempts: int = 3
    ) -> dict[str, Any]:
        def _once(force_refresh: bool = False) -> dict[str, Any]:
            if force_refresh:
                try:
                    self._tokens = refresh_chatgpt_tokens(self._tokens, proxy=self.proxy)
                except Exception as exc:
                    raise RuntimeError(f"ChatGPT token refresh failed: {exc}") from exc

            try:
                with self._client() as client:
                    with client.stream(
                        "POST",
                        CODEX_RESPONSES_URL,
                        headers=self._headers(),
                        json=body,
                    ) as resp:
                        if resp.status_code in (401, 403) and not force_refresh:
                            raise PermissionError(f"auth {resp.status_code}")
                        if resp.status_code >= 400:
                            err = resp.read().decode("utf-8", errors="replace")[:500]
                            raise RuntimeError(
                                f"ChatGPT OAuth request failed: {resp.status_code} {err}"
                            )
                        return _consume_sse_stream(resp)
            except httpx.ConnectTimeout as exc:
                hint = (
                    f" proxy={self.proxy}"
                    if self.proxy
                    else " (no HTTP proxy; traffic follows the OS default route, e.g. Tailscale)"
                )
                raise RuntimeError(
                    "Cannot reach chatgpt.com Codex backend (connect timeout)."
                    + hint
                ) from exc
            except httpx.TimeoutException as exc:
                raise RuntimeError(f"ChatGPT OAuth timed out: {exc}") from exc
            except ssl.SSLError as exc:
                raise RuntimeError(f"ChatGPT OAuth TLS error: {exc}") from exc

        last: BaseException | None = None
        attempts = max(1, int(max_attempts))
        for attempt in range(attempts):
            try:
                return _once(False)
            except PermissionError:
                return _once(True)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < attempts - 1 and _is_transient_http(exc):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        assert last is not None
        raise last
