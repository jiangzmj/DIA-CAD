"""Load / refresh ChatGPT OAuth tokens from Codex auth.json."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# Codex CLI public OAuth client id (used by official Codex login)
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
TOKEN_URL = "https://auth.openai.com/oauth/token"


@dataclass
class ChatGPTTokens:
    access_token: str
    refresh_token: str | None
    account_id: str | None
    id_token: str | None = None
    path: Path | None = None

    @property
    def ok(self) -> bool:
        return bool(self.access_token)


def _account_from_jwt(access_token: str) -> str | None:
    """Best-effort parse of chatgpt_account_id from JWT payload (no verify)."""
    try:
        import base64

        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        pad = "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + pad))
        auth = payload.get("https://api.openai.com/auth") or {}
        return auth.get("chatgpt_account_id")
    except Exception:
        return None


def load_chatgpt_tokens(path: Path) -> ChatGPTTokens:
    if not path.exists():
        raise FileNotFoundError(
            f"ChatGPT auth file not found: {path}. "
            "Run `codex login` or set CHATGPT_AUTH_PATH."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens = data.get("tokens") or {}
    access = tokens.get("access_token") or data.get("access_token") or ""
    refresh = tokens.get("refresh_token") or data.get("refresh_token")
    account = (
        tokens.get("account_id")
        or data.get("account_id")
        or _account_from_jwt(access)
    )
    return ChatGPTTokens(
        access_token=access,
        refresh_token=refresh,
        account_id=account,
        id_token=tokens.get("id_token") or data.get("id_token"),
        path=path,
    )


def refresh_chatgpt_tokens(
    tokens: ChatGPTTokens,
    proxy: str | None = None,
) -> ChatGPTTokens:
    if not tokens.refresh_token:
        return tokens

    from harness.net import resolve_proxy

    proxy_url = resolve_proxy(proxy, auto_detect=True)
    client_kwargs: dict[str, Any] = {"timeout": 30.0, "trust_env": False}
    if proxy_url:
        client_kwargs["proxy"] = proxy_url

    with httpx.Client(**client_kwargs) as client:
        resp = client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens.refresh_token,
                "client_id": CODEX_CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code >= 400:
            # Keep existing access token; caller may still succeed until expiry.
            raise RuntimeError(
                f"token refresh HTTP {resp.status_code}: {resp.text[:200]}"
            )
        payload = resp.json()

    access = payload.get("access_token") or tokens.access_token
    refresh = payload.get("refresh_token") or tokens.refresh_token
    account = tokens.account_id or _account_from_jwt(access)
    updated = ChatGPTTokens(
        access_token=access,
        refresh_token=refresh,
        account_id=account,
        id_token=payload.get("id_token") or tokens.id_token,
        path=tokens.path,
    )

    if tokens.path and tokens.path.exists():
        try:
            data = json.loads(tokens.path.read_text(encoding="utf-8"))
            bucket = data.setdefault("tokens", {})
            bucket["access_token"] = updated.access_token
            if updated.refresh_token:
                bucket["refresh_token"] = updated.refresh_token
            if updated.id_token:
                bucket["id_token"] = updated.id_token
            if updated.account_id:
                bucket["account_id"] = updated.account_id
            data["last_refresh"] = time.time()
            tokens.path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    return updated


def ensure_fresh_tokens(
    path: Path,
    force_refresh: bool = False,
    proxy: str | None = None,
) -> ChatGPTTokens:
    tokens = load_chatgpt_tokens(path)
    if not tokens.ok:
        raise RuntimeError(f"No access_token in {path}")
    if force_refresh or not tokens.access_token:
        tokens = refresh_chatgpt_tokens(tokens, proxy=proxy)
    return tokens
