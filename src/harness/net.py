"""Resolve HTTP(S) proxy from the *current* machine network.

Not pinned at process start. Each call re-reads env, Clash ports, and the
default route so Clash vs Tailscale vs TUN vs direct can change live.

Priority:
1. Explicit argument / config.yaml ``network.proxy``
2. Live HTTPS_PROXY / HTTP_PROXY if that proxy is still reachable
3. Clash mixed-port (7890/…) only if it is listening *and* a TUN (Tailscale /
   Clash TUN) is not already the default route
4. None — follow the OS default route (Tailscale, Clash TUN, or NIC)
"""

from __future__ import annotations

import os
import socket
import subprocess
from urllib.parse import urlparse

CLASH_HTTP_PORTS = (7890, 7897, 10809, 20171)
_TUN_DEV_PREFIXES = (
    "tailscale",
    "tun",
    "utun",
    "wg",
    "meta",
    "clash",
    "mihomo",
    "singbox",
    "wintun",
)


def _port_open(host: str, port: int, timeout: float = 0.15) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _proxy_reachable(url: str) -> bool:
    """Localhost proxies must still be listening; remote URLs are assumed live."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = parsed.hostname or ""
    port = parsed.port
    if not host or port is None:
        return False
    if host in {"127.0.0.1", "localhost", "::1"}:
        return _port_open("127.0.0.1" if host == "::1" else host, port)
    return True


def detect_local_proxy() -> str | None:
    """Clash / mihomo mixed HTTP port, if currently listening. Not cached."""
    for host, port in (("127.0.0.1", p) for p in CLASH_HTTP_PORTS):
        if _port_open(host, port):
            return f"http://{host}:{port}"
    return None


def default_route_device() -> str | None:
    """Linux ``ip route get 1.1.1.1`` device, or None if unknown."""
    try:
        proc = subprocess.run(
            ["ip", "-4", "route", "get", "1.1.1.1"],
            capture_output=True,
            text=True,
            timeout=0.4,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    parts = (proc.stdout or "").split()
    if "dev" not in parts:
        return None
    idx = parts.index("dev")
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def tun_owns_default_route(device: str | None = None) -> bool:
    """True when Tailscale / Clash-TUN / wireguard already captures the default route."""
    dev = (device if device is not None else default_route_device()) or ""
    name = dev.lower()
    return any(name == p or name.startswith(p) for p in _TUN_DEV_PREFIXES)


def _env_proxy() -> str | None:
    for key in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        value = os.environ.get(key)
        if value:
            return value
    return None


def resolve_proxy(explicit: str | None = None, *, auto_detect: bool = True) -> str | None:
    """Pick a proxy for *this* request. Safe to call on every HTTP call."""
    if explicit:
        return explicit if _proxy_reachable(explicit) else None

    if tun_owns_default_route():
        # Tailscale exit / Clash TUN already owns the packet path. Do not also
        # pin httpx to a leftover HTTP proxy from an earlier Clash session.
        return None

    env = _env_proxy()
    if env and _proxy_reachable(env):
        return env

    if auto_detect:
        return detect_local_proxy()
    return None


def httpx_proxy_mounts(proxy_url: str | None) -> dict[str, str] | None:
    """httpx Client(proxy=...) accepts a single URL; keep helper for clarity."""
    return None if not proxy_url else {"all://": proxy_url}
