"""Send a push alert via ntfy (https://ntfy.sh or a self-hosted instance).

Pure, injectable-transport function: pass a fake/mock `session` in tests to
avoid any real network access, or omit it to use `requests` directly.
"""

from __future__ import annotations

from typing import Protocol

from src.errors import AlertError


class _PostsRequests(Protocol):
    def post(self, url: str, *, data: bytes, headers: dict[str, str], timeout: float) -> object: ...


def send_ntfy_alert(
    message: str,
    *,
    base_url: str,
    topic: str,
    priority: str = "urgent",
    title: str | None = None,
    token: str | None = None,
    session: _PostsRequests | None = None,
) -> None:
    """POST `message` to the configured ntfy topic. Raises AlertError on failure."""
    import requests

    url = f"{base_url.rstrip('/')}/{topic}"
    headers = {"Priority": priority}
    if title:
        headers["Title"] = title
    if token:
        headers["Authorization"] = f"Bearer {token}"

    client = session if session is not None else requests
    try:
        response = client.post(url, data=message.encode("utf-8"), headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise AlertError(f"ntfy alert failed: {exc}") from exc
