"""Stable grouping for Telegram camera messages belonging to one trigger."""

from __future__ import annotations

import re

EVENT_TIMESTAMP_RE = re.compile(r"@\s*(\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def event_key(camera_id: str, caption: str | None) -> str | None:
    """Return the shared key in Initial/Stopped captions, when present."""
    if not caption:
        return None
    match = EVENT_TIMESTAMP_RE.search(caption)
    return f"{camera_id}|{match.group(1)}" if match else None


def message_event_key(camera_id: str, caption: str | None, message_id: int) -> str:
    """Return a durable event key, falling back to a message-local event."""
    return event_key(camera_id, caption) or f"{camera_id}|message:{message_id}"


def event_phase(caption: str | None) -> str:
    """Classify the camera caption's lifecycle marker."""
    lowered = (caption or "").lower()
    if "initial" in lowered:
        return "initial"
    if "stopped" in lowered or "timeout" in lowered:
        return "complete"
    return "standalone"
