"""Classify a raw Telegram message into a clip, a camera health event, or noise.

Camera-ID resolution is intentionally conservative: an unmatched caption becomes
the configured unknown_camera_id rather than a best-effort guess, because a wrong
guess silently corrupts the corpus while "unknown" is visible and fixable later
by adding an alias to config/cameras.yaml.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import CamerasConfig

# Order matters: checked in sequence, first match wins. Extend as real caption
# formats are observed during Phase 0a.
_HEALTH_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("battery_dead", ("battery low", "battery dead", "battery critical", "low battery")),
    ("power_out", ("power out", "power loss", "power failure", "offline")),
    ("back_online", ("back online", "power restored", "reconnected")),
)


@dataclass(frozen=True)
class ParsedMessage:
    kind: str  # "clip" | "system_event" | "ignored"
    camera_id: str | None = None
    event_type: str | None = None


def parse_message(*, text: str | None, has_media: bool, cameras: CamerasConfig) -> ParsedMessage:
    caption = text or ""

    if has_media:
        return ParsedMessage(kind="clip", camera_id=_resolve_camera(caption, cameras))

    event_type = _match_health_event(caption)
    if event_type is not None:
        return ParsedMessage(
            kind="system_event", camera_id=_resolve_camera(caption, cameras), event_type=event_type
        )

    return ParsedMessage(kind="ignored")


def _resolve_camera(caption: str, cameras: CamerasConfig) -> str:
    match = cameras.resolve_alias(caption)
    if match is not None:
        return match.id

    lowered = caption.lower()
    for camera in cameras.cameras:
        for alias in (camera.id, *camera.aliases):
            # Word boundaries prevent e.g. alias "camera 1" matching inside "camera 10".
            if re.search(rf"\b{re.escape(alias.lower())}\b", lowered):
                return camera.id
    return cameras.unknown_camera_id


def _match_health_event(caption: str) -> str | None:
    lowered = caption.lower()
    for event_type, keywords in _HEALTH_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return event_type
    return None
