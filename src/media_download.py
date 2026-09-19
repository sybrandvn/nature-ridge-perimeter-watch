"""Atomic Telegram media downloads shared by runtime callers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def is_video_message(message: Any) -> bool:
    if getattr(message, "video", None):
        return True
    document = getattr(message, "document", None)
    mime_type = getattr(document, "mime_type", "") if document is not None else ""
    return bool(mime_type and mime_type.lower().startswith("video/"))


async def download_media_atomic(client: Any, message: Any, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        result = await client.download_media(message, file=str(temporary))
        if result is None or not temporary.is_file():
            raise RuntimeError("Telegram returned no downloaded media")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
