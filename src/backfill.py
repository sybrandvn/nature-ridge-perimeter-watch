"""Metadata backfill: consume channel messages and persist clips/system_events.

Promoted from scripts/meta_backfill.py (Phase 1 step 19) now that the approach is
validated by the Phase 0 spike. Pure/async-generic and fully unit-testable without a
live Telegram connection: it only needs an async iterator of RawMessage-like objects.
Real Telethon wiring (client construction, credentials, CLI) stays in
scripts/meta_backfill.py, which is now a thin wrapper around this module.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from src import db
from src.config import CamerasConfig
from src.message_parsing import parse_message

logger = logging.getLogger("backfill")


class RawMessage(Protocol):
    id: int
    date: Any  # tz-aware datetime
    message: str | None
    video: Any
    document: Any


def extract_primitives(msg: RawMessage) -> tuple[int, str, str | None, bool]:
    """Return (message_id, iso_timestamp, caption_text, has_media)."""
    has_media = bool(getattr(msg, "video", None) or getattr(msg, "document", None))
    timestamp = msg.date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return msg.id, timestamp, getattr(msg, "message", None), has_media


async def run_backfill(
    messages: Any,
    *,
    channel_id: str,
    cameras: CamerasConfig,
    conn: Any,
) -> dict[str, int]:
    """Consume an async iterator of RawMessage-like objects and write to the db.

    Idempotent: re-running over the same channel history just re-upserts the
    same rows, so partial/interrupted backfills are safe to resume from scratch.
    """
    counts = {"clip": 0, "system_event": 0, "ignored": 0}
    async for msg in messages:
        message_id, timestamp, text, has_media = extract_primitives(msg)
        parsed = parse_message(text=text, has_media=has_media, cameras=cameras)

        if parsed.kind == "clip":
            db.upsert_clip(
                conn,
                channel_id=channel_id,
                message_id=message_id,
                camera_id=parsed.camera_id,
                timestamp=timestamp,
                caption=text,
                file_path=None,
                source="backfill",
            )
        elif parsed.kind == "system_event":
            db.upsert_system_event(
                conn,
                channel_id=channel_id,
                message_id=message_id,
                timestamp=timestamp,
                camera_id=parsed.camera_id,
                event_type=parsed.event_type,
                raw_text=text,
            )

        counts[parsed.kind] += 1
        logger.info(
            "backfill_message",
            extra={"message_id": message_id, "kind": parsed.kind, "camera_id": parsed.camera_id},
        )

    conn.commit()
    return counts
