"""Phase 0a: capture full channel history as metadata only (no video downloads).

Populates `clips` (file_path=None, source="backfill") and `system_events` in the
sqlite db so camera-order inference, roster building, and later full backfill all
have real data to work from before a single video is downloaded.

The message-shape parsing (`src.message_parsing`) and the per-message DB-write
logic (`run_backfill` below) are pure/async-generic and fully unit-testable
without a live Telegram connection. Only `main()` touches the real Telethon
client, and it is not covered by tests since it requires real credentials.

Run:
    uv run python scripts/meta_backfill.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db  # noqa: E402
from src.config import (  # noqa: E402
    CamerasConfig,
    load_app_config,
    load_cameras_config,
    resolve_channel_ref,
)
from src.message_parsing import parse_message  # noqa: E402

logger = logging.getLogger("meta_backfill")


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
            json.dumps(
                {"message_id": message_id, "kind": parsed.kind, "camera_id": parsed.camera_id}
            )
        )

    conn.commit()
    return counts


async def main() -> None:  # pragma: no cover - requires real Telegram credentials
    from telethon import TelegramClient

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app_cfg = load_app_config()
    cameras_cfg = load_cameras_config("config/cameras.yaml")
    conn = db.connect(app_cfg.db_path)

    client = TelegramClient(
        str(app_cfg.telegram_session_path), app_cfg.telegram_api_id, app_cfg.telegram_api_hash
    )
    async with client:
        source_channel = resolve_channel_ref(app_cfg.source_channel)
        counts = await run_backfill(
            client.iter_messages(source_channel, reverse=True),
            channel_id=str(app_cfg.source_channel),
            cameras=cameras_cfg,
            conn=conn,
        )
    conn.close()
    logger.info(json.dumps({"summary": counts}))


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
