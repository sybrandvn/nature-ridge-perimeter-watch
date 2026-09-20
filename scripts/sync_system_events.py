"""Backfill recognized non-video system events without touching clip rows."""

from __future__ import annotations

import asyncio
import json

from src import db
from src.config import load_app_config, load_cameras_config, resolve_channel_ref
from src.media_download import is_video_message
from src.message_parsing import parse_message


async def run() -> dict[str, int]:  # pragma: no cover - live Telegram CLI
    from telethon import TelegramClient

    config = load_app_config()
    cameras = load_cameras_config("config/cameras.yaml")
    conn = db.connect(config.db_path)
    counts = {"recognized": 0, "ignored": 0}
    client = TelegramClient(
        str(config.telegram_session_path), config.telegram_api_id, config.telegram_api_hash
    )
    try:
        async with client:
            channel = resolve_channel_ref(str(config.source_channel))
            async for message in client.iter_messages(channel, reverse=True):
                if is_video_message(message):
                    continue
                text = str(getattr(message, "message", None) or "").strip()
                if not text:
                    continue
                parsed = parse_message(text=text, has_media=False, cameras=cameras)
                if parsed.kind != "system_event":
                    counts["ignored"] += 1
                    continue
                db.upsert_system_event(
                    conn,
                    channel_id=str(config.source_channel),
                    message_id=int(message.id),
                    timestamp=message.date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    camera_id=parsed.camera_id,
                    event_type=str(parsed.event_type),
                    raw_text=text,
                )
                counts["recognized"] += 1
                if counts["recognized"] % 500 == 0:
                    conn.commit()
        conn.commit()
        return counts
    finally:
        conn.close()


def main() -> None:  # pragma: no cover
    print(json.dumps(asyncio.run(run()), sort_keys=True))


if __name__ == "__main__":
    main()
