"""Inventory recognized and ignored non-media source-message shapes."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter

from src.config import load_app_config, load_cameras_config, resolve_channel_ref
from src.media_download import is_video_message
from src.message_parsing import parse_message


def normalized_shape(text: str) -> str:
    value = re.sub(r"\s*@\s*\d{2}-\d{2}-\d{2,4}.*$", "", text.strip())
    value = re.sub(r"\[\d+\]", "[device]", value)
    value = re.sub(r"\s+", " ", value)
    return value[:240]


async def run(*, limit: int) -> dict[str, object]:  # pragma: no cover - live Telegram CLI
    from telethon import TelegramClient

    config = load_app_config()
    cameras = load_cameras_config("config/cameras.yaml")
    client = TelegramClient(
        str(config.telegram_session_path), config.telegram_api_id, config.telegram_api_hash
    )
    recognized: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    async with client:
        channel = resolve_channel_ref(str(config.source_channel))
        async for message in client.iter_messages(channel):
            if is_video_message(message):
                continue
            text = str(getattr(message, "message", None) or "").strip()
            if not text:
                continue
            parsed = parse_message(text=text, has_media=False, cameras=cameras)
            target = recognized if parsed.kind == "system_event" else ignored
            target[normalized_shape(text)] += 1
    return {
        "recognized": recognized.most_common(limit),
        "ignored": ignored.most_common(limit),
        "recognized_total": sum(recognized.values()),
        "ignored_total": sum(ignored.values()),
    }


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(limit=max(1, args.limit))), indent=2))


if __name__ == "__main__":
    main()
