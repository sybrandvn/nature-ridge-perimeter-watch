"""Phase 0a: capture full channel history as metadata only (no video downloads).

Populates `clips` (file_path=None, source="backfill") and `system_events` in the
sqlite db so camera-order inference, roster building, and later full backfill all
have real data to work from before a single video is downloaded.

The message-shape parsing (`src.message_parsing`) and the per-message DB-write
logic live in `src.backfill` (promoted from this script, Phase 1 step 19) and are
fully unit-tested without a live Telegram connection. Only `main()` here touches
the real Telethon client, and it is not covered by tests since it requires real
credentials.

Run:
    uv run python scripts/meta_backfill.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db  # noqa: E402
from src.backfill import run_backfill  # noqa: E402
from src.config import load_app_config, load_cameras_config, resolve_channel_ref  # noqa: E402
from src.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger("meta_backfill")


async def main() -> None:  # pragma: no cover - requires real Telegram credentials
    from telethon import TelegramClient

    configure_logging()
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
    logger.info("backfill_summary", extra={"summary": counts})


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
