"""List Telegram channels/groups you're a member of, to find SOURCE_CHANNEL.

Run this once your TELEGRAM_API_ID/HASH are set, before SOURCE_CHANNEL itself
is known. It prints each channel/group's numeric id, @username (if any), and
title, so you can pick the right one and paste it into .env.

Run:
    uv run python scripts/list_channels.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_app_config  # noqa: E402


def should_list_dialog(dialog: Any) -> bool:
    """Only channels/groups can be a SOURCE_CHANNEL, not individual users."""
    return bool(getattr(dialog, "is_channel", False) or getattr(dialog, "is_group", False))


def format_dialog_row(dialog: Any) -> str:
    username = getattr(dialog.entity, "username", None)
    handle = f"@{username}" if username else "-"
    return f"{dialog.id:<16} {handle:<25} {dialog.name}"


async def main() -> None:  # pragma: no cover - requires real Telegram credentials
    from telethon import TelegramClient

    app_cfg = load_app_config(require_telegram=False)
    if app_cfg.telegram_api_id is None or app_cfg.telegram_api_hash is None:
        raise SystemExit("Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env first.")

    client = TelegramClient(
        str(app_cfg.telegram_session_path), app_cfg.telegram_api_id, app_cfg.telegram_api_hash
    )
    async with client:
        print(f"{'id':<16} {'username':<25} title")
        print("-" * 70)
        async for dialog in client.iter_dialogs():
            if should_list_dialog(dialog):
                print(format_dialog_row(dialog))


if __name__ == "__main__":
    asyncio.run(main())
