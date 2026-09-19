"""Send a Telegram alert message via the Bot API.

Pure, injectable-transport function: pass a fake/mock `bot` in tests to avoid
any real network access, or omit it to construct a real `telegram.Bot`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from src.errors import AlertError


class _SendsMessages(Protocol):
    async def send_message(self, chat_id: int | str, text: str) -> object: ...

    async def send_video(
        self, chat_id: int | str, video: Any, caption: str
    ) -> object: ...


async def send_telegram_alert(
    message: str,
    *,
    bot_token: str,
    chat_id: str,
    bot: _SendsMessages | None = None,
    video_path: str | Path | None = None,
) -> None:
    """Send `message` to `chat_id` using `bot_token`. Raises AlertError on failure."""
    from telegram.error import TelegramError

    client = bot
    if client is None:
        from telegram import Bot

        client = Bot(token=bot_token)

    try:
        if video_path is None:
            await client.send_message(chat_id=chat_id, text=message)
        else:
            with Path(video_path).open("rb") as video:
                await client.send_video(chat_id=chat_id, video=video, caption=message)
    except (TelegramError, OSError) as exc:
        raise AlertError(f"Telegram alert failed: {exc}") from exc
