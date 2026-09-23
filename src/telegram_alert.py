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
    debug_message_id: int | None = None,
    bot_username: str | None = None,
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
            reply_markup = None
            if debug_message_id is not None and bot_username:
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup

                reply_markup = InlineKeyboardMarkup(
                    [[InlineKeyboardButton(
                        "Debug view",
                        url=(
                            f"https://t.me/{bot_username.removeprefix('@')}"
                            f"?start=debug_{debug_message_id}"
                        ),
                    )]]
                )
            with Path(video_path).open("rb") as video:
                await client.send_video(
                    chat_id=chat_id,
                    video=video,
                    caption=message,
                    reply_markup=reply_markup,
                )
    except (TelegramError, OSError) as exc:
        raise AlertError(f"Telegram alert failed: {exc}") from exc
