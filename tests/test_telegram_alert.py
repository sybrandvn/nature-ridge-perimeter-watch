from unittest.mock import AsyncMock

import pytest

from src.errors import AlertError
from src.telegram_alert import send_telegram_alert


async def test_send_telegram_alert_calls_bot_with_chat_and_text():
    bot = AsyncMock()
    await send_telegram_alert(
        "fence breach on cam01b", bot_token="unused", chat_id="-1001234", bot=bot
    )
    bot.send_message.assert_awaited_once_with(chat_id="-1001234", text="fence breach on cam01b")


async def test_send_telegram_alert_wraps_telegram_error():
    from telegram.error import TelegramError

    bot = AsyncMock()
    bot.send_message.side_effect = TelegramError("chat not found")

    with pytest.raises(AlertError, match="chat not found"):
        await send_telegram_alert("hi", bot_token="unused", chat_id="bad", bot=bot)
