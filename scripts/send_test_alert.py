"""Manual alert send-test (Phase 4): verify Telegram bot + ntfy credentials work.

Sends exactly one real Telegram message and one real ntfy notification using
the configured .env values, so you can confirm TELEGRAM_BOT_TOKEN,
ALERT_CHANNEL_ID, NTFY_BASE_URL/NTFY_TOPIC, and NTFY_TOKEN are all correct
before wiring alerts into anything automated.

Run:
    uv run python scripts/send_test_alert.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_app_config  # noqa: E402
from src.errors import AlertError  # noqa: E402
from src.ntfy_alert import send_ntfy_alert  # noqa: E402
from src.telegram_alert import send_telegram_alert  # noqa: E402

TEST_MESSAGE = "Nature Ridge Perimeter Watch: this is a manual alert test."


async def main() -> None:  # pragma: no cover - requires real credentials/network
    cfg = load_app_config(require_telegram=False)

    if cfg.telegram_bot_token and cfg.alert_channel_id:
        try:
            await send_telegram_alert(
                TEST_MESSAGE,
                bot_token=cfg.telegram_bot_token,
                chat_id=cfg.alert_channel_id,
            )
            print("Telegram: sent OK")
        except AlertError as exc:
            print(f"Telegram: FAILED -- {exc}")
    else:
        print("Telegram: skipped (TELEGRAM_BOT_TOKEN / ALERT_CHANNEL_ID not set)")

    if cfg.ntfy_base_url and cfg.ntfy_topic:
        try:
            send_ntfy_alert(
                TEST_MESSAGE,
                base_url=cfg.ntfy_base_url,
                topic=cfg.ntfy_topic,
                priority=cfg.ntfy_priority,
                title="Perimeter Watch test",
                token=cfg.ntfy_token,
            )
            print("ntfy: sent OK")
        except AlertError as exc:
            print(f"ntfy: FAILED -- {exc}")
    else:
        print("ntfy: skipped (NTFY_BASE_URL / NTFY_TOPIC not set)")


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
