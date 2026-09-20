"""Manual alert send-test: verify transports or send representative video examples.

Sends exactly one real Telegram message and one real ntfy notification using
the configured .env values, so you can confirm TELEGRAM_BOT_TOKEN,
ALERT_CHANNEL_ID, NTFY_BASE_URL/NTFY_TOPIC, and NTFY_TOKEN are all correct
before wiring alerts into anything automated.

Run:
    uv run python scripts/send_test_alert.py
    uv run python scripts/send_test_alert.py --examples
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db  # noqa: E402
from src.config import load_app_config  # noqa: E402
from src.errors import AlertError  # noqa: E402
from src.ntfy_alert import send_ntfy_alert  # noqa: E402
from src.telegram_alert import send_telegram_alert  # noqa: E402

TEST_MESSAGE = "Nature Ridge Perimeter Watch: this is a manual alert test."
EXAMPLE_ALERTS = (
    (7360, "animal_candidate", "outside_colour"),
    (21520, "incident_candidate", "outside_no_colour"),
)


async def _send_examples(cfg) -> None:
    if not cfg.telegram_bot_token or not cfg.alert_channel_id:
        print("Telegram examples: skipped (TELEGRAM_BOT_TOKEN / ALERT_CHANNEL_ID not set)")
        return
    with db.connect(cfg.db_path) as conn:
        for message_id, category, reason in EXAMPLE_ALERTS:
            row = conn.execute(
                """
                SELECT camera_id, timestamp, file_path FROM clips
                WHERE message_id = ? AND file_path IS NOT NULL
                """,
                (message_id,),
            ).fetchone()
            if row is None or not Path(str(row["file_path"])).is_file():
                print(f"Telegram example {message_id}: skipped (video unavailable)")
                continue
            message = (
                "🧪 SMOKE TEST — no action required\n"
                f"Perimeter alert: {category}\n"
                f"Camera: {row['camera_id']}\n"
                f"Time: {row['timestamp']}\n"
                f"Reason: {reason}\n"
                f"Event: smoke-test:{row['camera_id']}:{message_id}"
            )
            await send_telegram_alert(
                message,
                bot_token=cfg.telegram_bot_token,
                chat_id=cfg.alert_channel_id,
                video_path=str(row["file_path"]),
                debug_message_id=message_id,
            )
            print(f"Telegram example {message_id}: sent OK")


async def main(argv: list[str] | None = None) -> None:  # pragma: no cover - real credentials
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        action="store_true",
        help="send labelled animal and incident videos to the Telegram alert destination",
    )
    args = parser.parse_args(argv)
    cfg = load_app_config(require_telegram=False)

    if args.examples:
        await _send_examples(cfg)
        return

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
