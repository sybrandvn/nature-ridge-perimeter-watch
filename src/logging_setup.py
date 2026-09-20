"""Structured JSON logging, shared by every script's CLI entrypoint.

Replaces the old per-call-site `logger.info(json.dumps({...}))` pattern with a
formatter: call sites just log a short event name plus `extra={...}` fields, and
every line on stdout is a single JSON object.
"""

from __future__ import annotations

import json
import logging
import re

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}
_TELEGRAM_TOKEN_RE = re.compile(r"(?i)bot\d{6,}:[a-z0-9_-]{20,}")


def _redact(value: object) -> object:
    if isinstance(value, str):
        return _TELEGRAM_TOKEN_RE.sub("bot<redacted>", value)
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        return json.dumps(_redact(payload), default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Point the root logger at a single stdout handler emitting JSON lines."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # HTTPX logs full request URLs. Telegram Bot API URLs contain the bot token,
    # so INFO request logging is both noisy and credential disclosure.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
