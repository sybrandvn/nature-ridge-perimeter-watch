"""Structured JSON logging, shared by every script's CLI entrypoint.

Replaces the old per-call-site `logger.info(json.dumps({...}))` pattern with a
formatter: call sites just log a short event name plus `extra={...}` fields, and
every line on stdout is a single JSON object.
"""

from __future__ import annotations

import json
import logging

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


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
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Point the root logger at a single stdout handler emitting JSON lines."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
