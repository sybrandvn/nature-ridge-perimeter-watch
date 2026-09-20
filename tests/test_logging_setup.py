import json
import logging

from src.logging_setup import JsonFormatter, configure_logging


def test_json_formatter_includes_message_and_extra_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="something_happened",
        args=(),
        exc_info=None,
    )
    record.camera_id = "cam01"
    record.message_id = 42

    payload = json.loads(formatter.format(record))

    assert payload == {
        "level": "INFO",
        "logger": "test_logger",
        "message": "something_happened",
        "camera_id": "cam01",
        "message_id": 42,
    }


def test_json_formatter_redacts_telegram_tokens_from_messages_and_extras():
    token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    record = logging.LogRecord(
        "httpx",
        logging.INFO,
        "",
        0,
        f"POST https://api.telegram.org/bot{token}/getMe",
        (),
        None,
    )
    record.request_url = f"https://api.telegram.org/bot{token}/sendVideo"
    rendered = JsonFormatter().format(record)
    payload = json.loads(rendered)
    assert token not in rendered
    assert "bot<redacted>" in payload["message"]
    assert "bot<redacted>" in payload["request_url"]


def test_configure_logging_suppresses_http_request_info_logs():
    configure_logging()
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING
