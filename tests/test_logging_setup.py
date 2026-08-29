import json
import logging

from src.logging_setup import JsonFormatter


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
