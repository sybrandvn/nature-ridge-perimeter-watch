from unittest.mock import MagicMock

import pytest
import requests

from src.errors import AlertError
from src.ntfy_alert import send_ntfy_alert


def _session(response: MagicMock) -> MagicMock:
    session = MagicMock()
    session.post.return_value = response
    return session


def test_send_ntfy_alert_posts_to_topic_url_with_headers():
    response = MagicMock()
    session = _session(response)

    send_ntfy_alert(
        "fence breach on cam01b",
        base_url="https://ntfy.sh/",
        topic="nature-ridge",
        priority="urgent",
        title="Perimeter Watch",
        token="secret-token",
        session=session,
    )

    session.post.assert_called_once_with(
        "https://ntfy.sh/nature-ridge",
        data=b"fence breach on cam01b",
        headers={
            "Priority": "urgent",
            "Title": "Perimeter Watch",
            "Authorization": "Bearer secret-token",
        },
        timeout=10,
    )
    response.raise_for_status.assert_called_once()


def test_send_ntfy_alert_omits_optional_headers():
    response = MagicMock()
    session = _session(response)

    send_ntfy_alert("hi", base_url="https://ntfy.sh", topic="t", session=session)

    _, kwargs = session.post.call_args
    assert kwargs["headers"] == {"Priority": "urgent"}


def test_send_ntfy_alert_wraps_request_exception():
    session = MagicMock()
    session.post.side_effect = requests.ConnectionError("refused")

    with pytest.raises(AlertError, match="refused"):
        send_ntfy_alert("hi", base_url="https://ntfy.sh", topic="t", session=session)


def test_send_ntfy_alert_wraps_http_error_status():
    response = MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError("500 server error")
    session = _session(response)

    with pytest.raises(AlertError, match="500 server error"):
        send_ntfy_alert("hi", base_url="https://ntfy.sh", topic="t", session=session)
