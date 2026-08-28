from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.infer_camera_order import format_report, load_events
from src import db
from src.sequence import infer_camera_order


def _seed(conn, camera_id: str, message_id: int, timestamp: str) -> None:
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=message_id,
        camera_id=camera_id,
        timestamp=timestamp,
        caption=None,
        file_path=None,
        source="backfill",
    )


def test_load_events_excludes_unknown_by_default(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn, "cam_a", 1, "2024-01-01T00:00:00.000000Z")
    _seed(conn, "unknown", 2, "2024-01-01T00:00:05.000000Z")

    events = load_events(conn, exclude_camera_ids=frozenset({"unknown"}))

    assert [e.camera_id for e in events] == ["cam_a"]
    conn.close()


def test_load_events_parses_timestamps_to_epoch_seconds(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn, "cam_a", 1, "2024-01-01T00:00:10.000000Z")

    events = load_events(conn, exclude_camera_ids=frozenset())

    assert events[0].timestamp > 0
    conn.close()


def test_format_report_includes_order_and_metrics(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    order = ["camA", "camB", "camC"]
    base = datetime(2024, 1, 1, tzinfo=UTC)
    message_id = 1
    for _ in range(5):
        walk = order + list(reversed(order[:-1]))
        for cam in walk:
            timestamp = base.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            _seed(conn, cam, message_id, timestamp)
            message_id += 1
            base += timedelta(seconds=30)
        base += timedelta(hours=8)
    events = load_events(conn, exclude_camera_ids=frozenset())
    result = infer_camera_order(events)

    report = format_report(result, stability=0.9)

    assert "->".join(result.order) in report.replace(" ", "")
    assert "Split-half stability: 0.90" in report
    conn.close()
