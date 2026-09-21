from datetime import UTC, datetime

from src import db
from src.activity_reports import build_activity_report, render_activity_report


def _clip(conn, *, message_id, camera_id, timestamp, caption="motion"):
    db.upsert_clip(
        conn,
        channel_id="source",
        message_id=message_id,
        camera_id=camera_id,
        timestamp=timestamp,
        caption=caption,
        file_path=None,
        source="live",
    )


def test_build_activity_report_counts_one_event_per_sibling_pair(tmp_path):
    conn = db.connect(tmp_path / "reports.db")
    # Initial/Stopped pair sharing the same embedded alert timestamp -> one event.
    _clip(
        conn,
        message_id=1,
        camera_id="cam01",
        timestamp="2026-09-05T18:00:00Z",
        caption="Motion (Initial*) @ 05-09-26 20:00:00",
    )
    _clip(
        conn,
        message_id=2,
        camera_id="cam01",
        timestamp="2026-09-05T18:00:05Z",
        caption="Motion (Stopped*) @ 05-09-26 20:00:00",
    )
    _clip(conn, message_id=3, camera_id="cam02", timestamp="2026-09-10T04:30:00Z")

    report = build_activity_report(conn, "month", datetime(2026, 9, 21, 10, 0, tzinfo=UTC))
    assert report.event_count == 2
    assert report.active_days == 2
    assert sum(report.hourly_counts) == 2
    assert report.timeline_labels[0] == "1"
    assert report.timeline_counts[4] == 1  # 5 Sep, SAST = day 5
    assert report.timeline_counts[9] == 1  # 10 Sep, SAST = day 10


def test_build_activity_report_excludes_clips_outside_the_period(tmp_path):
    conn = db.connect(tmp_path / "reports.db")
    _clip(conn, message_id=1, camera_id="cam01", timestamp="2026-08-31T21:00:00Z")
    _clip(conn, message_id=2, camera_id="cam01", timestamp="2026-09-01T04:00:00Z")

    report = build_activity_report(conn, "month", datetime(2026, 9, 21, 10, 0, tzinfo=UTC))
    assert report.event_count == 1


def test_build_activity_report_year_buckets_by_month(tmp_path):
    conn = db.connect(tmp_path / "reports.db")
    _clip(conn, message_id=1, camera_id="cam01", timestamp="2026-01-15T10:00:00Z")
    _clip(conn, message_id=2, camera_id="cam01", timestamp="2026-06-15T10:00:00Z")

    report = build_activity_report(conn, "year", datetime(2026, 9, 21, 10, 0, tzinfo=UTC))
    assert report.period == "year"
    assert len(report.timeline_labels) == 12
    assert report.timeline_counts[0] == 1
    assert report.timeline_counts[5] == 1
    assert report.event_count == 2


def test_activity_report_caption_reports_peak_hour():
    conn = db.connect(":memory:")
    _clip(conn, message_id=1, camera_id="cam01", timestamp="2026-09-05T18:00:00Z")
    report = build_activity_report(conn, "month", datetime(2026, 9, 21, 10, 0, tzinfo=UTC))
    assert "Busiest hour" in report.caption
    assert "SAST" in report.caption


def test_activity_report_caption_handles_no_activity():
    conn = db.connect(":memory:")
    report = build_activity_report(conn, "month", datetime(2026, 9, 21, 10, 0, tzinfo=UTC))
    assert report.peak_hour is None
    assert "No activity recorded" in report.caption
    assert report.event_count == 0


def test_render_activity_report_produces_a_valid_png():
    conn = db.connect(":memory:")
    _clip(conn, message_id=1, camera_id="cam01", timestamp="2026-09-05T18:00:00Z")
    _clip(conn, message_id=2, camera_id="cam01", timestamp="2026-09-06T04:00:00Z")
    report = build_activity_report(conn, "month", datetime(2026, 9, 21, 10, 0, tzinfo=UTC))
    png = render_activity_report(report)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_activity_report_handles_empty_period_without_crashing():
    conn = db.connect(":memory:")
    report = build_activity_report(conn, "year", datetime(2026, 9, 21, 10, 0, tzinfo=UTC))
    png = render_activity_report(report)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
