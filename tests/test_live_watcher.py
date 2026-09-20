import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src import db
from src.bot import MediaClip
from src.classify import ClassificationResult
from src.clip_analysis import ClipAnalysis
from src.config import (
    Camera,
    CamerasConfig,
    CameraZone,
    load_app_config_from_mapping,
    load_thresholds_config,
)
from src.live_watcher import LiveWatcher

NOW = datetime(2026, 9, 19, 18, 0, tzinfo=UTC)


class Client:
    def __init__(self):
        self.messages = {}

    async def download_media(self, message, *, file):
        from pathlib import Path

        Path(file).write_bytes(b"video")
        return file

    async def get_messages(self, channel, *, ids):
        return self.messages.get(ids)


def _message(message_id, caption, *, mime="video/mp4"):
    return SimpleNamespace(
        id=message_id,
        date=NOW,
        message=caption,
        video=None,
        document=SimpleNamespace(mime_type=mime),
    )


def _runtime(tmp_path, analyses):
    db_path = tmp_path / "watch.db"
    conn = db.connect(db_path)
    config = load_app_config_from_mapping(
        {
            "TELEGRAM_API_ID": "1",
            "TELEGRAM_API_HASH": "x",
            "SOURCE_CHANNEL": "source",
            "TELEGRAM_BOT_TOKEN": "token",
            "ALERT_CHANNEL_ID": "alerts",
            "LIVE_MEDIA_DIR": str(tmp_path / "clips"),
        }
    )
    camera = Camera(
        id="cam01",
        aliases=("Camera 1",),
        order=0,
        zone=CameraZone(fence=None, outside=None, depth_cutoff=0.0, ignore=()),
        threshold_overrides={},
    )

    def analyze(**kwargs):
        category, reason = analyses.pop(0)
        return ClipAnalysis(
            ClassificationResult(category, reason, {}),
            {"green_light_ratio": 0.0},
            False,
            False,
        )

    bot = AsyncMock()
    watcher = LiveWatcher(
        conn=conn,
        db_path=db_path,
        client=Client(),
        channel_id="source",
        cameras=CamerasConfig((camera,), "unknown"),
        thresholds=load_thresholds_config("config/thresholds.yaml"),
        config=config,
        reference_entries=[],
        now=lambda: NOW,
        analysis_fn=analyze,
        telegram_bot=bot,
    )
    return conn, watcher, bot


def test_resident_and_neighbour_are_quiet_channel_alerts(tmp_path):
    _conn, watcher, _bot = _runtime(tmp_path, [])
    watcher.config = replace(
        watcher.config,
        ntfy_base_url="https://ntfy.example",
        ntfy_topic="alerts",
    )
    assert watcher._transports("resident_candidate") == ("telegram",)
    assert watcher._transports("neighbour_candidate") == ("telegram",)
    assert watcher._transports("animal_candidate") == ("telegram", "ntfy")
    assert watcher._transports("guard_candidate") == ()


async def test_initial_and_complete_resolve_to_urgent_sibling_and_send_video(tmp_path):
    conn, watcher, bot = _runtime(
        tmp_path,
        [("guard_candidate", "green_light"), ("animal_candidate", "outside_colour")],
    )
    suffix = "Camera 1 @ 19-09-26 20:00:00"
    await asyncio.wait_for(watcher.handle_message(_message(1, f"(Initial) {suffix}")), 5)
    assert conn.execute("SELECT status FROM live_events").fetchone()[0] == "pending"
    await asyncio.wait_for(watcher.handle_message(_message(2, f"(Stopped) {suffix}")), 5)
    event = conn.execute("SELECT * FROM live_events").fetchone()
    assert event["status"] == "finalized"
    assert event["final_category"] == "animal_candidate"
    assert event["representative_message_id"] == 2
    bot.send_video.assert_awaited_once()
    assert conn.execute("SELECT status FROM live_deliveries").fetchone()[0] == "delivered"


async def test_startup_incident_warns_early_then_reports_likely_guard(tmp_path):
    conn, watcher, bot = _runtime(
        tmp_path,
        [("incident_candidate", "outside_no_colour"), ("guard_candidate", "green_light")],
    )
    suffix = "Camera 1 @ 19-09-26 20:00:00"

    await asyncio.wait_for(watcher.handle_message(_message(1, f"(Initial) {suffix}")), 5)
    event = conn.execute("SELECT * FROM live_events").fetchone()
    assert event["status"] == "pending"
    first = bot.send_video.await_args_list[0].kwargs
    assert "Possible incident" in first["caption"]
    assert "unconfirmed evidence" in first["caption"]

    await asyncio.wait_for(watcher.handle_message(_message(2, f"(Stopped) {suffix}")), 5)
    event = conn.execute("SELECT * FROM live_events").fetchone()
    assert event["final_category"] == "incident_candidate"
    assert event["resolution_state"] == "likely_resolved"
    deliveries = list(
        conn.execute("SELECT * FROM live_deliveries ORDER BY notification_kind")
    )
    assert {row["notification_kind"] for row in deliveries} == {
        "preliminary",
        "resolution",
    }
    assert all(row["status"] == "delivered" for row in deliveries)
    second = bot.send_video.await_args_list[1].kwargs
    assert "likely guard activity" in second["caption"]
    assert second["video"].name.endswith("/2.mp4")


async def test_startup_animal_waits_for_completed_sibling_before_alerting(tmp_path):
    conn, watcher, bot = _runtime(
        tmp_path,
        [("animal_candidate", "outside_colour"), ("animal_candidate", "outside_colour")],
    )
    suffix = "Camera 1 @ 19-09-26 20:00:00"

    await asyncio.wait_for(watcher.handle_message(_message(1, f"(Initial) {suffix}")), 5)
    assert conn.execute("SELECT status FROM live_events").fetchone()[0] == "pending"
    bot.send_video.assert_not_awaited()

    await asyncio.wait_for(watcher.handle_message(_message(2, f"(Stopped) {suffix}")), 5)
    bot.send_video.assert_awaited_once()
    assert "Perimeter alert: animal" in bot.send_video.await_args.kwargs["caption"]


async def test_out_of_order_initial_uses_existing_completion_without_preliminary(tmp_path):
    conn, watcher, bot = _runtime(
        tmp_path,
        [("guard_candidate", "green_light"), ("incident_candidate", "outside_no_colour")],
    )
    suffix = "Camera 1 @ 19-09-26 20:00:00"

    await asyncio.wait_for(watcher.handle_message(_message(1, f"(Stopped) {suffix}")), 5)
    bot.send_video.assert_not_awaited()
    await asyncio.wait_for(watcher.handle_message(_message(2, f"(Initial) {suffix}")), 5)

    event = conn.execute("SELECT * FROM live_events").fetchone()
    assert event["status"] == "finalized"
    assert event["resolution_state"] == "likely_resolved"
    bot.send_video.assert_awaited_once()
    assert "Possible incident" not in bot.send_video.await_args.kwargs["caption"]
    assert "Review: likely guard activity" in bot.send_video.await_args.kwargs["caption"]


async def test_likely_resolved_final_uses_nonurgent_ntfy_priority(tmp_path):
    _conn, watcher, bot = _runtime(
        tmp_path,
        [("animal_candidate", "outside_colour"), ("guard_candidate", "green_light")],
    )
    watcher.config = replace(
        watcher.config,
        ntfy_base_url="https://ntfy.example",
        ntfy_topic="alerts",
        ntfy_priority="urgent",
    )
    response = MagicMock()
    watcher.ntfy_session = MagicMock()
    watcher.ntfy_session.post.return_value = response
    suffix = "Camera 1 @ 19-09-26 20:00:00"

    await asyncio.wait_for(watcher.handle_message(_message(1, f"(Initial) {suffix}")), 5)
    bot.send_video.assert_not_awaited()
    await asyncio.wait_for(watcher.handle_message(_message(2, f"(Stopped) {suffix}")), 5)

    headers = watcher.ntfy_session.post.call_args.kwargs["headers"]
    assert headers["Priority"] == "default"
    assert "likely guard" in headers["Title"]


async def test_only_incident_camera_events_use_urgent_ntfy_priority(tmp_path):
    for name, category, expected in (
        ("incident", "incident_candidate", "urgent"),
        ("animal", "animal_candidate", "default"),
    ):
        _conn, watcher, _bot = _runtime(tmp_path / name, [(category, "reason")])
        watcher.config = replace(
            watcher.config,
            ntfy_base_url="https://ntfy.example",
            ntfy_topic="alerts",
            ntfy_priority="urgent",
        )
        watcher.ntfy_session = MagicMock()
        watcher.ntfy_session.post.return_value = MagicMock()

        await asyncio.wait_for(watcher.handle_message(_message(1, "Camera 1 motion")), 5)

        headers = watcher.ntfy_session.post.call_args.kwargs["headers"]
        assert headers["Priority"] == expected


async def test_startup_incident_with_environment_completion_requests_review(tmp_path):
    conn, watcher, bot = _runtime(
        tmp_path,
        [
            ("incident_candidate", "outside_no_colour"),
            ("environment_candidate", "outside_vegetation"),
        ],
    )
    suffix = "Camera 1 @ 19-09-26 20:00:00"
    await asyncio.wait_for(watcher.handle_message(_message(1, f"(Initial) {suffix}")), 5)
    await asyncio.wait_for(watcher.handle_message(_message(2, f"(Stopped) {suffix}")), 5)

    event = conn.execute("SELECT * FROM live_events").fetchone()
    assert event["resolution_state"] == "conflicting"
    caption = bot.send_video.await_args_list[1].kwargs["caption"]
    assert "conflicting evidence" in caption
    assert "has not been cleared" in caption


async def test_startup_incident_timeout_is_reported_as_unconfirmed(tmp_path):
    conn, watcher, bot = _runtime(
        tmp_path, [("incident_candidate", "outside_no_colour")]
    )
    await asyncio.wait_for(
        watcher.handle_message(_message(1, "(Initial) Camera 1 @ 19-09-26 20:00:00")), 5
    )
    watcher.now = lambda: NOW.replace(minute=6)
    await watcher.tick()

    event = conn.execute("SELECT * FROM live_events").fetchone()
    assert event["resolution_state"] == "unconfirmed"
    assert "completed sibling was not received" in bot.send_video.await_args_list[1].kwargs[
        "caption"
    ]


async def test_late_completed_guard_reopens_unconfirmed_incident(tmp_path):
    conn, watcher, bot = _runtime(
        tmp_path,
        [
            ("incident_candidate", "outside_no_colour"),
            ("guard_candidate", "green_light"),
            ("incident_candidate", "outside_no_colour"),
        ],
    )
    suffix = "Camera 1 @ 19-09-26 20:00:00"
    await asyncio.wait_for(watcher.handle_message(_message(1, f"(Initial) {suffix}")), 5)

    watcher.now = lambda: NOW.replace(minute=6)
    await watcher.tick()
    assert conn.execute("SELECT resolution_state FROM live_events").fetchone()[0] == "unconfirmed"

    await asyncio.wait_for(watcher.handle_message(_message(2, f"(Stopped) {suffix}")), 5)
    event = conn.execute("SELECT * FROM live_events").fetchone()
    assert event["status"] == "finalized"
    assert event["resolution_state"] == "likely_resolved"
    assert len(bot.send_video.await_args_list) == 3
    assert "likely guard activity" in bot.send_video.await_args_list[-1].kwargs["caption"]

    await asyncio.wait_for(watcher.handle_message(_message(3, f"(Stopped) {suffix}")), 5)
    event = conn.execute("SELECT * FROM live_events").fetchone()
    assert event["resolution_state"] == "confirmed"
    assert len(bot.send_video.await_args_list) == 4
    assert "evidence confirmed" in bot.send_video.await_args_list[-1].kwargs["caption"]


async def test_system_maintenance_message_is_stored_and_sent_to_telegram(tmp_path):
    conn, watcher, bot = _runtime(tmp_path, [])
    message = _message(
        1,
        "Power Failure @ 21-09-26 19:00:00",
        mime="application/octet-stream",
    )
    await asyncio.wait_for(watcher.handle_message(message), 5)
    await asyncio.wait_for(watcher.handle_message(message), 5)

    event = conn.execute("SELECT * FROM system_events").fetchone()
    assert event["event_type"] == "power_out"
    assert conn.execute("SELECT COUNT(*) FROM live_deliveries").fetchone()[0] == 0
    delivery = conn.execute("SELECT * FROM system_deliveries").fetchone()
    assert delivery["status"] == "delivered"
    bot.send_message.assert_awaited_once()
    assert "Security-system alert: Power failure" in bot.send_message.await_args.kwargs["text"]
    bot.send_video.assert_not_awaited()


async def test_system_failure_routes_to_default_priority_ntfy(tmp_path):
    _conn, watcher, _bot = _runtime(tmp_path, [])
    watcher.config = replace(
        watcher.config,
        ntfy_base_url="https://ntfy.example",
        ntfy_topic="alerts",
        ntfy_priority="urgent",
    )
    watcher.ntfy_session = MagicMock()
    watcher.ntfy_session.post.return_value = MagicMock()

    await asyncio.wait_for(
        watcher.handle_message(
            _message(1, "Tamper Event [17]", mime="application/octet-stream")
        ),
        5,
    )

    headers = watcher.ntfy_session.post.call_args.kwargs["headers"]
    assert headers["Priority"] == "default"
    assert headers["Title"] == "System: Tamper detected"


async def test_system_restore_sends_low_priority_correction(tmp_path):
    _conn, watcher, bot = _runtime(tmp_path, [])
    watcher.config = replace(
        watcher.config,
        ntfy_base_url="https://ntfy.example",
        ntfy_topic="alerts",
    )
    watcher.ntfy_session = MagicMock()
    watcher.ntfy_session.post.return_value = MagicMock()

    await asyncio.wait_for(
        watcher.handle_message(
            _message(1, "Power Failure Restore", mime="application/octet-stream")
        ),
        5,
    )

    assert "System update: Power restored" in bot.send_message.await_args.kwargs["text"]
    headers = watcher.ntfy_session.post.call_args.kwargs["headers"]
    assert headers["Priority"] == "default"
    assert headers["Title"] == "System: Power restored"


async def test_duplicate_message_is_not_downloaded_or_analyzed_twice(tmp_path):
    analyses = [("guard_candidate", "green_light")]
    conn, watcher, bot = _runtime(tmp_path, analyses)
    message = _message(1, "Camera 1 motion")
    await asyncio.wait_for(watcher.handle_message(message), 5)
    await asyncio.wait_for(watcher.handle_message(message), 5)
    assert analyses == []
    assert conn.execute("SELECT count(*) FROM live_event_clips").fetchone()[0] == 1
    bot.send_video.assert_not_awaited()


async def test_non_video_document_is_ignored(tmp_path):
    analyses = []
    conn, watcher, _bot = _runtime(tmp_path, analyses)
    await asyncio.wait_for(
        watcher.handle_message(_message(1, "Camera 1 report", mime="application/pdf")), 5
    )
    assert conn.execute("SELECT status FROM live_messages").fetchone()[0] == "ignored"
    assert conn.execute("SELECT count(*) FROM clips").fetchone()[0] == 0


async def test_retrieve_media_restores_retained_clip_from_source(tmp_path):
    conn, watcher, _bot = _runtime(tmp_path, [])
    db.upsert_clip(
        conn,
        channel_id="source",
        message_id=9,
        camera_id="cam01",
        timestamp="2026-09-19T17:00:00Z",
        caption="Camera 1 motion",
        file_path=None,
        source="backfill",
    )
    watcher.client.messages[9] = _message(9, "Camera 1 motion")
    path = await watcher.retrieve_media(
        MediaClip("source", 9, "cam01", "2026-09-19T17:00:00Z", None)
    )
    assert path is not None and path.read_bytes() == b"video"
    assert db.get_clip(conn, "source", 9)["file_path"] == str(path)
