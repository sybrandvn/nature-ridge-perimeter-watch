import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src import db
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
    async def download_media(self, message, *, file):
        from pathlib import Path

        Path(file).write_bytes(b"video")
        return file


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
