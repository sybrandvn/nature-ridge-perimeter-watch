from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src import db, live_state
from src.bot import BotQueries, QueryBot, build_query_bot, operating_period
from src.config import Camera, CamerasConfig, CameraZone, load_app_config_from_mapping


def _setup(tmp_path, *, now=datetime(2026, 9, 19, 20, 0, tzinfo=UTC)):
    conn = db.connect(tmp_path / "bot.db")
    config = load_app_config_from_mapping(
        {
            "TELEGRAM_API_ID": "1",
            "TELEGRAM_API_HASH": "x",
            "SOURCE_CHANNEL": "source",
            "BOT_TRUSTEE_IDS": "11",
            "BOT_SECURITY_IDS": "22",
        }
    )
    zone = CameraZone(fence=None, outside=None, depth_cutoff=0.0, ignore=())
    cameras = CamerasConfig(
        cameras=(
            Camera(id="cam01", aliases=(), order=0, zone=zone, threshold_overrides={}),
            Camera(id="cam02", aliases=(), order=1, zone=zone, threshold_overrides={}),
        ),
        unknown_camera_id="unknown",
    )
    return conn, config, cameras, BotQueries(
        conn=conn, config=config, cameras=cameras, now=lambda: now
    )


def _final_event(conn, *, message_id, camera_id, timestamp, category, reason):
    key = f"{camera_id}|{message_id}"
    db.upsert_clip(
        conn,
        channel_id="source",
        message_id=message_id,
        camera_id=camera_id,
        timestamp=timestamp,
        caption="motion",
        file_path=f"{message_id}.mp4",
        source="live",
    )
    now = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    live_state.add_analyzed_clip(
        conn,
        event_key=key,
        camera_id=camera_id,
        deadline_at=now,
        channel_id="source",
        message_id=message_id,
        phase="standalone",
        category=category,
        reason=reason,
        blinding_foreground=False,
        features=None,
    )
    live_state.finalize_event(
        conn,
        event_key=key,
        category=category,
        reason=reason,
        representative_channel_id="source",
        representative_message_id=message_id,
        transports=(),
        now=now,
    )


def test_operating_period_is_schedule_aware(tmp_path):
    _conn, config, _cameras, _queries = _setup(tmp_path)
    start, end, active = operating_period(
        config, datetime(2026, 9, 19, 20, 0, tzinfo=UTC)
    )
    assert active is True
    assert start.isoformat() == "2026-09-19T16:00:00+00:00"
    assert end.isoformat() == "2026-09-20T04:00:00+00:00"


def test_tonight_and_animals_read_final_live_events(tmp_path):
    conn, _config, _cameras, queries = _setup(tmp_path)
    _final_event(
        conn,
        message_id=1,
        camera_id="cam01",
        timestamp="2026-09-19T18:30:00Z",
        category="animal_candidate",
        reason="outside_colour",
    )
    assert "animal_candidate=1" in queries.tonight()
    animals = queries.animals()
    assert "cam01" in animals
    assert "outside_colour" in animals


def test_health_uses_system_events_and_marks_window_silence(tmp_path):
    conn, _config, _cameras, queries = _setup(tmp_path)
    db.upsert_system_event(
        conn,
        channel_id="source",
        message_id=1,
        timestamp="2026-09-19T18:10:00Z",
        camera_id="cam01",
        event_type="power_out",
        raw_text="offline",
    )
    health = queries.health()
    assert "cam01: power_out" in health
    assert "cam02: no clips recorded" in health


def test_power_history_separates_restores_and_collapses_duplicates(tmp_path):
    conn, _config, _cameras, queries = _setup(tmp_path)
    for message_id, timestamp, text in (
        (1, "2026-08-20T17:42:35Z", "Power Failure Restore @ 20-08-26 19:39:52"),
        (2, "2026-08-20T17:43:32Z", "Power Failure @ 20-08-26 19:39:52"),
        (3, "2026-08-20T17:44:32Z", "Power Failure @ 20-08-26 19:39:52"),
    ):
        db.upsert_system_event(
            conn,
            channel_id="source",
            message_id=message_id,
            timestamp=timestamp,
            camera_id="unknown",
            event_type="power_out",
            raw_text=text,
        )
    report = queries.power(limit=10)
    assert "2 failures, 1 restores" in report
    assert report.count("POWER FAILURE") == 1
    assert report.count("restored") == 1


def test_battery_status_pairs_unscoped_restore_and_keeps_explicit_lows(tmp_path):
    conn, _config, _cameras, queries = _setup(tmp_path)
    events = (
        (1, "2026-01-01T10:00:00Z", "Device Battery Low [5]"),
        (2, "2026-01-02T10:00:00Z", "Device Battery Low Restore"),
        (3, "2026-03-17T10:00:00Z", "Device Battery Low [6]"),
        (4, "2026-03-18T10:00:00Z", "Device Battery Low Restore [Panel]"),
    )
    for message_id, timestamp, text in events:
        db.upsert_system_event(
            conn,
            channel_id="source",
            message_id=message_id,
            timestamp=timestamp,
            camera_id="unknown",
            event_type="battery_dead",
            raw_text=text,
        )
    report = queries.batteries()
    assert "Current replacement candidates" in report
    assert "device 6: 2026-03-17 12:00" in report
    assert "device 5:" not in report
    assert "panel:" not in report


def test_panel_and_fault_reports_use_synced_events(tmp_path):
    conn, _config, _cameras, queries = _setup(tmp_path)
    events = (
        (1, "2026-09-18T18:00:00Z", "panel_armed", "Panel Armed"),
        (2, "2026-09-19T06:00:00Z", "panel_disarmed", "Panel Disarmed"),
        (3, "2026-09-19T07:00:00Z", "tamper", "Tamper Event [17]"),
        (4, "2026-09-19T08:00:00Z", "tamper_restored", "Tamper Restore Event"),
        (
            5,
            "2026-09-19T09:00:00Z",
            "communication_failure",
            "ALERT TESTING FAILURE (FTT)",
        ),
    )
    for message_id, timestamp, event_type, text in events:
        db.upsert_system_event(
            conn,
            channel_id="source",
            message_id=message_id,
            timestamp=timestamp,
            camera_id="unknown",
            event_type=event_type,
            raw_text=text,
        )
    panel = queries.panel(limit=10)
    assert "latest: disarmed" in panel
    assert "1 armed, 1 disarmed" in panel
    faults = queries.faults(limit=10)
    assert "tamper=1, restored=1" in faults
    assert "CONTROL-ROOM COMMUNICATION FAILURE" in faults


async def test_unknown_users_are_refused_every_command(tmp_path):
    _conn, _config, _cameras, queries = _setup(tmp_path)
    controller = QueryBot(queries)
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=999), effective_message=message)
    await controller.about(update, None)
    message.reply_text.assert_awaited_once_with("Not authorized.")


async def test_security_is_refused_patrols_at_handler_level(tmp_path):
    _conn, _config, _cameras, queries = _setup(tmp_path)
    controller = QueryBot(queries)
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=22), effective_message=message)
    await controller.patrols(update, None)
    message.reply_text.assert_awaited_once_with("Not authorized.")


async def test_trustee_can_access_patrols(tmp_path):
    _conn, _config, _cameras, queries = _setup(tmp_path)
    controller = QueryBot(queries)
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=11), effective_message=message)
    await controller.patrols(update, None)
    response = message.reply_text.await_args.args[0]
    assert "No multi-camera guard passes" in response


def test_map_is_explicitly_approximate(tmp_path):
    _conn, _config, _cameras, queries = _setup(tmp_path)
    assert queries.map() == "Approximate fence order (not geographic)\n1:cam01 — 2:cam02"


def test_last_clip_returns_newest_metadata_even_when_media_needs_retrieval(tmp_path):
    conn, _config, _cameras, queries = _setup(tmp_path)
    existing = tmp_path / "existing.mp4"
    existing.write_bytes(b"video")
    for message_id, timestamp, path in (
        (1, "2026-09-19T18:00:00Z", existing),
        (2, "2026-09-19T19:00:00Z", tmp_path / "missing.mp4"),
        (3, "2026-09-19T20:00:00Z", None),
    ):
        db.upsert_clip(
            conn,
            channel_id="source",
            message_id=message_id,
            camera_id="cam01",
            timestamp=timestamp,
            caption="motion",
            file_path=None if path is None else str(path),
            source="live",
        )
    clip = queries.last_clip("cam01")
    assert clip is not None
    assert clip.message_id == 3
    assert clip.timestamp == "2026-09-19T20:00:00Z"
    assert clip.path is None


async def test_security_can_request_last_camera_video(tmp_path):
    conn, _config, _cameras, queries = _setup(tmp_path)
    path = tmp_path / "latest.mp4"
    path.write_bytes(b"video")
    db.upsert_clip(
        conn,
        channel_id="source",
        message_id=1,
        camera_id="cam01",
        timestamp="2026-09-19T18:30:00Z",
        caption="motion",
        file_path=str(path),
        source="live",
    )
    controller = QueryBot(queries)
    message = SimpleNamespace(reply_text=AsyncMock(), reply_video=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=22), effective_message=message)
    await controller.last(update, SimpleNamespace(args=["cam01"]))
    kwargs = message.reply_video.await_args.kwargs
    assert kwargs["video"].name == str(path)
    assert "cam01" in kwargs["caption"]
    assert kwargs["supports_streaming"] is True
    message.reply_text.assert_not_awaited()


async def test_last_rejects_unknown_camera_without_exposing_paths(tmp_path):
    _conn, _config, _cameras, queries = _setup(tmp_path)
    controller = QueryBot(queries)
    message = SimpleNamespace(reply_text=AsyncMock(), reply_video=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=11), effective_message=message)
    await controller.last(update, SimpleNamespace(args=["cam99"]))
    message.reply_text.assert_awaited_once_with("Unknown camera: cam99")
    message.reply_video.assert_not_awaited()


async def test_last_requires_camera_argument(tmp_path):
    _conn, _config, _cameras, queries = _setup(tmp_path)
    controller = QueryBot(queries)
    message = SimpleNamespace(reply_text=AsyncMock(), reply_video=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=11), effective_message=message)
    await controller.last(update, SimpleNamespace(args=[]))
    response = message.reply_text.await_args.args[0]
    assert response.startswith("Usage: /last <camera_id>")
    assert "cam01" in response


async def test_last_retrieves_missing_source_media(tmp_path):
    conn, _config, _cameras, queries = _setup(tmp_path)
    db.upsert_clip(
        conn,
        channel_id="source",
        message_id=8,
        camera_id="cam01",
        timestamp="2026-09-19T20:30:00Z",
        caption="motion",
        file_path=None,
        source="backfill",
    )
    downloaded = tmp_path / "retrieved.mp4"

    async def loader(clip):
        assert clip.message_id == 8
        downloaded.write_bytes(b"video")
        return downloaded

    controller = QueryBot(queries, media_loader=loader)
    message = SimpleNamespace(reply_text=AsyncMock(), reply_video=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=11), effective_message=message)
    await controller.last(update, SimpleNamespace(args=["cam01"]))
    assert message.reply_video.await_args.kwargs["video"].name == str(downloaded)


async def test_animal_and_incident_commands_send_labelled_videos(tmp_path):
    conn, _config, _cameras, queries = _setup(tmp_path)
    for message_id, camera, label in ((1, "cam01", "animal"), (2, "cam02", "incident")):
        path = tmp_path / f"{message_id}.mp4"
        path.write_bytes(b"video")
        db.upsert_clip(
            conn,
            channel_id="source",
            message_id=message_id,
            camera_id=camera,
            timestamp=f"2026-09-19T18:{message_id:02d}:00Z",
            caption="motion",
            file_path=str(path),
            source="backfill",
        )
        db.upsert_label(conn, channel_id="source", message_id=message_id, label=label)
    controller = QueryBot(queries)
    message = SimpleNamespace(reply_text=AsyncMock(), reply_video=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=22), effective_message=message)
    await controller.animal(update, SimpleNamespace(args=[]))
    assert "Animal candidate" in message.reply_video.await_args.kwargs["caption"]
    message.reply_video.reset_mock()
    await controller.incidents(update, SimpleNamespace(args=["1"]))
    assert "Incident candidate" in message.reply_video.await_args.kwargs["caption"]


async def test_history_groups_siblings_and_event_sends_representative_video(tmp_path):
    conn, _config, _cameras, queries = _setup(tmp_path)
    for message_id, phase, timestamp in (
        (10, "Initial", "2024-03-15T22:47:30Z"),
        (11, "Stopped", "2024-03-15T22:50:18Z"),
    ):
        path = tmp_path / f"{message_id}.mp4"
        path.write_bytes(b"video")
        db.upsert_clip(
            conn,
            channel_id="source",
            message_id=message_id,
            camera_id="cam01",
            timestamp=timestamp,
            caption=f"({phase}) cam01 @ 16-03-24 00:47:00",
            file_path=str(path),
            source="backfill",
        )
        db.upsert_label(conn, channel_id="source", message_id=message_id, label="incident")
    events = queries.category_events("incident")
    assert len(events) == 1
    assert events[0].clip.message_id == 11
    assert events[0].sibling_count == 2
    listing = queries.history("incident", page=1)
    assert "11 · 2024-03-16 00:50 · cam01, 2 clips" in listing

    controller = QueryBot(queries)
    message = SimpleNamespace(reply_text=AsyncMock(), reply_video=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=11), effective_message=message)
    await controller.event(update, SimpleNamespace(args=["11"]))
    assert "Incident history event 11" in message.reply_video.await_args.kwargs["caption"]


async def test_history_validates_category_and_event_id(tmp_path):
    _conn, _config, _cameras, queries = _setup(tmp_path)
    controller = QueryBot(queries)
    message = SimpleNamespace(reply_text=AsyncMock(), reply_video=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=11), effective_message=message)
    await controller.history(update, SimpleNamespace(args=["cars"]))
    message.reply_text.assert_awaited_once_with("Usage: /history <animal|incident> [page]")
    message.reply_text.reset_mock()
    await controller.event(update, SimpleNamespace(args=["999"]))
    message.reply_text.assert_awaited_once_with(
        "No animal or incident event is listed with ID 999."
    )


async def test_grouped_menu_navigates_sections(tmp_path):
    _conn, _config, _cameras, queries = _setup(tmp_path)
    controller = QueryBot(queries)
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(effective_user=SimpleNamespace(id=11), effective_message=message)
    await controller.menu(update, None)
    kwargs = message.reply_text.await_args.kwargs
    labels = [button.text for row in kwargs["reply_markup"].inline_keyboard for button in row]
    assert labels == ["Monitoring", "Events", "System", "Site", "About"]

    query = SimpleNamespace(
        data="menu:system",
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )
    callback_update = SimpleNamespace(
        effective_user=SimpleNamespace(id=11),
        effective_message=message,
        callback_query=query,
    )
    await controller.menu_callback(callback_update, None)
    query.answer.assert_awaited_once()
    assert query.edit_message_text.await_args.kwargs["text"] == "Alarm-system events"


def test_visible_bot_commands_are_compact_menu_entry_points(tmp_path):
    _conn, _config, _cameras, queries = _setup(tmp_path)
    application = build_query_bot(queries, "123456:example-token")
    commands = [command.command for command in application.bot_data["commands"]]
    assert commands == ["menu", "about", "event", "last", "history"]
