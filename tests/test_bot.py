from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src import db, live_state
from src.bot import BotQueries, QueryBot, operating_period
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
