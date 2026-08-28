from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.meta_backfill import run_backfill
from src import db
from src.config import Camera, CamerasConfig, CameraZone

_ZONE = CameraZone(fence=None, far_side=None, depth_cutoff=1.0, ignore=())


@dataclass
class FakeMessage:
    id: int
    date: datetime
    message: str | None = None
    video: object | None = None
    document: object | None = None


class FakeMessages:
    """Minimal async iterator standing in for Telethon's client.iter_messages()."""

    def __init__(self, messages: list[FakeMessage]):
        self._messages = messages

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for msg in self._messages:
            yield msg


@dataclass
class _FakeVideo:
    marker: bool = field(default=True)


@pytest.fixture
def cameras() -> CamerasConfig:
    return CamerasConfig(
        cameras=(
            Camera(
                id="cam_north", aliases=("North Gate",), order=0, zone=_ZONE, threshold_overrides={}
            ),
        ),
        unknown_camera_id="unknown",
    )


@pytest.fixture
def conn(tmp_path: Path):
    connection = db.connect(tmp_path / "test.db")
    yield connection
    connection.close()


async def _run(messages: list[FakeMessage], cameras: CamerasConfig, conn) -> dict[str, int]:
    return await run_backfill(
        FakeMessages(messages), channel_id="chan1", cameras=cameras, conn=conn
    )


async def test_run_backfill_classifies_and_persists_all_kinds(cameras, conn):
    messages = [
        FakeMessage(
            id=1, date=datetime(2024, 1, 1, tzinfo=UTC), message="North Gate", video=_FakeVideo()
        ),
        FakeMessage(
            id=2, date=datetime(2024, 1, 1, 0, 1, tzinfo=UTC), message="North Gate battery low"
        ),
        FakeMessage(
            id=3, date=datetime(2024, 1, 1, 0, 2, tzinfo=UTC), message="good evening team"
        ),
    ]

    counts = await _run(messages, cameras, conn)

    assert counts == {"clip": 1, "system_event": 1, "ignored": 1}
    clip = db.get_clip(conn, "chan1", 1)
    assert clip["camera_id"] == "cam_north"
    assert clip["source"] == "backfill"
    assert clip["file_path"] is None

    events = list(db.iter_system_events(conn))
    assert len(events) == 1
    assert events[0]["event_type"] == "battery_dead"
    assert events[0]["camera_id"] == "cam_north"


async def test_run_backfill_is_idempotent_on_rerun(cameras, conn):
    messages = [
        FakeMessage(
            id=1, date=datetime(2024, 1, 1, tzinfo=UTC), message="North Gate", video=_FakeVideo()
        ),
    ]

    await _run(messages, cameras, conn)
    await _run(messages, cameras, conn)

    assert len(list(db.iter_clips(conn))) == 1


async def test_run_backfill_unknown_camera_still_recorded(cameras, conn):
    messages = [
        FakeMessage(
            id=1,
            date=datetime(2024, 1, 1, tzinfo=UTC),
            message="unrecognised caption",
            video=_FakeVideo(),
        ),
    ]

    await _run(messages, cameras, conn)

    clip = db.get_clip(conn, "chan1", 1)
    assert clip["camera_id"] == "unknown"
