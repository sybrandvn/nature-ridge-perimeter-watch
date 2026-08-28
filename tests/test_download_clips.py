from pathlib import Path

import pytest

from scripts.download_clips import (
    clip_download_path,
    download_one,
    run_download,
    select_download_candidates,
)
from src import db

CHANNEL = "chan1"


@pytest.fixture
def conn(tmp_path: Path):
    connection = db.connect(tmp_path / "test.db")
    yield connection
    connection.close()


def _seed(conn, camera_id: str, message_id: int, timestamp: str, file_path: str | None = None):
    db.upsert_clip(
        conn,
        channel_id=CHANNEL,
        message_id=message_id,
        camera_id=camera_id,
        timestamp=timestamp,
        caption=None,
        file_path=file_path,
        source="backfill",
    )


class FakeMedia:
    pass


class FakeMessage:
    def __init__(self, *, video=None, document=None):
        self.video = video
        self.document = document


class FakeClient:
    """Stands in for Telethon's TelegramClient for the two calls we make."""

    def __init__(self, messages_by_id: dict[int, FakeMessage]):
        self._messages_by_id = messages_by_id
        self.downloaded: list[tuple[int, str]] = []

    async def get_messages(self, channel_ref, *, ids):
        return self._messages_by_id.get(ids)

    async def download_media(self, message, *, file):
        self.downloaded.append((id(message), file))
        Path(file).write_bytes(b"fake video bytes")


# --------------------------------------------------------------------------
# select_download_candidates
# --------------------------------------------------------------------------


def test_select_download_candidates_filters_by_camera(conn):
    _seed(conn, "cam01", 1, "2026-07-21T20:00:00Z")
    _seed(conn, "cam02", 2, "2026-07-21T20:00:00Z")

    rows = select_download_candidates(conn, cameras=["cam01"])

    assert [r["message_id"] for r in rows] == [1]


def test_select_download_candidates_skips_already_downloaded(conn):
    _seed(conn, "cam01", 1, "2026-07-21T20:00:00Z", file_path="data/history/cam01/1.mp4")
    _seed(conn, "cam01", 2, "2026-07-21T20:05:00Z")

    rows = select_download_candidates(conn, cameras=["cam01"])

    assert [r["message_id"] for r in rows] == [2]


def test_select_download_candidates_respects_since_until_window(conn):
    _seed(conn, "cam01", 1, "2026-07-20T23:00:00Z")
    _seed(conn, "cam01", 2, "2026-07-21T20:00:00Z")
    _seed(conn, "cam01", 3, "2026-07-22T00:00:00Z")

    rows = select_download_candidates(
        conn, cameras=["cam01"], since="2026-07-21T00:00:00Z", until="2026-07-22T00:00:00Z"
    )

    assert [r["message_id"] for r in rows] == [2]


def test_select_download_candidates_caps_per_camera(conn):
    for message_id in range(1, 6):
        _seed(conn, "cam01", message_id, f"2026-07-21T20:0{message_id}:00Z")

    rows = select_download_candidates(conn, cameras=["cam01"], limit_per_camera=2)

    assert [r["message_id"] for r in rows] == [1, 2]


# --------------------------------------------------------------------------
# clip_download_path
# --------------------------------------------------------------------------


def test_clip_download_path_layout():
    assert clip_download_path("data/history", "cam01b", 42) == Path("data/history/cam01b/42.mp4")


# --------------------------------------------------------------------------
# download_one
# --------------------------------------------------------------------------


async def test_download_one_writes_dest_via_atomic_rename(tmp_path):
    message = FakeMessage(video=FakeMedia())
    client = FakeClient({1: message})
    dest = tmp_path / "cam01" / "1.mp4"

    result = await download_one(client, "chan_ref", 1, dest)

    assert result is True
    assert dest.exists()
    assert not dest.with_suffix(".mp4.part").exists()
    assert client.downloaded == [(id(message), str(dest.with_suffix(".mp4.part")))]


async def test_download_one_skips_message_with_no_media(tmp_path):
    message = FakeMessage(video=None, document=None)
    client = FakeClient({1: message})
    dest = tmp_path / "cam01" / "1.mp4"

    result = await download_one(client, "chan_ref", 1, dest)

    assert result is False
    assert not dest.exists()


async def test_download_one_skips_missing_message(tmp_path):
    client = FakeClient({})
    dest = tmp_path / "cam01" / "1.mp4"

    result = await download_one(client, "chan_ref", 999, dest)

    assert result is False
    assert not dest.exists()


# --------------------------------------------------------------------------
# run_download
# --------------------------------------------------------------------------


async def test_run_download_updates_file_path_and_counts(conn, tmp_path):
    _seed(conn, "cam01", 1, "2026-07-21T20:00:00Z")
    _seed(conn, "cam01", 2, "2026-07-21T20:05:00Z")
    client = FakeClient(
        {
            1: FakeMessage(video=FakeMedia()),
            2: FakeMessage(video=None, document=None),
        }
    )
    candidates = select_download_candidates(conn, cameras=["cam01"])

    counts = await run_download(
        conn,
        client,
        channel_ref="chan_ref",
        channel_id=CHANNEL,
        candidates=candidates,
        out_dir=str(tmp_path),
    )

    assert counts == {"downloaded": 1, "skipped_no_media": 1}
    assert db.get_clip(conn, CHANNEL, 1)["file_path"] == str(tmp_path / "cam01" / "1.mp4")
    assert db.get_clip(conn, CHANNEL, 2)["file_path"] is None
