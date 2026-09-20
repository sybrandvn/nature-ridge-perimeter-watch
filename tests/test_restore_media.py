from pathlib import Path
from types import SimpleNamespace

from scripts.restore_media import missing_clips, restore_missing_media
from src import db


def _row(conn, message_id, *, file_path=None):
    db.upsert_clip(
        conn,
        channel_id="source",
        message_id=message_id,
        camera_id="cam01",
        timestamp=f"2026-09-19T18:{message_id:02d}:00Z",
        caption="motion",
        file_path=file_path,
        source="backfill",
    )


def test_missing_clips_selects_only_rows_without_media(tmp_path):
    conn = db.connect(tmp_path / "db.sqlite")
    _row(conn, 1)
    _row(conn, 2, file_path="already.mp4")
    assert [row["message_id"] for row in missing_clips(conn)] == [1]


async def test_restore_downloads_atomically_and_updates_database(tmp_path):
    conn = db.connect(tmp_path / "db.sqlite")
    _row(conn, 1)
    _row(conn, 2)

    class Client:
        async def get_messages(self, channel, *, ids):
            return [
                SimpleNamespace(id=1, video=True, document=None),
                SimpleNamespace(id=2, video=None, document=None),
            ]

        async def download_media(self, message, *, file):
            Path(file).write_bytes(b"video")
            return file

    result = await restore_missing_media(
        conn,
        Client(),
        channel_ref="source",
        clips=missing_clips(conn),
        out_dir=tmp_path / "history",
        sleep=lambda _delay: _no_sleep(),
    )
    assert result == {"selected": 2, "restored": 1, "no_media": 1, "failed": 0}
    path = tmp_path / "history/cam01/1.mp4"
    assert path.read_bytes() == b"video"
    assert db.get_clip(conn, "source", 1)["file_path"] == str(path)
    assert not path.with_suffix(".mp4.part").exists()


async def _no_sleep():
    return None
