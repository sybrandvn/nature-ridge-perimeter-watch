from types import SimpleNamespace

import pytest

from src.media_download import download_media_atomic, is_video_message


def test_is_video_message_rejects_non_video_documents():
    assert is_video_message(
        SimpleNamespace(video=None, document=SimpleNamespace(mime_type="video/mp4"))
    )
    assert not is_video_message(
        SimpleNamespace(video=None, document=SimpleNamespace(mime_type="application/pdf"))
    )


async def test_download_media_atomic_renames_complete_file(tmp_path):
    class Client:
        async def download_media(self, message, *, file):
            from pathlib import Path

            Path(file).write_bytes(b"complete")
            return file

    destination = tmp_path / "clip.mp4"
    await download_media_atomic(Client(), object(), destination)
    assert destination.read_bytes() == b"complete"
    assert not (tmp_path / "clip.mp4.part").exists()


async def test_download_media_atomic_removes_partial_on_failure(tmp_path):
    class Client:
        async def download_media(self, message, *, file):
            from pathlib import Path

            Path(file).write_bytes(b"partial")
            raise RuntimeError("network")

    with pytest.raises(RuntimeError, match="network"):
        await download_media_atomic(Client(), object(), tmp_path / "clip.mp4")
    assert not (tmp_path / "clip.mp4.part").exists()
