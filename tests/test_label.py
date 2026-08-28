from pathlib import Path

from scripts.label import run_labeling_session
from src import db


def _seed(
    conn,
    message_id: int,
    camera_id: str = "cam01",
    file_path: str | None = None,
    timestamp: str = "2026-01-01T20:00:00Z",
) -> None:
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=message_id,
        camera_id=camera_id,
        timestamp=timestamp,
        caption="Motion detected",
        file_path=file_path,
        source="backfill",
    )


def test_run_labeling_session_writes_labels_via_prompt(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn, 1)
    _seed(conn, 2)

    responses = iter([("guard", "night patrol"), ("animal", None)])
    labeled = run_labeling_session(conn, prompt_fn=lambda _clip: next(responses))

    assert labeled == 2
    assert db.get_label(conn, "chan1", 1)["label"] == "guard"
    assert db.get_label(conn, "chan1", 2)["label"] == "animal"
    conn.close()


def test_run_labeling_session_stops_on_quit(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn, 1)
    _seed(conn, 2)

    labeled = run_labeling_session(conn, prompt_fn=lambda _clip: None)

    assert labeled == 0
    assert db.get_label(conn, "chan1", 1) is None
    conn.close()


def test_run_labeling_session_respects_limit(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn, 1)
    _seed(conn, 2)
    _seed(conn, 3)

    labeled = run_labeling_session(conn, prompt_fn=lambda _clip: ("guard", None), limit=2)

    assert labeled == 2
    conn.close()


def test_run_labeling_session_filters_by_camera(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn, 1, camera_id="cam01")
    _seed(conn, 2, camera_id="cam02")

    labeled = run_labeling_session(
        conn, prompt_fn=lambda _clip: ("guard", None), camera_id="cam01"
    )

    assert labeled == 1
    assert db.get_label(conn, "chan1", 1) is not None
    assert db.get_label(conn, "chan1", 2) is None
    conn.close()


def test_run_labeling_session_with_file_only_skips_clips_without_a_file(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn, 1, file_path=None)
    _seed(conn, 2, file_path="data/history/cam01/2.mp4")

    labeled = run_labeling_session(
        conn, prompt_fn=lambda _clip: ("guard", None), with_file_only=True
    )

    assert labeled == 1
    assert db.get_label(conn, "chan1", 1) is None
    assert db.get_label(conn, "chan1", 2)["label"] == "guard"
    conn.close()


def test_run_labeling_session_surfaces_known_incident_clips_first(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn, 100, camera_id="cam06", timestamp="2026-01-01T20:00:00Z")
    _seed(conn, 21520, camera_id="cam06", timestamp="2026-07-21T20:31:00Z")
    _seed(conn, 200, camera_id="cam06", timestamp="2026-08-01T20:00:00Z")

    seen: list[int] = []

    def prompt_fn(clip):
        seen.append(clip["message_id"])
        return "guard", None

    run_labeling_session(conn, prompt_fn=prompt_fn)

    assert seen == [21520, 100, 200]
    conn.close()
