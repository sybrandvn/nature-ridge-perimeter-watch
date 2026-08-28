from pathlib import Path

from scripts.label import run_labeling_session
from src import db


def _seed(conn, message_id: int, camera_id: str = "cam01") -> None:
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=message_id,
        camera_id=camera_id,
        timestamp="2026-01-01T20:00:00Z",
        caption="Motion detected",
        file_path=None,
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
