from pathlib import Path

from scripts.label import _hyperlink, _resolve_label, run_labeling_session
from src import db
from src.db import VALID_LABELS

SHORT_DURATIONS = {
    "data/history/cam01/short1.mp4": 0.8,
    "data/history/cam01/short2.mp4": 1.0,
    "data/history/cam01/short3.mp4": 1.2,
    "data/history/cam01/short4.mp4": 0.6,
}


def _fake_duration_fn(path: str) -> float | None:
    return SHORT_DURATIONS.get(path, 8.0)


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


def test_resolve_label_accepts_full_word_and_shortcut_letter():
    for label in VALID_LABELS:
        assert _resolve_label(label) == label
        assert _resolve_label(label[0]) == label
        assert _resolve_label(label[0].upper()) == label


def test_resolve_label_rejects_unknown_input():
    assert _resolve_label("bogus") is None
    assert _resolve_label("") is None


def test_hyperlink_wraps_path_in_osc8_escape_with_file_uri(tmp_path: Path):
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"")

    link = _hyperlink(str(target))

    assert link.startswith("\033]8;;file://")
    assert str(target) in link
    assert link.endswith("\033]8;;\033\\")


def test_short_clips_prompted_individually_until_confirm_count_reached(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    for i, path in enumerate(SHORT_DURATIONS, start=1):
        _seed(conn, i, file_path=path)
    _seed(conn, 5, file_path="data/history/cam01/normal.mp4")  # 8.0s per fake duration_fn

    prompted: list[int] = []

    def prompt_fn(clip):
        prompted.append(clip["message_id"])
        return "environment", None

    confirm_calls: list[tuple[str, int]] = []

    def bulk_confirm_fn(label, remaining):
        confirm_calls.append((label, remaining))
        return True

    labeled = run_labeling_session(
        conn,
        prompt_fn=prompt_fn,
        short_clip_seconds=2.0,
        confirm_short_count=3,
        bulk_confirm_fn=bulk_confirm_fn,
        duration_fn=_fake_duration_fn,
    )

    # 3 short clips prompted individually, the 4th bulk-applied without a prompt.
    assert prompted == [1, 2, 3, 5]
    assert confirm_calls == [("environment", 1)]
    assert labeled == 5
    assert db.get_label(conn, "chan1", 4)["notes"] == "bulk: matches short blank-clip pattern"
    conn.close()


def test_short_clip_bulk_not_offered_when_labels_differ(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    for i, path in enumerate(SHORT_DURATIONS, start=1):
        _seed(conn, i, file_path=path)

    responses = iter(["environment", "unknown", "environment", "environment"])

    def prompt_fn(clip):
        return next(responses), None

    confirm_calls = []
    labeled = run_labeling_session(
        conn,
        prompt_fn=prompt_fn,
        short_clip_seconds=2.0,
        confirm_short_count=3,
        bulk_confirm_fn=lambda label, remaining: confirm_calls.append((label, remaining)) or True,
        duration_fn=_fake_duration_fn,
    )

    assert labeled == 4
    assert confirm_calls == []
    conn.close()


def test_short_clip_detection_disabled_by_default(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    for i, path in enumerate(SHORT_DURATIONS, start=1):
        _seed(conn, i, file_path=path)

    prompted: list[int] = []

    def prompt_fn(clip):
        prompted.append(clip["message_id"])
        return "environment", None

    run_labeling_session(conn, prompt_fn=prompt_fn, duration_fn=_fake_duration_fn)

    assert prompted == [1, 2, 3, 4]
    conn.close()


def test_priority_clip_never_bulk_labeled_even_if_short(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn, 21519, camera_id="cam06", file_path="data/history/cam06/21519.mp4")
    for i, path in enumerate(SHORT_DURATIONS, start=1):
        _seed(conn, 100 + i, camera_id="cam06", file_path=path)

    def fake_duration(path):
        if path == "data/history/cam06/21519.mp4":
            return 0.8
        return _fake_duration_fn(path)

    prompted: list[int] = []

    def prompt_fn(clip):
        prompted.append(clip["message_id"])
        return "incident" if clip["message_id"] == 21519 else "environment", None

    run_labeling_session(
        conn,
        prompt_fn=prompt_fn,
        short_clip_seconds=2.0,
        confirm_short_count=2,
        bulk_confirm_fn=lambda label, remaining: True,
        duration_fn=fake_duration,
    )

    assert 21519 in prompted  # always prompted individually, never bulk-applied
    conn.close()
