import random
from pathlib import Path

from scripts.label import _event_key, _resolve_label, run_labeling_session
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
    caption: str = "Motion detected",
) -> None:
    db.upsert_clip(
        conn,
        channel_id="chan1",
        message_id=message_id,
        camera_id=camera_id,
        timestamp=timestamp,
        caption=caption,
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


def test_event_key_groups_initial_and_stopped_captions():
    initial = "Cam Alert*: (Initial*) NATURE RIDGE COMPLEX, MOTIONVIEWER 10 @ 14-03-24 20:44:22"
    stopped = "Cam Alert*: (Stopped*) NATURE RIDGE COMPLEX, MOTIONVIEWER 10 @ 14-03-24 20:44:22"

    assert _event_key("cam10", initial) == _event_key("cam10", stopped)
    assert _event_key("cam10", initial) != _event_key("cam06", initial)  # different camera


def test_event_key_none_without_embedded_timestamp():
    assert _event_key("cam01", "Motion detected") is None
    assert _event_key("cam01", None) is None


def test_run_labeling_session_suggests_label_for_paired_alert(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    initial = "Cam Alert*: (Initial*) NATURE RIDGE COMPLEX, MOTIONVIEWER 10 @ 14-03-24 20:44:22"
    stopped = "Cam Alert*: (Stopped*) NATURE RIDGE COMPLEX, MOTIONVIEWER 10 @ 14-03-24 20:44:22"
    _seed(conn, 1, caption=initial)
    _seed(conn, 2, caption=stopped)

    seen_suggestions: list[str | None] = []
    responses = iter([("environment", None), ("environment", None)])

    def prompt_fn(clip):
        seen_suggestions.append(clip.get("_suggested_label"))
        return next(responses)

    labeled = run_labeling_session(conn, prompt_fn=prompt_fn)

    assert labeled == 2
    assert seen_suggestions == [None, "environment"]
    conn.close()


def test_event_suggestion_persists_across_sessions(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    initial = "Cam Alert*: (Initial*) NATURE RIDGE COMPLEX, MOTIONVIEWER 10 @ 14-03-24 20:44:22"
    stopped = "Cam Alert*: (Stopped*) NATURE RIDGE COMPLEX, MOTIONVIEWER 10 @ 14-03-24 20:44:22"
    _seed(conn, 1, caption=initial)
    _seed(conn, 2, caption=stopped)

    run_labeling_session(conn, prompt_fn=lambda _clip: ("guard", None), limit=1)

    seen_suggestions: list[str | None] = []

    def prompt_fn(clip):
        seen_suggestions.append(clip.get("_suggested_label"))
        return "guard", None

    run_labeling_session(conn, prompt_fn=prompt_fn)

    assert seen_suggestions == ["guard"]
    conn.close()


def test_run_labeling_session_spreads_rest_across_cameras(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    for cam_num, cam in enumerate(("cam01", "cam02", "cam03"), start=1):
        for i in range(3):
            _seed(
                conn,
                cam_num * 100 + i,
                camera_id=cam,
                timestamp=f"2026-01-01T{20 + i:02d}:00:00Z",
            )

    seen_cameras: list[str] = []

    def prompt_fn(clip):
        seen_cameras.append(clip["camera_id"])
        return "guard", None

    run_labeling_session(conn, prompt_fn=prompt_fn, rng=random.Random(0))

    assert len(seen_cameras) == 9
    # round-robin: each consecutive triple covers all three cameras, never the same
    # camera three times in a row like plain timestamp order would produce.
    for i in range(0, 9, 3):
        assert set(seen_cameras[i : i + 3]) == {"cam01", "cam02", "cam03"}


def test_run_labeling_session_single_camera_filter_not_shuffled(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn, 1, camera_id="cam01", timestamp="2026-01-01T20:00:00Z")
    _seed(conn, 2, camera_id="cam01", timestamp="2026-01-01T21:00:00Z")
    _seed(conn, 3, camera_id="cam01", timestamp="2026-01-01T22:00:00Z")

    seen_ids: list[int] = []

    def prompt_fn(clip):
        seen_ids.append(clip["message_id"])
        return "guard", None

    run_labeling_session(conn, prompt_fn=prompt_fn, camera_id="cam01", rng=random.Random(0))

    assert seen_ids == [1, 2, 3]  # untouched: only one camera present
    conn.close()


def test_priority_clips_stay_first_when_spread_across_cameras(tmp_path: Path):
    conn = db.connect(tmp_path / "t.db")
    _seed(conn, 21519, camera_id="cam06", timestamp="2026-01-01T20:00:00Z")
    for cam_num, cam in enumerate(("cam01", "cam02"), start=1):
        _seed(conn, cam_num * 100 + 1, camera_id=cam, timestamp="2026-01-01T19:00:00Z")
        _seed(conn, cam_num * 100 + 2, camera_id=cam, timestamp="2026-01-01T21:00:00Z")

    seen: list[tuple[str, int]] = []

    def prompt_fn(clip):
        seen.append((clip["camera_id"], clip["message_id"]))
        return "incident" if clip["camera_id"] == "cam06" else "guard", None

    run_labeling_session(conn, prompt_fn=prompt_fn, rng=random.Random(0))

    assert seen[0] == ("cam06", 21519)
    conn.close()
