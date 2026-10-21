from datetime import datetime

from src import db, live_state
from src.retention import clean_debug_cache, clean_local_media


def _clip(conn, root, *, message_id, camera="cam01", timestamp, label=None):
    path = root / camera / f"{message_id}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")
    db.upsert_clip(
        conn,
        channel_id="source",
        message_id=message_id,
        camera_id=camera,
        timestamp=timestamp,
        caption="motion",
        file_path=str(path),
        source="live",
    )
    if label:
        db.upsert_label(conn, channel_id="source", message_id=message_id, label=label)
    return path


def test_retention_keeps_urgent_labels_and_latest_per_camera(tmp_path):
    conn = db.connect(tmp_path / "db.sqlite")
    old = _clip(conn, tmp_path, message_id=1, timestamp="2026-09-19T18:00:00Z")
    animal = _clip(
        conn,
        tmp_path,
        message_id=2,
        timestamp="2026-09-19T18:10:00Z",
        label="animal",
    )
    latest = _clip(conn, tmp_path, message_id=3, timestamp="2026-09-19T18:20:00Z")
    other_camera = _clip(
        conn,
        tmp_path,
        message_id=4,
        camera="cam02",
        timestamp="2026-09-19T18:05:00Z",
    )
    result = clean_local_media(conn, data_root=tmp_path)
    assert result.deleted == 1
    assert not old.exists()
    assert animal.exists() and latest.exists() and other_camera.exists()
    assert db.get_clip(conn, "source", 1)["file_path"] is None


def test_retention_keeps_pending_and_final_urgent_live_events(tmp_path):
    conn = db.connect(tmp_path / "db.sqlite")
    pending_path = _clip(conn, tmp_path, message_id=1, timestamp="2026-09-19T18:00:00Z")
    final_path = _clip(conn, tmp_path, message_id=2, timestamp="2026-09-19T18:10:00Z")
    _clip(conn, tmp_path, message_id=3, timestamp="2026-09-19T18:20:00Z")
    now = datetime.fromisoformat("2026-09-19T18:00:00+00:00")
    for message_id, category in ((1, "guard_candidate"), (2, "incident_candidate")):
        key = f"event-{message_id}"
        live_state.add_analyzed_clip(
            conn,
            event_key=key,
            camera_id="cam01",
            deadline_at=now,
            channel_id="source",
            message_id=message_id,
            phase="initial",
            category=category,
            reason="test",
            blinding_foreground=False,
            features=None,
        )
        if message_id == 2:
            live_state.finalize_event(
                conn,
                event_key=key,
                category=category,
                reason="test",
                representative_channel_id="source",
                representative_message_id=message_id,
                transports=(),
                now=now,
            )
    clean_local_media(conn, data_root=tmp_path)
    assert pending_path.exists()
    assert final_path.exists()


def test_retention_refuses_to_delete_outside_data_root(tmp_path):
    data_root = tmp_path / "data"
    outside = tmp_path / "outside.mp4"
    outside_new = tmp_path / "outside-new.mp4"
    outside.write_bytes(b"keep")
    outside_new.write_bytes(b"keep")
    conn = db.connect(data_root / "db.sqlite")
    for message_id, timestamp, path in (
        (1, "2026-09-19T18:00:00Z", outside),
        (2, "2026-09-19T18:10:00Z", outside_new),
    ):
        db.upsert_clip(
            conn,
            channel_id="source",
            message_id=message_id,
            camera_id="cam01",
            timestamp=timestamp,
            caption="motion",
            file_path=str(path),
            source="live",
        )
    result = clean_local_media(conn, data_root=data_root)
    assert result.unsafe == 2
    assert outside.exists()
    assert outside_new.exists()


def test_debug_retention_keeps_urgent_and_latest_renders(tmp_path):
    conn = db.connect(tmp_path / "db.sqlite")
    _clip(conn, tmp_path, message_id=1, timestamp="2026-09-19T18:00:00Z")
    _clip(
        conn,
        tmp_path,
        message_id=2,
        timestamp="2026-09-19T18:10:00Z",
        label="animal",
    )
    _clip(conn, tmp_path, message_id=3, timestamp="2026-09-19T18:20:00Z")
    debug = tmp_path / "debug" / "cam01"
    debug.mkdir(parents=True)
    old = debug / "1-aaaaaaaaaaaa-r1.mp4"
    animal_old = debug / "2-aaaaaaaaaaaa-r1.mp4"
    animal_new = debug / "2-bbbbbbbbbbbb-r2.mp4"
    latest = debug / "3-aaaaaaaaaaaa-r1.mp4"
    unknown = debug / "operator-copy.mp4"
    for path in (old, animal_old, animal_new, latest, unknown):
        path.write_bytes(b"debug")
    animal_old.touch()
    animal_new.touch()

    result = clean_debug_cache(conn, debug_root=tmp_path / "debug")

    assert result.scanned == 4
    assert result.kept == 2
    assert result.deleted == 2
    assert not old.exists() and not animal_old.exists()
    assert animal_new.exists() and latest.exists() and unknown.exists()
