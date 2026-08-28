import sqlite3

import pytest

from src import db
from src.errors import DbError

CHANNEL = "source_channel"


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    yield connection
    connection.close()


def test_connect_sets_pragmas_and_creates_parent_dir(tmp_path):
    nested = tmp_path / "nested" / "dir" / "test.db"
    connection = db.connect(nested)
    try:
        assert nested.exists()
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()


def test_init_schema_is_idempotent(conn):
    db.init_schema(conn)
    db.init_schema(conn)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    assert row["version"] == db.SCHEMA_VERSION


def test_schema_version_mismatch_raises(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    connection.execute("UPDATE schema_version SET version = 999")
    connection.close()

    reopened = sqlite3.connect(str(tmp_path / "test.db"))
    reopened.row_factory = sqlite3.Row
    with pytest.raises(DbError, match="schema_version"):
        db.init_schema(reopened)
    reopened.close()


# --------------------------------------------------------------------------
# clips
# --------------------------------------------------------------------------


def test_upsert_and_get_clip_round_trip(conn):
    db.upsert_clip(
        conn,
        channel_id=CHANNEL,
        message_id=1,
        camera_id="cam01",
        timestamp="2026-01-01T20:00:00Z",
        caption="Camera 1 motion",
        file_path="data/history/cam01/1.mp4",
        source="backfill",
    )
    row = db.get_clip(conn, CHANNEL, 1)
    assert row["camera_id"] == "cam01"
    assert row["source"] == "backfill"


def test_upsert_clip_updates_existing_row(conn):
    db.upsert_clip(
        conn,
        channel_id=CHANNEL,
        message_id=1,
        camera_id="cam01",
        timestamp="2026-01-01T20:00:00Z",
        caption=None,
        file_path=None,
        source="backfill",
    )
    db.upsert_clip(
        conn,
        channel_id=CHANNEL,
        message_id=1,
        camera_id="cam01",
        timestamp="2026-01-01T20:00:00Z",
        caption=None,
        file_path="data/history/cam01/1.mp4",
        source="backfill",
    )
    rows = list(db.iter_clips(conn))
    assert len(rows) == 1
    assert rows[0]["file_path"] == "data/history/cam01/1.mp4"


def test_upsert_clip_rejects_invalid_source(conn):
    with pytest.raises(DbError, match="source"):
        db.upsert_clip(
            conn,
            channel_id=CHANNEL,
            message_id=1,
            camera_id="cam01",
            timestamp="2026-01-01T20:00:00Z",
            caption=None,
            file_path=None,
            source="bogus",
        )


def test_iter_clips_filters_by_camera(conn):
    for camera_id, message_id in (("cam01", 1), ("cam02", 2)):
        db.upsert_clip(
            conn,
            channel_id=CHANNEL,
            message_id=message_id,
            camera_id=camera_id,
            timestamp="2026-01-01T20:00:00Z",
            caption=None,
            file_path=None,
            source="backfill",
        )
    rows = list(db.iter_clips(conn, camera_id="cam01"))
    assert [r["message_id"] for r in rows] == [1]


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def test_upsert_label_rejects_invalid_value(conn):
    with pytest.raises(DbError, match="label"):
        db.upsert_label(conn, channel_id=CHANNEL, message_id=1, label="bogus")


def test_label_round_trip_and_export_import(conn, tmp_path):
    db.upsert_label(conn, channel_id=CHANNEL, message_id=1, label="guard")
    db.upsert_label(conn, channel_id=CHANNEL, message_id=2, label="incident", notes="crawler")

    export_path = tmp_path / "labels.jsonl"
    count = db.export_labels_jsonl(conn, export_path)
    assert count == 2

    fresh = db.connect(tmp_path / "rebuilt.db")
    imported = db.import_labels_jsonl(fresh, export_path)
    assert imported == 2
    row = db.get_label(fresh, CHANNEL, 2)
    assert row["label"] == "incident"
    assert row["notes"] == "crawler"
    fresh.close()


def test_upsert_label_overwrites_previous_label(conn):
    db.upsert_label(conn, channel_id=CHANNEL, message_id=1, label="unknown")
    db.upsert_label(conn, channel_id=CHANNEL, message_id=1, label="animal")
    row = db.get_label(conn, CHANNEL, 1)
    assert row["label"] == "animal"


# --------------------------------------------------------------------------
# system_events
# --------------------------------------------------------------------------


def test_system_event_round_trip(conn):
    db.upsert_system_event(
        conn,
        channel_id=CHANNEL,
        message_id=10,
        timestamp="2026-01-01T20:00:00Z",
        camera_id="cam01",
        event_type="battery_dead",
        raw_text="Cam 1 battery critical",
    )
    rows = list(db.iter_system_events(conn))
    assert len(rows) == 1
    assert rows[0]["event_type"] == "battery_dead"


# --------------------------------------------------------------------------
# feature cache (blob_tracks)
# --------------------------------------------------------------------------


def test_feature_cache_hit_and_miss(conn):
    db.put_cached_track(
        conn,
        channel_id=CHANNEL,
        message_id=1,
        extractor_version="v1",
        motion_fingerprint="fp-a",
        features={"aspect_ratio": 2.1},
    )
    hit = db.get_cached_track(
        conn, channel_id=CHANNEL, message_id=1, extractor_version="v1", motion_fingerprint="fp-a"
    )
    assert hit == {"aspect_ratio": 2.1}

    # Different motion fingerprint (extraction settings changed) -> cache miss.
    miss = db.get_cached_track(
        conn, channel_id=CHANNEL, message_id=1, extractor_version="v1", motion_fingerprint="fp-b"
    )
    assert miss is None


def test_feature_cache_upsert_overwrites(conn):
    db.put_cached_track(
        conn,
        channel_id=CHANNEL,
        message_id=1,
        extractor_version="v1",
        motion_fingerprint="fp-a",
        features={"aspect_ratio": 1.0},
    )
    db.put_cached_track(
        conn,
        channel_id=CHANNEL,
        message_id=1,
        extractor_version="v1",
        motion_fingerprint="fp-a",
        features={"aspect_ratio": 2.0},
    )
    hit = db.get_cached_track(
        conn, channel_id=CHANNEL, message_id=1, extractor_version="v1", motion_fingerprint="fp-a"
    )
    assert hit == {"aspect_ratio": 2.0}


# --------------------------------------------------------------------------
# backtest runs / results
# --------------------------------------------------------------------------


def test_backtest_run_lifecycle_and_isolation(conn):
    db.create_run(
        conn,
        run_id="run-1",
        thresholds_path="config/thresholds.yaml",
        thresholds_hash="hash1",
        cameras_path="config/cameras.yaml",
        cameras_hash="hash2",
    )
    db.add_backtest_result(
        conn,
        run_id="run-1",
        channel_id=CHANNEL,
        message_id=1,
        camera_id="cam01",
        predicted_class="far_side_alert",
        reason_codes=["far_side_pixel_fraction"],
        features={"aspect_ratio": 2.0},
    )
    db.finish_run(conn, "run-1", status="completed")

    db.create_run(
        conn,
        run_id="run-2",
        thresholds_path="config/thresholds.yaml",
        thresholds_hash="hash3",
        cameras_path="config/cameras.yaml",
        cameras_hash="hash2",
    )
    db.add_backtest_result(
        conn,
        run_id="run-2",
        channel_id=CHANNEL,
        message_id=1,
        camera_id="cam01",
        predicted_class="guard_side",
        reason_codes=["near_side"],
        features={"aspect_ratio": 2.0},
    )

    run1_results = list(db.iter_backtest_results(conn, "run-1"))
    run2_results = list(db.iter_backtest_results(conn, "run-2"))
    assert len(run1_results) == 1
    assert len(run2_results) == 1
    assert run1_results[0]["predicted_class"] == "far_side_alert"
    assert run2_results[0]["predicted_class"] == "guard_side"

    runs = {r["run_id"]: r["status"] for r in db.iter_backtest_runs(conn)}
    assert runs == {"run-1": "completed", "run-2": "running"}


def test_add_backtest_result_rejects_invalid_prediction(conn):
    db.create_run(
        conn,
        run_id="run-1",
        thresholds_path="x",
        thresholds_hash="x",
        cameras_path="x",
        cameras_hash="x",
    )
    with pytest.raises(DbError, match="predicted_class"):
        db.add_backtest_result(
            conn,
            run_id="run-1",
            channel_id=CHANNEL,
            message_id=1,
            camera_id="cam01",
            predicted_class="bogus",
            reason_codes=[],
            features={},
        )


def test_finish_run_rejects_invalid_status(conn):
    db.create_run(
        conn,
        run_id="run-1",
        thresholds_path="x",
        thresholds_hash="x",
        cameras_path="x",
        cameras_hash="x",
    )
    with pytest.raises(DbError, match="status"):
        db.finish_run(conn, "run-1", status="bogus")
