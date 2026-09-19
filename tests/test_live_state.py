from datetime import UTC, datetime, timedelta

from src import db, live_state


def _now():
    return datetime(2026, 9, 19, 18, 0, tzinfo=UTC)


def test_message_claim_is_idempotent_and_failed_work_retries(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    assert live_state.begin_message(conn, "source", 1) is True
    assert live_state.begin_message(conn, "source", 1) is False
    live_state.finish_message(conn, "source", 1, status="failed", error="download")
    assert live_state.begin_message(conn, "source", 1) is True
    live_state.finish_message(conn, "source", 1, status="processed", event_key="cam01|x")
    assert live_state.begin_message(conn, "source", 1) is False


def test_event_finalization_enqueues_transports_once(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    now = _now()
    db.upsert_clip(
        conn,
        channel_id="source",
        message_id=1,
        camera_id="cam01",
        timestamp="2026-09-19T18:00:00Z",
        caption="Initial",
        file_path="clip.mp4",
        source="live",
    )
    live_state.add_analyzed_clip(
        conn,
        event_key="cam01|x",
        camera_id="cam01",
        deadline_at=now,
        channel_id="source",
        message_id=1,
        phase="initial",
        category="animal_candidate",
        reason="outside_colour",
        blinding_foreground=False,
        features={"x": 1.0},
    )
    assert live_state.due_event_keys(conn, now) == ["cam01|x"]
    assert live_state.finalize_event(
        conn,
        event_key="cam01|x",
        category="animal_candidate",
        reason="outside_colour",
        representative_channel_id="source",
        representative_message_id=1,
        transports=("telegram", "ntfy"),
        now=now,
    )
    assert not live_state.finalize_event(
        conn,
        event_key="cam01|x",
        category="animal_candidate",
        reason="outside_colour",
        representative_channel_id="source",
        representative_message_id=1,
        transports=("telegram",),
        now=now,
    )
    assert {row["transport"] for row in live_state.due_deliveries(conn, now)} == {
        "telegram",
        "ntfy",
    }


def test_interrupted_send_becomes_ambiguous_not_retried(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    now = _now()
    live_state.add_analyzed_clip(
        conn,
        event_key="cam01|x",
        camera_id="cam01",
        deadline_at=now,
        channel_id="source",
        message_id=1,
        phase="complete",
        category="incident_candidate",
        reason="outside_no_colour",
        blinding_foreground=False,
        features=None,
    )
    live_state.finalize_event(
        conn,
        event_key="cam01|x",
        category="incident_candidate",
        reason="outside_no_colour",
        representative_channel_id="source",
        representative_message_id=1,
        transports=("telegram",),
        now=now,
    )
    assert live_state.claim_delivery(conn, "cam01|x", "telegram")
    assert live_state.recover_interrupted(conn) == 1
    assert live_state.due_deliveries(conn, now + timedelta(days=1)) == []
    row = conn.execute("SELECT * FROM live_deliveries").fetchone()
    assert row["status"] == "ambiguous"
