import sqlite3

from scripts.migrate_schema_v9 import migrate


def test_migrate_v8_preserves_delivery_as_final_notification(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.db", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (8);
        CREATE TABLE live_events (
            event_key TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            status TEXT NOT NULL,
            deadline_at TEXT NOT NULL,
            final_category TEXT,
            final_reason TEXT,
            representative_channel_id TEXT,
            representative_message_id INTEGER,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE live_deliveries (
            event_key TEXT NOT NULL REFERENCES live_events (event_key),
            transport TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TEXT NOT NULL,
            last_error TEXT,
            updated_at TEXT,
            PRIMARY KEY (event_key, transport)
        );
        CREATE INDEX idx_live_deliveries_due
            ON live_deliveries (status, next_attempt_at);
        INSERT INTO live_events VALUES (
            'cam01|x', 'cam01', 'finalized', '2026-09-20T00:00:00Z',
            'incident_candidate', 'outside_no_colour', 'source', 42, NULL, NULL
        );
        INSERT INTO live_deliveries VALUES (
            'cam01|x', 'telegram', 'pending', 0,
            '2026-09-20T00:00:00Z', NULL, '2026-09-20T00:00:00Z'
        );
        """
    )

    assert migrate(conn) == {"migrated": 1}
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 9
    event = conn.execute("SELECT * FROM live_events").fetchone()
    assert event["resolution_state"] == "confirmed"
    delivery = conn.execute("SELECT * FROM live_deliveries").fetchone()
    assert delivery["notification_kind"] == "final"
    assert delivery["category"] == "incident_candidate"
    assert delivery["representative_message_id"] == 42
    assert migrate(conn) == {"already_migrated": 1}
