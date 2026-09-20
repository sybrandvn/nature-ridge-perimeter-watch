import sqlite3

from scripts.migrate_schema_v10 import migrate


def test_migrate_v9_adds_system_delivery_outbox(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.db", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (9);
        CREATE TABLE system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            message_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            camera_id TEXT,
            event_type TEXT NOT NULL,
            raw_text TEXT,
            UNIQUE (channel_id, message_id)
        );
        """
    )

    assert migrate(conn) == {"migrated": 1}
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 10
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(system_deliveries)")
    }
    assert {"channel_id", "message_id", "transport", "status", "next_attempt_at"} <= columns
    assert migrate(conn) == {"already_migrated": 1}
