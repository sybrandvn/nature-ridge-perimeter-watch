import sqlite3

from scripts.migrate_schema_v8 import migrate
from src import db


def test_migrate_v7_adds_live_tables(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(db._SCHEMA_SQL)
    conn.execute("INSERT INTO schema_version (version) VALUES (7)")
    result = migrate(conn)
    assert result == {"migrated": 1}
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 8
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"live_messages", "live_events", "live_event_clips", "live_deliveries"} <= tables
    assert migrate(conn) == {"already_migrated": 1}
