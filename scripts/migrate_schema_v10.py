"""Add the durable system-event notification outbox and migrate v9 to v10."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_app_config  # noqa: E402


def migrate(conn: sqlite3.Connection) -> dict[str, int]:
    version = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()[0]
    if version == 10:
        return {"already_migrated": 1}
    if version != 9:
        raise RuntimeError(f"expected schema_version 9, found {version}")

    conn.execute("BEGIN")
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_deliveries (
                channel_id TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                transport TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('pending', 'sending', 'delivered', 'failed', 'ambiguous')
                ),
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL,
                last_error TEXT,
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                PRIMARY KEY (channel_id, message_id, transport),
                FOREIGN KEY (channel_id, message_id)
                    REFERENCES system_events (channel_id, message_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_system_deliveries_due
            ON system_deliveries (status, next_attempt_at)
            """
        )
        conn.execute("UPDATE schema_version SET version = 10")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return {"migrated": 1}


def main() -> None:  # pragma: no cover
    cfg = load_app_config(require_telegram=False)
    conn = sqlite3.connect(str(cfg.db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        print(migrate(conn))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
