"""Add staged alert notifications and migrate schema v8 to v9."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_app_config  # noqa: E402

_DELIVERIES_SQL = """
CREATE TABLE live_deliveries (
    event_key TEXT NOT NULL REFERENCES live_events (event_key),
    transport TEXT NOT NULL,
    notification_kind TEXT NOT NULL CHECK (
        notification_kind IN ('preliminary', 'final', 'resolution')
    ),
    category TEXT NOT NULL,
    reason TEXT NOT NULL,
    resolution_state TEXT NOT NULL CHECK (
        resolution_state IN (
            'pending', 'confirmed', 'conflicting', 'likely_resolved', 'unconfirmed'
        )
    ),
    representative_channel_id TEXT NOT NULL,
    representative_message_id INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'sending', 'delivered', 'failed', 'ambiguous')
    ),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    last_error TEXT,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (event_key, transport, notification_kind)
)
"""


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def migrate(conn: sqlite3.Connection) -> dict[str, int]:
    version = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()[0]
    if version == 9:
        return {"already_migrated": 1}
    if version != 8:
        raise RuntimeError(f"expected schema_version 8, found {version}")

    conn.execute("BEGIN")
    try:
        if "resolution_state" not in _columns(conn, "live_events"):
            conn.execute(
                """
                ALTER TABLE live_events ADD COLUMN resolution_state TEXT CHECK (
                    resolution_state IS NULL OR resolution_state IN (
                        'confirmed', 'conflicting', 'likely_resolved', 'unconfirmed'
                    )
                )
                """
            )
        conn.execute(
            """
            UPDATE live_events SET resolution_state = 'confirmed'
            WHERE status = 'finalized' AND resolution_state IS NULL
            """
        )

        if "notification_kind" not in _columns(conn, "live_deliveries"):
            conn.execute("ALTER TABLE live_deliveries RENAME TO live_deliveries_v8")
            conn.execute("DROP INDEX IF EXISTS idx_live_deliveries_due")
            conn.execute(_DELIVERIES_SQL)
            conn.execute(
                """
                INSERT INTO live_deliveries (
                    event_key, transport, notification_kind, category, reason,
                    resolution_state, representative_channel_id,
                    representative_message_id, status, attempt_count,
                    next_attempt_at, last_error, updated_at
                )
                SELECT d.event_key, d.transport, 'final',
                       COALESCE(e.final_category, 'unclassified'),
                       COALESCE(e.final_reason, 'legacy_delivery'),
                       COALESCE(e.resolution_state, 'confirmed'),
                       e.representative_channel_id, e.representative_message_id,
                       d.status, d.attempt_count, d.next_attempt_at,
                       d.last_error, d.updated_at
                FROM live_deliveries_v8 d JOIN live_events e USING (event_key)
                WHERE e.representative_channel_id IS NOT NULL
                  AND e.representative_message_id IS NOT NULL
                """
            )
            conn.execute("DROP TABLE live_deliveries_v8")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_deliveries_due
            ON live_deliveries (status, next_attempt_at)
            """
        )
        conn.execute("UPDATE schema_version SET version = 9")
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
