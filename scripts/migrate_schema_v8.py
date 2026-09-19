"""Add durable live-watcher state tables and migrate schema v7 to v8."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_app_config  # noqa: E402
from src.db import _SCHEMA_SQL  # noqa: E402


def migrate(conn: sqlite3.Connection) -> dict[str, int]:
    version = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()[0]
    if version == 8:
        return {"already_migrated": 1}
    if version != 7:
        raise RuntimeError(f"expected schema_version 7, found {version}")
    conn.execute("BEGIN")
    try:
        for statement in _SCHEMA_SQL.split(";"):
            statement = statement.strip()
            if statement and (
                "live_messages" in statement
                or "live_events" in statement
                or "live_event_clips" in statement
                or "live_deliveries" in statement
            ):
                conn.execute(statement)
        conn.execute("UPDATE schema_version SET version = 8")
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
