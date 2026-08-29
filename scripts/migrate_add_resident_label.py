"""One-off migration: widen `labels.label`'s CHECK constraint to allow `resident`.

No structural change (`schema_version` stays 5) -- just adds one more allowed value
to an existing column. SQLite can't ALTER a CHECK constraint in place, so this does
the same rename/recreate/copy-back dance as scripts/migrate_schema_v5.py, without any
row transformation (every existing row is copied through unchanged).

Added 2026-08-29 after an audit found `incident`-labelled cam01b clips that were
actually just residents/authorized people, not guards or a security event -- see
docs/plan.md's Ground truth labels section. Run once, then relabel those specific
rows with `db.upsert_label(..., label="resident")`.

Run:
    uv run python scripts/migrate_add_resident_label.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_app_config  # noqa: E402
from src.db import _SCHEMA_SQL, VALID_LABELS  # noqa: E402


def migrate(conn: sqlite3.Connection) -> dict[str, int]:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(labels)")}
    if "resident" not in VALID_LABELS:
        raise RuntimeError("src.db.VALID_LABELS no longer includes 'resident' -- stale script")
    already_widened = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='labels'"
    ).fetchone()["sql"]
    if "'resident'" in already_widened:
        return {"already_migrated": 1}
    if not columns:
        raise RuntimeError("labels table not found")

    rows = list(conn.execute("SELECT * FROM labels"))
    conn.execute("BEGIN")
    try:
        conn.execute("ALTER TABLE labels RENAME TO labels_pre_resident_backup")
        labels_sql = next(
            stmt
            for stmt in _SCHEMA_SQL.split(";\n\n")
            if "CREATE TABLE IF NOT EXISTS labels" in stmt
        )
        conn.execute(labels_sql)
        conn.executemany(
            """
            INSERT INTO labels (channel_id, message_id, label, startup_state, notes, labeled_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["channel_id"],
                    r["message_id"],
                    r["label"],
                    r["startup_state"],
                    r["notes"],
                    r["labeled_at"],
                )
                for r in rows
            ],
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return {"rows_copied": len(rows)}


def main() -> None:  # pragma: no cover - thin CLI wrapper
    app_cfg = load_app_config(require_telegram=False)
    conn = sqlite3.connect(str(app_cfg.db_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        stats = migrate(conn)
    finally:
        conn.close()
    print(stats)


if __name__ == "__main__":
    main()
