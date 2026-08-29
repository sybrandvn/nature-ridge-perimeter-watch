"""One-off migration: schema v4 -> v5, splitting `labels.label` into two axes.

v4 had a single `label` column mixing two different things: the event's class
(guard/animal/incident/environment/unknown) and a short/early clip's own visibility
(startup/startup_clear/startup_blank) -- see docs/handoff.md and the 2026-08-29
discussion of why that collided (a startup-family label silently discarded the
event's real class if the paired clip hadn't been reviewed).

v5 splits these into `label` (nullable, narrowed to the 5 real classes) and a new
`startup_state` (clear/blank/duplicate). This script transforms the existing v4
`labels` table in place:
  - guard/animal/incident/environment/unknown rows are carried over unchanged.
  - startup/startup_clear/startup_blank rows become startup_state=duplicate/clear/blank
    with label=NULL, UNLESS a sibling clip in the same event (same embedded alert
    timestamp) already has a real class, in which case that class is inherited.

Does not touch `clips`, `blob_tracks`, `backtest_runs`, or `backtest_results` --
only `labels` changes shape. The old table is kept as `labels_v4_backup` (not
dropped) as an extra safety net alongside the JSONL export this should be run after.

Run:
    uv run python scripts/migrate_schema_v5.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.label import _event_key  # noqa: E402
from src.config import load_app_config  # noqa: E402
from src.db import _SCHEMA_SQL  # noqa: E402

_STARTUP_STATE_BY_V4_LABEL = {
    "startup": "duplicate",
    "startup_clear": "clear",
    "startup_blank": "blank",
}
_REAL_CLASSES = {"guard", "animal", "incident", "environment", "unknown"}


def migrate(conn: sqlite3.Connection) -> dict[str, int]:
    version = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()[0]
    if version == 5:
        return {"already_migrated": 1}
    if version != 4:
        raise RuntimeError(f"expected schema_version 4, found {version}")

    clips_by_id = {
        (row["channel_id"], row["message_id"]): row
        for row in conn.execute("SELECT * FROM clips")
    }
    old_rows = list(conn.execute("SELECT * FROM labels"))

    # event_key -> real class, from whichever v4 row(s) already carry one.
    event_class: dict[str, str] = {}
    for row in old_rows:
        if row["label"] not in _REAL_CLASSES:
            continue
        clip = clips_by_id.get((row["channel_id"], row["message_id"]))
        if clip is None:
            continue
        key = _event_key(clip["camera_id"], clip["caption"])
        if key is not None:
            event_class[key] = row["label"]

    transformed = []
    stats = {"unchanged": 0, "startup_inherited": 0, "startup_pending": 0}
    for row in old_rows:
        v4_label = row["label"]
        if v4_label in _REAL_CLASSES:
            transformed.append((row["channel_id"], row["message_id"], v4_label, None, row["notes"]))
            stats["unchanged"] += 1
            continue
        startup_state = _STARTUP_STATE_BY_V4_LABEL[v4_label]
        clip = clips_by_id.get((row["channel_id"], row["message_id"]))
        key = _event_key(clip["camera_id"], clip["caption"]) if clip is not None else None
        label = event_class.get(key) if key is not None else None
        if label is not None:
            stats["startup_inherited"] += 1
        else:
            stats["startup_pending"] += 1
        transformed.append(
            (row["channel_id"], row["message_id"], label, startup_state, row["notes"])
        )

    conn.execute("BEGIN")
    try:
        conn.execute("ALTER TABLE labels RENAME TO labels_v4_backup")
        # Recreate just the (now-updated) labels table from the current schema script.
        labels_sql = next(
            stmt
            for stmt in _SCHEMA_SQL.split(";\n\n")
            if "CREATE TABLE IF NOT EXISTS labels" in stmt
        )
        conn.execute(labels_sql)
        conn.executemany(
            """
            INSERT INTO labels (channel_id, message_id, label, startup_state, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            transformed,
        )
        conn.execute("UPDATE schema_version SET version = 5")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return stats


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
