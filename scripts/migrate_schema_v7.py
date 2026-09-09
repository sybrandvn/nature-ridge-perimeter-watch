"""One-off migration: schema v6 -> v7, widening backtest_results.predicted_class
again to accept the new `neighbour_candidate` category (src/classify.py,
2026-09-09 -- see docs/detection_improvement_review.md section 2.2).

v6 widened this same CHECK constraint once already (guard_side/outside_alert/
outside_priority/ambiguous plus the eight *_candidate/no_motion/unclassified
categories classify() emitted at the time). scripts/migrate_schema_v6.py could
drop and recreate the table outright because it was empty when that migration
ran. It is NOT empty now (a real backtest run has since been recorded), so this
migration follows the OTHER established pattern instead (see
scripts/migrate_schema_v5.py): rename the old table, recreate it from the
CURRENT schema script (which already has the widened CHECK), copy every row
across unchanged (no data transform needed -- only the constraint changed, no
existing row's predicted_class is `neighbour_candidate` since that category did
not exist before this migration), and keep the old table as
`backtest_results_v6_backup` rather than dropping it.

Does not touch `clips`, `labels`, `system_events`, `blob_tracks`, or
`backtest_runs` -- only `backtest_results` changes shape.

Run:
    uv run python scripts/migrate_schema_v7.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_app_config  # noqa: E402
from src.db import _SCHEMA_SQL  # noqa: E402


def migrate(conn: sqlite3.Connection) -> dict[str, int | str]:
    version = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()[0]
    if version == 7:
        return {"already_migrated": 1}
    if version != 6:
        raise RuntimeError(f"expected schema_version 6, found {version}")

    (row_count,) = conn.execute("SELECT COUNT(*) FROM backtest_results").fetchone()

    # Recreate backtest_results (plus its index) from the CURRENT schema
    # script, which already carries the widened CHECK constraint. Split into
    # individual statements and run each via conn.execute(), NOT
    # executescript(): executescript() implicitly commits any pending
    # transaction before it runs anything, which would silently break the
    # BEGIN/COMMIT wrapping below and leave no transaction for ROLLBACK to
    # act on if a later statement failed. Same discipline as
    # scripts/migrate_schema_v6.py, which this otherwise differs from (that
    # one refused to run on a non-empty table; this one instead copies the
    # data across, matching scripts/migrate_schema_v5.py's pattern).
    backtest_results_block = next(
        stmt
        for stmt in _SCHEMA_SQL.split(";\n\n")
        if "CREATE TABLE IF NOT EXISTS backtest_results" in stmt
    )
    create_statements = [s.strip() for s in backtest_results_block.split(";") if s.strip()]

    conn.execute("BEGIN")
    try:
        conn.execute("ALTER TABLE backtest_results RENAME TO backtest_results_v6_backup")
        for statement in create_statements:
            conn.execute(statement)
        if row_count:
            conn.execute(
                """
                INSERT INTO backtest_results
                    (id, run_id, channel_id, message_id, camera_id,
                     predicted_class, reason_codes_json, features_json)
                SELECT id, run_id, channel_id, message_id, camera_id,
                       predicted_class, reason_codes_json, features_json
                FROM backtest_results_v6_backup
                """
            )
        conn.execute("UPDATE schema_version SET version = 7")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return {"migrated": 1, "backtest_results_rows_copied": row_count}


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
