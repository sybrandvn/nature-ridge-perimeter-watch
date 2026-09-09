"""One-off migration: schema v5 -> v6, widening backtest_results.predicted_class.

v5's `backtest_results.predicted_class` CHECK constraint only accepted the
four-class routing vocabulary docs/plan.md step 25 originally specified
(guard_side/outside_alert/outside_priority/ambiguous) -- a vocabulary that was
never built. src.classify.classify() emits eight different categories
(guard_candidate, animal_candidate, incident_candidate, environment_candidate,
insect_candidate, resident_candidate, unclassified, no_motion), so nothing
could actually call db.add_backtest_result() with a real classify() output
without hitting the CHECK constraint.

v6 widens the constraint to accept both vocabularies (see src/db.py's
VALID_PREDICTIONS comment for why this widens rather than maps one onto the
other -- short version: mapping eight measured categories onto four routing
classes would encode an alerting policy nobody has validated, and picking one
is gated on Ship readiness criterion #3 in docs/plan.md, still unset).

SQLite cannot ALTER a CHECK constraint in place, so this drops and recreates
just the `backtest_results` table from the CURRENT schema script (which
already has the widened CHECK) plus its index, then bumps schema_version.
Safe to do as a drop because this migration REFUSES to run if the table has
any rows -- verified at runtable time, not assumed from this docstring.

Does not touch `clips`, `labels`, `system_events`, `blob_tracks`, or
`backtest_runs` -- only `backtest_results` changes shape.

Run:
    uv run python scripts/migrate_schema_v6.py
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
    if version == 6:
        return {"already_migrated": 1}
    if version != 5:
        raise RuntimeError(f"expected schema_version 5, found {version}")

    (row_count,) = conn.execute("SELECT COUNT(*) FROM backtest_results").fetchone()
    if row_count:
        raise RuntimeError(
            f"backtest_results has {row_count} row(s) -- this migration drops and "
            "recreates that table, which is only safe while it is empty. Export "
            "whatever is in it first (nothing in this repo has written to it yet, "
            "so if this fires, something new did)."
        )

    # Recreate backtest_results (plus its index) from the CURRENT schema
    # script, which already carries the widened CHECK constraint. Split into
    # individual statements and run each via conn.execute(), NOT
    # executescript(): executescript() implicitly commits any pending
    # transaction before it runs anything, which would silently break the
    # BEGIN/COMMIT wrapping below and leave no transaction for ROLLBACK to
    # act on if a later statement failed.
    backtest_results_block = next(
        stmt
        for stmt in _SCHEMA_SQL.split(";\n\n")
        if "CREATE TABLE IF NOT EXISTS backtest_results" in stmt
    )
    statements = [s.strip() for s in backtest_results_block.split(";") if s.strip()]

    conn.execute("BEGIN")
    try:
        conn.execute("DROP TABLE backtest_results")
        for statement in statements:
            conn.execute(statement)
        conn.execute("UPDATE schema_version SET version = 6")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return {"migrated": 1, "backtest_results_rows_before": row_count}


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
