"""Correct stored system-event types from their preserved source text."""

from __future__ import annotations

from src import db
from src.config import load_app_config
from src.message_parsing import classify_health_event


def reclassify(conn) -> dict[str, int]:
    counts = {"examined": 0, "updated": 0, "unrecognized": 0}
    rows = list(conn.execute("SELECT id, event_type, raw_text FROM system_events"))
    for row in rows:
        counts["examined"] += 1
        event_type = classify_health_event(str(row["raw_text"] or ""))
        if event_type is None:
            counts["unrecognized"] += 1
            continue
        if event_type != row["event_type"]:
            conn.execute(
                "UPDATE system_events SET event_type = ? WHERE id = ?", (event_type, row["id"])
            )
            counts["updated"] += 1
    conn.commit()
    return counts


def main() -> None:  # pragma: no cover - operator CLI
    config = load_app_config(require_telegram=False)
    conn = db.connect(config.db_path)
    try:
        print(reclassify(conn))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
