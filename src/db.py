"""SQLite persistence: schema init and narrow repository functions.

No migration framework — schema uses CREATE TABLE IF NOT EXISTS plus a single
schema_version row. If a breaking schema change is ever needed, export labels
to JSONL first (the durability guarantee), delete the db file, and reimport.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.errors import DbError

SCHEMA_VERSION = 4

VALID_LABELS = (
    "guard",
    "animal",
    "incident",
    "environment",
    "unknown",
    "startup",
    "startup_clear",
    "startup_blank",
)
VALID_SOURCES = ("live", "backfill")
VALID_PREDICTIONS = ("guard_side", "outside_alert", "outside_priority", "ambiguous")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    camera_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    caption TEXT,
    file_path TEXT,
    source TEXT NOT NULL CHECK (source IN ('live', 'backfill')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (channel_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_clips_timestamp ON clips (timestamp);
CREATE INDEX IF NOT EXISTS idx_clips_camera_timestamp ON clips (camera_id, timestamp);

CREATE TABLE IF NOT EXISTS labels (
    channel_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    label TEXT NOT NULL CHECK (
        label IN (
            'guard', 'animal', 'incident', 'environment', 'unknown', 'startup',
            'startup_clear', 'startup_blank'
        )
    ),
    notes TEXT,
    labeled_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (channel_id, message_id)
);

CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    camera_id TEXT,
    event_type TEXT NOT NULL,
    raw_text TEXT,
    UNIQUE (channel_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events (timestamp);

CREATE TABLE IF NOT EXISTS blob_tracks (
    channel_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    extractor_version TEXT NOT NULL,
    motion_fingerprint TEXT NOT NULL,
    features_json TEXT NOT NULL,
    computed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (channel_id, message_id, extractor_version, motion_fingerprint)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    thresholds_path TEXT NOT NULL,
    thresholds_hash TEXT NOT NULL,
    cameras_path TEXT NOT NULL,
    cameras_hash TEXT NOT NULL,
    git_revision TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES backtest_runs (run_id),
    channel_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    camera_id TEXT NOT NULL,
    predicted_class TEXT NOT NULL CHECK (
        predicted_class IN ('guard_side', 'outside_alert', 'outside_priority', 'ambiguous')
    ),
    reason_codes_json TEXT NOT NULL,
    features_json TEXT NOT NULL,
    UNIQUE (run_id, channel_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_backtest_results_run ON backtest_results (run_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with WAL mode, foreign keys, and the schema initialized."""
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    elif row["version"] != SCHEMA_VERSION:
        raise DbError(
            f"Database schema_version {row['version']} does not match code "
            f"version {SCHEMA_VERSION}. Export labels to JSONL, delete the db "
            f"file, and reimport (see export_labels_jsonl/import_labels_jsonl)."
        )


# --------------------------------------------------------------------------
# clips
# --------------------------------------------------------------------------


def upsert_clip(
    conn: sqlite3.Connection,
    *,
    channel_id: str,
    message_id: int,
    camera_id: str,
    timestamp: str,
    caption: str | None,
    file_path: str | None,
    source: str,
) -> None:
    if source not in VALID_SOURCES:
        raise DbError(f"clips.source must be one of {VALID_SOURCES}, got {source!r}")
    conn.execute(
        """
        INSERT INTO clips (channel_id, message_id, camera_id, timestamp, caption, file_path, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (channel_id, message_id) DO UPDATE SET
            camera_id = excluded.camera_id,
            timestamp = excluded.timestamp,
            caption = excluded.caption,
            file_path = excluded.file_path,
            source = excluded.source
        """,
        (channel_id, message_id, camera_id, timestamp, caption, file_path, source),
    )


def get_clip(conn: sqlite3.Connection, channel_id: str, message_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM clips WHERE channel_id = ? AND message_id = ?",
        (channel_id, message_id),
    ).fetchone()


def set_clip_file_path(
    conn: sqlite3.Connection, *, channel_id: str, message_id: int, file_path: str
) -> None:
    """Record where a clip's video was downloaded to, without touching its other fields."""
    conn.execute(
        "UPDATE clips SET file_path = ? WHERE channel_id = ? AND message_id = ?",
        (file_path, channel_id, message_id),
    )


def iter_clips(
    conn: sqlite3.Connection, *, camera_id: str | None = None
) -> Iterator[sqlite3.Row]:
    if camera_id is None:
        yield from conn.execute("SELECT * FROM clips ORDER BY timestamp")
    else:
        yield from conn.execute(
            "SELECT * FROM clips WHERE camera_id = ? ORDER BY timestamp", (camera_id,)
        )


def iter_unlabeled_clips(
    conn: sqlite3.Connection,
    *,
    camera_id: str | None = None,
    with_file_only: bool = False,
) -> Iterator[sqlite3.Row]:
    """Clips with no row yet in `labels`, for driving a labeling CLI/session.

    `with_file_only=True` restricts to clips that already have a downloaded video
    (`file_path` set) -- most clips in the backfill don't, so an interactive labeling
    session should default to this or it walks thousands of unwatchable rows.
    """
    query = """
        SELECT clips.* FROM clips
        LEFT JOIN labels
            ON clips.channel_id = labels.channel_id AND clips.message_id = labels.message_id
        WHERE labels.message_id IS NULL
    """
    params: tuple[Any, ...] = ()
    if camera_id is not None:
        query += " AND clips.camera_id = ?"
        params = (*params, camera_id)
    if with_file_only:
        query += " AND clips.file_path IS NOT NULL"
    query += " ORDER BY clips.timestamp"
    yield from conn.execute(query, params)


def iter_clips_with_label(
    conn: sqlite3.Connection,
    label: str,
    *,
    camera_id: str | None = None,
    with_file_only: bool = False,
) -> Iterator[sqlite3.Row]:
    """Clips already carrying `label` in `labels`, for re-triaging a prior label into
    a finer-grained one (e.g. re-reviewing `startup` rows to split into
    `startup_clear`/`startup_blank`)."""
    query = """
        SELECT clips.* FROM clips
        JOIN labels
            ON clips.channel_id = labels.channel_id AND clips.message_id = labels.message_id
        WHERE labels.label = ?
    """
    params: tuple[Any, ...] = (label,)
    if camera_id is not None:
        query += " AND clips.camera_id = ?"
        params = (*params, camera_id)
    if with_file_only:
        query += " AND clips.file_path IS NOT NULL"
    query += " ORDER BY clips.timestamp"
    yield from conn.execute(query, params)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def upsert_label(
    conn: sqlite3.Connection,
    *,
    channel_id: str,
    message_id: int,
    label: str,
    notes: str | None = None,
) -> None:
    if label not in VALID_LABELS:
        raise DbError(f"label must be one of {VALID_LABELS}, got {label!r}")
    conn.execute(
        """
        INSERT INTO labels (channel_id, message_id, label, notes)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (channel_id, message_id) DO UPDATE SET
            label = excluded.label,
            notes = excluded.notes,
            labeled_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        (channel_id, message_id, label, notes),
    )


def get_label(conn: sqlite3.Connection, channel_id: str, message_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM labels WHERE channel_id = ? AND message_id = ?",
        (channel_id, message_id),
    ).fetchone()


def iter_labels(conn: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    yield from conn.execute("SELECT * FROM labels ORDER BY channel_id, message_id")


def export_labels_jsonl(conn: sqlite3.Connection, path: str | Path) -> int:
    """Write every label to JSONL. This is the durability guarantee for hand-entered
    ground truth: the database can be deleted and rebuilt without losing it."""
    count = 0
    with Path(path).open("w") as f:
        for row in iter_labels(conn):
            f.write(json.dumps(dict(row)) + "\n")
            count += 1
    return count


def import_labels_jsonl(conn: sqlite3.Connection, path: str | Path) -> int:
    count = 0
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            upsert_label(
                conn,
                channel_id=record["channel_id"],
                message_id=record["message_id"],
                label=record["label"],
                notes=record.get("notes"),
            )
            count += 1
    return count


# --------------------------------------------------------------------------
# system_events (camera health notifications)
# --------------------------------------------------------------------------


def upsert_system_event(
    conn: sqlite3.Connection,
    *,
    channel_id: str,
    message_id: int,
    timestamp: str,
    camera_id: str | None,
    event_type: str,
    raw_text: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO system_events
            (channel_id, message_id, timestamp, camera_id, event_type, raw_text)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (channel_id, message_id) DO UPDATE SET
            timestamp = excluded.timestamp,
            camera_id = excluded.camera_id,
            event_type = excluded.event_type,
            raw_text = excluded.raw_text
        """,
        (channel_id, message_id, timestamp, camera_id, event_type, raw_text),
    )


def iter_system_events(conn: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    yield from conn.execute("SELECT * FROM system_events ORDER BY timestamp")


# --------------------------------------------------------------------------
# feature cache (blob_tracks)
# --------------------------------------------------------------------------


def get_cached_track(
    conn: sqlite3.Connection,
    *,
    channel_id: str,
    message_id: int,
    extractor_version: str,
    motion_fingerprint: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT features_json FROM blob_tracks
        WHERE channel_id = ? AND message_id = ? AND extractor_version = ? AND motion_fingerprint = ?
        """,
        (channel_id, message_id, extractor_version, motion_fingerprint),
    ).fetchone()
    return json.loads(row["features_json"]) if row else None


def put_cached_track(
    conn: sqlite3.Connection,
    *,
    channel_id: str,
    message_id: int,
    extractor_version: str,
    motion_fingerprint: str,
    features: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO blob_tracks
            (channel_id, message_id, extractor_version, motion_fingerprint, features_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (channel_id, message_id, extractor_version, motion_fingerprint) DO UPDATE SET
            features_json = excluded.features_json,
            computed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        (channel_id, message_id, extractor_version, motion_fingerprint, json.dumps(features)),
    )


# --------------------------------------------------------------------------
# backtest runs / results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunHandles:
    run_id: str


def create_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    thresholds_path: str,
    thresholds_hash: str,
    cameras_path: str,
    cameras_hash: str,
    git_revision: str | None = None,
    notes: str | None = None,
) -> RunHandles:
    conn.execute(
        """
        INSERT INTO backtest_runs
            (run_id, thresholds_path, thresholds_hash, cameras_path, cameras_hash,
             git_revision, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, 'running', ?)
        """,
        (run_id, thresholds_path, thresholds_hash, cameras_path, cameras_hash, git_revision, notes),
    )
    return RunHandles(run_id=run_id)


def finish_run(conn: sqlite3.Connection, run_id: str, *, status: str) -> None:
    if status not in ("completed", "failed"):
        raise DbError(f"finish_run status must be 'completed' or 'failed', got {status!r}")
    conn.execute("UPDATE backtest_runs SET status = ? WHERE run_id = ?", (status, run_id))


def add_backtest_result(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    channel_id: str,
    message_id: int,
    camera_id: str,
    predicted_class: str,
    reason_codes: list[str],
    features: dict[str, Any],
) -> None:
    if predicted_class not in VALID_PREDICTIONS:
        raise DbError(
            f"predicted_class must be one of {VALID_PREDICTIONS}, got {predicted_class!r}"
        )
    conn.execute(
        """
        INSERT INTO backtest_results
            (run_id, channel_id, message_id, camera_id, predicted_class,
             reason_codes_json, features_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            channel_id,
            message_id,
            camera_id,
            predicted_class,
            json.dumps(reason_codes),
            json.dumps(features),
        ),
    )


def iter_backtest_results(conn: sqlite3.Connection, run_id: str) -> Iterator[sqlite3.Row]:
    yield from conn.execute(
        "SELECT * FROM backtest_results WHERE run_id = ? ORDER BY camera_id, message_id", (run_id,)
    )


def iter_backtest_runs(conn: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    yield from conn.execute("SELECT * FROM backtest_runs ORDER BY created_at")
