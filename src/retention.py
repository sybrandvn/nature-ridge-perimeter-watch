"""Aggressive, retrieval-safe cleanup of locally downloaded source videos."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src import db

URGENT_CATEGORIES = frozenset({"animal_candidate", "incident_candidate"})
URGENT_LABELS = frozenset({"animal", "incident"})


@dataclass(frozen=True)
class RetentionResult:
    scanned: int
    kept: int
    deleted: int
    missing: int
    unsafe: int


def _managed_path(raw: str, data_root: Path) -> Path | None:
    path = Path(raw)
    resolved = (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()
    root = data_root.resolve()
    return resolved if resolved.is_relative_to(root) else None


def clean_local_media(conn: sqlite3.Connection, *, data_root: Path) -> RetentionResult:
    """Keep urgent evidence and the newest existing video for each camera."""
    rows = list(
        conn.execute(
            """
            SELECT c.channel_id, c.message_id, c.camera_id, c.timestamp, c.file_path,
                   l.label, lec.category AS clip_category,
                   le.status AS event_status, le.final_category
            FROM clips c
            LEFT JOIN labels l
              ON l.channel_id = c.channel_id AND l.message_id = c.message_id
            LEFT JOIN live_event_clips lec
              ON lec.channel_id = c.channel_id AND lec.message_id = c.message_id
            LEFT JOIN live_events le ON le.event_key = lec.event_key
            WHERE c.file_path IS NOT NULL
            ORDER BY c.timestamp DESC, c.message_id DESC
            """
        )
    )
    resolved: dict[tuple[str, int], Path | None] = {}
    newest_by_camera: dict[str, tuple[str, int]] = {}
    protected: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row["channel_id"]), int(row["message_id"]))
        path = _managed_path(str(row["file_path"]), data_root)
        resolved[key] = path
        if path is not None and path.is_file() and str(row["camera_id"]) not in newest_by_camera:
            newest_by_camera[str(row["camera_id"])] = key
        if (
            row["label"] in URGENT_LABELS
            or row["clip_category"] in URGENT_CATEGORIES
            or row["event_status"] == "pending"
            or row["final_category"] in URGENT_CATEGORIES
        ):
            protected.add(key)

    keep = protected | set(newest_by_camera.values())
    deleted = missing = unsafe = kept = 0
    kept_paths = {resolved[key] for key in keep if resolved.get(key) is not None}
    for row in rows:
        key = (str(row["channel_id"]), int(row["message_id"]))
        path = resolved[key]
        if key in keep or (path is not None and path in kept_paths):
            kept += 1
            continue
        if path is None:
            unsafe += 1
            continue
        if path.is_file():
            path.unlink()
            deleted += 1
        else:
            missing += 1
        db.clear_clip_file_path(conn, channel_id=key[0], message_id=key[1])
    return RetentionResult(
        scanned=len(rows), kept=kept, deleted=deleted, missing=missing, unsafe=unsafe
    )
