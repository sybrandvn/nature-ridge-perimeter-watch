"""Aggressive, retrieval-safe cleanup of locally downloaded source videos."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src import db

URGENT_CATEGORIES = frozenset({"animal_candidate", "incident_candidate"})
URGENT_LABELS = frozenset({"animal", "incident"})
DEBUG_VIDEO_RE = re.compile(r"^(?P<message_id>\d+)-[0-9a-f]+-r\d+\.mp4$")


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


def _debug_keep_ids(conn: sqlite3.Connection) -> set[int]:
    """Return urgent clips plus the newest known clip for each camera."""
    rows = list(
        conn.execute(
            """
            SELECT c.message_id, c.camera_id, c.timestamp,
                   l.label, lec.category AS clip_category,
                   le.status AS event_status, le.final_category
            FROM clips c
            LEFT JOIN labels l
              ON l.channel_id = c.channel_id AND l.message_id = c.message_id
            LEFT JOIN live_event_clips lec
              ON lec.channel_id = c.channel_id AND lec.message_id = c.message_id
            LEFT JOIN live_events le ON le.event_key = lec.event_key
            ORDER BY c.timestamp DESC, c.message_id DESC
            """
        )
    )
    newest_by_camera: dict[str, int] = {}
    protected: set[int] = set()
    for row in rows:
        message_id = int(row["message_id"])
        newest_by_camera.setdefault(str(row["camera_id"]), message_id)
        if (
            row["label"] in URGENT_LABELS
            or row["clip_category"] in URGENT_CATEGORIES
            or row["event_status"] == "pending"
            or row["final_category"] in URGENT_CATEGORIES
        ):
            protected.add(message_id)
    return protected | set(newest_by_camera.values())


def clean_debug_cache(conn: sqlite3.Connection, *, debug_root: Path) -> RetentionResult:
    """Apply source-video retention policy to regenerable detector renders.

    Only cache files matching the renderer-owned filename format are managed.
    For a kept clip, retain the newest fingerprint/version and discard older
    variants. Temporary and unknown files are left alone to avoid racing an
    active render or deleting operator-owned material.
    """
    if not debug_root.exists():
        return RetentionResult(scanned=0, kept=0, deleted=0, missing=0, unsafe=0)
    keep_ids = _debug_keep_ids(conn)
    managed: list[tuple[Path, int]] = []
    for path in debug_root.rglob("*.mp4"):
        match = DEBUG_VIDEO_RE.fullmatch(path.name)
        if path.is_file() and match is not None:
            managed.append((path, int(match.group("message_id"))))

    newest_kept: dict[int, Path] = {}
    for path, message_id in managed:
        if message_id not in keep_ids:
            continue
        current = newest_kept.get(message_id)
        if current is None or (path.stat().st_mtime_ns, path.name) > (
            current.stat().st_mtime_ns,
            current.name,
        ):
            newest_kept[message_id] = path

    kept = deleted = 0
    for path, message_id in managed:
        if newest_kept.get(message_id) == path:
            kept += 1
        else:
            path.unlink()
            deleted += 1
    return RetentionResult(
        scanned=len(managed), kept=kept, deleted=deleted, missing=0, unsafe=0
    )
