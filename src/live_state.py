"""Durable state transitions for the live Telegram watcher."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime


def utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def message_status(conn: sqlite3.Connection, channel_id: str, message_id: int) -> str | None:
    row = conn.execute(
        "SELECT status FROM live_messages WHERE channel_id = ? AND message_id = ?",
        (channel_id, message_id),
    ).fetchone()
    return None if row is None else str(row["status"])


def begin_message(conn: sqlite3.Connection, channel_id: str, message_id: int) -> bool:
    """Claim an unseen or previously failed message for processing."""
    row = conn.execute(
        "SELECT status FROM live_messages WHERE channel_id = ? AND message_id = ?",
        (channel_id, message_id),
    ).fetchone()
    if row is not None and row["status"] in ("processing", "processed", "ignored"):
        return False
    conn.execute(
        """
        INSERT INTO live_messages (channel_id, message_id, status, error)
        VALUES (?, ?, 'processing', NULL)
        ON CONFLICT (channel_id, message_id) DO UPDATE SET
            status = 'processing', error = NULL,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        (channel_id, message_id),
    )
    return True


def finish_message(
    conn: sqlite3.Connection,
    channel_id: str,
    message_id: int,
    *,
    status: str,
    event_key: str | None = None,
    error: str | None = None,
) -> None:
    if status not in ("processed", "ignored", "failed"):
        raise ValueError(f"invalid final message status: {status}")
    conn.execute(
        """
        UPDATE live_messages SET status = ?, event_key = ?, error = ?,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE channel_id = ? AND message_id = ?
        """,
        (status, event_key, error, channel_id, message_id),
    )


def add_analyzed_clip(
    conn: sqlite3.Connection,
    *,
    event_key: str,
    camera_id: str,
    deadline_at: datetime,
    channel_id: str,
    message_id: int,
    phase: str,
    category: str,
    reason: str,
    blinding_foreground: bool,
    features: dict[str, float] | None,
) -> None:
    deadline = utc_text(deadline_at)
    conn.execute(
        """
        INSERT INTO live_events (event_key, camera_id, status, deadline_at)
        VALUES (?, ?, 'pending', ?)
        ON CONFLICT (event_key) DO UPDATE SET
            status = CASE
                WHEN live_events.status = 'finalized'
                 AND COALESCE(live_events.final_category, '') NOT IN
                     ('animal_candidate', 'incident_candidate')
                 AND ? IN ('animal_candidate', 'incident_candidate')
                THEN 'pending' ELSE live_events.status
            END,
            deadline_at = CASE
                WHEN live_events.status = 'pending'
                  OR (COALESCE(live_events.final_category, '') NOT IN
                        ('animal_candidate', 'incident_candidate')
                      AND ? IN ('animal_candidate', 'incident_candidate'))
                THEN excluded.deadline_at
                ELSE live_events.deadline_at
            END,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        (event_key, camera_id, deadline, category, category),
    )
    conn.execute(
        """
        INSERT INTO live_event_clips
            (event_key, channel_id, message_id, phase, category, reason,
             blinding_foreground, features_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (channel_id, message_id) DO NOTHING
        """,
        (
            event_key,
            channel_id,
            message_id,
            phase,
            category,
            reason,
            int(blinding_foreground),
            None if features is None else json.dumps(features, sort_keys=True),
        ),
    )


def due_event_keys(conn: sqlite3.Connection, now: datetime) -> list[str]:
    return [
        str(row["event_key"])
        for row in conn.execute(
            "SELECT event_key FROM live_events WHERE status = 'pending' AND deadline_at <= ?",
            (utc_text(now),),
        )
    ]


def event_clips(conn: sqlite3.Connection, event_key: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT live_event_clips.*, clips.file_path, clips.camera_id, clips.timestamp
            FROM live_event_clips
            LEFT JOIN clips USING (channel_id, message_id)
            WHERE event_key = ? ORDER BY message_id
            """,
            (event_key,),
        )
    )


def finalize_event(
    conn: sqlite3.Connection,
    *,
    event_key: str,
    category: str,
    reason: str,
    representative_channel_id: str,
    representative_message_id: int,
    transports: Iterable[str],
    now: datetime,
) -> bool:
    """Finalize once and enqueue each configured transport atomically."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        changed = conn.execute(
            """
            UPDATE live_events SET status = 'finalized', final_category = ?, final_reason = ?,
                representative_channel_id = ?, representative_message_id = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE event_key = ? AND status = 'pending'
            """,
            (
                category,
                reason,
                representative_channel_id,
                representative_message_id,
                event_key,
            ),
        ).rowcount
        if changed:
            for transport in transports:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO live_deliveries
                        (event_key, transport, status, next_attempt_at)
                    VALUES (?, ?, 'pending', ?)
                    """,
                    (event_key, transport, utc_text(now)),
                )
        conn.execute("COMMIT")
        return bool(changed)
    except Exception:
        conn.execute("ROLLBACK")
        raise


def recover_interrupted(conn: sqlite3.Connection) -> int:
    """Quarantine ambiguous sends and make interrupted message work retryable."""
    ambiguous = conn.execute(
        """
        UPDATE live_deliveries SET status = 'ambiguous',
            last_error = 'process stopped while transport call was in flight',
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE status = 'sending'
        """
    ).rowcount
    conn.execute(
        """
        UPDATE live_messages SET status = 'failed', error = 'process stopped during processing',
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE status = 'processing'
        """
    )
    return ambiguous


def due_deliveries(conn: sqlite3.Connection, now: datetime) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT d.*, e.final_category, e.final_reason,
                   e.representative_channel_id, e.representative_message_id,
                   c.file_path, c.camera_id, c.timestamp
            FROM live_deliveries d JOIN live_events e USING (event_key)
            LEFT JOIN clips c
              ON c.channel_id = e.representative_channel_id
             AND c.message_id = e.representative_message_id
            WHERE d.status IN ('pending', 'failed') AND d.next_attempt_at <= ?
            ORDER BY d.next_attempt_at
            """,
            (utc_text(now),),
        )
    )


def claim_delivery(conn: sqlite3.Connection, event_key: str, transport: str) -> bool:
    return bool(
        conn.execute(
            """
            UPDATE live_deliveries SET status = 'sending', attempt_count = attempt_count + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE event_key = ? AND transport = ? AND status IN ('pending', 'failed')
            """,
            (event_key, transport),
        ).rowcount
    )


def finish_delivery(
    conn: sqlite3.Connection,
    event_key: str,
    transport: str,
    *,
    delivered: bool,
    next_attempt_at: datetime,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE live_deliveries SET status = ?, next_attempt_at = ?, last_error = ?,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE event_key = ? AND transport = ? AND status = 'sending'
        """,
        (
            "delivered" if delivered else "failed",
            utc_text(next_attempt_at),
            error,
            event_key,
            transport,
        ),
    )
