"""Role-restricted Telegram query bot backed by live watcher state."""

from __future__ import annotations

import asyncio
import io
import logging
import re
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.activity_reports import (
    ActivityReport,
    PatrolReport,
    build_activity_report,
    build_patrol_report,
    render_activity_report,
    render_patrol_report,
)
from src.config import AppConfig, CamerasConfig
from src.event_keys import event_phase, message_event_key
from src.message_parsing import classify_health_event
from src.sequence import ClipEvent, segment_passes

logger = logging.getLogger("query_bot")
LOCAL_ZONE = ZoneInfo("Africa/Johannesburg")
MEDIA_EVENT_CATEGORIES = ("incident", "animal", "resident", "neighbour")
HISTORY_USAGE = "Usage: /history <animal|incident|resident|neighbour> [page]"


@dataclass(frozen=True)
class MediaClip:
    channel_id: str
    message_id: int
    camera_id: str
    timestamp: str
    path: Path | None


@dataclass(frozen=True)
class HistoryEvent:
    category: str
    clip: MediaClip
    sibling_count: int
    resolution_state: str | None = None
    initial_category: str | None = None
    complete_category: str | None = None


def _parse_hhmm(value: str) -> time:
    hour, minute = (int(part) for part in value.split(":"))
    return time(hour, minute)


def operating_period(config: AppConfig, now: datetime) -> tuple[datetime, datetime, bool]:
    """Return the relevant overnight window in UTC and whether it is active."""
    local = now.astimezone(LOCAL_ZONE)
    start_time = _parse_hhmm(config.operating_window_start)
    end_time = _parse_hhmm(config.operating_window_end)
    active = local.time() >= start_time or local.time() < end_time
    if local.time() >= start_time:
        start_date = local.date()
    else:
        start_date = local.date() - timedelta(days=1)
    start = datetime.combine(start_date, start_time, LOCAL_ZONE)
    end = datetime.combine(start_date + timedelta(days=1), end_time, LOCAL_ZONE)
    return start.astimezone(UTC), end.astimezone(UTC), active


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _local_timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(LOCAL_ZONE).strftime("%Y-%m-%d %H:%M")


def _stored_event_type(row: sqlite3.Row) -> str:
    precise = classify_health_event(str(row["raw_text"] or ""))
    if precise is not None:
        return precise
    legacy = {"battery_dead": "battery_low", "back_online": "power_restored"}
    return legacy.get(str(row["event_type"]), str(row["event_type"]))


def _battery_subject(raw_text: str) -> str:
    match = re.search(r"\[([^]]+)\]", raw_text)
    if match:
        value = match.group(1).strip()
        return "panel" if value.lower() == "panel" else f"device {value}"
    if "panel battery" in raw_text.lower():
        return "panel"
    return "unspecified device"


class BotQueries:
    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        config: AppConfig,
        cameras: CamerasConfig,
        now: Any = None,
    ) -> None:
        self.conn = conn
        self.config = config
        self.cameras = cameras
        self.now = now or (lambda: datetime.now(UTC))

    def role(self, user_id: int) -> str | None:
        if user_id in self.config.bot_trustee_ids:
            return "trustee"
        if user_id in self.config.bot_security_ids:
            return "security"
        return None

    def resolve_camera(self, value: str) -> str | None:
        camera = self.cameras.resolve_alias(value)
        return None if camera is None else camera.id

    def last_clip(self, camera_id: str) -> MediaClip | None:
        """Return the newest known clip; its media may need live retrieval."""
        row = self.conn.execute(
            """
            SELECT channel_id, message_id, camera_id, timestamp, file_path FROM clips
            WHERE camera_id = ? ORDER BY timestamp DESC, message_id DESC LIMIT 1
            """,
            (camera_id,),
        ).fetchone()
        if row is None:
            return None
        return MediaClip(
            channel_id=str(row["channel_id"]),
            message_id=int(row["message_id"]),
            camera_id=str(row["camera_id"]),
            timestamp=str(row["timestamp"]),
            path=None if row["file_path"] is None else Path(str(row["file_path"])),
        )

    def clip_by_message_id(self, message_id: int) -> MediaClip | None:
        """Return any known source clip, including clips outside event history."""
        row = self.conn.execute(
            """
            SELECT channel_id, message_id, camera_id, timestamp, file_path FROM clips
            WHERE message_id = ? ORDER BY timestamp DESC LIMIT 1
            """,
            (message_id,),
        ).fetchone()
        return None if row is None else self._media_clip(row)

    def _category_rows(self, category: str) -> list[sqlite3.Row]:
        if category not in MEDIA_EVENT_CATEGORIES:
            raise ValueError(f"unsupported media category: {category}")
        candidate = f"{category}_candidate"
        return list(
            self.conn.execute(
                """
                WITH matching AS (
                      SELECT lec.event_key, c.channel_id, c.message_id, c.camera_id,
                          c.timestamp, c.file_path, c.caption
                      FROM live_events e JOIN live_event_clips lec USING (event_key)
                      JOIN clips c
                     ON c.channel_id = lec.channel_id AND c.message_id = lec.message_id
                    WHERE e.status = 'finalized' AND e.final_category = ?
                    UNION
                      SELECT NULL AS event_key, c.channel_id, c.message_id, c.camera_id,
                          c.timestamp, c.file_path, c.caption
                    FROM labels l JOIN clips c
                      ON c.channel_id = l.channel_id AND c.message_id = l.message_id
                    WHERE l.label = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM live_event_clips existing
                          WHERE existing.channel_id = c.channel_id
                            AND existing.message_id = c.message_id
                      )
                )
                SELECT * FROM matching ORDER BY timestamp DESC, message_id DESC
                """,
                (candidate, category),
            )
        )

    @staticmethod
    def _media_clip(row: sqlite3.Row) -> MediaClip:
        return MediaClip(
            channel_id=str(row["channel_id"]),
            message_id=int(row["message_id"]),
            camera_id=str(row["camera_id"]),
            timestamp=str(row["timestamp"]),
            path=None if row["file_path"] is None else Path(str(row["file_path"])),
        )

    def category_clips(self, category: str, *, limit: int) -> list[MediaClip]:
        return [event.clip for event in self.category_events(category)[:limit]]

    def category_events(self, category: str) -> list[HistoryEvent]:
        groups: dict[str, list[sqlite3.Row]] = {}
        for row in self._category_rows(category):
            key = str(row["event_key"] or message_event_key(
                str(row["camera_id"]), row["caption"], int(row["message_id"])
            ))
            groups.setdefault(key, []).append(row)
        events = []
        for rows in groups.values():
            representative = max(
                rows,
                key=lambda row: (
                    event_phase(row["caption"]) == "complete",
                    str(row["timestamp"]),
                    int(row["message_id"]),
                ),
            )
            resolution = self._resolution_details(
                str(representative["channel_id"]), int(representative["message_id"])
            )
            events.append(
                HistoryEvent(
                    category=category,
                    clip=self._media_clip(representative),
                    sibling_count=len(rows),
                    resolution_state=(
                        None if resolution is None else str(resolution["resolution_state"])
                    ),
                    initial_category=(
                        None if resolution is None else resolution["initial_category"]
                    ),
                    complete_category=(
                        None if resolution is None else resolution["complete_category"]
                    ),
                )
            )
        return sorted(
            events,
            key=lambda event: (event.clip.timestamp, event.clip.message_id),
            reverse=True,
        )

    def _resolution_details(
        self, channel_id: str, message_id: int
    ) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT e.resolution_state,
                   (SELECT category FROM live_event_clips initial
                    WHERE initial.event_key = e.event_key AND initial.phase = 'initial'
                    ORDER BY initial.message_id DESC LIMIT 1) AS initial_category,
                   (SELECT category FROM live_event_clips complete
                    WHERE complete.event_key = e.event_key AND complete.phase = 'complete'
                    ORDER BY complete.message_id DESC LIMIT 1) AS complete_category
            FROM live_events e JOIN live_event_clips selected USING (event_key)
            WHERE selected.channel_id = ? AND selected.message_id = ?
            LIMIT 1
            """,
            (channel_id, message_id),
        ).fetchone()

    def history(self, category: str, *, page: int, page_size: int = 8) -> str:
        events = self.category_events(category)
        if not events:
            return f"No {category} history is recorded."
        pages = max(1, (len(events) + page_size - 1) // page_size)
        page = min(max(page, 1), pages)
        selected = events[(page - 1) * page_size : page * page_size]
        lines = [f"{category.title()} history — page {page}/{pages} ({len(events)} events)"]
        for event in selected:
            sibling_text = f", {event.sibling_count} clips" if event.sibling_count > 1 else ""
            resolution_text = (
                f", {event.resolution_state.replace('_', ' ')}"
                if event.resolution_state not in (None, "confirmed")
                else ""
            )
            lines.append(
                f"{event.clip.message_id} · {_local_timestamp(event.clip.timestamp)} · "
                f"{event.clip.camera_id}{sibling_text}{resolution_text}"
            )
        lines.append("Tap a video button below, or send /event <id>.")
        if pages > 1:
            lines.append(f"Next page: /history {category} {min(page + 1, pages)}")
        return "\n".join(lines)

    def history_clip(self, message_id: int) -> HistoryEvent | None:
        for category in MEDIA_EVENT_CATEGORIES:
            for event in self.category_events(category):
                if event.clip.message_id == message_id:
                    return event
        return None

    def about(self, role: str) -> str:
        return (
            "Nature Ridge Perimeter Watch\n"
            f"Access: {role}\n"
            "Classical motion analysis supplements the guard service; categories are candidates, "
            "not identity claims. Times are Africa/Johannesburg."
        )

    def contacts(self) -> str:
        def value(raw: str | None) -> str:
            return raw or "Not configured"

        return (
            "Security contacts\n"
            f"Company: {value(self.config.security_company_name)}\n"
            f"Company phone: {value(self.config.security_company_phone)}\n"
            f"Control room: {value(self.config.control_room_phone)}\n"
            f"Armed response: {value(self.config.armed_response_phone)}"
        )

    def activity_report(self, period: str) -> ActivityReport:
        if period not in {"month", "year"}:
            raise ValueError(f"unsupported report period: {period}")
        return build_activity_report(self.conn, period, self.now())

    def tonight(self) -> str:
        start, end, active = operating_period(self.config, self.now())
        rows = list(
            self.conn.execute(
                """
                SELECT e.final_category, COUNT(*) AS n
                FROM live_events e
                JOIN clips c
                  ON c.channel_id = e.representative_channel_id
                 AND c.message_id = e.representative_message_id
                WHERE e.status = 'finalized' AND c.timestamp >= ? AND c.timestamp < ?
                GROUP BY e.final_category ORDER BY n DESC
                """,
                (_utc_text(start), _utc_text(end)),
            )
        )
        pending = self.conn.execute(
            "SELECT COUNT(*) FROM live_events WHERE status = 'pending'"
        ).fetchone()[0]
        total = sum(int(row["n"]) for row in rows)
        counts = ", ".join(f"{row['final_category']}={row['n']}" for row in rows) or "none"
        state = "active" if active else "closed"
        return (
            f"Night window ({state})\n"
            f"{start.astimezone(LOCAL_ZONE):%Y-%m-%d %H:%M} to "
            f"{end.astimezone(LOCAL_ZONE):%Y-%m-%d %H:%M}\n"
            f"Finalized events: {total} ({counts})\nPending sibling groups: {pending}"
        )

    def health(self) -> str:
        start, _end, active = operating_period(self.config, self.now())
        latest_clips = {
            str(row["camera_id"]): str(row["latest"])
            for row in self.conn.execute(
                "SELECT camera_id, MAX(timestamp) AS latest FROM clips GROUP BY camera_id"
            )
        }
        latest_events = {
            str(row["camera_id"]): (_stored_event_type(row), str(row["timestamp"]))
            for row in self.conn.execute(
                """
                SELECT s.camera_id, s.event_type, s.timestamp, s.raw_text
                FROM system_events s
                JOIN (
                    SELECT camera_id, MAX(timestamp) AS latest
                    FROM system_events WHERE camera_id IS NOT NULL GROUP BY camera_id
                ) x ON x.camera_id = s.camera_id AND x.latest = s.timestamp
                """
            )
        }
        lines = ["Camera health" + (" (operating window)" if active else " (silence expected)")]
        for camera in sorted(self.cameras.cameras, key=lambda item: item.order):
            latest = latest_clips.get(camera.id)
            event = latest_events.get(camera.id)
            status = "no clips recorded"
            if latest:
                status = f"last {_local_timestamp(latest)}"
                if active and latest < _utc_text(start):
                    status = "no clip this window; " + status
            if event and event[0] in {"battery_low", "power_out"}:
                status = f"{event[0]} {_local_timestamp(event[1])}; {status}"
            elif event and event[0] in {"battery_restored", "power_restored"}:
                status = f"online {_local_timestamp(event[1])}; {status}"
            lines.append(f"{camera.id}: {status}")
        return "\n".join(lines)

    def power(self, *, limit: int = 10) -> str:
        rows = list(
            self.conn.execute(
                """
                SELECT timestamp, event_type, raw_text FROM system_events
                WHERE event_type IN ('power_out', 'power_restored', 'back_online')
                   OR lower(COALESCE(raw_text, '')) LIKE '%power failure%'
                ORDER BY timestamp DESC
                """
            )
        )
        unique: list[tuple[sqlite3.Row, str]] = []
        seen: set[str] = set()
        counts: dict[str, int] = {"power_out": 0, "power_restored": 0}
        for row in rows:
            event_type = _stored_event_type(row)
            if event_type not in counts:
                continue
            counts[event_type] += 1
            raw = " ".join(str(row["raw_text"] or "").lower().split())
            if raw in seen:
                continue
            seen.add(raw)
            if len(unique) < limit:
                unique.append((row, event_type))
        if not unique:
            return "No power notifications are recorded."
        latest = "failure" if unique[0][1] == "power_out" else "restored"
        lines = [
            f"Power notifications — latest: {latest}",
            f"History: {counts['power_out']} failures, {counts['power_restored']} restores",
        ]
        labels = {"power_out": "POWER FAILURE", "power_restored": "restored"}
        lines.extend(
            f"{_local_timestamp(str(row['timestamp']))} — {labels[event_type]}"
            for row, event_type in unique
        )
        lines.append("Duplicate notifications with identical source text are collapsed above.")
        return "\n".join(lines)

    def batteries(self) -> str:
        rows = list(
            self.conn.execute(
                """
                SELECT timestamp, event_type, raw_text FROM system_events
                WHERE event_type IN ('battery_dead', 'battery_low', 'battery_restored')
                   OR lower(COALESCE(raw_text, '')) LIKE '%battery low%'
                ORDER BY timestamp
                """
            )
        )
        active: dict[str, sqlite3.Row] = {}
        notifications: list[tuple[sqlite3.Row, str, str]] = []
        for row in rows:
            event_type = _stored_event_type(row)
            if event_type not in {"battery_low", "battery_restored"}:
                continue
            subject = _battery_subject(str(row["raw_text"] or ""))
            notifications.append((row, event_type, subject))
            if event_type == "battery_low":
                active[subject] = row
            elif subject != "unspecified device":
                active.pop(subject, None)
            elif active:
                # Older source messages omit the device on restore. Pair such
                # a restore with the most recently warned unresolved device.
                latest_subject = max(active, key=lambda key: str(active[key]["timestamp"]))
                active.pop(latest_subject, None)
        if not notifications:
            return "No battery notifications are recorded."
        lines = ["Battery replacement status"]
        cutoff = self.now() - timedelta(days=365)
        current = {
            subject: row
            for subject, row in active.items()
            if datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00")) >= cutoff
        }
        older_count = len(active) - len(current)
        if current:
            lines.append("Current replacement candidates:")
            for subject, row in sorted(
                current.items(), key=lambda item: str(item[1]["timestamp"]), reverse=True
            ):
                lines.append(f"{subject}: {_local_timestamp(str(row['timestamp']))}")
        else:
            lines.append("No unresolved low-battery warnings from the last year.")
        if older_count:
            lines.append(
                f"Older unmatched warnings: {older_count} "
                "(historical uncertainty, not current work)"
            )
        lines.append("Recent notifications:")
        seen: set[tuple[str, str]] = set()
        recent = []
        for row, event_type, subject in reversed(notifications):
            key = (str(row["raw_text"] or ""), event_type)
            if key in seen:
                continue
            seen.add(key)
            recent.append((row, event_type, subject))
            if len(recent) == 5:
                break
        for row, event_type, subject in recent:
            label = "LOW" if event_type == "battery_low" else "restored"
            lines.append(f"{_local_timestamp(str(row['timestamp']))} — {subject} {label}")
        lines.append(
            "Status is inferred from low/restore notifications; confirm replacements on the "
            "alarm panel."
        )
        return "\n".join(lines)

    def panel(self, *, limit: int = 10) -> str:
        rows = list(
            self.conn.execute(
                """
                SELECT timestamp, event_type, raw_text FROM system_events
                WHERE event_type IN ('panel_armed', 'panel_disarmed')
                ORDER BY timestamp DESC
                """
            )
        )
        if not rows:
            return "No panel arm/disarm notifications are recorded."
        latest = _stored_event_type(rows[0]).removeprefix("panel_")
        counts = {"panel_armed": 0, "panel_disarmed": 0}
        unique = []
        seen: set[str] = set()
        for row in rows:
            event_type = _stored_event_type(row)
            if event_type not in counts:
                continue
            counts[event_type] += 1
            raw = " ".join(str(row["raw_text"] or "").lower().split())
            if raw in seen:
                continue
            seen.add(raw)
            if len(unique) < limit:
                unique.append((row, event_type))
        lines = [
            f"Panel — latest: {latest}",
            f"History: {counts['panel_armed']} armed, {counts['panel_disarmed']} disarmed",
        ]
        lines.extend(
            f"{_local_timestamp(str(row['timestamp']))} — "
            f"{event_type.removeprefix('panel_')}"
            for row, event_type in unique
        )
        return "\n".join(lines)

    def faults(self, *, limit: int = 10) -> str:
        supported = {
            "tamper": "TAMPER",
            "tamper_restored": "tamper restored",
            "supervision_error": "SUPERVISION / DEVICE MISSING",
            "communication_failure": "CONTROL-ROOM COMMUNICATION FAILURE",
        }
        placeholders = ",".join("?" for _ in supported)
        rows = list(
            self.conn.execute(
                f"""
                SELECT timestamp, event_type, raw_text FROM system_events
                WHERE event_type IN ({placeholders}) ORDER BY timestamp DESC
                """,
                tuple(supported),
            )
        )
        if not rows:
            return "No tamper, supervision, or communication faults are recorded."
        counts = {event_type: 0 for event_type in supported}
        unique = []
        seen: set[str] = set()
        for row in rows:
            event_type = _stored_event_type(row)
            if event_type not in supported:
                continue
            counts[event_type] += 1
            raw = " ".join(str(row["raw_text"] or "").lower().split())
            if raw in seen:
                continue
            seen.add(raw)
            if len(unique) < limit:
                unique.append((row, event_type, _battery_subject(str(row["raw_text"] or ""))))
        lines = [
            "System fault history",
            (
                f"tamper={counts['tamper']}, restored={counts['tamper_restored']}, "
                f"supervision={counts['supervision_error']}, "
                f"communication={counts['communication_failure']}"
            ),
        ]
        for row, event_type, subject in unique:
            suffix = f" — {subject}" if subject != "unspecified device" else ""
            lines.append(
                f"{_local_timestamp(str(row['timestamp']))} — {supported[event_type]}{suffix}"
            )
        lines.append("Fault history is advisory; confirm current state with the control room.")
        return "\n".join(lines)

    def animals(self) -> str:
        rows = list(
            self.conn.execute(
                """
                SELECT c.camera_id, c.timestamp, e.final_reason
                FROM live_events e JOIN clips c
                  ON c.channel_id = e.representative_channel_id
                 AND c.message_id = e.representative_message_id
                WHERE e.status = 'finalized' AND e.final_category = 'animal_candidate'
                ORDER BY c.timestamp DESC LIMIT 10
                """
            )
        )
        if not rows:
            return "No live animal candidates recorded yet."
        lines = ["Recent animal candidates"]
        lines.extend(
            f"{_local_timestamp(row['timestamp'])} {row['camera_id']} ({row['final_reason']})"
            for row in rows
        )
        return "\n".join(lines)

    def map(self) -> str:
        cameras = sorted(self.cameras.cameras, key=lambda item: item.order)
        return "Approximate fence order (not geographic)\n" + " — ".join(
            f"{index + 1}:{camera.id}" for index, camera in enumerate(cameras)
        )

    def patrol_report(self) -> PatrolReport:
        start, end, _active = operating_period(self.config, self.now())
        rows = list(
            self.conn.execute(
                """
                SELECT c.camera_id, c.timestamp
                FROM live_events e JOIN clips c
                  ON c.channel_id = e.representative_channel_id
                 AND c.message_id = e.representative_message_id
                WHERE e.status = 'finalized' AND e.final_category = 'guard_candidate'
                  AND c.timestamp >= ? AND c.timestamp < ?
                ORDER BY c.timestamp
                """,
                (_utc_text(start), _utc_text(end)),
            )
        )
        events = [
            ClipEvent(
                camera_id=str(row["camera_id"]),
                timestamp=datetime.fromisoformat(
                    str(row["timestamp"]).replace("Z", "+00:00")
                ).timestamp(),
            )
            for row in rows
        ]
        passes = segment_passes(events, max_gap_seconds=600, min_cameras=3)
        camera_order = tuple(camera.id for camera in self.cameras.ordered())
        return build_patrol_report(
            passes, camera_order, window_start=start, window_end=end
        )


class QueryBot:
    def __init__(
        self,
        queries: BotQueries,
        *,
        media_loader: Callable[[MediaClip], Awaitable[Path | None]] | None = None,
        debug_loader: Callable[[MediaClip, Path], Awaitable[Path | None]] | None = None,
    ) -> None:
        self.queries = queries
        self.media_loader = media_loader
        self.debug_loader = debug_loader

    async def _authorize(
        self, update: Any, command: str, *, trustee_only: bool = False
    ) -> tuple[Any, str] | None:
        user = getattr(update, "effective_user", None)
        message = getattr(update, "effective_message", None)
        user_id = getattr(user, "id", None)
        if message is None or user_id is None:
            return None
        role = self.queries.role(int(user_id))
        if role is None or (trustee_only and role != "trustee"):
            chat = getattr(update, "effective_chat", None)
            logger.warning(
                "query_bot_access_denied",
                extra={
                    "user_id": user_id,
                    "chat_id": getattr(chat, "id", None),
                    "command": command,
                    "role": role,
                },
            )
            await message.reply_text("Not authorized.")
            return None
        return message, role

    async def _reply(self, update: Any, command: str, *, trustee_only: bool = False) -> None:
        authorized = await self._authorize(update, command, trustee_only=trustee_only)
        if authorized is None:
            return
        message, role = authorized
        if command == "about":
            text = self.queries.about(role)
        else:
            text = getattr(self.queries, command)()
        await message.reply_text(text)

    @staticmethod
    def _keyboard(rows: list[list[tuple[str, str]]]) -> Any:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(label, callback_data=data) for label, data in row]
                for row in rows
            ]
        )

    def _home_menu(self) -> tuple[str, Any]:
        return (
            "Nature Ridge Perimeter Watch\nChoose a section:",
            self._keyboard(
                [
                    [("Monitoring", "menu:monitor"), ("Events", "menu:events")],
                    [("System", "menu:system"), ("Site", "menu:site")],
                ]
            ),
        )

    def _section_menu(self, section: str) -> tuple[str, Any]:
        menus = {
            "monitor": (
                "Monitoring",
                [
                    [("Tonight", "menu:show:tonight"), ("Camera health", "menu:show:health")],
                    [("Latest camera video", "menu:cameras")],
                ],
            ),
            "events": (
                "Camera-classified events",
                [
                    [
                        ("Animal history", "menu:history:animal:1"),
                        ("Incident history", "menu:history:incident:1"),
                    ],
                    [
                        ("Resident history", "menu:history:resident:1"),
                        ("Neighbour history", "menu:history:neighbour:1"),
                    ],
                ],
            ),
            "system": (
                "Alarm-system events",
                [
                    [("Power", "menu:show:power"), ("Batteries", "menu:show:batteries")],
                    [("Panel", "menu:show:panel"), ("Faults", "menu:show:faults")],
                ],
            ),
            "site": (
                "Site",
                [
                    [("Camera order", "menu:show:map"), ("Patrols", "menu:show:patrols")],
                    [("This month", "menu:report:month"), ("This year", "menu:report:year")],
                    [
                        ("Security contacts", "menu:show:contacts"),
                        ("About this system", "menu:show:about"),
                    ],
                ],
            ),
        }
        title, rows = menus[section]
        return title, self._keyboard([*rows, [("Back", "menu:home")]])

    def _camera_menu(self) -> tuple[str, Any]:
        cameras = self.queries.cameras.ordered()
        rows = [
            [
                (camera.id, f"menu:last:{camera.id}")
                for camera in cameras[index : index + 3]
            ]
            for index in range(0, len(cameras), 3)
        ]
        rows.append([("Back", "menu:monitor")])
        return "Choose a camera for its latest video:", self._keyboard(rows)

    def _history_menu(self, category: str, page: int) -> tuple[str, Any]:
        page_size = 8
        events = self.queries.category_events(category)
        pages = max(1, (len(events) + page_size - 1) // page_size)
        page = min(max(page, 1), pages)
        selected = events[(page - 1) * page_size : page * page_size]
        rows = [
            [
                (
                    f"▶ {_local_timestamp(event.clip.timestamp)} · {event.clip.camera_id}",
                    f"menu:event:{event.clip.message_id}",
                )
            ]
            for event in selected
        ]
        navigation = []
        if page > 1:
            navigation.append(("Previous", f"menu:history:{category}:{page - 1}"))
        if page < pages:
            navigation.append(("Next", f"menu:history:{category}:{page + 1}"))
        if navigation:
            rows.append(navigation)
        rows.append([("Back", "menu:events"), ("Main menu", "menu:home")])
        return self.queries.history(category, page=page, page_size=page_size), self._keyboard(rows)

    async def menu(self, update: Any, context: Any) -> None:
        authorized = await self._authorize(update, "menu")
        if authorized is None:
            return
        message, _role = authorized
        text, markup = self._home_menu()
        await message.reply_text(text, reply_markup=markup)

    async def start(self, update: Any, context: Any) -> None:
        await self.menu(update, context)

    async def menu_callback(self, update: Any, context: Any) -> None:
        query = getattr(update, "callback_query", None)
        if query is None:
            return
        await query.answer()
        authorized = await self._authorize(update, "menu")
        if authorized is None:
            return
        message, role = authorized
        data = str(getattr(query, "data", ""))
        if data == "menu:home":
            text, markup = self._home_menu()
        elif data in {"menu:monitor", "menu:events", "menu:system", "menu:site"}:
            text, markup = self._section_menu(data.removeprefix("menu:"))
        elif data == "menu:cameras":
            text, markup = self._camera_menu()
        elif data.startswith("menu:history:"):
            _prefix, _history, category, raw_page = data.split(":", 3)
            text, markup = self._history_menu(category, int(raw_page))
        elif data.startswith("menu:event:"):
            await self._send_history_event(message, int(data.rsplit(":", 1)[1]))
            return
        elif data.startswith("menu:debug:"):
            await self._send_debug_video(message, int(data.rsplit(":", 1)[1]))
            return
        elif data.startswith("menu:last:"):
            await self._send_last_camera(message, data.rsplit(":", 1)[1])
            return
        elif data.startswith("menu:report:"):
            if role != "trustee":
                await message.reply_text("Not authorized.")
                return
            await self._send_activity_report(message, data.rsplit(":", 1)[1])
            return
        elif data == "menu:show:patrols":
            if role != "trustee":
                await message.reply_text("Not authorized.")
                return
            await self._send_patrol_report(message)
            return
        elif data.startswith("menu:show:"):
            command = data.rsplit(":", 1)[1]
            text = (
                self.queries.about(role)
                if command == "about"
                else getattr(self.queries, command)()
            )
            section = {
                "about": "home",
                "tonight": "monitor",
                "health": "monitor",
                "power": "system",
                "batteries": "system",
                "panel": "system",
                "faults": "system",
                "map": "site",
                "contacts": "site",
            }[command]
            markup = self._keyboard(
                [[("Back", f"menu:{section}"), ("Main menu", "menu:home")]]
            )
        else:
            text, markup = self._home_menu()
        await query.edit_message_text(text=text, reply_markup=markup)

    async def about(self, update: Any, context: Any) -> None:
        await self._reply(update, "about")

    async def contacts(self, update: Any, context: Any) -> None:
        await self._reply(update, "contacts")

    async def tonight(self, update: Any, context: Any) -> None:
        await self._reply(update, "tonight")

    async def health(self, update: Any, context: Any) -> None:
        await self._reply(update, "health")

    async def power(self, update: Any, context: Any) -> None:
        authorized = await self._authorize(update, "power")
        if authorized is None:
            return
        message, _role = authorized
        count = 10
        args = getattr(context, "args", ()) or ()
        if args:
            try:
                count = int(args[0])
            except ValueError:
                await message.reply_text("Usage: /power [count]")
                return
        await message.reply_text(self.queries.power(limit=min(max(count, 1), 20)))

    async def batteries(self, update: Any, context: Any) -> None:
        await self._reply(update, "batteries")

    async def panel(self, update: Any, context: Any) -> None:
        await self._counted_text(update, context, command="panel")

    async def faults(self, update: Any, context: Any) -> None:
        await self._counted_text(update, context, command="faults")

    async def _counted_text(self, update: Any, context: Any, *, command: str) -> None:
        authorized = await self._authorize(update, command)
        if authorized is None:
            return
        message, _role = authorized
        count = 10
        args = getattr(context, "args", ()) or ()
        if args:
            try:
                count = int(args[0])
            except ValueError:
                await message.reply_text(f"Usage: /{command} [count]")
                return
        query = getattr(self.queries, command)
        await message.reply_text(query(limit=min(max(count, 1), 20)))

    async def map(self, update: Any, context: Any) -> None:
        await self._reply(update, "map")

    async def patrols(self, update: Any, context: Any) -> None:
        authorized = await self._authorize(update, "patrols", trustee_only=True)
        if authorized is None:
            return
        message, _role = authorized
        await self._send_patrol_report(message)

    async def month(self, update: Any, context: Any) -> None:
        await self._activity_report(update, "month")

    async def year(self, update: Any, context: Any) -> None:
        await self._activity_report(update, "year")

    async def _activity_report(self, update: Any, period: str) -> None:
        authorized = await self._authorize(update, period, trustee_only=True)
        if authorized is None:
            return
        message, _role = authorized
        await self._send_activity_report(message, period)

    async def _send_activity_report(self, message: Any, period: str) -> None:
        report = self.queries.activity_report(period)
        png = await asyncio.to_thread(render_activity_report, report)
        photo = io.BytesIO(png)
        photo.name = f"activity-{period}.png"
        await message.reply_photo(photo=photo, caption=report.caption)

    async def _send_patrol_report(self, message: Any) -> None:
        report = self.queries.patrol_report()
        png = await asyncio.to_thread(render_patrol_report, report)
        photo = io.BytesIO(png)
        photo.name = "patrols.png"
        await message.reply_photo(photo=photo, caption=report.caption)

    async def last(self, update: Any, context: Any) -> None:
        authorized = await self._authorize(update, "last")
        if authorized is None:
            return
        message, _role = authorized
        raw_camera = " ".join(getattr(context, "args", ()) or ()).strip()
        if not raw_camera:
            camera_ids = ", ".join(camera.id for camera in self.queries.cameras.ordered())
            await message.reply_text(f"Usage: /last <camera_id>\nCameras: {camera_ids}")
            return
        camera_id = self.queries.resolve_camera(raw_camera)
        if camera_id is None:
            await message.reply_text(f"Unknown camera: {raw_camera}")
            return
        await self._send_last_camera(message, camera_id)

    async def _send_last_camera(self, message: Any, camera_id: str) -> None:
        clip = self.queries.last_clip(camera_id)
        if clip is None:
            await message.reply_text(f"No video metadata is available for {camera_id}.")
            return
        path = await self._ensure_media(clip)
        if path is None:
            await message.reply_text(f"The latest {camera_id} video is unavailable at its source.")
            return
        await self._send_video(message, clip, path, heading="Latest camera video")

    async def _ensure_media(self, clip: MediaClip) -> Path | None:
        if clip.path is not None and clip.path.is_file():
            return clip.path
        if self.media_loader is None:
            return None
        try:
            return await self.media_loader(clip)
        except Exception as exc:
            logger.exception(
                "query_bot_media_retrieval_failed",
                extra={
                    "camera_id": clip.camera_id,
                    "message_id": clip.message_id,
                    "error": str(exc),
                },
            )
            return None

    async def _send_video(
        self,
        message: Any,
        clip: MediaClip,
        path: Path,
        *,
        heading: str,
        offer_debug: bool = True,
    ) -> bool:
        caption = (
            f"{heading}: {clip.camera_id}\n"
            f"{_local_timestamp(clip.timestamp)}"
        )
        try:
            with path.open("rb") as video:
                await message.reply_video(
                    video=video,
                    caption=caption,
                    supports_streaming=True,
                    read_timeout=60,
                    write_timeout=60,
                    reply_markup=(
                        self._keyboard(
                            [[("Debug view", f"menu:debug:{clip.message_id}")]]
                        )
                        if offer_debug and self.debug_loader is not None
                        else None
                    ),
                )
        except Exception as exc:
            logger.exception(
                "query_bot_video_failed",
                extra={"camera_id": clip.camera_id, "error": str(exc)},
            )
            await message.reply_text(f"The {clip.camera_id} video could not be sent.")
            return False
        return True

    async def _category_videos(
        self,
        update: Any,
        context: Any,
        *,
        category: str,
        default_count: int,
    ) -> None:
        command = {
            "animal": "animals",
            "incident": "incidents",
            "resident": "residents",
            "neighbour": "neighbours",
        }[category]
        authorized = await self._authorize(update, command)
        if authorized is None:
            return
        message, _role = authorized
        count = default_count
        args = getattr(context, "args", ()) or ()
        if args:
            try:
                count = int(args[0])
            except ValueError:
                await message.reply_text(f"Usage: /{command} [count]")
                return
        count = min(max(count, 1), 5)
        clips = self.queries.category_clips(category, limit=count)
        if not clips:
            await message.reply_text(f"No {category} videos are recorded yet.")
            return
        sent = 0
        for clip in clips:
            path = await self._ensure_media(clip)
            if path is None:
                continue
            sent += int(
                await self._send_video(
                    message,
                    clip,
                    path,
                    heading=f"{category.title()} candidate",
                )
            )
        if sent == 0:
            await message.reply_text(
                f"{len(clips)} {category} event(s) are recorded, but their source videos "
                "could not be retrieved."
            )

    async def animal(self, update: Any, context: Any) -> None:
        await self._category_videos(update, context, category="animal", default_count=1)

    async def animals(self, update: Any, context: Any) -> None:
        await self._category_videos(update, context, category="animal", default_count=3)

    async def incidents(self, update: Any, context: Any) -> None:
        await self._category_videos(update, context, category="incident", default_count=3)

    async def residents(self, update: Any, context: Any) -> None:
        await self._category_videos(update, context, category="resident", default_count=3)

    async def neighbours(self, update: Any, context: Any) -> None:
        await self._category_videos(update, context, category="neighbour", default_count=3)

    async def history(self, update: Any, context: Any) -> None:
        authorized = await self._authorize(update, "history")
        if authorized is None:
            return
        message, _role = authorized
        args = list(getattr(context, "args", ()) or ())
        if not args:
            await message.reply_text(HISTORY_USAGE)
            return
        category = args[0].lower().removesuffix("s")
        if category not in MEDIA_EVENT_CATEGORIES:
            await message.reply_text(HISTORY_USAGE)
            return
        try:
            page = 1 if len(args) < 2 else int(args[1])
        except ValueError:
            await message.reply_text(HISTORY_USAGE)
            return
        text, markup = self._history_menu(category, max(page, 1))
        await message.reply_text(text, reply_markup=markup)

    async def event(self, update: Any, context: Any) -> None:
        authorized = await self._authorize(update, "event")
        if authorized is None:
            return
        message, _role = authorized
        args = list(getattr(context, "args", ()) or ())
        try:
            message_id = int(args[0]) if len(args) == 1 else None
        except ValueError:
            message_id = None
        if message_id is None:
            await message.reply_text(
                "Usage: /event <id>\nFind IDs in the Events menu or with /history <category>"
            )
            return
        await self._send_history_event(message, message_id)

    async def debug(self, update: Any, context: Any) -> None:
        authorized = await self._authorize(update, "debug")
        if authorized is None:
            return
        message, _role = authorized
        args = list(getattr(context, "args", ()) or ())
        try:
            message_id = int(args[0]) if len(args) == 1 else None
        except ValueError:
            message_id = None
        if message_id is None:
            await message.reply_text("Usage: /debug <video-id>")
            return
        await self._send_debug_video(message, message_id)

    async def _send_debug_video(self, message: Any, message_id: int) -> None:
        clip = self.queries.clip_by_message_id(message_id)
        if clip is None:
            await message.reply_text(f"No video is listed with ID {message_id}.")
            return
        if self.debug_loader is None:
            await message.reply_text("Debug video rendering is unavailable.")
            return
        source = await self._ensure_media(clip)
        if source is None:
            await message.reply_text(f"Video {message_id} is unavailable at its source.")
            return
        await message.reply_text(f"Generating debug view for {clip.camera_id}/{message_id}…")
        try:
            path = await self.debug_loader(clip, source)
        except Exception as exc:
            logger.exception(
                "query_bot_debug_render_failed",
                extra={"camera_id": clip.camera_id, "message_id": message_id, "error": str(exc)},
            )
            path = None
        if path is None or not path.is_file():
            await message.reply_text(f"Debug view for {clip.camera_id}/{message_id} failed.")
            return
        await self._send_video(
            message,
            clip,
            path,
            heading=f"Detector debug {message_id}",
            offer_debug=False,
        )

    async def _send_history_event(self, message: Any, message_id: int) -> None:
        event = self.queries.history_clip(message_id)
        if event is None:
            await message.reply_text(f"No camera-classified event is listed with ID {message_id}.")
            return
        path = await self._ensure_media(event.clip)
        if path is None:
            await message.reply_text(f"Event {message_id} is unavailable at its source.")
            return
        heading = f"{event.category.title()} history event {message_id}"
        if event.resolution_state not in (None, "confirmed"):
            heading += f"\nResolution: {event.resolution_state.replace('_', ' ')}"
            if event.initial_category is not None:
                heading += f"\nStartup: {event.initial_category.removesuffix('_candidate')}"
            if event.complete_category is not None:
                heading += f"\nCompleted: {event.complete_category.removesuffix('_candidate')}"
        await self._send_video(
            message,
            event.clip,
            path,
            heading=heading,
        )


def build_query_bot(
    queries: BotQueries,
    token: str,
    *,
    media_loader: Callable[[MediaClip], Awaitable[Path | None]] | None = None,
    debug_loader: Callable[[MediaClip, Path], Awaitable[Path | None]] | None = None,
) -> Any:
    from telegram import BotCommand
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler

    controller = QueryBot(queries, media_loader=media_loader, debug_loader=debug_loader)
    application = Application.builder().token(token).build()
    for command in (
        "about",
        "contacts",
        "start",
        "menu",
        "tonight",
        "health",
        "power",
        "batteries",
        "panel",
        "faults",
        "animal",
        "animals",
        "incidents",
        "residents",
        "neighbours",
        "history",
        "event",
        "debug",
        "map",
        "patrols",
        "last",
        "month",
        "year",
    ):
        application.add_handler(CommandHandler(command, getattr(controller, command)))
    application.add_handler(CallbackQueryHandler(controller.menu_callback, pattern=r"^menu:"))
    application.bot_data["commands"] = [
        BotCommand("menu", "Open the grouped menu"),
        BotCommand("tonight", "Show tonight's event summary"),
        BotCommand("health", "Show camera health now"),
        BotCommand("about", "Explain what the system reports"),
    ]
    return application
