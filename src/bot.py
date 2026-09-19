"""Role-restricted Telegram query bot backed by live watcher state."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from src.config import AppConfig, CamerasConfig
from src.sequence import ClipEvent, segment_passes

logger = logging.getLogger("query_bot")
LOCAL_ZONE = ZoneInfo("Africa/Johannesburg")


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

    def about(self, role: str) -> str:
        return (
            "Nature Ridge Perimeter Watch\n"
            f"Access: {role}\n"
            "Classical motion analysis supplements the guard service; categories are candidates, "
            "not identity claims. Times are Africa/Johannesburg."
        )

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
            str(row["camera_id"]): (str(row["event_type"]), str(row["timestamp"]))
            for row in self.conn.execute(
                """
                SELECT s.camera_id, s.event_type, s.timestamp
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
            if event and event[0] in {"battery_dead", "power_out"}:
                status = f"{event[0]} {_local_timestamp(event[1])}; {status}"
            elif event and event[0] == "back_online":
                status = f"online {_local_timestamp(event[1])}; {status}"
            lines.append(f"{camera.id}: {status}")
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

    def patrols(self) -> str:
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
        if not passes:
            return "No multi-camera guard passes detected in the night window."
        all_camera_ids = {camera.id for camera in self.cameras.cameras}
        lines = [f"Candidate guard passes: {len(passes)}"]
        for index, patrol in enumerate(passes, 1):
            covered = {event.camera_id for event in patrol.events}
            first = datetime.fromtimestamp(patrol.events[0].timestamp, UTC).astimezone(LOCAL_ZONE)
            last = datetime.fromtimestamp(patrol.events[-1].timestamp, UTC).astimezone(LOCAL_ZONE)
            gaps = sorted(all_camera_ids - covered)
            gap_text = ",".join(gaps) if gaps else "none"
            lines.append(
                f"{index}. {first:%H:%M}-{last:%H:%M}, cameras={len(covered)}, gaps={gap_text}"
            )
        lines.append("Candidate analytics only; verify against operational records.")
        return "\n".join(lines)


class QueryBot:
    def __init__(self, queries: BotQueries) -> None:
        self.queries = queries

    async def _reply(self, update: Any, command: str, *, trustee_only: bool = False) -> None:
        user = getattr(update, "effective_user", None)
        message = getattr(update, "effective_message", None)
        user_id = getattr(user, "id", None)
        if message is None or user_id is None:
            return
        role = self.queries.role(int(user_id))
        if role is None or (trustee_only and role != "trustee"):
            logger.warning(
                "query_bot_access_denied",
                extra={"user_id": user_id, "command": command, "role": role},
            )
            await message.reply_text("Not authorized.")
            return
        if command == "about":
            text = self.queries.about(role)
        else:
            text = getattr(self.queries, command)()
        await message.reply_text(text)

    async def about(self, update: Any, context: Any) -> None:
        await self._reply(update, "about")

    async def tonight(self, update: Any, context: Any) -> None:
        await self._reply(update, "tonight")

    async def health(self, update: Any, context: Any) -> None:
        await self._reply(update, "health")

    async def animals(self, update: Any, context: Any) -> None:
        await self._reply(update, "animals")

    async def map(self, update: Any, context: Any) -> None:
        await self._reply(update, "map")

    async def patrols(self, update: Any, context: Any) -> None:
        await self._reply(update, "patrols", trustee_only=True)


def build_query_bot(queries: BotQueries, token: str) -> Any:
    from telegram import BotCommand
    from telegram.ext import Application, CommandHandler

    controller = QueryBot(queries)
    application = Application.builder().token(token).build()
    for command in ("about", "tonight", "health", "animals", "map", "patrols"):
        application.add_handler(CommandHandler(command, getattr(controller, command)))
    application.bot_data["commands"] = [
        BotCommand("about", "What this system reports"),
        BotCommand("tonight", "Tonight's event summary"),
        BotCommand("health", "Camera health and last activity"),
        BotCommand("animals", "Recent animal candidates"),
        BotCommand("map", "Approximate camera order"),
        BotCommand("patrols", "Guard-pass candidates (trustees only)"),
    ]
    return application
