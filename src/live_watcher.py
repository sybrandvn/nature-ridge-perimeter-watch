"""Durable live-message processing, event resolution, and alert dispatch."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src import db, live_state
from src.clip_analysis import ClipAnalysis, analyze_clip
from src.config import AppConfig, CamerasConfig, ThresholdsConfig, resolve_channel_ref
from src.event_keys import event_phase, message_event_key
from src.event_resolution import PRELIMINARY_CATEGORIES, resolve_event
from src.media_download import download_media_atomic, is_video_message
from src.message_parsing import parse_message
from src.ntfy_alert import send_ntfy_alert
from src.reference_bg import load_manifest
from src.retention import RetentionResult, clean_debug_cache, clean_local_media
from src.telegram_alert import send_telegram_alert

logger = logging.getLogger("live_watcher")
TELEGRAM_ALERT_CATEGORIES = frozenset(
    {"animal_candidate", "incident_candidate", "resident_candidate", "neighbour_candidate"}
)
NTFY_ALERT_CATEGORIES = frozenset({"animal_candidate", "incident_candidate"})
SYSTEM_EVENT_LABELS = {
    "battery_low": "Battery low",
    "battery_restored": "Battery restored",
    "power_out": "Power failure",
    "power_restored": "Power restored",
    "tamper": "Tamper detected",
    "tamper_restored": "Tamper restored",
    "supervision_error": "Device missing / supervision fault",
    "communication_failure": "Control-room communication failure",
    "panel_disarmed": "Panel disarmed",
    "panel_armed": "Panel armed",
}
SYSTEM_FAILURE_EVENTS = frozenset(
    {
        "battery_low",
        "power_out",
        "tamper",
        "supervision_error",
        "communication_failure",
        "panel_disarmed",
    }
)


def _message_timestamp(message: Any) -> tuple[datetime, str]:
    value = message.date
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    value = value.astimezone(UTC)
    return value, value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def format_alert(row: sqlite3.Row) -> str:
    category = str(row["category"]).removesuffix("_candidate").replace("_", " ")
    initial = str(row["initial_category"] or "not recorded").removesuffix("_candidate")
    complete = str(row["complete_category"] or "not received").removesuffix("_candidate")
    kind = str(row["notification_kind"])
    state = str(row["resolution_state"])

    if kind == "preliminary":
        heading = f"Possible {category} — startup clip"
        status = "Awaiting the completed clip; this is early, unconfirmed evidence."
    elif state == "confirmed" and kind == "resolution":
        heading = f"Update: {category} evidence confirmed"
        status = f"Startup: {initial}; completed clip: {complete}."
    elif state == "likely_resolved":
        prefix = "Update" if kind == "resolution" else "Review"
        heading = f"{prefix}: likely guard activity"
        status = (
            f"Startup: {initial}; completed clip: {complete}. "
            "The original urgent evidence remains recorded for review."
        )
    elif state == "conflicting":
        prefix = "Update" if kind == "resolution" else "Alert"
        heading = f"{prefix}: conflicting evidence — review required"
        status = f"Startup: {initial}; completed clip: {complete}. This has not been cleared."
    elif state == "unconfirmed":
        heading = f"Unconfirmed {category}"
        status = "The completed sibling was not received before the wait expired."
    else:
        heading = f"Perimeter alert: {category}"
        status = "Completed or standalone evidence."

    return (
        f"{heading}\n"
        f"Status: {status}\n"
        f"Camera: {row['camera_id']}\n"
        f"Time: {row['timestamp']}\n"
        f"Reason: {row['evidence_reason'] or row['reason']}\n"
        f"Event: {row['event_key']}"
    )


def alert_title(row: sqlite3.Row) -> str:
    state = str(row["resolution_state"])
    if str(row["notification_kind"]) == "preliminary":
        return f"Possible perimeter {str(row['category']).removesuffix('_candidate')}"
    if state == "likely_resolved":
        return "Perimeter update: likely guard"
    if state == "conflicting":
        return "Perimeter update: review required"
    if state == "unconfirmed":
        return "Perimeter update: unconfirmed"
    return f"Perimeter alert: {str(row['category']).removesuffix('_candidate')}"


def format_system_alert(row: sqlite3.Row) -> str:
    event_type = str(row["event_type"])
    label = SYSTEM_EVENT_LABELS.get(event_type, event_type.replace("_", " ").title())
    heading = "Security-system alert" if event_type in SYSTEM_FAILURE_EVENTS else "System update"
    camera = str(row["camera_id"] or "unknown")
    source = "Site / panel" if camera == "unknown" else camera
    return (
        f"{heading}: {label}\n"
        f"Source: {source}\n"
        f"Time: {row['timestamp']}\n"
        f"Message: {row['raw_text'] or label}\n"
        f"Source ID: {row['message_id']}"
    )


class LiveWatcher:
    """Process messages and event state serially for deterministic ordering."""

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        db_path: Path,
        client: Any,
        channel_id: str,
        cameras: CamerasConfig,
        thresholds: ThresholdsConfig,
        config: AppConfig,
        reference_entries: Any = None,
        now: Callable[[], datetime] | None = None,
        analysis_fn: Callable[..., ClipAnalysis] = analyze_clip,
        telegram_bot: Any = None,
        ntfy_session: Any = None,
    ) -> None:
        self.conn = conn
        self.db_path = db_path
        self.client = client
        self.channel_id = channel_id
        self.cameras = cameras
        self.thresholds = thresholds
        self.config = config
        self.reference_entries = (
            load_manifest("data/reference_bg") if reference_entries is None else reference_entries
        )
        self.now = now or (lambda: datetime.now(UTC))
        self.analysis_fn = analysis_fn
        self.telegram_bot = telegram_bot
        self.ntfy_session = ntfy_session
        self._lock = asyncio.Lock()
        self._next_retention_at: datetime | None = None

    def recover(self) -> int:
        ambiguous = live_state.recover_interrupted(self.conn)
        live_state.enqueue_missing_deliveries(
            self.conn,
            transports=self._transports("incident_candidate"),
            now=self.now(),
        )
        self.clean_media_if_due(force=True)
        return ambiguous

    def clean_media_if_due(self, *, force: bool = False) -> RetentionResult | None:
        if not self.config.media_retention_enabled:
            return None
        now = self.now()
        if not force and self._next_retention_at is not None and now < self._next_retention_at:
            return None
        result = clean_local_media(self.conn, data_root=self.db_path.parent)
        debug_result = clean_debug_cache(
            self.conn, debug_root=self.db_path.parent / "debug"
        )
        self._next_retention_at = now + timedelta(
            seconds=self.config.media_retention_interval_seconds
        )
        if result.deleted or result.missing or result.unsafe or debug_result.deleted:
            logger.info(
                "live_media_retention",
                extra={
                    "scanned": result.scanned,
                    "kept": result.kept,
                    "deleted": result.deleted,
                    "missing": result.missing,
                    "unsafe": result.unsafe,
                    "debug_scanned": debug_result.scanned,
                    "debug_kept": debug_result.kept,
                    "debug_deleted": debug_result.deleted,
                },
            )
        return result

    async def retrieve_media(self, clip: Any) -> Path | None:
        """Restore one DB-known source clip through Telethon after retention."""
        async with self._lock:
            if str(clip.channel_id) != self.channel_id:
                return None
            row = db.get_clip(self.conn, self.channel_id, int(clip.message_id))
            if row is None:
                return None
            if row["file_path"]:
                existing = Path(str(row["file_path"]))
                if existing.is_file():
                    return existing
            message = await self.client.get_messages(
                resolve_channel_ref(str(self.config.source_channel)), ids=int(clip.message_id)
            )
            if message is None or not is_video_message(message):
                return None
            destination = (
                self.config.live_media_dir / str(clip.camera_id) / f"{clip.message_id}.mp4"
            )
            await download_media_atomic(self.client, message, destination)
            db.set_clip_file_path(
                self.conn,
                channel_id=self.channel_id,
                message_id=int(clip.message_id),
                file_path=str(destination),
            )
            logger.info(
                "live_media_retrieved",
                extra={"camera_id": clip.camera_id, "message_id": clip.message_id},
            )
            return destination

    def _analyze(
        self, *, channel_id: str, message_id: int, path: Path, timestamp: str, camera: Any
    ) -> ClipAnalysis:
        return self.analysis_fn(
            conn=self.conn,
            channel_id=channel_id,
            message_id=message_id,
            video_path=path,
            timestamp=timestamp,
            camera=camera,
            thresholds=self.thresholds,
            reference_entries=self.reference_entries,
        )

    async def handle_message(self, message: Any) -> None:
        async with self._lock:
            message_id = int(message.id)
            if not live_state.begin_message(self.conn, self.channel_id, message_id):
                return
            try:
                await self._process_claimed(message)
            except Exception as exc:
                live_state.finish_message(
                    self.conn,
                    self.channel_id,
                    message_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                logger.exception("live_message_failed", extra={"message_id": message_id})

    async def _process_claimed(self, message: Any) -> None:
        message_id = int(message.id)
        received_at, timestamp = _message_timestamp(message)
        text = getattr(message, "message", None)
        parsed = parse_message(
            text=text,
            has_media=is_video_message(message),
            cameras=self.cameras,
        )
        if parsed.kind == "system_event":
            event_type = str(parsed.event_type)
            live_state.record_system_event(
                self.conn,
                channel_id=self.channel_id,
                message_id=message_id,
                timestamp=timestamp,
                camera_id=parsed.camera_id,
                event_type=event_type,
                raw_text=text,
                transports=self._system_transports(event_type),
                now=self.now(),
            )
            live_state.finish_message(
                self.conn, self.channel_id, message_id, status="processed"
            )
            await self.tick()
            return
        if parsed.kind != "clip":
            live_state.finish_message(self.conn, self.channel_id, message_id, status="ignored")
            return

        camera_id = str(parsed.camera_id)
        camera = self.cameras.by_id(camera_id)
        db.upsert_clip(
            self.conn,
            channel_id=self.channel_id,
            message_id=message_id,
            camera_id=camera_id,
            timestamp=timestamp,
            caption=text,
            file_path=None,
            source="live",
        )
        if camera is None:
            live_state.finish_message(
                self.conn,
                self.channel_id,
                message_id,
                status="ignored",
                error=f"unknown camera: {camera_id}",
            )
            return

        path = self.config.live_media_dir / camera_id / f"{message_id}.mp4"
        if not path.is_file():
            await download_media_atomic(self.client, message, path)
        db.set_clip_file_path(
            self.conn,
            channel_id=self.channel_id,
            message_id=message_id,
            file_path=str(path),
        )
        analysis = self._analyze(
            channel_id=self.channel_id,
            message_id=message_id,
            path=path,
            timestamp=timestamp,
            camera=camera,
        )
        phase = event_phase(text)
        key = message_event_key(camera_id, text, message_id)
        completed_sibling_exists = phase == "initial" and any(
            row["phase"] == "complete" for row in live_state.event_clips(self.conn, key)
        )
        deadline = (
            received_at + timedelta(seconds=self.config.event_wait_seconds)
            if phase == "initial" and not completed_sibling_exists
            else self.now()
        )
        live_state.add_analyzed_clip(
            self.conn,
            event_key=key,
            camera_id=camera_id,
            deadline_at=deadline,
            channel_id=self.channel_id,
            message_id=message_id,
            phase=phase,
            category=analysis.classification.category,
            reason=analysis.classification.reason,
            blinding_foreground=analysis.blinding_foreground,
            features=analysis.features,
        )
        if phase == "initial" and analysis.classification.category in PRELIMINARY_CATEGORIES:
            live_state.enqueue_preliminary(
                self.conn,
                event_key=key,
                category=analysis.classification.category,
                reason=analysis.classification.reason,
                representative_channel_id=self.channel_id,
                representative_message_id=message_id,
                transports=self._transports(analysis.classification.category),
                now=self.now(),
            )
        live_state.finish_message(
            self.conn,
            self.channel_id,
            message_id,
            status="processed",
            event_key=key,
        )
        logger.info(
            "live_clip_analyzed",
            extra={
                "message_id": message_id,
                "camera_id": camera_id,
                "event_key": key,
                "phase": phase,
                "category": analysis.classification.category,
                "reason": analysis.classification.reason,
                "blinding_foreground": analysis.blinding_foreground,
            },
        )
        await self.tick()

    def _transports(self, category: str) -> tuple[str, ...]:
        if category not in TELEGRAM_ALERT_CATEGORIES | NTFY_ALERT_CATEGORIES:
            return ()
        transports = []
        if (
            category in TELEGRAM_ALERT_CATEGORIES
            and self.config.telegram_bot_token
            and self.config.alert_channel_id
        ):
            transports.append("telegram")
        if (
            category in NTFY_ALERT_CATEGORIES
            and self.config.ntfy_base_url
            and self.config.ntfy_topic
        ):
            transports.append("ntfy")
        return tuple(transports)

    def _system_transports(self, event_type: str) -> tuple[str, ...]:
        transports = []
        if self.config.telegram_bot_token and self.config.alert_channel_id:
            transports.append("telegram")
        if (
            event_type in SYSTEM_EVENT_LABELS
            and self.config.ntfy_base_url
            and self.config.ntfy_topic
        ):
            transports.append("ntfy")
        return tuple(transports)

    def finalize_due_events(self) -> int:
        count = 0
        now = self.now()
        for key in live_state.due_event_keys(self.conn, now):
            clips = live_state.event_clips(self.conn, key)
            if not clips:
                continue
            resolution = resolve_event(clips)
            representative = resolution.representative
            notification = resolution.notification_representative
            if live_state.finalize_event(
                self.conn,
                event_key=key,
                category=resolution.category,
                reason=resolution.reason,
                representative_channel_id=str(representative["channel_id"]),
                representative_message_id=int(representative["message_id"]),
                notification_channel_id=str(notification["channel_id"]),
                notification_message_id=int(notification["message_id"]),
                resolution_state=resolution.resolution_state,
                transports=self._transports(resolution.category),
                now=now,
            ):
                count += 1
                logger.info(
                    "live_event_finalized",
                    extra={
                        "event_key": key,
                        "category": resolution.category,
                        "resolution_state": resolution.resolution_state,
                        "clip_count": len(clips),
                    },
                )
        return count

    async def dispatch_due(self) -> int:
        delivered = 0
        now = self.now()
        for row in live_state.due_deliveries(self.conn, now):
            key = str(row["event_key"])
            transport = str(row["transport"])
            kind = str(row["notification_kind"])
            if not live_state.claim_delivery(self.conn, key, transport, kind):
                continue
            try:
                message = format_alert(row)
                if transport == "telegram":
                    await send_telegram_alert(
                        message,
                        bot_token=str(self.config.telegram_bot_token),
                        chat_id=str(self.config.alert_channel_id),
                        video_path=row["file_path"],
                        debug_message_id=int(row["representative_message_id"]),
                        bot=self.telegram_bot,
                    )
                elif transport == "ntfy":
                    send_ntfy_alert(
                        message,
                        base_url=str(self.config.ntfy_base_url),
                        topic=str(self.config.ntfy_topic),
                        priority=(
                            self.config.ntfy_priority
                            if row["category"] == "incident_candidate"
                            and (
                                kind == "preliminary"
                                or (kind == "final" and row["resolution_state"] == "confirmed")
                            )
                            else "default"
                        ),
                        title=alert_title(row),
                        token=self.config.ntfy_token,
                        session=self.ntfy_session,
                    )
                else:
                    raise RuntimeError(f"unknown delivery transport: {transport}")
            except Exception as exc:
                exponent = min(int(row["attempt_count"]), 20)
                delay = min(
                    self.config.delivery_retry_max_seconds,
                    self.config.delivery_retry_base_seconds * (2**exponent),
                )
                live_state.finish_delivery(
                    self.conn,
                    key,
                    transport,
                    kind,
                    delivered=False,
                    next_attempt_at=now + timedelta(seconds=delay),
                    error=f"{type(exc).__name__}: {exc}",
                )
                logger.exception(
                    "live_delivery_failed", extra={"event_key": key, "transport": transport}
                )
            else:
                live_state.finish_delivery(
                    self.conn,
                    key,
                    transport,
                    kind,
                    delivered=True,
                    next_attempt_at=now,
                )
                delivered += 1
        return delivered

    async def dispatch_due_system_events(self) -> int:
        delivered = 0
        now = self.now()
        for row in live_state.due_system_deliveries(self.conn, now):
            channel_id = str(row["channel_id"])
            message_id = int(row["message_id"])
            transport = str(row["transport"])
            if not live_state.claim_system_delivery(
                self.conn, channel_id, message_id, transport
            ):
                continue
            try:
                message = format_system_alert(row)
                if transport == "telegram":
                    await send_telegram_alert(
                        message,
                        bot_token=str(self.config.telegram_bot_token),
                        chat_id=str(self.config.alert_channel_id),
                        bot=self.telegram_bot,
                    )
                elif transport == "ntfy":
                    event_type = str(row["event_type"])
                    send_ntfy_alert(
                        message,
                        base_url=str(self.config.ntfy_base_url),
                        topic=str(self.config.ntfy_topic),
                        priority="default",
                        title=f"System: {SYSTEM_EVENT_LABELS.get(event_type, event_type)}",
                        token=self.config.ntfy_token,
                        session=self.ntfy_session,
                    )
                else:
                    raise RuntimeError(f"unknown delivery transport: {transport}")
            except Exception as exc:
                exponent = min(int(row["attempt_count"]), 20)
                delay = min(
                    self.config.delivery_retry_max_seconds,
                    self.config.delivery_retry_base_seconds * (2**exponent),
                )
                live_state.finish_system_delivery(
                    self.conn,
                    channel_id,
                    message_id,
                    transport,
                    delivered=False,
                    next_attempt_at=now + timedelta(seconds=delay),
                    error=f"{type(exc).__name__}: {exc}",
                )
                logger.exception(
                    "system_delivery_failed",
                    extra={
                        "message_id": message_id,
                        "event_type": row["event_type"],
                        "transport": transport,
                    },
                )
            else:
                live_state.finish_system_delivery(
                    self.conn,
                    channel_id,
                    message_id,
                    transport,
                    delivered=True,
                    next_attempt_at=now,
                )
                delivered += 1
        return delivered

    async def tick(self) -> None:
        self.finalize_due_events()
        await self.dispatch_due()
        await self.dispatch_due_system_events()
        self.clean_media_if_due()

    async def maintenance(self) -> None:
        async with self._lock:
            await self.tick()
