"""Run the durable Telethon source-channel watcher and alert dispatcher."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from scripts.render_debug import DEBUG_RENDER_VERSION, render_clip
from src import db
from src.bot import BotQueries, build_query_bot
from src.config import (
    load_app_config,
    load_cameras_config,
    load_thresholds_config,
    resolve_channel_ref,
)
from src.features import daylight_hint, is_daylight
from src.live_watcher import LiveWatcher
from src.logging_setup import configure_logging
from src.motion import extraction_fingerprint
from src.reference_bg import era_of, load_reference_image, reference_for

logger = logging.getLogger("watch")


def render_bot_debug_video(
    clip,
    source: Path,
    *,
    cameras,
    thresholds,
    reference_entries,
    output_root: Path,
) -> Path | None:
    """Render and cache the production detector view for one bot video."""
    camera = cameras.by_id(clip.camera_id)
    if camera is None:
        return None
    reference = None
    entry = reference_for(
        reference_entries,
        camera.id,
        clip.timestamp,
        era=era_of(camera, clip.timestamp),
        daylight=is_daylight(clip.timestamp),
    )
    if entry is not None:
        reference = load_reference_image(Path("data/reference_bg"), entry)
    fingerprint = extraction_fingerprint(
        video_path=source,
        motion_fingerprint=thresholds.motion_fingerprint(),
        zone=camera.zone_at(clip.timestamp),
        reference_background=reference,
        daylight_hint=daylight_hint(clip.timestamp),
    )
    destination = (
        output_root
        / camera.id
        / f"{clip.message_id}-{fingerprint[:12]}-r{DEBUG_RENDER_VERSION}.mp4"
    )
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.part.mp4")
    temporary.unlink(missing_ok=True)
    result = render_clip(
        str(source),
        camera.zone_at(clip.timestamp),
        out_path=str(temporary),
        title=f"{camera.id}/{clip.message_id}",
        scale=2,
        reference_background=reference,
        timestamp=clip.timestamp,
        motion_thresholds=thresholds.motion_thresholds(),
    )
    if result is None:
        temporary.unlink(missing_ok=True)
        return None
    Path(result).replace(destination)
    for old in destination.parent.glob(f"{clip.message_id}-*-r*.mp4"):
        if old != destination:
            old.unlink()
    return destination


def write_heartbeat(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text("ok\n")
    temporary.replace(path)


async def _maintenance_loop(watcher: LiveWatcher, heartbeat: Path, interval: float) -> None:
    while True:
        await watcher.maintenance()
        write_heartbeat(heartbeat)
        await asyncio.sleep(interval)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", default="config/cameras.yaml")
    parser.add_argument("--thresholds", default="config/thresholds.yaml")
    return parser.parse_args(argv)


async def run(argv: list[str] | None = None) -> None:  # pragma: no cover - real Telegram loop
    from telethon import TelegramClient, events

    configure_logging()
    args = parse_args(argv)
    config = load_app_config()
    cameras = load_cameras_config(args.cameras)
    thresholds = load_thresholds_config(args.thresholds)
    conn = db.connect(config.db_path)
    client = TelegramClient(
        str(config.telegram_session_path), config.telegram_api_id, config.telegram_api_hash
    )
    watcher = LiveWatcher(
        conn=conn,
        db_path=config.db_path,
        client=client,
        channel_id=str(config.source_channel),
        cameras=cameras,
        thresholds=thresholds,
        config=config,
    )
    ambiguous = watcher.recover()
    if ambiguous:
        logger.error("ambiguous_deliveries_require_review", extra={"count": ambiguous})
    write_heartbeat(config.watcher_heartbeat_path)

    query_application = None
    query_initialized = False
    if config.telegram_bot_token:
        try:
            query_application = build_query_bot(
                BotQueries(conn=conn, config=config, cameras=cameras),
                config.telegram_bot_token,
                media_loader=watcher.retrieve_media,
                debug_loader=lambda clip, source: asyncio.to_thread(
                    render_bot_debug_video,
                    clip,
                    source,
                    cameras=cameras,
                    thresholds=thresholds,
                    reference_entries=watcher.reference_entries,
                    output_root=config.live_media_dir.parent / "debug",
                ),
            )
            await query_application.initialize()
            query_initialized = True
            watcher.telegram_bot_username = query_application.bot.username
            await query_application.bot.set_my_commands(
                query_application.bot_data["commands"]
            )
            await query_application.start()
            if query_application.updater is None:
                raise RuntimeError("Telegram query bot has no polling updater")
            await query_application.updater.start_polling()
            logger.info(
                "query_bot_started",
                extra={
                    "trustee_count": len(config.bot_trustee_ids),
                    "security_count": len(config.bot_security_ids),
                },
            )
        except Exception:
            logger.exception("query_bot_start_failed")
            if query_application is not None and query_initialized:
                await query_application.shutdown()
            query_application = None

    async def on_message(event):
        await watcher.handle_message(event.message)

    client.add_event_handler(
        on_message,
        events.NewMessage(chats=resolve_channel_ref(str(config.source_channel))),
    )
    maintenance = asyncio.create_task(
        _maintenance_loop(
            watcher, config.watcher_heartbeat_path, config.watcher_poll_seconds
        )
    )
    try:
        async with client:
            logger.info("watcher_started", extra={"source_channel": config.source_channel})
            await client.run_until_disconnected()
    finally:
        maintenance.cancel()
        await asyncio.gather(maintenance, return_exceptions=True)
        if query_application is not None:
            if query_application.updater is not None:
                await query_application.updater.stop()
            await query_application.stop()
            await query_application.shutdown()
        conn.close()


def main() -> None:  # pragma: no cover
    asyncio.run(run())


if __name__ == "__main__":
    main()
