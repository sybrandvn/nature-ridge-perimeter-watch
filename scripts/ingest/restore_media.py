"""Resumably restore every missing clip from the Telegram source channel.

Rows are selected only when ``clips.file_path`` is NULL. Each successful media
download uses a temporary file and atomic rename before updating SQLite, so an
interruption is safe: rerun the same command and it resumes remaining rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from src import db
from src.config import load_app_config, resolve_channel_ref
from src.media_download import download_media_atomic, is_video_message

ProgressFn = Callable[[dict[str, int]], None]
SleepFn = Callable[[float], Awaitable[None]]


def missing_clips(
    conn: Any, *, camera_id: str | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    query = "SELECT * FROM clips WHERE file_path IS NULL"
    params: list[Any] = []
    if camera_id is not None:
        query += " AND camera_id = ?"
        params.append(camera_id)
    query += " ORDER BY timestamp, message_id"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    return [dict(row) for row in conn.execute(query, params)]


async def _retry(
    operation: Callable[[], Awaitable[Any]], *, attempts: int, sleep: SleepFn
) -> Any:
    for attempt in range(attempts):
        try:
            return await operation()
        except Exception as exc:
            if attempt + 1 >= attempts:
                raise
            requested = float(getattr(exc, "seconds", 0) or 0)
            delay = max(float(2**attempt), requested + 1.0)
            await sleep(min(delay, 600.0))
    raise AssertionError("unreachable")


async def restore_missing_media(
    conn: Any,
    client: Any,
    *,
    channel_ref: Any,
    clips: list[dict[str, Any]],
    out_dir: Path,
    batch_size: int = 100,
    concurrency: int = 8,
    attempts: int = 5,
    sleep: SleepFn = asyncio.sleep,
    progress: ProgressFn | None = None,
) -> dict[str, int]:
    if batch_size <= 0 or concurrency <= 0 or attempts <= 0:
        raise ValueError("batch_size, concurrency and attempts must be greater than zero")
    counts = {"selected": len(clips), "restored": 0, "no_media": 0, "failed": 0}
    semaphore = asyncio.Semaphore(concurrency)
    for offset in range(0, len(clips), batch_size):
        batch = clips[offset : offset + batch_size]
        ids = [int(clip["message_id"]) for clip in batch]

        async def fetch_batch(ids=ids):
            return await client.get_messages(channel_ref, ids=ids)

        try:
            messages = await _retry(fetch_batch, attempts=attempts, sleep=sleep)
        except Exception:
            counts["failed"] += len(batch)
            if progress:
                progress(dict(counts))
            continue
        if not isinstance(messages, (list, tuple)):
            messages = [messages]
        by_id = {
            int(message.id): message
            for message in messages
            if message is not None and getattr(message, "id", None) is not None
        }
        async def restore_one(clip: dict[str, Any], by_id=by_id) -> None:
            async with semaphore:
                message_id = int(clip["message_id"])
                message = by_id.get(message_id)
                if message is None or not is_video_message(message):
                    counts["no_media"] += 1
                    return
                destination = out_dir / str(clip["camera_id"]) / f"{message_id}.mp4"

                async def download(message=message, destination=destination):
                    return await download_media_atomic(client, message, destination)

                try:
                    await _retry(download, attempts=attempts, sleep=sleep)
                except Exception:
                    counts["failed"] += 1
                    return
                db.set_clip_file_path(
                    conn,
                    channel_id=str(clip["channel_id"]),
                    message_id=message_id,
                    file_path=str(destination),
                )
                counts["restored"] += 1

        await asyncio.gather(*(restore_one(clip) for clip in batch))
        if progress:
            progress(dict(counts))
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="data/history")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


async def run(argv: list[str] | None = None) -> dict[str, int]:  # pragma: no cover - network CLI
    from telethon import TelegramClient

    args = parse_args(argv)
    config = load_app_config()
    conn = db.connect(config.db_path)
    clips = missing_clips(conn, camera_id=args.camera, limit=args.limit)
    if args.dry_run:
        result = {"selected": len(clips), "restored": 0, "no_media": 0, "failed": 0}
        print(json.dumps(result, sort_keys=True))
        conn.close()
        return result

    client = TelegramClient(
        str(config.telegram_session_path), config.telegram_api_id, config.telegram_api_hash
    )

    def show_progress(counts: dict[str, int]) -> None:
        completed = counts["restored"] + counts["no_media"] + counts["failed"]
        if completed % 500 == 0 or completed == counts["selected"]:
            print(json.dumps(counts, sort_keys=True), flush=True)

    try:
        async with client:
            result = await restore_missing_media(
                conn,
                client,
                channel_ref=resolve_channel_ref(str(config.source_channel)),
                clips=clips,
                out_dir=Path(args.out_dir),
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                attempts=args.attempts,
                progress=show_progress,
            )
    finally:
        conn.close()
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:  # pragma: no cover
    asyncio.run(run())


if __name__ == "__main__":
    main()
