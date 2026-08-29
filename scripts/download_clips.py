"""Phase 0c: download a small clip subset for the CV feasibility spike.

Downloads actual video files for a hand-picked set of cameras (typically a
narrow date window around a known incident) using rows already present in
`clips` from `scripts/meta_backfill.py`. This is deliberately small-scale --
a few clips per camera, enough to hand-label and run `scripts/spike.py`
against -- not the resumable full-history backfill (that's Phase 1's
`src/backfill.py`, which handles every message type, flood-wait/reconnect,
and a deterministic manifest).

Selection: clips matching `--camera` with no `file_path` yet, optionally
restricted to a `[--since, --until)` timestamp window, capped at
`--limit-per-camera` per camera. Downloads to
`data/history/{camera_id}/{message_id}.mp4` via a `.part` temp file then an
atomic rename, and records the path back into `clips.file_path` so
`scripts/label.py` and `scripts/spike.py` can find it.

A single clip that fails to download (Telegram-side timeout, oversized file,
etc.) is logged as `status: failed` and skipped rather than aborting the run --
useful for unattended full-history passes, but this still has no flood-wait
backoff or reconnect handling, so a wholesale connection loss will still need
a rerun (safe to do: candidate selection re-queries `file_path IS NULL`).

Run:
    uv run python scripts/download_clips.py --camera cam01b --camera cam07 --camera cam05 \\
        --since 2026-07-20 --until 2026-07-22 --limit-per-camera 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db  # noqa: E402
from src.config import load_app_config, resolve_channel_ref  # noqa: E402

logger = logging.getLogger("download_clips")


def select_download_candidates(
    conn: Any,
    *,
    cameras: list[str],
    since: str | None = None,
    until: str | None = None,
    limit_per_camera: int = 10,
) -> list[dict[str, Any]]:
    """Clips needing a download: matching camera, no file yet, in-window, capped per camera."""
    candidates: list[dict[str, Any]] = []
    for camera_id in cameras:
        count = 0
        for row in db.iter_clips(conn, camera_id=camera_id):
            if row["file_path"]:
                continue
            if since is not None and row["timestamp"] < since:
                continue
            if until is not None and row["timestamp"] >= until:
                continue
            candidates.append(dict(row))
            count += 1
            if count >= limit_per_camera:
                break
    return candidates


def clip_download_path(out_dir: str, camera_id: str, message_id: int) -> Path:
    return Path(out_dir) / camera_id / f"{message_id}.mp4"


async def download_one(client: Any, channel_ref: Any, message_id: int, dest: Path) -> bool:
    """Download one message's media to `dest` via a `.part` temp file + atomic rename.

    Returns False, leaving `dest` untouched, if the message no longer has media.
    """
    message = await client.get_messages(channel_ref, ids=message_id)
    if message is None or not (getattr(message, "video", None) or message.document):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")
    await client.download_media(message, file=str(part_path))
    part_path.rename(dest)
    return True


async def run_download(
    conn: Any,
    client: Any,
    *,
    channel_ref: Any,
    channel_id: str,
    candidates: list[dict[str, Any]],
    out_dir: str,
) -> dict[str, int]:
    counts = {"downloaded": 0, "skipped_no_media": 0, "failed": 0}
    for clip in candidates:
        dest = clip_download_path(out_dir, clip["camera_id"], clip["message_id"])
        try:
            downloaded = await download_one(client, channel_ref, clip["message_id"], dest)
        except Exception as exc:
            # A single bad message (Telegram-side timeout, oversized file, etc.) shouldn't
            # abort the whole run -- log it and keep going so a full-history pass can finish.
            counts["failed"] += 1
            logger.info(
                json.dumps(
                    {"message_id": clip["message_id"], "status": "failed", "error": str(exc)}
                )
            )
            continue
        if not downloaded:
            counts["skipped_no_media"] += 1
            logger.info(
                json.dumps({"message_id": clip["message_id"], "status": "skipped_no_media"})
            )
            continue
        db.set_clip_file_path(
            conn, channel_id=channel_id, message_id=clip["message_id"], file_path=str(dest)
        )
        counts["downloaded"] += 1
        logger.info(
            json.dumps(
                {"message_id": clip["message_id"], "status": "downloaded", "file_path": str(dest)}
            )
        )
    conn.commit()
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera", action="append", required=True, dest="cameras", help="Camera id (repeatable)"
    )
    parser.add_argument("--since", default=None, help="ISO timestamp lower bound, inclusive")
    parser.add_argument("--until", default=None, help="ISO timestamp upper bound, exclusive")
    parser.add_argument("--limit-per-camera", type=int, default=10)
    parser.add_argument("--out-dir", default="data/history")
    return parser.parse_args(argv)


async def main() -> None:  # pragma: no cover - requires real Telegram credentials
    from telethon import TelegramClient

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    app_cfg = load_app_config()
    conn = db.connect(app_cfg.db_path)

    candidates = select_download_candidates(
        conn,
        cameras=args.cameras,
        since=args.since,
        until=args.until,
        limit_per_camera=args.limit_per_camera,
    )
    if not candidates:
        print("No matching clips without a local file_path found.")
        conn.close()
        return

    client = TelegramClient(
        str(app_cfg.telegram_session_path), app_cfg.telegram_api_id, app_cfg.telegram_api_hash
    )
    async with client:
        channel_ref = resolve_channel_ref(app_cfg.source_channel)
        counts = await run_download(
            conn,
            client,
            channel_ref=channel_ref,
            channel_id=str(app_cfg.source_channel),
            candidates=candidates,
            out_dir=args.out_dir,
        )
    conn.close()
    logger.info(json.dumps({"summary": counts}))


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
