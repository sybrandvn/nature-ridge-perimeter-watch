"""Validate the container's configuration, persistent storage and media stack.

This is both the default image command and the healthcheck contract for the
future watcher. Readiness validates the real camera/threshold configuration,
writability of the mounted data directory, and SQLite schema compatibility.
``--heartbeat`` additionally checks a watcher-owned heartbeat file without
pretending that a watcher exists today. ``--media-smoke`` performs an explicit
H.264 write/read round trip for deployment verification.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from src import db
from src.config import load_cameras_config, load_thresholds_config
from src.video_encode import Mp4Writer


def check_readiness(
    *,
    data_dir: Path,
    db_path: Path,
    cameras_path: Path,
    thresholds_path: Path,
) -> dict[str, object]:
    """Raise on an unusable deployment and return inspectable readiness facts."""
    cameras = load_cameras_config(cameras_path)
    thresholds = load_thresholds_config(thresholds_path)

    data_dir.mkdir(parents=True, exist_ok=True)
    for relative in ("history", "live", "logs", "reference_bg", "reports"):
        (data_dir / relative).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=data_dir, prefix=".write-check-", delete=True) as probe:
        probe.write(b"ok")
        probe.flush()

    conn = db.connect(db_path)
    try:
        schema_version = int(conn.execute("SELECT version FROM schema_version").fetchone()[0])
        conn.execute("SELECT 1").fetchone()
    finally:
        conn.close()

    return {
        "ready": True,
        "camera_count": len(cameras.cameras),
        "schema_version": schema_version,
        "motion_fingerprint": thresholds.motion_fingerprint()[:12],
        "data_dir": str(data_dir),
        "db_path": str(db_path),
    }


def check_heartbeat(path: Path, *, max_age_seconds: float, now: float | None = None) -> float:
    """Return heartbeat age, raising when it is missing or stale."""
    if not path.is_file():
        raise RuntimeError(f"watcher heartbeat is missing: {path}")
    age = (time.time() if now is None else now) - path.stat().st_mtime
    if age < 0:
        age = 0.0
    if age > max_age_seconds:
        raise RuntimeError(
            f"watcher heartbeat is stale: {age:.1f}s old (maximum {max_age_seconds:.1f}s)"
        )
    return age


def check_media_round_trip(*, work_dir: Path) -> dict[str, object]:
    """Encode H.264 through the production writer and decode a frame again."""
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "container-media-smoke.mp4"
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    cv2.rectangle(frame, (12, 16), (44, 48), (40, 255, 40), thickness=-1)
    writer = Mp4Writer(str(output), fps=5.0, width=96, height=64)
    try:
        writer.write(frame)
        writer.write(np.roll(frame, 8, axis=1))
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(output))
    try:
        opened = capture.isOpened()
        readable, decoded = capture.read()
    finally:
        capture.release()
    if not opened or not readable or decoded is None:
        raise RuntimeError(f"encoded H.264 output is not readable: {output}")
    return {
        "media_smoke": True,
        "output": str(output),
        "bytes": output.stat().st_size,
        "decoded_shape": list(decoded.shape),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", action="store_true", help="validate config, storage and DB")
    parser.add_argument("--data-dir", default="/app/data")
    parser.add_argument(
        "--db-path",
        default=os.environ.get("DB_PATH", "/app/data/perimeter_watch.db"),
    )
    parser.add_argument("--cameras", default="config/cameras.yaml")
    parser.add_argument("--thresholds", default="config/thresholds.yaml")
    parser.add_argument("--heartbeat", help="require this watcher heartbeat file to be fresh")
    parser.add_argument("--max-heartbeat-age", type=float, default=120.0)
    parser.add_argument("--media-smoke", action="store_true", help="verify H.264 encode/decode")
    parser.add_argument("--media-work-dir", default="/tmp/perimeter-watch-media-smoke")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result: dict[str, object] = {}
    try:
        if args.readiness or (not args.heartbeat and not args.media_smoke):
            result.update(
                check_readiness(
                    data_dir=Path(args.data_dir),
                    db_path=Path(args.db_path),
                    cameras_path=Path(args.cameras),
                    thresholds_path=Path(args.thresholds),
                )
            )
        if args.heartbeat:
            result["heartbeat_age_seconds"] = check_heartbeat(
                Path(args.heartbeat), max_age_seconds=args.max_heartbeat_age
            )
        if args.media_smoke:
            result.update(check_media_round_trip(work_dir=Path(args.media_work_dir)))
    except Exception as error:
        print(
            json.dumps(
                {"ready": False, "error": f"{type(error).__name__}: {error}"},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
