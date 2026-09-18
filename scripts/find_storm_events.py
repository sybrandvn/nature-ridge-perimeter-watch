"""Cross-camera corroboration for environment/storm candidate discovery.

Scans every downloaded clip (labelled or not) for `src.classify.classify`'s
`environment_candidate` rule, then groups clips from fence-order-adjacent
cameras (`config/cameras.yaml`'s `order`) that both fire within a short time
window into distinct multi-camera "events" (`src.storm_events`). A lone
camera's own noise doesn't corroborate; real wind/storm activity affecting a
stretch of fence tends to.

Validated 2026-09-04 on this repo's real corpus, across two independent review
batches, at the default `--window-minutes 15 --neighbor-distance 2`: 100%
precision (11/11 clips correctly `environment`). Widening the search (e.g.
`--window-minutes 30 --neighbor-distance 3`) finds more real events but also a
different false-positive mode: a guard walking a stretch of adjacent cameras
triggers them in sequence within the window, indistinguishable in shape from a
real multi-camera storm on some camera clusters. Always review the output
visually (`scripts/label.py --message-ids-file`) before trusting it -- this
script only proposes candidates, it never labels anything. See
`/memories/repo/nature-ridge-conventions.md` for the full validation history
and which camera clusters are known-reliable vs. known-guard-patrol-prone.

Run:
    uv run python scripts/find_storm_events.py --workers 8 \\
        --out data/reports/storm_events_2026-09-04.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backtest import ExtractFn, iter_clips_with_files  # noqa: E402
from src import db  # noqa: E402
from src.classify import classify  # noqa: E402
from src.config import CamerasConfig, load_app_config, load_cameras_config  # noqa: E402
from src.features import daylight_hint, is_twilight  # noqa: E402
from src.scoring import extract_clip_features  # noqa: E402
from src.storm_events import ClipSignal, find_corroborated_events  # noqa: E402

REPORT_COLUMNS = (
    "n_clips",
    "n_cameras",
    "cameras",
    "start",
    "end",
    "labels_present",
    "message_ids",
)


def collect_signals(
    conn: Any,
    cameras: CamerasConfig,
    *,
    camera_id: str | None = None,
    extract_fn: ExtractFn = extract_clip_features,
) -> list[dict[str, Any]]:
    """Single-process signal collection -- the testable, injectable core. The
    CLI uses a multiprocessing variant of this same logic for full-corpus runs
    (see `_collect_signals_parallel`), matching `scripts.rank_candidates`."""
    unknown_cameras: set[str] = set()
    rows: list[dict[str, Any]] = []
    for clip in iter_clips_with_files(conn, camera_id=camera_id):
        camera = cameras.by_id(clip["camera_id"])
        if camera is None:
            unknown_cameras.add(clip["camera_id"])
            continue
        features = extract_fn(
            clip["file_path"],
            camera.zone_at(clip["timestamp"]),
            daylight_hint=daylight_hint(clip["timestamp"]),
        )
        rows.append(
            {
                "camera_id": clip["camera_id"],
                "message_id": clip["message_id"],
                "timestamp": clip["timestamp"],
                "label": clip["label"],
                "is_candidate": classify(features) == "environment_candidate",
            }
        )
    for camera_id_ in sorted(unknown_cameras):
        print(f"skipped: no config/cameras.yaml entry for {camera_id_!r}", file=sys.stderr)
    return rows


_WORKER_CAMERAS: CamerasConfig | None = None


def _init_worker(cameras_path: str) -> None:
    global _WORKER_CAMERAS
    _WORKER_CAMERAS = load_cameras_config(cameras_path)


def _worker(clip: dict[str, Any]) -> dict[str, Any] | None:
    camera = _WORKER_CAMERAS.by_id(clip["camera_id"]) if _WORKER_CAMERAS else None
    if camera is None:
        return None
    try:
        features = extract_clip_features(
            clip["file_path"],
            camera.zone_at(clip["timestamp"]),
            daylight_hint=daylight_hint(clip["timestamp"]),
        )
    except Exception:
        features = None
    return {
        "camera_id": clip["camera_id"],
        "message_id": clip["message_id"],
        "timestamp": clip["timestamp"],
        "label": clip["label"],
        "is_candidate": classify(features) == "environment_candidate",
    }


def _collect_signals_parallel(
    conn: Any, cameras_path: str, *, workers: int
) -> list[dict[str, Any]]:
    clips = list(iter_clips_with_files(conn))
    print(f"clips with a file: {len(clips)}")
    rows: list[dict[str, Any]] = []
    with Pool(workers, initializer=_init_worker, initargs=(cameras_path,)) as pool:
        for i, row in enumerate(pool.imap_unordered(_worker, clips, chunksize=40)):
            if row is not None:
                rows.append(row)
            if i % 2000 == 0:
                print(i, flush=True)
    return rows


def events_from_rows(
    rows: list[dict[str, Any]],
    cameras: CamerasConfig,
    *,
    window_minutes: float,
    neighbor_distance: int,
    exclude_twilight_minutes: float | None = None,
) -> list[list[ClipSignal]]:
    """`exclude_twilight_minutes`, when given, treats any clip within that many
    minutes of dawn or dusk (`src.features.is_twilight`) as a non-candidate --
    it can neither seed nor join an event, though it still appears in output
    rows as an ordinary non-candidate clip. Confirmed 2026-09-04: `blob_count`'s
    guard false-fire rate spikes to 38% in the 15-60 minute post-sunset window,
    which otherwise pollutes a wider (larger window/distance) sweep with
    guard-patrol-route false events -- see repo memory.
    """
    order = {c.id: c.order for c in cameras.cameras if c.order is not None}
    clips = [
        ClipSignal(
            camera_id=r["camera_id"],
            message_id=int(r["message_id"]),
            timestamp=r["timestamp"],
            is_candidate=(
                r["is_candidate"]
                and (
                    exclude_twilight_minutes is None
                    or not is_twilight(r["timestamp"], margin_minutes=exclude_twilight_minutes)
                )
            ),
        )
        for r in rows
        if r["timestamp"]
    ]
    return find_corroborated_events(
        clips, order, window_minutes=window_minutes, neighbor_distance=neighbor_distance
    )


def write_events(
    events: list[list[ClipSignal]],
    label_by_key: dict[tuple[str, int], str | None],
    out_path: str,
) -> list[tuple[str, int]]:
    """Writes the event CSV and returns the still-reviewable message ids (any
    member not already labelled `environment`) for a `.message_ids` sidecar,
    same convention as `scripts.rank_candidates`."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    to_review: list[tuple[str, int]] = []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for members in events:
            cams = sorted({c.camera_id for c in members})
            labels = sorted({label_by_key.get((c.camera_id, c.message_id)) or "" for c in members})
            writer.writerow(
                {
                    "n_clips": len(members),
                    "n_cameras": len(cams),
                    "cameras": ",".join(cams),
                    "start": members[0].timestamp,
                    "end": members[-1].timestamp,
                    "labels_present": ",".join(labels),
                    "message_ids": ";".join(f"{c.camera_id}/{c.message_id}" for c in members),
                }
            )
            for c in members:
                if label_by_key.get((c.camera_id, c.message_id)) != "environment":
                    to_review.append((c.camera_id, c.message_id))
    ids_path = str(Path(out_path).with_suffix(".message_ids"))
    with open(ids_path, "w") as f:
        for camera_id, message_id in to_review:
            f.write(f"{message_id}  # {camera_id}\n")
    return to_review


def main() -> None:  # pragma: no cover - requires the real corpus/db
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera", default=None, help="Restrict signal collection to one camera id"
    )
    parser.add_argument("--workers", type=int, default=1, help="1 runs single-process")
    parser.add_argument("--window-minutes", type=float, default=15.0)
    parser.add_argument("--neighbor-distance", type=int, default=2)
    parser.add_argument(
        "--exclude-twilight-minutes",
        type=float,
        default=None,
        help="treat clips within this many minutes of dawn/dusk as non-candidates "
        "(e.g. 60 avoids the measured blob_count dusk false-fire window)",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    app_cfg = load_app_config(require_telegram=False)
    cameras_cfg = load_cameras_config("config/cameras.yaml")
    conn = db.connect(app_cfg.db_path)

    if args.workers > 1 and args.camera is None:
        rows = _collect_signals_parallel(conn, "config/cameras.yaml", workers=args.workers)
    else:
        rows = collect_signals(conn, cameras_cfg, camera_id=args.camera)
    conn.close()

    events = events_from_rows(
        rows,
        cameras_cfg,
        window_minutes=args.window_minutes,
        neighbor_distance=args.neighbor_distance,
        exclude_twilight_minutes=args.exclude_twilight_minutes,
    )
    label_by_key = {(r["camera_id"], int(r["message_id"])): r["label"] for r in rows}
    to_review = write_events(events, label_by_key, args.out)

    print(f"found {len(events)} multi-camera corroborated events")
    print(f"wrote {args.out} and {len(to_review)} reviewable ids to "
          f"{Path(args.out).with_suffix('.message_ids')}")


if __name__ == "__main__":
    main()
