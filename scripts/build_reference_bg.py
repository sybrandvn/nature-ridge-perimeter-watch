"""Build the per-camera reference backgrounds used to tell scenery from subjects.

Reads every downloaded clip from the database, groups them by camera, geometry
era, lighting and calendar quarter (see `src.reference_bg`), and writes one
median-of-medians PNG per bucket plus a manifest.

Output is derived data -- rebuild it rather than editing it, and rebuild it
after a camera is remounted, since a remount starts a new era in
`config/cameras.yaml` and the old reference no longer describes the scene.

Run:
    uv run python scripts/build_reference_bg.py
    uv run python scripts/build_reference_bg.py --camera cam05 --out data/reference_bg
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from src import db  # noqa: E402
from src.config import load_cameras_config  # noqa: E402
from src.reference_bg import (  # noqa: E402
    MANIFEST_NAME,
    MIN_CLIPS,
    ReferenceEntry,
    bucket_clips,
    clip_median_background,
    combine,
    sample_evenly,
)


def build(
    db_path: str, cameras_path: str, out_root: Path, only_camera: str | None, quiet: bool
) -> list[ReferenceEntry]:
    cameras = load_cameras_config(cameras_path)
    conn = db.connect(db_path)
    rows = conn.execute(
        "select camera_id, timestamp, file_path from clips where file_path is not null "
        "order by camera_id, timestamp"
    ).fetchall()

    by_camera: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        camera_id, timestamp, file_path = row[0], row[1], row[2]
        if only_camera and camera_id != only_camera:
            continue
        if Path(file_path).exists():
            by_camera.setdefault(camera_id, []).append((timestamp, file_path))

    entries: list[ReferenceEntry] = []
    for camera_id, clips in sorted(by_camera.items()):
        camera = cameras.by_id(camera_id)
        buckets = bucket_clips(clips, camera)
        timestamps = dict(clips and [(path, ts) for ts, path in clips])
        for index, (key, paths) in enumerate(sorted(buckets.items(), key=lambda kv: kv[0][1:])):
            era, daylight, _year, _quarter = key
            if len(paths) < MIN_CLIPS:
                if not quiet:
                    print(f"  {camera_id}: skip bucket {key} ({len(paths)} < {MIN_CLIPS} clips)")
                continue
            chosen = sample_evenly(paths)
            medians = [m for m in (clip_median_background(p) for p in chosen) if m is not None]
            reference = combine(medians)
            if reference is None:
                continue
            name = f"{'day' if daylight else 'night'}_{index:02d}.png"
            relative = Path(camera_id) / name
            (out_root / camera_id).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_root / relative), reference)
            covered = sorted(timestamps[p] for p in paths if p in timestamps)
            entries.append(
                ReferenceEntry(
                    camera_id=camera_id,
                    daylight=daylight,
                    era=era,
                    start=covered[0],
                    end=covered[-1],
                    clip_count=len(medians),
                    path=str(relative),
                )
            )
            if not quiet:
                print(
                    f"  {camera_id}: {name} from {len(medians)} of {len(paths)} clips "
                    f"({covered[0][:10]} to {covered[-1][:10]})"
                )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / MANIFEST_NAME).write_text(json.dumps([asdict(e) for e in entries], indent=2))
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/perimeter_watch.db")
    parser.add_argument("--cameras", default="config/cameras.yaml")
    parser.add_argument("--out", default="data/reference_bg")
    parser.add_argument("--camera", default=None, help="Build for one camera only")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    out_root = Path(args.out)
    entries = build(args.db, args.cameras, out_root, args.camera, args.quiet)
    print(f"wrote {len(entries)} reference backgrounds to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
