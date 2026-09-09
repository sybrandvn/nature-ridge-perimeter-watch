"""Phase 0c screening: apply the gate-2 heuristic rules across every downloaded
clip -- labelled or not -- to surface animal/incident/guard candidates for a
human to review. This is a screening tool, not a classifier: every threshold
below is one already validated in docs/gate2_separability_finding.md against
real labelled footage, not invented for this script.

The classification rules themselves, and the full derivation of every threshold
they use, live in src/classify.py -- moved there 2026-09-09 (plan.md step 25).
This file is the screening/reporting harness around them: it walks downloaded
clips, extracts features, applies src.classify, and writes a CSV.

Run:
    uv run python scripts/backtest.py --out data/reports/backtest_2026-08-29.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.spike import FEATURE_COLUMNS, extract_clip_features  # noqa: E402
from src import db  # noqa: E402
from src.backtester import record_run  # noqa: E402
from src.classify import classify_detailed, is_blinding_foreground  # noqa: E402
from src.config import CamerasConfig, load_app_config, load_cameras_config  # noqa: E402
from src.features import daylight_hint, is_daylight  # noqa: E402
from src.reference_bg import (  # noqa: E402
    era_of,
    load_manifest,
    load_reference_image,
    reference_for,
)

ExtractFn = Callable[..., "dict[str, float] | None"]

# Identity/verdict columns, then EVERY numeric feature. Deliberately derived
# from FEATURE_COLUMNS rather than hand-listed: a hand-picked subset silently
# drops whatever was added last (recovered_fraction and longest_detection_run
# were both being computed and discarded before 2026-09-06), and re-running the
# scoring pass just to see one more column wastes minutes.
_IDENTITY_COLUMNS = (
    "channel_id",
    "message_id",
    "camera_id",
    "label",
    "category",
    "reason",
    "blinding_foreground",
)
_NON_NUMERIC = ("channel_id", "message_id", "camera_id", "label", "time_of_day", "is_daylight")
REPORT_COLUMNS = (
    *_IDENTITY_COLUMNS,
    *(c for c in FEATURE_COLUMNS if c not in _NON_NUMERIC),
)


def iter_clips_with_files(
    conn: Any, *, camera_id: str | None = None, labelled_only: bool = False
) -> Iterator[dict[str, Any]]:
    for row in db.iter_clips(conn, camera_id=camera_id):
        if not row["file_path"]:
            continue
        label_row = db.get_label(conn, row["channel_id"], row["message_id"])
        label = label_row["label"] if label_row is not None else None
        if labelled_only and label is None:
            continue
        yield {
            "channel_id": row["channel_id"],
            "message_id": row["message_id"],
            "camera_id": row["camera_id"],
            "timestamp": row["timestamp"],
            "file_path": row["file_path"],
            "label": label,
        }


def _reference_background(entries, root, camera, timestamp):
    """This camera's reference background for the clip's era and lighting, or
    None when nothing covers it. Same resolution as scripts/render_debug.py,
    so the screening run and the render agree on what the detector saw."""
    if not entries or timestamp is None:
        return None
    entry = reference_for(
        entries,
        camera.id,
        timestamp,
        era=era_of(camera, timestamp),
        daylight=is_daylight(timestamp),
    )
    return None if entry is None else load_reference_image(Path(root), entry)


def run_backtest(
    conn: Any,
    cameras: CamerasConfig,
    *,
    camera_id: str | None = None,
    labelled_only: bool = False,
    reference_entries: Any = None,
    reference_root: str = "data/reference_bg",
    extract_fn: ExtractFn = extract_clip_features,
) -> Iterator[dict[str, Any]]:
    unknown_cameras: set[str] = set()
    for clip in iter_clips_with_files(conn, camera_id=camera_id, labelled_only=labelled_only):
        camera = cameras.by_id(clip["camera_id"])
        if camera is None:
            unknown_cameras.add(clip["camera_id"])
            continue
        extra: dict[str, Any] = {}
        reference = _reference_background(
            reference_entries, reference_root, camera, clip["timestamp"]
        )
        if reference is not None:
            extra["reference_background"] = reference
        hint = daylight_hint(clip["timestamp"])
        if hint is not None:
            extra["daylight_hint"] = hint
        features = extract_fn(clip["file_path"], camera.zone_at(clip["timestamp"]), **extra)
        if features is not None and clip["timestamp"] is not None:
            # classify() needs the real exogenous signal, not an image
            # statistic -- see the module docstring's resident_candidate note.
            features["is_daylight"] = is_daylight(clip["timestamp"])
        result = classify_detailed(features)
        row = {
            "channel_id": clip["channel_id"],
            "message_id": clip["message_id"],
            "camera_id": clip["camera_id"],
            "label": clip["label"],
            "category": result.category,
            "reason": result.reason,
            "blinding_foreground": is_blinding_foreground(features),
        }
        for col in REPORT_COLUMNS[len(_IDENTITY_COLUMNS) :]:
            row[col] = None if features is None else features.get(col)
        yield row
    for camera_id_ in sorted(unknown_cameras):
        print(f"skipped: no config/cameras.yaml entry for {camera_id_!r}", file=sys.stderr)


def write_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:  # pragma: no cover - requires real downloaded footage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=None, help="Restrict to one camera id")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--labelled-only",
        action="store_true",
        help="Restrict to clips with a human label -- for before/after regression"
        " runs against the ~400-clip labelled corpus instead of all downloaded history",
    )
    parser.add_argument(
        "--reference-bg",
        default="data/reference_bg",
        help="per-camera reference background directory (build with scripts/build_reference_bg.py)",
    )
    parser.add_argument(
        "--no-reference-bg",
        action="store_true",
        help="disable the reference-background scenery veto, for before/after comparison",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="skip recording this pass as an immutable backtest_runs/backtest_results row "
        "(src.backtester) -- for a scratch/exploratory run nobody needs to find again",
    )
    args = parser.parse_args()

    app_cfg = load_app_config(require_telegram=False)
    cameras_cfg = load_cameras_config("config/cameras.yaml")
    conn = db.connect(app_cfg.db_path)

    reference_entries = [] if args.no_reference_bg else load_manifest(args.reference_bg)
    rows = list(
        run_backtest(
            conn,
            cameras_cfg,
            camera_id=args.camera,
            labelled_only=args.labelled_only,
            reference_entries=reference_entries,
            reference_root=args.reference_bg,
        )
    )

    run_id = None
    if not args.no_record:
        run_id = record_run(conn, rows)
    conn.close()

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    if run_id is not None:
        print(f"Recorded as backtest run {run_id!r}")
    from collections import Counter

    print(Counter(r["category"] for r in rows))


if __name__ == "__main__":
    main()
