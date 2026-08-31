"""Phase 0c screening: apply the gate-2 heuristic rules across every downloaded
clip -- labelled or not -- to surface animal/incident/guard candidates for a
human to review. This is a screening tool, not a classifier: every threshold
below is one already validated in docs/gate2_separability_finding.md against
real labelled footage, not invented for this script.

Rules:
  - guard_candidate: green_light_ratio > 0.05 or green_light_flicker > 0.02
    (measured 2026-08-31 against the current tracker/detector on 200 labelled
    guard clips: 8.7% recall, degraded from the doc's original 61% -- see
    "DAYLIGHT GATE IS SELF-DEFEATING" in repo memory. Left unchanged anyway:
    every alternative gate design tried and rejected 2026-08-30, see
    "Daylight-gate fix investigated and ABANDONED" in repo memory.)
  - environment_candidate: blob_count > 10 (added 2026-08-31, checked before
    animal_or_incident_candidate/insect_candidate so a stormy/windy clip's
    scattered foliage blobs don't get read as a shape signal. Measured against
    222 labelled guard+environment clips: 59% environment recall (13/22), 4.5%
    guard false-fire (9/200), 0/15 leak against the full labelled animal+incident
    set -- blob_count is the strongest single discriminator found for this pair,
    AUC 0.933. Higher thresholds trade recall for guard false-fire; 10 is the
    highest threshold with zero animal+incident leak (cam15/15454's porcupine
    sits at exactly blob_count=10).)
  - animal_or_incident_candidate: aspect_ratio < 0.95 and green_light_ratio < 0.05
    (measured 2026-08-31 on the current tracker: this rule's own premise has
    inverted -- animal+incident median aspect_ratio is now 1.24-1.41, HIGHER
    than guard's 1.05, not lower as the original finding doc assumed. Left
    unchanged pending a real re-derivation of this rule, not touched this pass;
    still fires on 74/200 guard clips, a known contamination of this candidate
    pool -- do not trust its precision without re-measuring.) This category
    doesn't split animal vs incident -- there's no validated rule for that.
  - insect_candidate: jitter > 50 and solidity < 0.85
    (measured 2026-08-31: 0/22 environment clips now reach jitter>50 under the
    current tracker -- this rule is effectively dead, the persistent-tracking
    work from 2026-08-30 smooths out the erratic jitter it used to key on.
    Left in place, not removed, pending a decision on a replacement.)
  - no_motion: extract_clip_features found nothing to track
  - unclassified: motion detected but none of the above rules fired

Known limits (read before trusting a "storm" claim from this tool -- there
isn't one): there is no separate `storm` label and none should be added --
a storm/wind/rain trigger is just one cause of an `environment` clip (user
confirmed 2026-08-31). `environment_candidate` above identifies `environment`
clips generally (via scattered-blob-count, not a storm-specific signal); an
earlier ad-hoc "whole-frame motion fraction" idea was tried 2026-08-29
specifically as a storm discriminator against the 4 clips the user's notes
flagged as storms (cam15/9944, 9954, 9956, 18948) and did NOT separate them
from guard/environment clips on that metric. Every rule above also comes from a sample
of only 4-12 positive clips per class; treat every count this script produces
as a triage pointer, never as ground truth.

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

from scripts.spike import extract_clip_features  # noqa: E402
from src import db  # noqa: E402
from src.config import CamerasConfig, load_app_config, load_cameras_config  # noqa: E402

ExtractFn = Callable[..., "dict[str, float] | None"]

REPORT_COLUMNS = (
    "channel_id",
    "message_id",
    "camera_id",
    "label",
    "category",
    "aspect_ratio",
    "solidity",
    "green_light_ratio",
    "green_light_flicker",
    "jitter",
    "persistence",
    "outside_pixel_fraction",
    "path_length",
    "blob_count",
)


def classify(features: dict[str, float] | None) -> str:
    """Pure rule lookup -- see the module docstring for what each rule means and
    where its thresholds come from. `guard_candidate` is checked first: the
    green-light exclusion in `animal_or_incident_candidate` is what raises that
    rule's precision, so a clip matching both is a guard, not a double-count.
    `environment_candidate` is checked next, before the shape-based rules, so a
    stormy/windy clip's scattered blobs don't get read as a shape signal."""
    if features is None:
        return "no_motion"
    if features["green_light_ratio"] > 0.05 or features["green_light_flicker"] > 0.02:
        return "guard_candidate"
    if features["blob_count"] > 10:
        return "environment_candidate"
    if features["aspect_ratio"] < 0.95 and features["green_light_ratio"] < 0.05:
        return "animal_or_incident_candidate"
    if features["jitter"] > 50 and features["solidity"] < 0.85:
        return "insect_candidate"
    return "unclassified"


def iter_clips_with_files(conn: Any, *, camera_id: str | None = None) -> Iterator[dict[str, Any]]:
    for row in db.iter_clips(conn, camera_id=camera_id):
        if not row["file_path"]:
            continue
        label_row = db.get_label(conn, row["channel_id"], row["message_id"])
        yield {
            "channel_id": row["channel_id"],
            "message_id": row["message_id"],
            "camera_id": row["camera_id"],
            "timestamp": row["timestamp"],
            "file_path": row["file_path"],
            "label": label_row["label"] if label_row is not None else None,
        }


def run_backtest(
    conn: Any,
    cameras: CamerasConfig,
    *,
    camera_id: str | None = None,
    extract_fn: ExtractFn = extract_clip_features,
) -> Iterator[dict[str, Any]]:
    unknown_cameras: set[str] = set()
    for clip in iter_clips_with_files(conn, camera_id=camera_id):
        camera = cameras.by_id(clip["camera_id"])
        if camera is None:
            unknown_cameras.add(clip["camera_id"])
            continue
        features = extract_fn(clip["file_path"], camera.zone_at(clip["timestamp"]))
        row = {
            "channel_id": clip["channel_id"],
            "message_id": clip["message_id"],
            "camera_id": clip["camera_id"],
            "label": clip["label"],
            "category": classify(features),
        }
        for col in REPORT_COLUMNS[5:]:
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
    args = parser.parse_args()

    app_cfg = load_app_config(require_telegram=False)
    cameras_cfg = load_cameras_config("config/cameras.yaml")
    conn = db.connect(app_cfg.db_path)

    rows = list(run_backtest(conn, cameras_cfg, camera_id=args.camera))
    conn.close()

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    from collections import Counter

    print(Counter(r["category"] for r in rows))


if __name__ == "__main__":
    main()
