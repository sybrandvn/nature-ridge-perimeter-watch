"""Phase 0c screening: apply the gate-2 heuristic rules across every downloaded
clip -- labelled or not -- to surface animal/incident/guard candidates for a
human to review. This is a screening tool, not a classifier: every threshold
below is one already validated in docs/gate2_separability_finding.md against
real labelled footage, not invented for this script.

Rules:
  - guard_candidate: green_light_ratio > 0.05 or green_light_flicker > 0.02
    (61% recall / 2% false-fire / 0 animal+incident leakage on the labelled sample)
  - animal_or_incident_candidate: aspect_ratio < 0.95 and green_light_ratio < 0.05
    (75% recall / 22% precision on 12 labelled positives -- a screening signal,
    not a verdict; guard_candidate wins if both match, since the green-light
    exclusion is what raises this rule's precision)
  - insect_candidate: jitter > 50 and solidity < 0.85
    (55% of labelled `environment` clips)
  - no_motion: extract_clip_features found nothing to track
  - unclassified: motion detected but none of the above rules fired

Known limits (read before trusting a "storm" claim from this tool -- there
isn't one): no validated storm/rain/wind feature exists yet. An ad-hoc
"whole-frame motion fraction" idea was tried 2026-08-29 against the 4 clips
the user's notes flagged as storms (cam15/9944, 9954, 9956, 18948) and did
NOT separate them from guard/environment clips (some guard/environment clips
scored higher on the same metric). Storm clips fall into `unclassified` here,
not a dedicated category -- inventing one without a validated feature would
be a guess dressed up as a result. Every rule above also comes from a sample
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
)


def classify(features: dict[str, float] | None) -> str:
    """Pure rule lookup -- see the module docstring for what each rule means and
    where its thresholds come from. `guard_candidate` is checked first: the
    green-light exclusion in `animal_or_incident_candidate` is what raises that
    rule's precision, so a clip matching both is a guard, not a double-count."""
    if features is None:
        return "no_motion"
    if features["green_light_ratio"] > 0.05 or features["green_light_flicker"] > 0.02:
        return "guard_candidate"
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
        features = extract_fn(clip["file_path"], camera.zone)
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
