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
    animal_candidate/incident_candidate/insect_candidate so a stormy/windy clip's
    scattered foliage blobs don't get read as a shape signal. Measured against
    222 labelled guard+environment clips: 59% environment recall (13/22), 4.5%
    guard false-fire (9/200), 0/15 leak against the full labelled animal+incident
    set -- blob_count is the strongest single discriminator found for this pair,
    AUC 0.933. Higher thresholds trade recall for guard false-fire; 10 is the
    highest threshold with zero animal+incident leak (cam15/15454's porcupine
    sits at exactly blob_count=10).)
  - environment_candidate (metric physics gate): implausible_height_fraction > 0.5
    (added 2026-09-06, checked right after blob_count, same reasoning -- a
    physically-impossible reading should route to "not a real subject" before
    the shape rules get a chance to call it animal/incident). Measured on 416
    labelled clips through src.ground_calibration (15 of 18 cameras opted in):
    incident's own worst clip sits at 0.333, so 0.5 has real margin; 0/10
    incident and 0/8 calibrated-animal leak. Catches 4/91 calibrated environment
    clips outright, TWO of which (cam12/4032, cam10/4045) were previously
    misrouted as incident_candidate -- a genuine, non-redundant catch, not a
    duplicate of blob_count (verified by checking each catch's prior category).
    Only fires for calibrated cameras (`uncalibrated == 0.0`); the 3 without a
    usable picket trace (cam01b, cam15, cam16) are untouched by this rule and
    fall through to the pixel-space rules exactly as before.
  - animal_candidate / incident_candidate: outside_pixel_fraction > 0.6 and
    median_fence_distance > 0.1, split further by color_fraction > 0.15
    (re-derived 2026-09-04, replacing the old aspect_ratio<0.95 rule -- its
    premise had fully inverted under the current tracker, see below).
    Measured on 378 labelled+detected clips (guard 264, environment 95,
    animal 9, incident 10), through the SAME rule order as classify() itself
    (i.e. only rows guard_candidate/environment_candidate didn't already
    catch): recall 26.3% -> 73.7%, guard false-fire 31.8% -> 15.9% (both a
    clear improvement over the old rule measured the same way). Cost:
    environment false-fire rose 21.1% -> 31.6%, acceptable since
    environment_candidate is checked first and already catches most of it.
    `outside_pixel_fraction` alone (AUC 0.700 animal+incident vs guard+
    environment) was NOT safe on its own -- guards routinely register as
    "outside" too (they walk close to the fence, visible through the mesh,
    and shine a flashlight across it), matching this exact predicted failure
    mode in docs/gate2_separability_finding.md. Requiring `median_fence_
    distance > 0.1` alongside it is what actually suppresses that confound.
    The animal/incident split itself is DERIVED FROM ONLY 19 TOTAL CLIPS (9
    animal, 10 incident) -- treat it as a strong LEAD, not a certified rule.
    `color_fraction > 0.15` (AUC 0.828 animal-vs-incident) separates them
    cleanly in this small sample: 0/10 incident clips exceed 0.04, so every
    incident in this corpus happened in full night IR; 7/9 animal clips
    exceed 0.15 (daylight/dusk sightings). The other rule-order note still
    applies: `aspect_ratio` itself is now USELESS for this pair (AUC 0.518,
    barely better than random) -- do not reuse it, that premise inverted for
    real this time, confirmed on a much bigger sample than the original
    2026-08-31 measurement.
  - insect_candidate: jitter > 50 and solidity < 0.85
    (measured 2026-08-31: 0/22 environment clips now reach jitter>50 under the
    current tracker -- this rule is effectively dead, the persistent-tracking
    work from 2026-08-30 smooths out the erratic jitter it used to key on.
    Left in place, not removed, pending a decision on a replacement.)
  - guard_candidate (inside-only fallback): zone_classifiable_fraction > 0 and
    outside_pixel_fraction == 0.0 (added 2026-09-06). Deliberately the LAST
    rule, so it can only ever reclassify what would otherwise fall through to
    `unclassified` -- it cannot override any positive classification above.
    The guard patrols INSIDE the fence, so a blob the zone geometry actually
    classified, and classified entirely inside, is a guard rather than an
    unknown. Requires zone_classifiable_fraction > 0 because
    outside_pixel_fraction returns 0.0 for BOTH "all inside" and "nothing was
    classifiable" (whole blob in an ignore region / beyond depth_cutoff) --
    without that guard this rule would confidently suppress blobs it never
    actually classified.
    Measured at EVENT level on the 416-clip labelled corpus (217 events,
    grouped by shared physical trigger via scripts.label._event_key, since a
    startup_state=blank precursor clip is expected to read as nothing on its
    own): 0/5 incident events lost, 145/193 currently-unclassified guard clips
    become a confident guard_candidate. This is why the event-level view
    matters -- per clip, 3 of 10 incident clips DO read fully inside
    (cam06/21519, cam09/21521, cam10/21523), but in all three cases the
    sibling clip of the same event reads fully outside, so no incident event
    is lost.

REJECTED, do not re-add without new data: gating the HIGH-priority incident
channel on `metric_aspect` (height/width in metres). It looked strong --
AUC 0.816 separating the 5 alerting incident events from the 49 alerting
guard+environment ones, and a `metric_aspect < 1.0` gate would have cut
false alerts in the incident channel from 25 events to 16 while keeping all
5 incidents. It was rejected because the LOWEST incident values are the
crawling ones -- cam10/21524 "2 men crawling away" at 1.08, cam09/21522
"2 men crawling and shuffling toward the fence" at 1.14 -- i.e. crawlers sit
at the very bottom of the range, only 0.08 above the threshold. Crawling
under the fence is the signature threat this system exists to catch, so a
thin margin there is not worth a 36% false-alert reduction.
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
    "zone_classifiable_fraction",
    "median_fence_distance",
    "color_fraction",
    "path_length",
    "blob_count",
    "implausible_height_fraction",
    "off_plane_fraction",
    "height_consistency",
    "depth_progression",
    "depth_range_m",
    "subject_height_m",
    "subject_width_m",
    "subject_area_m2",
    "metric_aspect",
    "distance_median_m",
    "speed_mps",
    "uncalibrated",
)


def classify(features: dict[str, float] | None) -> str:
    """Pure rule lookup -- see the module docstring for what each rule means and
    where its thresholds come from. `guard_candidate` is checked first since
    most rules below assume a real flashlight sighting has already been pulled
    out. `environment_candidate` (both the blob_count and the metric physics
    gate) is checked next, before the shape-based rules, so a stormy/windy
    clip's scattered blobs -- or a geometrically-impossible reading -- don't
    get read as a shape signal."""
    if features is None:
        return "no_motion"
    if features["green_light_ratio"] > 0.05 or features["green_light_flicker"] > 0.02:
        return "guard_candidate"
    if features["blob_count"] > 10:
        return "environment_candidate"
    if (
        features.get("uncalibrated", 1.0) == 0.0
        and features.get("implausible_height_fraction", 0.0) > 0.5
    ):
        return "environment_candidate"
    if features["outside_pixel_fraction"] > 0.6 and features["median_fence_distance"] > 0.1:
        return "animal_candidate" if features["color_fraction"] > 0.15 else "incident_candidate"
    if features["jitter"] > 50 and features["solidity"] < 0.85:
        return "insect_candidate"
    if (
        features.get("zone_classifiable_fraction", 0.0) > 0.0
        and features["outside_pixel_fraction"] == 0.0
    ):
        return "guard_candidate"
    return "unclassified"


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


def run_backtest(
    conn: Any,
    cameras: CamerasConfig,
    *,
    camera_id: str | None = None,
    labelled_only: bool = False,
    extract_fn: ExtractFn = extract_clip_features,
) -> Iterator[dict[str, Any]]:
    unknown_cameras: set[str] = set()
    for clip in iter_clips_with_files(conn, camera_id=camera_id, labelled_only=labelled_only):
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
    parser.add_argument(
        "--labelled-only",
        action="store_true",
        help="Restrict to clips with a human label -- for before/after regression"
        " runs against the ~400-clip labelled corpus instead of all downloaded history",
    )
    args = parser.parse_args()

    app_cfg = load_app_config(require_telegram=False)
    cameras_cfg = load_cameras_config("config/cameras.yaml")
    conn = db.connect(app_cfg.db_path)

    rows = list(
        run_backtest(conn, cameras_cfg, camera_id=args.camera, labelled_only=args.labelled_only)
    )
    conn.close()

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    from collections import Counter

    print(Counter(r["category"] for r in rows))


if __name__ == "__main__":
    main()
