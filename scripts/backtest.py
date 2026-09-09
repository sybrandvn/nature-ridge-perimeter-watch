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
import json
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
# `is_daylight` is deliberately NOT in this list (fixed 2026-09-09) -- it is
# the real exogenous signal classify_detailed() reads to split
# resident_candidate from guard_candidate (see run_backtest below), and
# dropping it from REPORT_COLUMNS/features_json meant no recorded backtest
# row could be replayed faithfully: re-running classify_detailed() on a
# stored features_json would silently default every clip to features.get(
# "is_daylight", False) == night. Kept a plain float (0.0/1.0), matching
# every other REPORT_COLUMNS value's type, rather than a bare bool.
_NON_NUMERIC = ("channel_id", "message_id", "camera_id", "label", "time_of_day")
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
            # Cast to float so it round-trips through REPORT_COLUMNS/CSV/
            # features_json the same way every other feature does.
            features["is_daylight"] = float(is_daylight(clip["timestamp"]))
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


# The alert channel: every category the routing plan actually pages someone
# for. Kept here rather than imported from src.classify.SUPPRESSED-style
# constant because this is this SCRIPT's own reporting vocabulary -- see
# docs/plan.md's Ground truth labels / Ship readiness sections for why
# "incident"/"animal" are positive and everything else labelled is negative.
ALERT_CATEGORIES = frozenset({"incident_candidate", "animal_candidate"})
POSITIVE_LABELS = frozenset({"incident", "animal"})


def summarize_labelled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Real metrics against ground truth, added 2026-09-09 because nothing in
    this repo previously compared `category` to the `label` column every row
    already carries -- every precision/recall/AUC number quoted anywhere in
    docs/handoff.md or docs/gate2_separability_finding.md was computed by hand,
    out of band, in a session that survives only as a doc entry now. This is
    per-clip (matching what a live system actually sees, one clip at a time --
    see docs/detection_improvement_review.md's own caution about event-level
    figures overstating what a live system gets), restricted to rows carrying
    a real human `label`.

    Returns a JSON-safe dict: `confusion` (label -> category -> count),
    `alert_channel` (precision/recall/f1 plus raw counts, positive =
    incident+animal, negative = every other labelled category), and
    `per_camera_leak` (for guard and environment specifically: how many of
    that camera's own labelled clips reach the alert channel).
    """
    labelled = [r for r in rows if r.get("label")]
    confusion: dict[str, dict[str, int]] = {}
    for row in labelled:
        by_label = confusion.setdefault(row["label"], {})
        by_label[row["category"]] = by_label.get(row["category"], 0) + 1

    tp = fp = fn = tn = 0
    for row in labelled:
        alerted = row["category"] in ALERT_CATEGORIES
        positive = row["label"] in POSITIVE_LABELS
        if positive and alerted:
            tp += 1
        elif positive and not alerted:
            fn += 1
        elif not positive and alerted:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_camera_leak: dict[str, dict[str, Any]] = {}
    for label in ("guard", "environment"):
        by_camera: dict[str, list[int]] = {}
        for row in labelled:
            if row["label"] != label:
                continue
            counts = by_camera.setdefault(row["camera_id"], [0, 0])
            counts[0] += 1
            if row["category"] in ALERT_CATEGORIES:
                counts[1] += 1
        per_camera_leak[label] = {
            camera_id: {
                "n": n,
                "leaked": leaked,
                "leak_rate": round(leaked / n, 4) if n else 0.0,
            }
            for camera_id, (n, leaked) in sorted(by_camera.items())
        }

    return {
        "n_labelled": len(labelled),
        "confusion": confusion,
        "alert_channel": {
            "positive_labels": sorted(POSITIVE_LABELS),
            "alert_categories": sorted(ALERT_CATEGORIES),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "per_camera_leak": per_camera_leak,
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Console rendering of summarize_labelled()'s output -- kept separate so
    a caller (or a test) can get the structured dict without the print noise.
    """
    print(f"\n--- confusion matrix ({summary['n_labelled']} labelled clips) ---")
    for label in sorted(summary["confusion"]):
        by_category = summary["confusion"][label]
        n = sum(by_category.values())
        breakdown = ", ".join(
            f"{cat}={n_cat}" for cat, n_cat in sorted(by_category.items(), key=lambda kv: -kv[1])
        )
        print(f"  {label:12s} n={n:<4d} {breakdown}")

    alert = summary["alert_channel"]
    print(
        f"\n--- alert channel (positive={'+'.join(alert['positive_labels'])}, "
        f"alert={'+'.join(alert['alert_categories'])}) ---"
    )
    print(
        f"  tp={alert['tp']} fp={alert['fp']} fn={alert['fn']} tn={alert['tn']}  "
        f"precision={alert['precision']:.3f} recall={alert['recall']:.3f} f1={alert['f1']:.3f}"
    )

    for label, by_camera in summary["per_camera_leak"].items():
        leaking = {cid: v for cid, v in by_camera.items() if v["leaked"] > 0}
        if not leaking:
            print(f"\n--- {label} leak into the alert channel: none ---")
            continue
        print(f"\n--- {label} leak into the alert channel, by camera ---")
        for camera_id, v in sorted(leaking.items(), key=lambda kv: -kv[1]["leak_rate"]):
            print(f"  {camera_id:8s} {v['leaked']:4d}/{v['n']:<4d} = {v['leak_rate'] * 100:5.1f}%")


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

    summary = summarize_labelled(rows)
    if summary["n_labelled"]:
        print_summary(summary)
        summary_path = Path(args.out).with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"Wrote metrics summary to {summary_path}")
    else:
        print("\nNo labelled rows in this run -- skipping the confusion matrix/metrics summary.")


if __name__ == "__main__":
    main()
