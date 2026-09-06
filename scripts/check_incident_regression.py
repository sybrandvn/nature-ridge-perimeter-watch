"""Regression check: every INCIDENT EVENT in
tests/fixtures/incident_regression.jsonl must have at least one clip that
scripts.backtest.classify does not route to a suppressed category.

Checked per EVENT, not per clip: the label schema shares one label across every
clip from the same physical trigger (see docs/plan.md "Ground truth labels"),
and a short "startup_state=blank" precursor clip is EXPECTED to classify as
no_motion/unclassified on its own even for a genuine incident -- its paired
longer clip is what actually carries the detectable motion. Confirmed on this
exact fixture 2026-09-06: cam06/21519, cam09/21521 and cam10/21523 (three blank
or near-blank precursors) all read unclassified while their siblings
(21520/21522/21524) correctly read incident_candidate -- a per-clip check would
misreport this as 3 failures when event-level recall is actually 5/5.

Not a pytest test -- it needs real downloaded footage (data/history/, gitignored,
not present in a fresh clone/CI), same reasoning as scripts/backtest.py's own
main(). Run this by hand after any change to classify() or its features, and
before opting any new camera in to metric_calibration.

SUPPRESSED = {"guard_candidate", "environment_candidate", "no_motion",
"unclassified"} -- these are the categories the routing plan suppresses from
the alert channel entirely. Every incident EVENT must have at least one clip
that avoids all of them; this script is the automated guardrail for that, so a
threshold change that quietly drops one is caught immediately rather than
discovered against live footage later.

Run:
    uv run python scripts/check_incident_regression.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backtest import classify  # noqa: E402
from scripts.spike import extract_clip_features  # noqa: E402
from src import db  # noqa: E402
from src.config import load_app_config, load_cameras_config  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "incident_regression.jsonl"
SUPPRESSED = {"guard_candidate", "environment_candidate", "no_motion", "unclassified"}


def main() -> None:
    fixture_rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    cameras_cfg = load_cameras_config("config/cameras.yaml")
    app_cfg = load_app_config(require_telegram=False)
    conn = db.connect(app_cfg.db_path)

    checked = 0
    event_results: dict[str, list[bool]] = defaultdict(list)  # event -> [suppressed, ...]
    event_label: dict[str, str] = {}
    for entry in fixture_rows:
        row = conn.execute(
            "select file_path from clips where camera_id=? and message_id=?",
            (entry["camera_id"], entry["message_id"]),
        ).fetchone()
        if row is None or not row["file_path"]:
            print(f"SKIP  {entry['camera_id']}/{entry['message_id']}: no local file")
            continue
        camera = cameras_cfg.by_id(entry["camera_id"])
        if camera is None:
            print(f"SKIP  {entry['camera_id']}/{entry['message_id']}: unknown camera")
            continue
        zone = camera.zone_at(entry["timestamp"])
        features = extract_clip_features(row["file_path"], zone)
        category = classify(features)
        checked += 1
        suppressed = category in SUPPRESSED
        event = entry["event"]
        event_results[event].append(suppressed)
        event_label[event] = entry["label"]
        print(f"{'sup ' if suppressed else 'ok  '}  {entry['label']:9s} {entry['camera_id']:8s} "
              f"{entry['message_id']:7d}  -> {category}   ({event})")

    conn.close()
    print(f"\nchecked {checked}/{len(fixture_rows)} fixture clips, "
          f"{len(event_results)} events")

    failures = [
        event
        for event, suppressed_flags in event_results.items()
        if event_label[event] == "incident" and all(suppressed_flags)
    ]
    if failures:
        print(f"\nFAILED: {len(failures)} incident EVENT(s) with every clip suppressed:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    n_incident_events = sum(1 for lbl in event_label.values() if lbl == "incident")
    print(f"PASS: all {n_incident_events} incident events have >=1 non-suppressed clip")


if __name__ == "__main__":
    main()
