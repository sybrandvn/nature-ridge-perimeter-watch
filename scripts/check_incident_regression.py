"""Regression check: every INCIDENT EVENT (hard) and ANIMAL EVENT (hard, with a
small, explicit, documented exception list) in tests/fixtures/incident_regression.jsonl
must have at least one clip that src.classify.classify does not route to a
suppressed category.

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

SUPPRESSED = {"guard_candidate", "resident_candidate", "environment_candidate", "no_motion",
"unclassified"} -- these are the categories the routing plan suppresses from
the alert channel entirely. Every incident EVENT must have at least one clip
that avoids all of them; this script is the automated guardrail for that, so a
threshold change that quietly drops one is caught immediately rather than
discovered against live footage later.

Animal events get the SAME hard check, per docs/plan.md's Ship readiness
criterion #2 ("no labelled animal event is silently dropped ... requires an
explicit, documented exception, not a silent regression"). Until 2026-09-09
this script only asserted on incident events -- the fixture's 11 animal and
10 resident rows were loaded, classified and printed, but never checked, so an
animal regression could pass silently. Wiring the assertion up surfaced three
previously-invisible failures beyond the one (cam10/7632) that was already a
documented, deliberate tradeoff. All three are now fixed by measured recovery
rules in src.classify (2026-09-16):

  - cam10/7632 (event cam10-2024-09-14T16:24): median_fence_distance=0.504,
    just above MEDIAN_FENCE_DISTANCE_MAX=0.40. Already a deliberate, signed-off
    exception -- see docs/plan.md's 2026-09-07 checkpoint and src/classify.py's
    docstring. Listed in KNOWN_ANIMAL_EXCEPTIONS below.
  - cam10/9405 (event cam10-2024-12-01T16:12): median_fence_distance=0.097,
    just BELOW MEDIAN_FENCE_DISTANCE_MIN=0.10 -- an animal standing at the
    fence, not past it. Fixed by the narrow daylight/colour/compact
    near_fence_animal branch rather than globally lowering the standard floor.
    Its compact-blob ceiling excludes the labelled wind clip cam15/9644, so it
    recovers four labelled animal clips with no labelled environment alert.
  - cam15-2025-07-15T00:29 (15453+15454, the porcupine): both clips read
    outside_pixel_fraction=0.5 (single-track geometry), exactly the fence
    line, never a clean majority. The rejected multi-object OR gate remains
    rejected. Fixed instead by fence_straddle_no_colour: track-level outside
    frames + an actual fence crossing corroborate the 50/50 best silhouette,
    with compact/low-scene-motion night gates. It is the only newly alerting
    clip in the 16,886-clip corpus after earlier rules.
  - cam10/17146 (bird on the fence rail): outside_pixel_fraction=0.0 AND the
    multi-object reading agrees (0.0), while implausible_height_fraction=0.8.
    Fixed by inside_elevated_animal before the generic metric environment gate:
    a calibrated, compact, colour-bearing daylight subject can violate the
    ground plane because it is perched. Across the full corpus this reaches
    only the bird's two clips and one other labelled animal clip.

None of those three needed an exception. cam10/7632 remains the one explicit,
previously signed-off animal-event exception below.

Also fixed 2026-09-09: this check now passes a reference_background, matching
scripts/backtest.py and scripts/rank_candidates.py -- previously it was the
only one of the three real callers scoring clips through a fresh-median-only
detector, so it validated a configuration nobody actually ran.

Run:
    uv run python scripts/check_incident_regression.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db  # noqa: E402
from src.classify import classify  # noqa: E402
from src.config import load_app_config, load_cameras_config  # noqa: E402
from src.features import daylight_hint, is_daylight  # noqa: E402
from src.reference_bg import (  # noqa: E402
    era_of,
    load_manifest,
    load_reference_image,
    reference_for,
)
from src.scoring import extract_clip_features  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "incident_regression.jsonl"
REFERENCE_BG_ROOT = Path(__file__).resolve().parents[1] / "data" / "reference_bg"
SUPPRESSED = {
    "guard_candidate",
    "resident_candidate",
    "environment_candidate",
    "no_motion",
    "unclassified",
}

# Deliberate, individually-signed-off exceptions to the animal-event hard
# check -- an event key added here must carry its own dated rationale, same
# discipline docs/plan.md already applies to cam10/7632 (MEDIAN_FENCE_DISTANCE_MAX,
# 2026-09-07). Never add an entry here just to make this script pass; every
# entry is a documented, deliberate acceptance of one specific known miss, not
# a way to silence a new one.
KNOWN_ANIMAL_EXCEPTIONS: dict[str, str] = {
    "cam10-2024-09-14T16:24": (
        "cam10/7632: median_fence_distance 0.504 sits just above "
        "MEDIAN_FENCE_DISTANCE_MAX=0.40. Accepted 2026-09-07 (docs/plan.md's "
        "checkpoint of that date) -- confirmed the animal genuinely was that "
        "far from the fence (recomputed from genuine, non-recovered centroids "
        "only: 0.498, barely moves), and no persistence/path_length "
        "combination recovers it without re-admitting 6-15 guard/environment "
        "clips per animal clip saved."
    ),
}


def _reference_background(entries, camera, timestamp):
    """Same resolution scripts/backtest.py uses -- see its own helper of the
    same name. Duplicated rather than imported because scripts/backtest.py's
    version is a module-private helper tied to its own CLI args; this script
    has no CLI, so the defaults (data/reference_bg, era_of/reference_for's own
    fallback-to-None-across-era-or-sparsity behaviour) are hardcoded instead."""
    if not entries or timestamp is None:
        return None
    entry = reference_for(
        entries,
        camera.id,
        timestamp,
        era=era_of(camera, timestamp),
        daylight=is_daylight(timestamp),
    )
    return None if entry is None else load_reference_image(REFERENCE_BG_ROOT, entry)


def main() -> None:
    fixture_rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    cameras_cfg = load_cameras_config("config/cameras.yaml")
    app_cfg = load_app_config(require_telegram=False)
    conn = db.connect(app_cfg.db_path)
    reference_entries = load_manifest(str(REFERENCE_BG_ROOT))

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
        reference = _reference_background(reference_entries, camera, entry["timestamp"])
        features = extract_clip_features(
            row["file_path"],
            zone,
            daylight_hint=daylight_hint(entry.get("timestamp")),
            reference_background=reference,
        )
        if features is not None and entry.get("timestamp"):
            features["is_daylight"] = is_daylight(entry["timestamp"])
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

    incident_failures = [
        event
        for event, suppressed_flags in event_results.items()
        if event_label[event] == "incident" and all(suppressed_flags)
    ]
    animal_failures = [
        event
        for event, suppressed_flags in event_results.items()
        if event_label[event] == "animal"
        and all(suppressed_flags)
        and event not in KNOWN_ANIMAL_EXCEPTIONS
    ]
    known_animal_exceptions_hit = [
        event
        for event, suppressed_flags in event_results.items()
        if event_label[event] == "animal"
        and all(suppressed_flags)
        and event in KNOWN_ANIMAL_EXCEPTIONS
    ]

    if incident_failures or animal_failures:
        if incident_failures:
            print(
                f"\nFAILED: {len(incident_failures)} incident EVENT(s) with every clip suppressed:"
            )
            for f in incident_failures:
                print(f"  {f}")
        if animal_failures:
            print(
                f"\nFAILED: {len(animal_failures)} animal EVENT(s) with every clip suppressed "
                "and no documented exception:"
            )
            for f in animal_failures:
                print(f"  {f}")
            print(
                "  -- either fix the underlying detection gap, or add a dated, individually"
                "\n     justified entry to KNOWN_ANIMAL_EXCEPTIONS in this script (see its"
                "\n     docstring) -- never add one just to silence this failure."
            )
        sys.exit(1)

    n_incident_events = sum(1 for lbl in event_label.values() if lbl == "incident")
    n_animal_events = sum(1 for lbl in event_label.values() if lbl == "animal")
    print(f"PASS: all {n_incident_events} incident events have >=1 non-suppressed clip")
    print(
        f"PASS: all {n_animal_events} animal events have >=1 non-suppressed clip "
        f"or a documented exception"
    )
    if known_animal_exceptions_hit:
        print(f"  ({len(known_animal_exceptions_hit)} covered by a documented exception:")
        for event in known_animal_exceptions_hit:
            print(f"     {event}: {KNOWN_ANIMAL_EXCEPTIONS[event]}")
        print("  )")


if __name__ == "__main__":
    main()
