"""Phase 0c screening: apply the gate-2 heuristic rules across every downloaded
clip -- labelled or not -- to surface animal/incident/guard candidates for a
human to review. This is a screening tool, not a classifier: every threshold
below is one already validated in docs/gate2_separability_finding.md against
real labelled footage, not invented for this script.

Rules:
  - guard_candidate: green_light_ratio > 0.05 or green_light_flicker > 0.02
    (measured 2026-08-31 against the current tracker/detector on 200 labelled
    guard clips: 8.7% recall for THIS RULE ALONE, degraded from the doc's
    original 61% -- see "DAYLIGHT GATE IS SELF-DEFEATING" in repo memory. Left
    unchanged anyway: every alternative gate design tried and rejected
    2026-08-30, see "Daylight-gate fix investigated and ABANDONED" in repo
    memory. NOTE: this rule's own weakness is no longer the system's guard
    recall -- measured 2026-09-07 across 376 labelled guard clips, classify()
    reaches 71.8% guard recall overall, of which the inside-only rule supplies
    34.8%, the warmup-flashlight rule 29.5%, and this rule only 7.4%.)
  - guard_candidate (warmup flashlight): warmup_flashlight_ratio > 0.002
    (added 2026-09-06). Fixes the dominant false-candidate mode found by
    reviewing the first ranked queue by hand: the guard walks out of shot
    BEFORE the IR gain settles, so every frame they appear in is dropped as
    flare, and the scored frames contain only whatever moved next -- a vine, a
    wind-blown bush, or a camera artifact on the final frame. Eight of the
    eleven reviewed false candidates had exactly ONE genuine detection frame,
    several of them the last scored frame.
    `warmup_flashlight_ratio` scores the DROPPED frames for the flashlight (see
    scripts.spike.extract_clip_features), which no other feature looks at.
    Measured over the whole labelled corpus, with the same daylight gate the
    other green-light features use: animal 11/11 exactly 0.0, incident max
    0.00046, guard median 0.00049 and max 0.154. A 0.002 threshold sits 4.3x
    above the highest positive and flags 118/283 guards with **0/21
    animal+incident leak**.
    Placed with the other guard rule, ABOVE the animal/incident geometry rule,
    because these clips DO pass that geometry test -- that is exactly why they
    reached the queue. Known risk, accepted: a genuine incident that a guard
    responds to within the same clip would be routed to guard. The existing
    green_light rule above already carries that same risk, and the 4.3x margin
    is the mitigation; revisit if a labelled incident ever exceeds 0.002.
  - environment_candidate: blob_count > 10 (added 2026-08-31, checked before
    animal_candidate/incident_candidate/insect_candidate so a stormy/windy clip's
    scattered foliage blobs don't get read as a shape signal. Measured against
    222 labelled guard+environment clips: 59% environment recall (13/22), 4.5%
    guard false-fire (9/200), 0/15 leak against the full labelled animal+incident
    set -- blob_count is the strongest single discriminator found for this pair,
    AUC 0.933. Higher thresholds trade recall for guard false-fire; 10 is the
    highest threshold with zero animal+incident leak (cam15/15454's porcupine
    sits at exactly blob_count=10).)
  - environment_candidate (sustained scattered motion): blob_count_median > 4
    (added 2026-09-07). The blob_count rule above is a PEAK over frames, which
    is what a storm needs but also fires on a single flare-settle frame at the
    start of an otherwise quiet clip -- a bug diagnosed but left unfixed on
    2026-09-05 (cam04/10887 reads [16, 5, 5, 4, 3...], cam03/18512 reads
    [20, 1, 1, 1, ...]). Rather than change the peak rule's already-validated
    threshold, the median counterpart is added alongside it: it only rises when
    the scattered motion actually PERSISTS, which is what wind does and a
    settle transient does not. Measured at event level on the 678-clip / 351-
    event labelled corpus, through this exact rule order: guard leak into the
    alert channel 39 -> 32 events, environment leak 16 -> 9, with 5/5 incident
    events, 7/7 incident CLIPS, 19 animal events and 15/15 animal clips all
    unchanged. 4 is the tightest threshold with zero per-clip protected-class
    cost: at 3 and below, cam08/4054 (blob_count_median 4.0, "man putting on
    backup, intruder, outside fence") is gated, and while its sibling 4055
    still carries the event, a live system sees one clip at a time -- so the
    per-clip margin is what the threshold is chosen on, not the event one.
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
    GATED FIRST, added 2026-09-07, by two conditions that both say "whatever
    is outside the fence here is not a subject" -- and both were chosen on
    PER-CLIP protected-class margin, not event-level survival:
      * `blob_white_fraction >= 0.4` -- a blinded lens never alerts. This is
        the same threshold, deliberately, that `is_blinding_foreground` uses
        for the maintenance queue; the finding is that when a bright
        vegetation/web obstruction is against the lens, the "outside blob" IS
        the obstruction. The flag stays orthogonal for reporting (a blinded
        clip can still be a correct guard_candidate), this only stops it
        reaching the ALERT channel. Worst real value among clips that
        currently alert: incident 0.310 (cam08/4054), animal 0.182
        (cam10/18787) -- so 0/7 incident and 0/15 animal clips are affected.
        Alone: guard leak 39 -> 32 events, environment 16 -> 10.
      * `motion_pixel_fraction_median > 0.12` -- sustained whole-frame motion
        is wind or an unsettled IR ramp, not a compact intruder. The peak
        `motion_pixel_fraction` was measured first and REJECTED as the gate:
        it is the single strongest discriminator in this population (AUC
        0.236, i.e. 0.764 inverted) but its worst real incident clip sits at
        0.123 against a guard/environment median of 0.172, and any threshold
        tight enough to be useful also costs an animal event. The median is
        both safer and free -- worst incident 0.0833 (cam08/4055), worst
        animal 0.0015 (cam01/16027), a 100x gap on the animal side. Alone:
        guard leak 39 -> 33 events, environment 16 -> 11.
    NOTE the `long_flare_frames >= 18` half of `is_blinding_foreground` was
    tried as a third gate here and DELIBERATELY NOT SHIPPED: it buys only 1-2
    more events, and cam06/21520 -- the crawl incident, the signature threat
    this system exists to catch -- sits at 15, a 20% margin. Same reasoning
    that rejected the `metric_aspect` gate below. It remains a maintenance
    signal only.
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
    UPPER-BOUNDED at `MEDIAN_FENCE_DISTANCE_MAX = 0.40` (added 2026-09-07,
    user decision -- see docs/plan.md's 2026-09-07 checkpoint): a real
    intruder approaches the fence (the 10 incident clips run 0.117-0.353,
    comfortably under 0.40) while the environment clips leaking through this
    rule sit out in the field (leaking-environment median 0.406, AUC 0.840
    separating the two). Measured through this exact rule order on the full
    labelled corpus: environment leak into this channel drops 29->14 clips,
    all 5 incident EVENTS still alert (per-clip 7/10 unaffected by the
    bound), and the only cost is one animal EVENT with no sibling clip to
    cover it, `cam10/7632` (median_fence_distance 0.504, itself noted
    "visually marginal/hard to confirm, enters during IR flare"). Accepted
    -- breaks the standing "never lose an animal" constraint for this one
    known clip, a deliberate, explicit tradeoff, not a silent regression.
    ANIMAL_CANDIDATE branch additionally UPPER-BOUNDED at `ANIMAL_ROW_AREA_MAX
    = 3000` on `row_normalised_area` (added 2026-09-07, after the user
    labelled a fresh review batch and specifically flagged "animal blobs are
    small, especially further away, where branches pick up larger"). Measured
    on the re-grown labelled corpus (571 clips): real animal median
    `row_normalised_area` is 666 vs environment-that-would-otherwise-read-
    animal_candidate's median 10471 -- a >15x gap. Deliberately NEVER applied
    to incident_candidate: the same signal does NOT separate incident from
    environment there (incident median 10704, almost identical to
    environment's), and checking it directly against real incident clips
    showed 2-6/7 would be wrongly gated at every threshold tried -- touching
    that branch would violate docs/plan.md's "shape may never suppress an
    outside alert" policy for a real reason, not just caution. On the
    animal_candidate side only: redirects 15/19 (79%) of environment clips
    that would otherwise misread as animal_candidate, at the highest
    threshold with ZERO real animal clips lost (0/10) -- verified by sweeping
    500 to 10000, animal loss only appears below 3000.
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
  - guard_candidate (inside-only fallback) / resident_candidate: zone_classifiable_fraction
    > 0 and outside_pixel_fraction == 0.0 (added 2026-09-06, split by real
    `is_daylight` 2026-09-07). Deliberately the LAST rule, so it can only ever
    reclassify what would otherwise fall through to `unclassified` -- it
    cannot override any positive classification above. The guard patrols
    INSIDE the fence, so a blob the zone geometry actually classified, and
    classified entirely inside, is a guard rather than an unknown -- UNLESS
    it happens in real daylight, in which case a resident going about their
    business is at least as likely as a night patrol, so it reads
    `resident_candidate` instead. Requires zone_classifiable_fraction > 0
    because outside_pixel_fraction returns 0.0 for BOTH "all inside" and
    "nothing was classifiable" (whole blob in an ignore region / beyond
    depth_cutoff) -- without that guard this rule would confidently suppress
    blobs it never actually classified.
    Measured at EVENT level on the 416-clip labelled corpus (217 events,
    grouped by shared physical trigger via scripts.label._event_key, since a
    startup_state=blank precursor clip is expected to read as nothing on its
    own): 0/5 incident events lost, 145/193 currently-unclassified guard clips
    become a confident guard_candidate. This is why the event-level view
    matters -- per clip, 3 of 10 incident clips DO read fully inside
    (cam06/21519, cam09/21521, cam10/21523), but in all three cases the
    sibling clip of the same event reads fully outside, so no incident event
    is lost.
    The daylight split deliberately uses the REAL `is_daylight(timestamp)`
    signal (threaded into `features["is_daylight"]` by every caller of
    `classify()`, not derived from the image), NOT `color_fraction` -- checked
    first and rejected: `color_fraction > 0.15` on this same rule's hits would
    reclassify 85/132 (64%) of currently-correct guard clips as resident,
    because a guard's own flashlight raises whole-frame saturation the same
    way ambient daylight does (the same "DAYLIGHT GATE IS SELF-DEFEATING"
    confound as `green_light_ratio`'s own gate). With the real sun-time
    signal instead, only 9/132 (6.8%) of guard's rule-7 hits are real
    daylight, and both outcomes here are suppressed/non-alerting categories
    either way -- this is a labelling-quality improvement with zero change to
    what alerts. Only 2 of resident's 6 rule-7 hits are real daylight (n=2,
    explicitly a LEAD not a certified split -- residents are seen at night
    too, this only catches the subset that happens to be daylight).

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

`is_blinding_foreground` (added 2026-09-06) is a SEPARATE, independent flag,
not part of the category chain above: a bright vegetation/web obstruction
against the lens (`blob_white_fraction >= 0.4`) or an IR ramp that never
naturally settled (`long_flare_frames >= 18`) means the camera needs physical
cleaning, regardless of whatever category the clip also gets -- a guard can
still be genuinely present and correctly `guard_candidate` in the very same
clip. See its own docstring for the measured zero-leak numbers.

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
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.spike import FEATURE_COLUMNS, extract_clip_features  # noqa: E402
from src import db  # noqa: E402
from src.config import CamerasConfig, load_app_config, load_cameras_config  # noqa: E402
from src.features import is_daylight  # noqa: E402
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
    "blinding_foreground",
)
_NON_NUMERIC = ("channel_id", "message_id", "camera_id", "label", "time_of_day", "is_daylight")
REPORT_COLUMNS = (
    *_IDENTITY_COLUMNS,
    *(c for c in FEATURE_COLUMNS if c not in _NON_NUMERIC),
)

# See classify()'s docstring ("animal_candidate / incident_candidate") for how this was measured.
MEDIAN_FENCE_DISTANCE_MAX = 0.40
# See classify()'s docstring for how this was measured -- only bounds the
# animal_candidate branch, deliberately never incident_candidate.
ANIMAL_ROW_AREA_MAX = 3000.0
# See classify()'s "environment_candidate (sustained scattered motion)" note.
BLOB_COUNT_MEDIAN_MAX = 4.0
# See classify()'s "sustained whole-frame motion" note.
MOTION_PIXEL_FRACTION_MEDIAN_MAX = 0.12
# See classify()'s "blinded lens never alerts" note. Same threshold as
# is_blinding_foreground()'s own, deliberately -- one obstruction definition.
BLINDING_BLOB_WHITE_FRACTION = 0.4


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
    if features.get("warmup_flashlight_ratio", 0.0) > 0.002:
        return "guard_candidate"
    if features["blob_count"] > 10:
        return "environment_candidate"
    if features.get("blob_count_median", 0.0) > BLOB_COUNT_MEDIAN_MAX:
        return "environment_candidate"
    if (
        features.get("uncalibrated", 1.0) == 0.0
        and features.get("implausible_height_fraction", 0.0) > 0.5
    ):
        return "environment_candidate"
    if (
        features["outside_pixel_fraction"] > 0.6
        and features["median_fence_distance"] > 0.1
        and features["median_fence_distance"] < MEDIAN_FENCE_DISTANCE_MAX
    ):
        if features.get("blob_white_fraction", 0.0) >= BLINDING_BLOB_WHITE_FRACTION:
            return "environment_candidate"
        if (
            features.get("motion_pixel_fraction_median", 0.0)
            > MOTION_PIXEL_FRACTION_MEDIAN_MAX
        ):
            return "environment_candidate"
        if features["color_fraction"] > 0.15:
            if features.get("row_normalised_area", 0.0) > ANIMAL_ROW_AREA_MAX:
                return "environment_candidate"
            return "animal_candidate"
        return "incident_candidate"
    if features["jitter"] > 50 and features["solidity"] < 0.85:
        return "insect_candidate"
    if (
        features.get("zone_classifiable_fraction", 0.0) > 0.0
        and features["outside_pixel_fraction"] == 0.0
    ):
        return "resident_candidate" if features.get("is_daylight", False) else "guard_candidate"
    return "unclassified"


def is_blinding_foreground(features: dict[str, float] | None) -> bool:
    """True if a bright obstruction (vegetation, a web) right against the lens
    is dominating the tracked blob -- a maintenance signal (clean the camera),
    independent of and orthogonal to `classify()`'s category: a clip can be
    BOTH a real guard sighting AND blinding (e.g. the guard is still visible
    via their flashlight past the obstruction), so this is never folded into
    the mutually-exclusive category chain above.

    Measured 2026-09-06 on the full labelled corpus: `blob_white_fraction`'s
    max ever seen is 0.182 for animal and 0.310 for incident, both well under
    0.4; `long_flare_frames` (the ABSOLUTE frame count the IR ramp never
    settled within, not the fraction -- that's capped at 40% of every clip
    and common everywhere) crossing 18 is similarly rare outside this pattern.
    Combined: 7/10 of a hand-picked blinding debug set caught, ZERO leak into
    animal or incident, 14.5% guard / 29.3% environment / 20% resident false-fire.
    """
    if features is None:
        return False
    return features.get("blob_white_fraction", 0.0) >= 0.4 or (
        features.get("long_flare_frames", 0.0) >= 18
    )


# Real alert classes always win over every suppressed one, per docs/plan.md's
# own rule ("nothing about blob shape may downgrade or suppress an outside
# alert"). Below those, the order is a reasonable-but-uncertified default for
# reporting only.
_EVENT_CATEGORY_PRIORITY = (
    "incident_candidate",
    "animal_candidate",
    "environment_candidate",
    "insect_candidate",
    "guard_candidate",
    "resident_candidate",
    "unclassified",
    "no_motion",
)


def classify_event(categories: Iterable[str]) -> str:
    """Reduce every sibling clip's own `classify()` category for one physical
    event (an "(Initial*)"/"(Stopped*)" pair sharing one embedded timestamp,
    see `scripts.label._event_key`) down to a single event-level verdict, so
    a truncated preview clip's weaker read can never hide what a fuller
    sibling actually shows (measured 2026-09-07: cam01a/18603-18604 and
    cam08/10852-10853 both had the short preview alone read `incident_
    candidate` while the full clip is plainly `guard_candidate`).

    `incident_candidate`/`animal_candidate` always win, matching this repo's
    own alerting rule that shape/trajectory may never suppress an outside
    alert -- so this can only ever RAISE an event's verdict toward the alert
    channel relative to any single clip, never lower it. This function is
    reporting/analysis-only; it is never called from the live per-clip
    `classify()` path, since a real system sees clips one at a time and
    can't know a sibling's category before it exists.

    Empty input (an id with no clips at all) returns "no_motion", matching
    `classify()`'s own convention for "nothing to go on".
    """
    seen = set(categories)
    if not seen:
        return "no_motion"
    for category in _EVENT_CATEGORY_PRIORITY:
        if category in seen:
            return category
    return next(iter(seen))  # defensive: an unrecognised category string


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
        features = extract_fn(clip["file_path"], camera.zone_at(clip["timestamp"]), **extra)
        if features is not None and clip["timestamp"] is not None:
            # classify() needs the real exogenous signal, not an image
            # statistic -- see the module docstring's resident_candidate note.
            features["is_daylight"] = is_daylight(clip["timestamp"])
        row = {
            "channel_id": clip["channel_id"],
            "message_id": clip["message_id"],
            "camera_id": clip["camera_id"],
            "label": clip["label"],
            "category": classify(features),
            "blinding_foreground": is_blinding_foreground(features),
        }
        for col in REPORT_COLUMNS[6:]:
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
    conn.close()

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    from collections import Counter

    print(Counter(r["category"] for r in rows))


if __name__ == "__main__":
    main()
